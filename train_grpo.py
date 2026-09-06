import sys
import logging
from datetime import datetime
import os
from utils.logger import setup_logger

log_file = setup_logger()
from utils.torchoptics_pin import assert_torchoptics_pinned
assert_torchoptics_pinned()  # torchOptics 가 고정 커밋(8e50d6a)이 아니면 여기서 죽는다. 최신 torchOptics 는 사전학습 BinaryNet 을 깨뜨린다.
logging.info("GRPO Training Script Initialized")

import glob
import copy
import shutil
import time
import warnings

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.utils.data import Dataset, DataLoader

import torchvision

import torchOptics.optics as tt
import torchOptics.metrics as tm

from env import BinaryHologramEnv
from optics_constants import OPTICS_META, PROP_Z   # 광학 상수 (의존성 없는 모듈; env.py 도 여기서 가져온다)
from utils.overrides import (sweepable_names, collect_overrides, check_unread,
                             apply_overrides, flatten, axes_string, run_stamp)
import json

IPS = 256
CH = 8
warnings.filterwarnings('ignore')

current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
torch.backends.cudnn.enabled = False


# ============================================================
# 설정 — 여기만 수정한다. 값을 바꿀 때는 이 딕셔너리의 기본값을 고쳐서 커밋한다.
# 주석의 대안 값은 그대로 바꿔 넣으면 실제로 적용된다.
# ============================================================
CONFIG = {
    # --- 데이터 ---
    "train_dir": '/nfs/dataset/DIV2K/DIV2K_train_HR/DIV2K_train_HR/',
    "valid_dir": '/nfs/dataset/DIV2K/DIV2K_valid_HR/DIV2K_valid_HR/',
    # 광학 상수(dx, wl, z)는 여기 없다 — env.py 의 OPTICS_META / PROP_Z 를 모든 시뮬레이션 지점이 공유한다.
    # 물리 장치가 정하는 값이라 튜닝 대상이 아니고, CONFIG 밖에 두어 주입 자체가 불가능하다.
    "padding": 0,
    "batch_size": 1,
    # --- 사전학습 BinaryNet (torchOptics 8e50d6a 에서만 동작) ---
    "pretrained_path": 'result_v/2024-12-19 20:37:52.499731_pre_reinforce_8_0.002/'
                       '2024-12-19 20:37:52.499731_pre_reinforce_8_0.002',
    # --- 환경 (env.py, PPO 시절과 동일 값). 실험의 정의(종료 조건·성공 임계)라 성능 손잡이가 아니다.
    #     바꾸면 에피소드/성공의 의미가 달라져 PPO 시절·기존 GRPO 런과 비교가 끊긴다. 스윕으로 바꿀 수는 있지만 그걸 알고 바꿀 것.
    "env": dict(max_steps=10000, T_PSNR=30.0, T_steps=1, T_PSNR_DIFF=1/4, num_samples=10000),   # T_PSNR 은 dB 실수 필드
    # --- GRPO ---
    "group_size": 16,           # 그룹 내 샘플 수 G           (예: 8, 16, 32)
    "sim_batch_size": 4,        # 시뮬레이션 서브 배치 크기    (GPU 메모리에 맞게, 예: 2, 4, 8)
    "lr": 1e-4,
    "clip_range": 0.2,          # PPO-style 클리핑
    "kl_coef": 0.04,            # KL(π_θ || π_ref) 계수
    "update_epochs": 4,         # GRPO 업데이트당 epoch 수
    "max_grad_norm": 0.5,
    "max_kl": 0.1,              # KL 이 이 값을 넘으면 그 스텝의 남은 epoch 조기 중단 (9c3a8b0 에서 하드코딩됐던 값)
    "ref_update_interval": 10,  # π_ref 갱신 주기 (에피소드)
    "grpo_update_interval": 1,  # 스텝당 GRPO 업데이트 빈도 (1 = 매 스텝)
    "mid_channels": 64,         # 정책 FCN 폭. 바꾸면 기존 체크포인트와 호환되지 않는다
    # --- 실험 정의를 바꾸는 스위치: 켜면 save_dir 이름이 바뀌어 기존 런과 섞이지 않는다 ---
    "mask_in_update": False,    # True: 업데이트 루프의 new/ref log-prob 에도 샘플링과 같은 실패 마스크를 적용 (첫 epoch 의 ratio 가 부동소수 오차 범위에서 1)
                                # False: ep2800 까지 돌린 기존 런과 같은 동작 (첫 epoch 에도 ratio = 1 - 마스크된 확률질량)
    "seed": None,               # None: 비고정 (기존 동작). 정수(예: 0)면 random/numpy/torch 시드 고정.
                                # 재개(resume) 시에는 시드가 다시 처음부터 적용돼 데이터 순서가 첫 런과 같아진다 — 재현용이지 이어달리기용이 아님
    # --- 학습 구조 (PLAN.md). "v1" = 기존 GRPO(상태당 16샘플을 각각 시뮬레이션). "v2" = 오라클 보상 + 배치 GRPO ---
    "trainer": "v1",            # "v1" | "v2".  v2 로 켜면 save_dir 이 grpo_models_v2_<policy_kind>/ 로 바뀐다
    "v2": dict(
        policy_kind="unet",         # "fcn"(v1 과 같은 5층 3x3, 수용 영역 9px) | "unet"(4단, 기본 후보) | "fno"(공유 스펙트럼 필터: PSF 회귀에 가까움, 학습 속도 비교 arm)
        feature_spec="field",       # "legacy"(26ch, v1 관측; v2 에서는 state_record 채널이 항상 0) | "field"(51ch: 복소 필드·오차·곱 특징)
        state_gate=True,            # 로짓 = h·L1 + (1-h)·L0 로 상태가 고른다 (fno 는 필수, unet/fcn 은 보조). 관측 h 를 쓰므로 답 누설 아님
        objective="grpo",           # "grpo" | "exact"(진단: Σπ(a)A(a) 직접 최대화 — GRPO 가 아니며 그 결과를 GRPO 로 보고하지 않는다)
        reward_transform="relu",    # "relu"(max(R,0): DBS 가 실제로 실현하는 스텝 이득 = 평가 지표와 같은 함수) | "raw"(비교 arm)
        adv_baseline="sample",      # "sample"(G 표본 통계 = DeepSeek GRPO 그대로) | "policy"(π_old 가중 모집단, G→∞ 극한) | "uniform"(균등 모집단)
        images_per_batch=8,         # K: 동시에 굴리는 이미지 수 (예: 4, 8, 16). 표본은 공짜지만 유효표본 수는 K 가 정한다
        group_size=64,              # G: 상태당 정책 샘플 수 (예: 16, 64, 128)
        steps_per_image=200,        # 이미지당 DBS 스텝 뒤 새 이미지로 교체 (예: 200, 600)
        update_epochs=2,            # 반복당 epoch
        minibatch_states=4,         # 미니배치에 넣는 상태 수 (GPU 메모리에 맞게)
        lr=3e-4,
        clip_range=0.2,
        kl_coef=0.04,
        max_grad_norm=0.5,
        ref_update_iters=50,        # π_ref 갱신 주기(반복). 0 = 고정 참조(초기 정책 = 엔트로피 정규화 해석). v1 과 같은 이동 참조가 기본
        entropy_coef=0.0,           # 엔트로피 보너스. A 가 z-score 라 0.01 이하는 거의 무의미 (예: 0, 0.05, 0.2)
        advance="best",             # 상태 전진: "best"(G 개 중 R>0 최고) | "sample"(정책 샘플 1개 — 시험 시와 같은 분포)
        num_iters=20000,            # 이번 실행에서 추가로 도는 반복 수 (재개 시 누적 아님)
        val_images=8,               # 고정 검증 상태 수 V (검증 이미지 앞 V 장; 뒤 절반은 val_advance_steps 만큼 진행한 '중반 상태')
        val_advance_steps=100,      # 검증 상태 절반을 무작위-채택 DBS 로 이만큼 진행
        val_every=50,
        save_every=500,
        startup_check=True,         # 시작 시 첫 상태에서 오라클 vs 실제 시뮬레이션 8픽셀 대조. 불일치면 죽는다 (폴백 없음)
        unet_base=32,               # unet 폭 (예: 16, 32)
        fno_hidden=16,              # fno 트렁크 폭
    ),
    # --- 실행 ---
    "num_episodes": 8000,
    "save_dir": "./grpo_models/",   # 파일 스위치를 켜면 접미사가 붙는다: mask_in_update → _maskfix, seed → _seed{N}.
                                    # 앞 셀 주입(스윕)이 있으면 save_dir/sweep_<축>_<시각>_j<잡ID>/ 로 갈라진다 (아래 __main__ 참조)
    "save_interval": 100,
    "resume_training": True,        # save_dir/grpo_latest.pt 가 있으면 이어서 학습
}

# 이 실험의 정의 — 앞 셀 주입(스윕)이 덮으려 하면 ValueError 로 멈춘다 (소유자 결정 2026-09-06).
# 광학 상수(OPTICS_META, PROP_Z)는 env.py 의 모듈 상수라 CONFIG 에 없고, 따라서 주입 자체가 불가능하다.
FORCED_KEYS = ("pretrained_path",)

# 앞 셀에서 주입할 때 쓰는 접두사. 점 대신 '__':  grpo__lr = 3e-5,  grpo__env__T_PSNR_DIFF = 0.5,  grpo__mask_in_update = True
# 이 접두사가 붙었는데 CONFIG 에 없는 이름은 오타로 보고 멈춘다.
SWEEP_PREFIX = "grpo__"


def build_config(injected):
    """세 층을 한 dict 에 쌓아 한 번에 적용한다:  CONFIG(파일) ← 주입(앞 셀 전역) ← 불변식(FORCED_KEYS).

    - 층마다 따로 고치지 않고 한 dict 에 모으는 이유: 우선순위가 이 함수 한 곳에서 읽히고, 최종 조합을
      통째로 기록(overrides.json)할 수 있고, 불변식을 맨 뒤에 얹을 수 있다.
    - 반환 (cfg, overrides). cfg 는 CONFIG 의 깊은 복사본, overrides 는 주입으로 실제 바뀐 키만.
    """
    names = sweepable_names(CONFIG, SWEEP_PREFIX)          # CONFIG 가 곧 목록 — 손목록 없음

    # %run 을 -i 없이 쓰면 새 네임스페이스라 앞 셀의 grpo__* 가 이 함수에 안 보인다.
    # 그 경우 조용히 기본값으로 도는 대신, 사용자 네임스페이스에 주입이 있으면 멈춘다.
    import builtins
    _get_ip = getattr(builtins, "get_ipython", None)
    if _get_ip is not None:
        _user_ns = getattr(_get_ip(), "user_ns", None)
        if _user_ns is not None and _user_ns is not injected:
            hidden = [n for n in _user_ns if n.startswith(SWEEP_PREFIX) and _user_ns[n] is not None]
            if hidden:
                raise ValueError(
                    f"앞 셀의 {hidden} 가 이 실행에는 보이지 않습니다 (%run 을 -i 없이 쓴 것으로 보임). "
                    "셀에 붙여넣거나 `%run -i train_grpo.py` 로 돌리세요."
                )

    check_unread(injected, names, SWEEP_PREFIX)            # 접두사 붙은 오타는 여기서 멈춘다
    file_values = flatten(CONFIG)
    overrides, shadowed = collect_overrides(injected, names, file_values)   # 적용은 아직

    # 트레이너 전용 키 가드: v1 전용 키를 v2 런에(또는 v2.* 를 v1 런에) 주입하면 폴더 이름에는 들어가는데 학습은 안 읽는다
    # — '스윕 축이 조용히 무시되는' 사고. 목록은 손으로 두지 않고 CONFIG 에서 만든다.
    trainer = overrides.get("trainer", file_values["trainer"])
    v2_keys = {k for k in file_values if k.startswith("v2.")}
    shared = {"train_dir", "valid_dir", "padding", "batch_size", "pretrained_path", "mid_channels", "seed",
              "trainer", "save_dir", "resume_training"}
    v1_only = {k for k in file_values if k not in v2_keys and k not in shared}
    wrong = sorted(k for k in overrides if (trainer == "v2" and k in v1_only) or (trainer != "v2" and k in v2_keys))
    if wrong:
        raise ValueError(f"trainer={trainer!r} 런에서 읽히지 않는 키가 주입됐습니다: {wrong}. "
                         "v2 는 v2.* 키만, v1 은 최상위 키만 읽습니다 (공유 키: " + ", ".join(sorted(shared)) + ").")

    # 불변식: 값은 CONFIG 의 것 그대로. 주입이 다른 값을 주면 조용히 덮지 않고 멈춘다.
    clash = {k: overrides[k] for k in FORCED_KEYS if k in overrides and overrides[k] != file_values[k]}
    if clash:
        raise ValueError(
            "이 실험의 정의를 바꾸려 했습니다: "
            + ", ".join(f"{k}={v!r} (고정값 {file_values[k]!r})" for k, v in clash.items())
            + "\n  다른 사전학습 모델로 돌리려면 CONFIG 의 기본값을 고쳐서 커밋하세요."
        )

    if shadowed:   # 바깥 값이 파일 값을 덮은 것은 의도된 순서지만, 모르고 있으면 사고가 나므로 크게 찍는다
        print("=" * 70)
        print("[!] 앞 셀에서 주입된 값이 아래 파일 CONFIG 값을 덮었습니다:")
        for key, was, now in shadowed:
            print(f"    {key}: 파일={was!r}  ->  주입={now!r}  (주입 적용)")
        print("    파일 값으로 돌리려면 앞 셀에서 그 이름을 지우세요.")
        print("=" * 70)

    cfg = copy.deepcopy(CONFIG)
    apply_overrides(cfg, overrides)                         # 마지막에 한 번. 없는 키면 KeyError
    print("[Final OVERRIDES]:", overrides if overrides else "{}  (주입 없음 - 파일 CONFIG 그대로)")
    return cfg, overrides


# ============================================================
# Pre-trained BinaryNet (홀로그램 초기값 생성용, train.py와 동일)
# ============================================================
class BinaryNet(nn.Module):
    def __init__(self, num_hologram, final='Sigmoid', in_planes=3,
                 channels=[32, 64, 128, 256, 512, 1024, 2048, 4096],
                 convReLU=True, convBN=True, poolReLU=True, poolBN=True,
                 deconvReLU=True, deconvBN=True):
        super(BinaryNet, self).__init__()

        def CRB2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                  bias=True, relu=True, bn=True):
            layers = [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size, stride=stride, padding=padding,
                                bias=bias)]
            if relu:
                layers += [nn.Tanh()]
            if bn:
                layers += [nn.BatchNorm2d(num_features=out_channels)]
            return nn.Sequential(*layers)

        def TRB2d(in_channels, out_channels, kernel_size=2, stride=2,
                  bias=True, relu=True, bn=True):
            layers = [nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=2, stride=2, padding=0, bias=True)]
            if bn:
                layers += [nn.BatchNorm2d(num_features=out_channels)]
            if relu:
                layers += [nn.ReLU()]
            return nn.Sequential(*layers)

        self.enc1_1 = CRB2d(in_planes, channels[0], relu=convReLU, bn=convBN)
        self.enc1_2 = CRB2d(channels[0], channels[0], relu=convReLU, bn=convBN)
        self.pool1 = CRB2d(channels[0], channels[0], stride=2, relu=poolReLU, bn=poolBN)

        self.enc2_1 = CRB2d(channels[0], channels[1], relu=convReLU, bn=convBN)
        self.enc2_2 = CRB2d(channels[1], channels[1], relu=convReLU, bn=convBN)
        self.pool2 = CRB2d(channels[1], channels[1], stride=2, relu=poolReLU, bn=poolBN)

        self.enc3_1 = CRB2d(channels[1], channels[2], relu=convReLU, bn=convBN)
        self.enc3_2 = CRB2d(channels[2], channels[2], relu=convReLU, bn=convBN)
        self.pool3 = CRB2d(channels[2], channels[2], stride=2, relu=poolReLU, bn=poolBN)

        self.enc4_1 = CRB2d(channels[2], channels[3], relu=convReLU, bn=convBN)
        self.enc4_2 = CRB2d(channels[3], channels[3], relu=convReLU, bn=convBN)
        self.pool4 = CRB2d(channels[3], channels[3], stride=2, relu=poolReLU, bn=poolBN)

        self.enc5_1 = CRB2d(channels[3], channels[4], relu=convReLU, bn=convBN)
        self.enc5_2 = CRB2d(channels[4], channels[4], relu=convReLU, bn=convBN)

        self.deconv4 = TRB2d(channels[4], channels[3], relu=deconvReLU, bn=deconvBN, stride=2)
        self.dec4_1 = CRB2d(channels[4], channels[3], relu=convReLU, bn=convBN)
        self.dec4_2 = CRB2d(channels[3], channels[3], relu=convReLU, bn=convBN)

        self.deconv3 = TRB2d(channels[3], channels[2], relu=deconvReLU, bn=deconvBN, stride=2)
        self.dec3_1 = CRB2d(channels[3], channels[2], relu=convReLU, bn=convBN)
        self.dec3_2 = CRB2d(channels[2], channels[2], relu=convReLU, bn=convBN)

        self.deconv2 = TRB2d(channels[2], channels[1], relu=deconvReLU, bn=deconvBN, stride=2)
        self.dec2_1 = CRB2d(channels[2], channels[1], relu=convReLU, bn=convBN)
        self.dec2_2 = CRB2d(channels[1], channels[1], relu=convReLU, bn=convBN)

        self.deconv1 = TRB2d(channels[1], channels[0], relu=deconvReLU, bn=deconvBN, stride=2)
        self.dec1_1 = CRB2d(channels[1], channels[0], relu=convReLU, bn=convBN)
        self.dec1_2 = CRB2d(channels[0], channels[0], relu=convReLU, bn=convBN)

        self.classifier = CRB2d(channels[0], num_hologram, relu=False, bn=False)

    def forward(self, x):
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)
        enc5_2 = self.enc5_2(enc5_1)

        deconv4 = self.deconv4(enc5_2)
        concat4 = torch.cat((deconv4, enc4_2), dim=1)
        dec4_1 = self.dec4_1(concat4)
        dec4_2 = self.dec4_2(dec4_1)

        deconv3 = self.deconv3(dec4_2)
        concat3 = torch.cat((deconv3, enc3_2), dim=1)
        dec3_1 = self.dec3_1(concat3)
        dec3_2 = self.dec3_2(dec3_1)

        deconv2 = self.deconv2(dec3_2)
        concat2 = torch.cat((deconv2, enc2_2), dim=1)
        dec2_1 = self.dec2_1(concat2)
        dec2_2 = self.dec2_2(dec2_1)

        deconv1 = self.deconv1(dec2_2)
        concat1 = torch.cat((deconv1, enc1_2), dim=1)
        dec1_1 = self.dec1_1(concat1)
        dec1_2 = self.dec1_2(dec1_1)

        out = self.classifier(dec1_2)
        out = nn.Sigmoid()(out)
        return out


# ============================================================
# Dataset (train.py와 동일)
# ============================================================
class Dataset512(Dataset):
    def __init__(self, target_dir, meta, transform=None, isTrain=True, padding=0):
        self.target_dir = target_dir
        self.transform = transform
        self.meta = meta
        self.isTrain = isTrain
        self.target_list = sorted(glob.glob(target_dir + '*.png'))
        self.center_crop = torchvision.transforms.CenterCrop(IPS)
        self.random_crop = torchvision.transforms.RandomCrop((IPS, IPS))
        self.padding = padding

    def __len__(self):
        return len(self.target_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        target = tt.imread(self.target_list[idx], meta=self.meta, gray=True).unsqueeze(0)
        if target.shape[-1] < IPS or target.shape[-2] < IPS:
            target = torchvision.transforms.Resize(IPS)(target)
        if self.isTrain:
            target = self.random_crop(target)
            target = torchvision.transforms.functional.pad(
                target, (self.padding, self.padding, self.padding, self.padding))
        else:
            target = self.center_crop(target)
            target = torchvision.transforms.functional.pad(
                target, (self.padding, self.padding, self.padding, self.padding))
        return target, self.target_list[idx]


# ============================================================
# GRPO Policy Network
# Critic 없이 FCN 으로 (CH*IPS*IPS) 개 액션에 대한 logit 출력
# ============================================================
class GRPOPolicy(nn.Module):
    def __init__(self, num_channels=CH, img_size=IPS, mid_channels=64):
        super().__init__()
        # state(CH) + state_record(CH) + pre_model(CH) + recon(1) + target(1)
        in_channels = 3 * num_channels + 2

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(mid_channels, mid_channels * 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(mid_channels * 2, mid_channels * 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(mid_channels * 2, mid_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(mid_channels, num_channels, 1),
        )

    def forward(self, x):
        """(B, in_ch, H, W) -> (B, CH*H*W) logits"""
        logits = self.features(x).reshape(x.size(0), -1)
        return torch.clamp(logits, -20.0, 20.0)

    def get_distribution(self, x):
        logits = self.forward(x)
        return torch.distributions.Categorical(logits=logits)


# ============================================================
# GRPO Trainer
#
# DeepSeek-Math / DeepSeek-R1 의 GRPO 알고리즘을 step-level RL 에 적용.
# 핵심: Critic(Value Network) 없이, 같은 상태에서 G개 액션을 샘플링 →
#        보상을 그룹 내 정규화하여 상대적 어드밴티지 계산 →
#        Clipped surrogate + KL 정규화로 정책 업데이트.
# ============================================================
class GRPOTrainer:
    def __init__(
        self,
        policy: GRPOPolicy,
        env: BinaryHologramEnv,
        group_size: int = 16,
        sim_batch_size: int = 4,
        lr: float = 1e-4,
        clip_range: float = 0.2,
        kl_coef: float = 0.04,
        update_epochs: int = 4,
        max_grad_norm: float = 0.5,
        ref_update_interval: int = 10,
        grpo_update_interval: int = 1,
        max_kl: float = 0.1,
        mask_in_update: bool = False,
        device: str = 'cuda',
    ):
        self.device = device
        self.policy = policy.to(device)

        # π_ref: 레퍼런스 정책 (frozen copy)
        self.ref_policy = copy.deepcopy(policy).to(device)
        self.ref_policy.eval()
        for p in self.ref_policy.parameters():
            p.requires_grad = False

        self.env = env
        self.group_size = group_size
        self.sim_batch_size = sim_batch_size
        self.clip_range = clip_range
        self.kl_coef = kl_coef
        self.update_epochs = update_epochs
        self.max_grad_norm = max_grad_norm
        self.ref_update_interval = ref_update_interval
        self.grpo_update_interval = grpo_update_interval
        self.max_kl = max_kl
        self.mask_in_update = mask_in_update

        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self.episode_count = 0

        # 진단 카운터 (에피소드마다 리셋, 에피소드 로그에 출력)
        self.kl_stops = 0    # KL > max_kl 로 epoch 루프를 조기 중단한 횟수
        self.nan_skips = 0   # new_log_probs 에 NaN 이 나와 epoch 루프를 중단한 횟수
        self.masked_sum = 0  # grpo_step 시점의 실패 마스크 크기 누적 (에피소드 평균 계산용)

        # 실패 마스킹: 같은 상태에서 이미 시도하여 실패한 액션을 제외
        self.num_pixels = CH * IPS * IPS
        self.failed_mask = torch.zeros(self.num_pixels, device=device)
        self._prev_state_hash = None

    # ----------------------------------------------------------
    # Observation → Tensor 변환
    # ----------------------------------------------------------
    def obs_to_tensor(self, obs):
        """Dict obs → (1, 3*CH+2, IPS, IPS) GPU tensor"""
        parts = []
        for key in ('state', 'state_record', 'pre_model', 'recon_image', 'target_image'):
            parts.append(torch.as_tensor(obs[key], dtype=torch.float32))
        return torch.cat(parts, dim=1).to(self.device)

    # ----------------------------------------------------------
    # 그룹 보상 평가 (상태 변경 없이 PSNR 변화량 계산)
    # ----------------------------------------------------------
    def evaluate_group_rewards(self, actions, z=PROP_Z):
        """
        G개 액션 각각에 대해 픽셀 플립 → PSNR 변화 계산.
        sim_batch_size 단위로 서브 배치 처리하여 GPU 메모리 제어.
        """
        G = len(actions)
        states_batch = np.tile(self.env.state, (G, 1, 1, 1))  # (G, CH, IPS, IPS)

        for i, action in enumerate(actions):
            a = action.item() if isinstance(action, torch.Tensor) else action
            ch, px = divmod(a, IPS * IPS)
            row, col = divmod(px, IPS)
            states_batch[i, ch, row, col] = 1 - states_batch[i, ch, row, col]

        rewards = np.zeros(G)
        sb = self.sim_batch_size

        for start in range(0, G, sb):
            end = min(start + sb, G)
            sub_batch = torch.tensor(
                states_batch[start:end], dtype=torch.float32, device=self.device)
            sub_batch = tt.Tensor(sub_batch, meta=OPTICS_META)

            with torch.no_grad():
                sim = tt.simulate(sub_batch, z).abs() ** 2
                result = torch.mean(sim, dim=1, keepdim=True)

            for i in range(end - start):
                psnr_i = tt.relativeLoss(
                    result[i:i+1], self.env.target_image, tm.get_PSNR)
                rewards[start + i] = float(psnr_i - self.env.previous_psnr)

            del sub_batch, sim, result
            torch.cuda.empty_cache()

        return rewards

    # ----------------------------------------------------------
    # GRPO 핵심 스텝
    # ----------------------------------------------------------
    def _update_failed_mask(self, obs):
        """상태가 바뀌면(= 성공적 플립) 실패 마스크 초기화"""
        state_hash = obs['state'].tobytes()
        if state_hash != self._prev_state_hash:
            self.failed_mask.zero_()
            self._prev_state_hash = state_hash

    @staticmethod
    def _distribution(policy, obs_tensor, mask):
        """정책 로짓으로 Categorical 을 만든다. mask(bool, (N,))가 있으면 해당 액션을 -1e9 로 눌러 제외한다.
        in-place 대입 대신 masked_fill 을 써서 autograd 그래프를 유지한다."""
        logits = policy(obs_tensor)  # (1, N)
        if mask is not None:
            logits = logits.masked_fill(mask, -1e9)
        return torch.distributions.Categorical(logits=logits)

    def grpo_step(self, obs):
        """
        1) π_old 에서 G개 액션 샘플링 (실패 마스킹 적용)
        2) 그룹 보상 평가
        3) 상대적 어드밴티지 = (r - mean) / std
        4) Clipped surrogate + KL(π_θ || π_ref) 로 정책 업데이트
        5) 그룹 내 최고 보상 액션 반환
        """
        self._update_failed_mask(obs)
        obs_tensor = self.obs_to_tensor(obs)

        # 1) 그룹 샘플링 (실패한 액션 마스킹)
        self.policy.eval()
        with torch.no_grad():
            logits = self.policy(obs_tensor).squeeze(0)  # (CH*IPS*IPS,)
            logits[self.failed_mask.bool()] = -1e9
            masked_dist = torch.distributions.Categorical(logits=logits)
            actions = masked_dist.sample((self.group_size,))  # (G,)
            old_log_probs = masked_dist.log_prob(actions)     # (G,)
        self.policy.train()

        # 2) 보상 평가
        rewards = self.evaluate_group_rewards(actions)

        # 3) 그룹 상대적 어드밴티지
        adv = rewards - rewards.mean()
        std = rewards.std()
        if std > 1e-8:
            adv = adv / std
        advantages = torch.tensor(adv, dtype=torch.float32, device=self.device)

        actions_gpu = actions.to(self.device)
        old_lp_detached = old_log_probs.detach()

        # 업데이트 루프에서 쓸 마스크. mask_in_update=False(기존 동작)면 None → 마스크 없는 분포로 new/ref 를 계산한다.
        # True 면 샘플링에 쓴 것과 같은 마스크를 적용해 첫 epoch 의 ratio 가 (부동소수 오차 범위에서) 1 이 된다.
        update_mask = self.failed_mask.bool() if self.mask_in_update else None
        self.masked_sum += int(self.failed_mask.sum().item())

        # 레퍼런스 log prob (KL 계산용)
        with torch.no_grad():
            ref_dist = self._distribution(self.ref_policy, obs_tensor, update_mask)
            ref_log_probs = ref_dist.log_prob(actions_gpu).squeeze(-1)

        # 4) 정책 업데이트 (다중 에폭, KL early stopping 포함)
        total_loss = 0.0
        actual_epochs = 0
        max_kl = self.max_kl  # KL이 이 값을 넘으면 조기 중단 (CONFIG["max_kl"])

        for _ in range(self.update_epochs):
            dist_new = self._distribution(self.policy, obs_tensor, update_mask)
            new_log_probs = dist_new.log_prob(actions_gpu).squeeze(-1)

            # NaN 감지 → 해당 에폭 스킵
            if torch.isnan(new_log_probs).any():
                self.nan_skips += 1
                break

            ratio = torch.exp(new_log_probs - old_lp_detached)
            ratio = torch.clamp(ratio, 0.0, 10.0)  # ratio 폭주 방지

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_range,
                                1.0 + self.clip_range) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            log_ratio_ref = ref_log_probs - new_log_probs
            log_ratio_ref = torch.clamp(log_ratio_ref, -10.0, 10.0)
            kl_loss = (torch.exp(log_ratio_ref) - log_ratio_ref - 1.0).mean()

            # KL이 너무 커지면 조기 중단 (정책이 너무 빨리 변하는 것 방지)
            if kl_loss.item() > max_kl:
                self.kl_stops += 1
                break

            loss = policy_loss + self.kl_coef * kl_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            total_loss += loss.item()
            actual_epochs += 1

        avg_loss = total_loss / max(actual_epochs, 1)

        # 5) 최고 보상 액션 선택 + 실패 액션 마스킹
        best_idx = int(np.argmax(rewards))
        best_action = actions[best_idx].item()
        best_reward = rewards[best_idx]

        for i, r in enumerate(rewards):
            if r <= 0:
                self.failed_mask[actions[i].item()] = 1.0

        return best_action, best_reward, avg_loss

    # ----------------------------------------------------------
    # 메인 학습 루프
    # ----------------------------------------------------------
    def train(self, num_episodes=1000, save_dir="./grpo_models/", save_interval=100):
        os.makedirs(save_dir, exist_ok=True)

        for ep in range(num_episodes):
            obs, info = self.env.reset()
            self.episode_count += 1
            episode_reward = 0.0
            step_count = 0
            ep_start = time.time()
            grpo_updates = 0
            last_loss = 0.0

            # 에피소드 시작 시 마스크·진단 카운터 초기화
            self.failed_mask.zero_()
            self._prev_state_hash = None
            self.kl_stops = 0
            self.nan_skips = 0
            self.masked_sum = 0

            while True:
                step_count += 1

                if step_count % self.grpo_update_interval == 0:
                    # GRPO 업데이트 + 최적 액션 선택
                    best_action, grpo_reward, last_loss = self.grpo_step(obs)
                    grpo_updates += 1

                    if grpo_reward > 0:
                        obs, reward, terminated, truncated, _ = self.env.step(best_action)
                    else:
                        # 그룹 내 양수 보상 액션 없음 → 마스킹된 분포에서 샘플링
                        obs_tensor = self.obs_to_tensor(obs)
                        with torch.no_grad():
                            logits = self.policy(obs_tensor).squeeze(0)
                            logits[self.failed_mask.bool()] = -1e9
                            dist = torch.distributions.Categorical(logits=logits)
                            action = dist.sample().item()
                        self.failed_mask[action] = 1.0
                        obs, reward, terminated, truncated, _ = self.env.step(action)
                else:
                    # 업데이트 간격 사이: 마스킹된 분포에서 행동 선택
                    self._update_failed_mask(obs)
                    obs_tensor = self.obs_to_tensor(obs)
                    with torch.no_grad():
                        logits = self.policy(obs_tensor).squeeze(0)
                        logits[self.failed_mask.bool()] = -1e9
                        dist = torch.distributions.Categorical(logits=logits)
                        action = dist.sample().item()
                    self.failed_mask[action] = 1.0
                    obs, reward, terminated, truncated, _ = self.env.step(action)

                episode_reward += reward

                if terminated or truncated:
                    break

            elapsed = time.time() - ep_start
            # PSNRgain = 이 에피소드에서 실제로 올린 PSNR(dB). Reward= 는 env.py 의 순위 보상 합이라 GRPO 학습 신호가 아니다.
            psnr_gain = float(self.env.previous_psnr - self.env.initial_psnr)
            avg_masked = self.masked_sum / max(grpo_updates, 1)
            print(
                f"\033[41mEpisode {self.episode_count}: "
                f"Reward={episode_reward:.2f}, Steps={step_count}, "
                f"GRPO Updates={grpo_updates}, Loss={last_loss:.4f}, "
                f"Time={elapsed:.1f}s\033[0m"
                f" | PSNRgain={psnr_gain:+.4f}, KLstops={self.kl_stops}, "
                f"NaNskips={self.nan_skips}, AvgMasked={avg_masked:.1f}"
            )

            # 레퍼런스 정책 주기적 업데이트
            if self.episode_count % self.ref_update_interval == 0:
                self.ref_policy.load_state_dict(self.policy.state_dict())
                print(f"  [GRPO] π_ref updated at episode {self.episode_count}")

            # 체크포인트 저장
            if self.episode_count % save_interval == 0:
                self._save_checkpoint(save_dir, f"grpo_ep{self.episode_count}.pt")

        self._save_checkpoint(save_dir, "grpo_final.pt")

    # ----------------------------------------------------------
    # 저장 / 로드
    # ----------------------------------------------------------
    def _save_checkpoint(self, save_dir, filename):
        path = os.path.join(save_dir, filename)
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'ref_policy_state_dict': self.ref_policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_count': self.episode_count,
        }, path)
        print(f"  [GRPO] Checkpoint saved: {path}")

        latest = os.path.join(save_dir, "grpo_latest.pt")
        shutil.copyfile(path, latest)

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt['policy_state_dict'])
        self.ref_policy.load_state_dict(ckpt['ref_policy_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.episode_count = ckpt['episode_count']
        print(f"  [GRPO] Loaded checkpoint: {path}  (episode {self.episode_count})")


# ============================================================
# 메인 실행
# ============================================================
if __name__ == '__main__':
    # --- 세 층 적용: CONFIG(파일) ← 앞 셀 주입 ← 불변식. 오타·타입·불변식 위반은 여기서 멈춘다 ---
    cfg, overrides = build_config(globals())

    # --- 시드 (None 이면 기존처럼 비고정) ---
    if cfg["seed"] is not None:
        import random
        random.seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        torch.manual_seed(cfg["seed"])
        torch.cuda.manual_seed_all(cfg["seed"])

    # --- 저장 경로: 어느 설정의 산출물인지 이름만 보고 알 수 있어야 하고, 켰을 때만 이름이 바뀌어야 한다 ---
    # 파일에서 켠 스위치는 접미사로 붙인다 (주입으로 켠 것은 아래 sweep 이름의 축에 이미 들어간다).
    switch_suffix = (("_maskfix" if cfg["mask_in_update"] and "mask_in_update" not in overrides else "")
                     + (f"_seed{cfg['seed']}" if cfg["seed"] is not None and "seed" not in overrides else "")
                     + (f"_v2_{cfg['v2']['policy_kind']}" if cfg["trainer"] == "v2" and "trainer" not in overrides else ""))
    base_dir = cfg["save_dir"].rstrip("/\\") + switch_suffix
    if "save_dir" in overrides:
        # 앞 셀이 save_dir 을 직접 준 경우 — 스윕 arm 을 이어서 돌릴 때 쓴다. 그대로 쓴다.
        save_dir = cfg["save_dir"].rstrip("/\\") + "/"
        chosen = "앞 셀이 준 save_dir 그대로 (재개용)"
        os.makedirs(save_dir, exist_ok=True)
    elif overrides:
        # 주입이 하나라도 있으면 흔든 축 + 시각 + 잡 번호로 새 폴더. 정의상 새 폴더이므로 이미 있으면 다른 arm 과 겹친 것 → 죽는다.
        save_dir = f"{base_dir}/sweep_{axes_string(overrides)}_{run_stamp()}/"
        chosen = "주입된 축 + 스탬프로 새 폴더 (resume 은 앞 셀에서 grpo__save_dir 을 줄 때만)"
        os.makedirs(save_dir, exist_ok=False)
    else:
        # 주입 없음: 파일 스위치를 켠 것만 접미사로. 아무것도 안 켜면 예전 그대로 ./grpo_models/
        save_dir = f"{base_dir}/"
        chosen = "파일 CONFIG 기준" + (f" (스위치 접미사 {switch_suffix})" if switch_suffix else " (기존 런과 같은 폴더)")
        os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, "grpo_v2_latest.pt" if cfg["trainer"] == "v2" else "grpo_latest.pt")
    overrides_path = os.path.join(save_dir, "overrides.json")
    print(f"[GRPO] save_dir = {save_dir}   <- {chosen}")

    # 이어달리기 안전장치: 이 폴더의 직전 실행 설정(overrides.json)과 지금 설정이 다르면 멈춘다.
    # 다른 설정의 체크포인트를 조용히 이어받아 원래 이름으로 기록하는 사고를 막는다. 이어달리며 바꿔도 되는 키만 예외.
    RESUME_MUTABLE = {"num_episodes", "save_interval", "resume_training", "save_dir",
                      "v2.num_iters", "v2.save_every", "v2.val_every"}   # 실행 길이·주기만 — 학습 결과에 영향 없는 키
    if cfg["resume_training"] and os.path.exists(checkpoint_path) and os.path.exists(overrides_path):
        with open(overrides_path, encoding="utf-8") as f:
            prev = json.load(f).get("effective_config", {})
        now_flat, prev_flat = flatten(cfg), flatten(prev)
        diffs = {k: (prev_flat[k], v) for k, v in now_flat.items()
                 if k not in RESUME_MUTABLE and k.split(".")[0] not in RESUME_MUTABLE and k in prev_flat and prev_flat[k] != v}
        if diffs:
            raise ValueError(
                "설정이 다른 채로 이 폴더의 체크포인트를 이어받으려 합니다: "
                + ", ".join(f"{k}: 이전={a!r} 지금={b!r}" for k, (a, b) in diffs.items())
                + "\n  새 런이면 CONFIG['save_dir'] 을 바꾸거나 앞 셀에서 축을 주입해 새 폴더로 가고, "
                  "정말 이어달리려면 그 폴더의 overrides.json 을 지우세요."
            )

    print("[GRPO] CONFIG (effective):")
    for k, v in cfg.items():
        print(f"  {k} = {v!r}")

    # --- 데이터 ---
    train_dataset = Dataset512(target_dir=cfg["train_dir"], meta=OPTICS_META, isTrain=True, padding=cfg["padding"])
    valid_dataset = Dataset512(target_dir=cfg["valid_dir"], meta=OPTICS_META, isTrain=False, padding=cfg["padding"])
    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg["batch_size"], shuffle=False)

    # --- Pre-trained BinaryNet 로드 ---
    hologram_model = BinaryNet(
        num_hologram=CH, in_planes=1,
        convReLU=False, convBN=False,
        poolReLU=False, poolBN=False,
        deconvReLU=False, deconvBN=False,
    ).cuda()
    hologram_model.load_state_dict(torch.load(cfg["pretrained_path"]))
    hologram_model.eval()

    # ============================================================
    # v2: 오라클 보상 + 배치 GRPO (PLAN.md). env.py 의 에피소드/임계 정의는 쓰지 않는다 (평가 프로토콜은 그대로).
    # ============================================================
    if cfg["trainer"] == "v2":
        from grpo.oracle import FlipOracle
        from grpo.features import num_channels
        from grpo.policies import make_policy, count_params
        from grpo.dbs_state import DBSImage
        from grpo.trainer_v2 import GRPOTrainerV2

        v2 = cfg["v2"]
        device = torch.device("cuda")
        oracle = FlipOracle(OPTICS_META, PROP_Z, IPS, CH, device=device)

        def _prep(T, path):
            # Dataset512 는 tt.Tensor(meta 포함) 를 준다. 먼저 벗기지 않으면 pre/h0/U/특징/액션/보상이 전부 서브클래스가 되고
            # 0-차원 서브클래스 텐서는 f-string 포맷에서 TypeError 로 죽는다 (smoke 는 plain 텐서만 써서 이 경로를 못 본다)
            T = T.to(device).as_subclass(torch.Tensor)
            with torch.no_grad():
                pre = hologram_model(T)[0]                       # (C, n, n) 사전학습 확률
            return T[0, 0], (pre >= 0.5).float(), pre, os.path.basename(path[0])

        _train_iter = [iter(train_loader)]

        def new_image():
            try:
                T, path = next(_train_iter[0])
            except StopIteration:
                _train_iter[0] = iter(train_loader)
                T, path = next(_train_iter[0])
            return _prep(T, path)

        # 검증 상태: 검증 이미지 앞 V 장의 초기 홀로그램. 뒤 절반은 무작위-채택 DBS 로 val_advance_steps 진행한 '중반 상태'.
        val_states = []
        _val_iter = iter(valid_loader)
        _val_rng = np.random.default_rng(0)
        for i in range(v2["val_images"]):
            T, path = next(_val_iter)
            Tv, h0, pre, name = _prep(T, path)
            s = DBSImage(oracle, Tv, h0, name=name)
            s.pre_model = pre
            if i >= v2["val_images"] // 2 and v2["val_advance_steps"] > 0:
                s.random_accept_steps(v2["val_advance_steps"], _val_rng)
            val_states.append(s)
        print(f"[v2] val states: {len(val_states)} (뒤 {len(val_states) - v2['val_images'] // 2}개는 {v2['val_advance_steps']}스텝 진행 상태)")

        in_ch = num_channels(v2["feature_spec"], CH)
        policy = make_policy(v2["policy_kind"], in_ch, CH, IPS, feature_spec=v2["feature_spec"], state_gate=v2["state_gate"],
                             mid_channels=cfg["mid_channels"], unet_base=v2["unet_base"], fno_hidden=v2["fno_hidden"])
        print(f"[v2] policy={v2['policy_kind']} feature_spec={v2['feature_spec']} state_gate={v2['state_gate']} "
              f"in_ch={in_ch} params={count_params(policy):,}")
        trainer = GRPOTrainerV2(policy, oracle, v2, feature_spec=v2["feature_spec"], new_image_fn=new_image,
                                val_states=val_states, device=device, log_dir=save_dir)
        trainer.extra_meta = {"policy_kind": v2["policy_kind"], "feature_spec": v2["feature_spec"],
                              "in_channels": in_ch, "state_gate": v2["state_gate"], "config": cfg}
        if v2["startup_check"]:
            from grpo.oracle import startup_consistency_check
            startup_consistency_check(oracle, trainer.images[0].h, trainer.images[0].T)
        resume_from = checkpoint_path if (cfg["resume_training"] and os.path.exists(checkpoint_path)) else None
        if cfg["resume_training"] and resume_from is None:
            print(f"Warning: No checkpoint at {checkpoint_path}. Training from scratch.")
        if resume_from:
            trainer.load_checkpoint(resume_from)
        record = {"overrides": overrides, "effective_config": cfg, "save_dir": save_dir,
                  "resumed_from_episode": trainer.iteration if resume_from else None, "log_file": log_file,
                  "written_at": datetime.now().isoformat(timespec="seconds")}
        with open(overrides_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2, default=str)
        with open(os.path.join(save_dir, "overrides_history.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        trainer.train(num_iters=v2["num_iters"], save_dir=save_dir, save_every=v2["save_every"], val_every=v2["val_every"])
        raise SystemExit(0)

    # --- 환경 ---
    env = BinaryHologramEnv(
        target_function=hologram_model,
        trainloader=train_loader,
        **cfg["env"],
    )

    # --- GRPO 정책 & 트레이너 ---
    grpo_policy = GRPOPolicy(num_channels=CH, img_size=IPS, mid_channels=cfg["mid_channels"])

    trainer = GRPOTrainer(
        policy=grpo_policy,
        env=env,
        group_size=cfg["group_size"],
        sim_batch_size=cfg["sim_batch_size"],
        lr=cfg["lr"],
        clip_range=cfg["clip_range"],
        kl_coef=cfg["kl_coef"],
        update_epochs=cfg["update_epochs"],
        max_grad_norm=cfg["max_grad_norm"],
        ref_update_interval=cfg["ref_update_interval"],
        grpo_update_interval=cfg["grpo_update_interval"],
        max_kl=cfg["max_kl"],
        mask_in_update=cfg["mask_in_update"],
    )

    resumed_from = None
    if cfg["resume_training"] and os.path.exists(checkpoint_path):
        trainer.load_checkpoint(checkpoint_path)
        resumed_from = trainer.episode_count
    elif cfg["resume_training"]:
        print(f"Warning: No checkpoint at {checkpoint_path}. Training from scratch.")

    # 최종 조합을 산출물 옆에 남긴다 — "무슨 설정으로 만든 체크포인트인가" 를 사후에 확실히 하기 위해.
    # overrides.json 은 최신 실행, overrides_history.jsonl 은 실행마다 누적(덮어쓴 기록의 이력).
    record = {"overrides": overrides, "effective_config": cfg, "save_dir": save_dir,
              "resumed_from_episode": resumed_from, "log_file": log_file,
              "written_at": datetime.now().isoformat(timespec="seconds")}
    with open(overrides_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2, default=str)
    with open(os.path.join(save_dir, "overrides_history.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    # --- 학습 시작 ---
    trainer.train(
        num_episodes=cfg["num_episodes"],
        save_dir=save_dir,
        save_interval=cfg["save_interval"],
    )
