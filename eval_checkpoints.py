"""
GRPO 체크포인트 비교 평가
저장된 모든 체크포인트를 동일한 이미지 셋으로 테스트하여 최적 모델 탐색.
"""

import sys
import logging
from datetime import datetime
import os
from utils.logger import setup_logger

log_file = setup_logger()
from utils.torchoptics_pin import assert_torchoptics_pinned
assert_torchoptics_pinned()  # torchOptics 가 고정 커밋(8e50d6a)이 아니면 여기서 죽는다. 최신 torchOptics 는 사전학습 BinaryNet 을 깨뜨린다.
logging.info("GRPO Checkpoint Evaluation Initialized")

import glob
import time
import warnings
import re

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision

import torchOptics.optics as tt
import torchOptics.metrics as tm

from grpo.eval_utils import policy_input, load_policy_checkpoint, make_oracle_action_fn

IPS = 256
CH = 8
warnings.filterwarnings('ignore')
torch.backends.cudnn.enabled = False


# ============================================================
# BinaryNet
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
        enc1_1 = self.enc1_1(x);   enc1_2 = self.enc1_2(enc1_1);   pool1 = self.pool1(enc1_2)
        enc2_1 = self.enc2_1(pool1);enc2_2 = self.enc2_2(enc2_1);   pool2 = self.pool2(enc2_2)
        enc3_1 = self.enc3_1(pool2);enc3_2 = self.enc3_2(enc3_1);   pool3 = self.pool3(enc3_2)
        enc4_1 = self.enc4_1(pool3);enc4_2 = self.enc4_2(enc4_1);   pool4 = self.pool4(enc4_2)
        enc5_1 = self.enc5_1(pool4);enc5_2 = self.enc5_2(enc5_1)
        deconv4 = self.deconv4(enc5_2);  dec4_2 = self.dec4_2(self.dec4_1(torch.cat((deconv4, enc4_2), 1)))
        deconv3 = self.deconv3(dec4_2);  dec3_2 = self.dec3_2(self.dec3_1(torch.cat((deconv3, enc3_2), 1)))
        deconv2 = self.deconv2(dec3_2);  dec2_2 = self.dec2_2(self.dec2_1(torch.cat((deconv2, enc2_2), 1)))
        deconv1 = self.deconv1(dec2_2);  dec1_2 = self.dec1_2(self.dec1_1(torch.cat((deconv1, enc1_2), 1)))
        return nn.Sigmoid()(self.classifier(dec1_2))


# ============================================================
# Dataset
# ============================================================
class Dataset512(Dataset):
    def __init__(self, target_dir, meta, transform=None, isTrain=True, padding=0):
        self.target_dir = target_dir
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
        else:
            target = self.center_crop(target)
        target = torchvision.transforms.functional.pad(
            target, (self.padding, self.padding, self.padding, self.padding))
        return target, self.target_list[idx]


# ============================================================
# GRPO Policy
# ============================================================
class GRPOPolicy(nn.Module):
    def __init__(self, num_channels=CH, img_size=IPS, mid_channels=64):
        super().__init__()
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
        logits = self.features(x).reshape(x.size(0), -1)
        return torch.clamp(logits, -20.0, 20.0)


# ============================================================
# 시뮬레이션 & DBS 함수
# ============================================================
def simulate_psnr(state, target_image, z=2e-3):
    binary = torch.tensor(state, dtype=torch.float32).cuda()
    binary = tt.Tensor(binary, meta={'dx': (7.56e-6, 7.56e-6), 'wl': 515e-9})
    with torch.no_grad():
        field = tt.simulate(binary, z)                 # (1, C, n, n) complex — v2 정책 특징·오라클 기준선용
        sim = field.abs() ** 2
        result = torch.mean(sim, dim=1, keepdim=True)
        psnr = tt.relativeLoss(result, target_image, tm.get_PSNR)
    return float(psnr), result, field


def make_grpo_action_fn(policy, device='cuda', spec="legacy", v2=False):
    """spec: 체크포인트의 feature_spec. 'legacy' 는 v1 과 같은 26채널 입력, 'field' 는 복소 필드 특징 (grpo.eval_utils.policy_input)."""
    policy.eval()
    num_pixels = CH * IPS * IPS
    failed_mask = torch.zeros(num_pixels, device=device)
    prev_state_hash = [None]

    def select(obs):
        state_bytes = obs['state'].tobytes()
        if state_bytes != prev_state_hash[0]:
            failed_mask.zero_()
            prev_state_hash[0] = state_bytes

        x = policy_input(obs, spec, device, v2=v2)

        with torch.no_grad():
            logits = policy(x).squeeze(0)
            logits[failed_mask.bool()] = -1e9
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample().item()

        failed_mask[action] = 1.0
        return action

    return select


def make_random_action_fn():
    """랜덤 DBS: 무작위 픽셀 선택"""
    num_pixels = CH * IPS * IPS
    def select(obs):
        return np.random.randint(num_pixels)
    return select


def run_dbs(state, pre_model, target_image, target_image_np,
            max_steps, select_action_fn, step_marks=(), success_marks=()):
    state = state.copy()
    state_record = np.zeros_like(state)

    initial_psnr, recon, field = simulate_psnr(state, target_image)
    current_psnr = initial_psnr
    recon_np = recon.cpu().numpy()

    flip_count = 0
    at_step, at_success = {}, {}      # 구간 표용: 스텝/성공 수 지점의 PSNR↑

    for step in range(1, max_steps + 1):
        obs = {
            "state": state,
            "state_record": state_record,
            "pre_model": pre_model,
            "recon_image": recon_np,
            "target_image": target_image_np,
            # v2 정책·오라클 기준선용 GPU 텐서. legacy 정책은 쓰지 않는다.
            "recon_t": recon, "field": field, "target_t": target_image,
        }

        action = select_action_fn(obs)
        ch, px = divmod(action, IPS * IPS)
        row, col = divmod(px, IPS)

        state[0, ch, row, col] = 1 - state[0, ch, row, col]
        psnr_after, recon_after, field_after = simulate_psnr(state, target_image)

        if psnr_after > current_psnr:
            current_psnr = psnr_after
            recon, field = recon_after, field_after
            recon_np = recon_after.cpu().numpy()
            state_record[0, ch, row, col] += 1
            flip_count += 1
            if flip_count in success_marks and flip_count not in at_success:
                at_success[flip_count] = current_psnr - initial_psnr
        else:
            state[0, ch, row, col] = 1 - state[0, ch, row, col]
        if step in step_marks:
            at_step[step] = current_psnr - initial_psnr

    return {
        "at_step": at_step, "at_success": at_success,
        "initial_psnr": initial_psnr,
        "final_psnr": current_psnr,
        "psnr_diff": current_psnr - initial_psnr,
        "flip_count": flip_count,
        "success_ratio": flip_count / max_steps if max_steps > 0 else 0,
    }


# ============================================================
# 체크포인트 목록 수집
# ============================================================
def find_checkpoints(model_dir):
    """grpo_ep*.pt (v1, 에피소드) 와 grpo_v2_it*.pt (v2, 반복) 를 번호순으로 반환"""
    files = sorted(glob.glob(os.path.join(model_dir, "grpo_ep*.pt")) + glob.glob(os.path.join(model_dir, "grpo_v2_it*.pt")))

    checkpoints = []
    for f in files:
        basename = os.path.basename(f)
        match = re.search(r'grpo_(?:ep|v2_it)(\d+)\.pt', basename)   # v1: grpo_ep{N}, v2: grpo_v2_it{N}
        if match:
            ep = int(match.group(1))
            checkpoints.append((ep, f))

    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


# ============================================================
# 메인
# ============================================================
if __name__ == '__main__':
    # ╔══════════════════════════════════════════════════════════╗
    # ║                    여기만 수정하세요                      ║
    # ╚══════════════════════════════════════════════════════════╝
    MODEL_DIR       = "./grpo_models_v2_unet/"          # 체크포인트 폴더 경로 (v1: "./grpo_models/")
    MAX_STEPS       = 20000                             # 이미지당 DBS 스텝 수. Random 곡선(+2~3 dB 에 10만~20만 시도) 기준 '의미 있는 국면'. 1차 평가는 500 (예: 500, 20000)
    STEP_MARKS      = (500, 2000, 5000, 10000, 20000)   # 이 스텝에서의 PSNR↑ 를 표로 (MAX_STEPS 이하만 의미)
    SUCCESS_MARKS   = (1000, 2000, 5000, 10000)         # 성공(채택) 횟수가 여기 닿았을 때의 PSNR↑ 를 표로 (같은 성공 수 비교)
    NUM_EVAL_IMAGES = 10                                # 평가에 사용할 이미지 수 (0 = 전체)
    EVAL_DIR        = '/nfs/dataset/DIV2K/DIV2K_valid_HR/DIV2K_valid_HR/'  # 평가 데이터셋 경로
    INCLUDE_RANDOM_BASELINE = True    # True: 같은 이미지·스텝으로 Random DBS 를 1회 돌려 기준선 행을 표·CSV 에 추가 (episode=0, checkpoint=random_baseline)
    INCLUDE_ORACLE_GREEDY   = True    # True: 오라클 탐욕 DBS(매 스텝 실제 최선 픽셀)를 상한 기준선으로 추가 (episode=-1, checkpoint=oracle_greedy). grpo/oracle_selftest.py 통과 후에만
    BASELINE_NAMES = ("random_baseline", "oracle_greedy")
    EVAL_SPLIT = "all"     # "all"(지금까지와 같음) | "select"(앞 SELECT_N 장: 체크포인트 고르기용) | "report"(나머지: 고른 체크포인트 보고용)
                            #   같은 이미지로 고르고 보고하면 최고값이 위로 치우친다 -> 본 실험 보고는 select 로 고르고 report 로 보고한다
    SELECT_N   = 20         # EVAL_SPLIT 이 select/report 일 때의 경계 (NUM_EVAL_IMAGES 는 그 전에 적용)
    EVAL_SEEDS = None       # None: 지금처럼 시드 미고정 1회 | (0, 1, 2): 시드마다 전체를 반복해 평균과 시드 간 표준편차 열(±std)을 표·CSV 에 추가
    CHECKPOINT_SELECT = "every:5000"   # None: 전부 | "every:5000": 반복 번호가 5000 의 배수인 것만 | (4500, 12000, 24500): 이 번호만. 2만 스텝이면 체크포인트당 ≈25분
    # ════════════════════════════════════════════════════════════

    meta = {'wl': 515e-9, 'dx': (7.56e-6, 7.56e-6)}
    padding = 0

    print(f"\n{'=' * 60}")
    print(f"  GRPO 체크포인트 비교 평가 설정")
    print(f"{'=' * 60}")
    print(f"  Model Dir:       {MODEL_DIR}")
    print(f"  Max Steps/Image: {MAX_STEPS}")
    print(f"  Eval Images:     {'전체' if NUM_EVAL_IMAGES == 0 else NUM_EVAL_IMAGES}")
    print(f"  Eval Data Dir:   {EVAL_DIR}")
    print(f"  Random Baseline: {INCLUDE_RANDOM_BASELINE}")
    print(f"  Oracle Greedy:   {INCLUDE_ORACLE_GREEDY}")
    if EVAL_SPLIT != "all" or EVAL_SEEDS is not None:
        print(f"  Eval Split:      {EVAL_SPLIT} (SELECT_N={SELECT_N})")
        print(f"  Eval Seeds:      {EVAL_SEEDS}")
    print(f"{'=' * 60}\n")

    # --- 데이터 ---
    valid_dataset = Dataset512(target_dir=EVAL_DIR, meta=meta, isTrain=False, padding=padding)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
    print(f"Validation images: {len(valid_dataset)}")

    # --- 평가용 이미지 미리 로드 (모든 체크포인트에서 동일 이미지 사용) ---
    n_load = len(valid_dataset) if NUM_EVAL_IMAGES == 0 else NUM_EVAL_IMAGES
    eval_images = []
    data_iter = iter(valid_loader)
    for _ in range(n_load):
        try:
            target_image, file_path = next(data_iter)
        except StopIteration:
            break
        eval_images.append({
            "target_image": target_image.cuda(),
            "target_image_np": target_image.cpu().numpy(),
            "file_name": os.path.basename(file_path[0]),
        })
    if EVAL_SPLIT == "select":
        eval_images = eval_images[:SELECT_N]
    elif EVAL_SPLIT == "report":
        eval_images = eval_images[SELECT_N:]
    elif EVAL_SPLIT != "all":
        raise ValueError(f"EVAL_SPLIT 은 all/select/report 중 하나: {EVAL_SPLIT!r}")
    if not eval_images:
        raise ValueError(f"평가 이미지가 0장 (EVAL_SPLIT={EVAL_SPLIT!r}, SELECT_N={SELECT_N}, NUM_EVAL_IMAGES={NUM_EVAL_IMAGES})")
    print(f"Eval images loaded: {len(eval_images)}" + (f" (split={EVAL_SPLIT})" if EVAL_SPLIT != "all" else ""))

    # --- BinaryNet 로드 ---
    hologram_model = BinaryNet(
        num_hologram=CH, in_planes=1,
        convReLU=False, convBN=False,
        poolReLU=False, poolBN=False,
        deconvReLU=False, deconvBN=False,
    ).cuda()
    hologram_model.load_state_dict(torch.load(
        'result_v/2024-12-19 20:37:52.499731_pre_reinforce_8_0.002/'
        '2024-12-19 20:37:52.499731_pre_reinforce_8_0.002'))
    hologram_model.eval()

    # --- 초기 홀로그램 미리 계산 (모든 체크포인트에서 동일) ---
    for img in eval_images:
        with torch.no_grad():
            model_output = hologram_model(img["target_image"])
        img["pre_model"] = model_output.cpu().numpy()
        img["initial_state"] = (img["pre_model"] >= 0.5).astype(np.int8)

    # --- 체크포인트 수집 ---
    checkpoints = find_checkpoints(MODEL_DIR)
    if not checkpoints:
        print(f"No checkpoints found in {MODEL_DIR}")
        sys.exit(1)
    if isinstance(CHECKPOINT_SELECT, str) and CHECKPOINT_SELECT.startswith("every:"):
        _every = int(CHECKPOINT_SELECT.split(":")[1])
        checkpoints = [(e, p) for e, p in checkpoints if e % _every == 0]
        if not checkpoints:
            raise ValueError(f"CHECKPOINT_SELECT={CHECKPOINT_SELECT!r} 에 맞는 체크포인트가 없다")
        print(f"CHECKPOINT_SELECT: {len(checkpoints)}개만 평가 (반복 번호 {_every} 의 배수)")
    elif CHECKPOINT_SELECT is not None:
        wanted = set(int(x) for x in CHECKPOINT_SELECT)
        missing = sorted(wanted - {e for e, _ in checkpoints})
        if missing:
            raise ValueError(f"CHECKPOINT_SELECT 에 없는 체크포인트: {missing} (있는 것: {[e for e, _ in checkpoints]})")
        checkpoints = [(e, p) for e, p in checkpoints if e in wanted]
        print(f"CHECKPOINT_SELECT: {len(checkpoints)}개만 평가 {sorted(wanted)}")
    print(f"Found {len(checkpoints)} checkpoints: ep{checkpoints[0][0]} ~ ep{checkpoints[-1][0]}")

    # --- 정책은 체크포인트마다 기록된 구성(policy_kind/feature_spec)으로 만든다. v1 은 레거시 GRPOPolicy ---

    # --- 체크포인트별 평가 ---
    all_results = []

    seeds = [None] if EVAL_SEEDS is None else list(EVAL_SEEDS)
    std_hdr = f"  {'±std(seed)':>10}" if EVAL_SEEDS is not None else ""
    print(f"\n{'━' * 90}")
    print(f"  {'Checkpoint':<20} {'Episode':>8}  {'Avg PSNR↑':>10}{std_hdr}  {'Avg Success%':>13}  "
          f"{'Avg Flips':>10}  {'Time':>8}")
    print(f"{'━' * 90}")

    def evaluate(make_action_fn):
        """eval_images 전체를 시드마다 1회 돈다 (EVAL_SEEDS=None 이면 지금까지처럼 시드 미고정 1회).
        반환: 시드 평균 지표, 시드 간 표준편차(psnr_diff_std; 1회면 0), 첫 시드의 이미지별 결과."""
        per_seed = []
        for sd in seeds:
            if sd is not None:
                np.random.seed(sd)
                torch.manual_seed(sd)
                torch.cuda.manual_seed_all(sd)
            results = []
            for img in eval_images:
                torch.cuda.empty_cache()
                results.append(run_dbs(
                    state=img["initial_state"],
                    pre_model=img["pre_model"],
                    target_image=img["target_image"],
                    target_image_np=img["target_image_np"],
                    max_steps=MAX_STEPS,
                    select_action_fn=make_action_fn(),
                    step_marks=STEP_MARKS, success_marks=SUCCESS_MARKS,
                ))
            per_seed.append(results)
        diffs = [np.mean([r["psnr_diff"] for r in rs]) for rs in per_seed]
        flat = [r for rs in per_seed for r in rs]
        return {
            "avg_psnr_diff": float(np.mean(diffs)),
            "psnr_diff_std": float(np.std(diffs)) if len(diffs) > 1 else 0.0,
            "avg_success_ratio": float(np.mean([r["success_ratio"] for r in flat])),
            "avg_flip_count": float(np.mean([r["flip_count"] for r in flat])),
            "per_image": per_seed[0],
            "at_step": {m: float(np.mean([r["at_step"][m] for r in flat if m in r["at_step"]])) if any(m in r["at_step"] for r in flat) else float("nan")
                        for m in STEP_MARKS},
            "at_success": {m: (float(np.mean([r["at_success"][m] for r in flat if m in r["at_success"]])) if any(m in r["at_success"] for r in flat) else float("nan"),
                            sum(1 for r in flat if m in r["at_success"]), len(flat)) for m in SUCCESS_MARKS},
        }

    def print_row(name, ep, res, elapsed):
        std_col = f"  {res['psnr_diff_std']:>10.4f}" if EVAL_SEEDS is not None else ""
        print(f"  {name:<20} {ep:>8}  "
              f"{res['avg_psnr_diff']:>+10.4f}{std_col}  {res['avg_success_ratio']:>12.2%}  "
              f"{res['avg_flip_count']:>10.1f}  {elapsed:>7.1f}s")

    baselines = []
    if INCLUDE_RANDOM_BASELINE:
        baselines.append(("random_baseline", 0, make_random_action_fn))
    if INCLUDE_ORACLE_GREEDY:
        baselines.append(("oracle_greedy", -1, lambda: make_oracle_action_fn(CH, IPS)))
    for bname, bep, bfn in baselines:
        # 기준선: 같은 이미지·같은 초기 홀로그램·같은 스텝 수. 시드는 EVAL_SEEDS 가 없으면 고정하지 않는다.
        t_start = time.time()
        res = evaluate(bfn)
        elapsed = time.time() - t_start
        all_results.append(dict(res, episode=bep, checkpoint=bname, time=elapsed))
        print_row(bname, "-", res, elapsed)

    for ep_num, ckpt_path in checkpoints:
        # 체크포인트 로드 — v1(policy_kind 없음)은 레거시 GRPOPolicy 로 예전과 똑같이, v2 는 기록된 policy_kind/feature_spec 으로
        grpo_policy, policy_spec, pinfo = load_policy_checkpoint(ckpt_path, GRPOPolicy, CH, IPS)
        t_start = time.time()
        res = evaluate(lambda: make_grpo_action_fn(grpo_policy, spec=policy_spec, v2=pinfo["v2"]))
        elapsed = time.time() - t_start
        all_results.append(dict(res, episode=ep_num, checkpoint=ckpt_path, time=elapsed))
        print_row(os.path.basename(ckpt_path), ep_num, res, elapsed)

    # --- 최고 성능 체크포인트 ---
    print(f"\n{'━' * 90}")

    # '최고 체크포인트' 는 체크포인트끼리만 고른다. 기준선이 더 좋은 경우는 아래 기준선 대비 요약이 따로 말한다.
    ckpt_rows = [r for r in all_results if r["checkpoint"] not in BASELINE_NAMES]
    best_psnr = max(ckpt_rows, key=lambda x: x["avg_psnr_diff"])
    best_success = max(ckpt_rows, key=lambda x: x["avg_success_ratio"])

    print(f"\n  🏆 PSNR 향상 최고:  ep{best_psnr['episode']}  "
          f"(+{best_psnr['avg_psnr_diff']:.4f} dB)  "
          f"→ {best_psnr['checkpoint']}")
    print(f"  🏆 성공률 최고:     ep{best_success['episode']}  "
          f"({best_success['avg_success_ratio']:.2%})  "
          f"→ {best_success['checkpoint']}")

    if INCLUDE_RANDOM_BASELINE:
        base = next(r for r in all_results if r["checkpoint"] == "random_baseline")
        ckpts = ckpt_rows
        n_psnr = sum(1 for r in ckpts if r["avg_psnr_diff"] > base["avg_psnr_diff"])
        n_succ = sum(1 for r in ckpts if r["avg_success_ratio"] > base["avg_success_ratio"])
        print(f"\n  Random 기준선 대비 (PSNR↑ {base['avg_psnr_diff']:+.4f} dB, 성공률 {base['avg_success_ratio']:.2%}):")
        print(f"    PSNR↑ 가 기준선보다 높은 체크포인트: {n_psnr}/{len(ckpts)}")
        print(f"    성공률이 기준선보다 높은 체크포인트: {n_succ}/{len(ckpts)}")
        # 같은 이미지끼리 짝지어 센 값이 평균 하나의 부등호보다 안정적이다
        n_img = len(base["per_image"])
        print(f"    체크포인트별 '기준선을 이긴 이미지 수' (같은 이미지끼리 PSNR↑ 비교, 총 {n_img}장):")
        for r in ckpts:
            wins = sum(1 for a, b in zip(r["per_image"], base["per_image"]) if a["psnr_diff"] > b["psnr_diff"])
            print(f"      ep{r['episode']:>5}: {wins}/{n_img}")
        if n_psnr == 0:
            print("    → 무처리(초기 홀로그램) 대비 개선이 있어도 Random DBS 대비 개선이 없으면 '학습/개선 안 됨' 으로 판정한다")
        if EVAL_SEEDS is None:
            print("    (주의: 기준선·체크포인트 모두 시드 미고정 1회 실행 - 차이가 작으면 EVAL_SEEDS 로 반복해 확인 필요)")
        else:
            print(f"    (시드 {len(seeds)}개 평균; ±std 열은 시드 간 표준편차)")

    if INCLUDE_ORACLE_GREEDY:
        orc = next(r for r in all_results if r["checkpoint"] == "oracle_greedy")
        n_steps = max(1, len(eval_images) * MAX_STEPS * len(seeds))
        print(f"\n  오라클 탐욕 상한 (PSNR↑ {orc['avg_psnr_diff']:+.4f} dB, 성공률 {orc['avg_success_ratio']:.2%}, "
              f"{orc['time'] / n_steps * 1000:.1f} ms/스텝) 대비 회수율  (정책 행의 ms/스텝 = 정책 1회 + 시뮬레이션 1회):")
        for r in ckpt_rows:
            rec = r["avg_psnr_diff"] / orc["avg_psnr_diff"] if orc["avg_psnr_diff"] > 0 else float("nan")
            print(f"      ep{r['episode']:>5}: {rec:.1%}   ({r['time'] / n_steps * 1000:.1f} ms/스텝)")

    # --- 구간 표: 같은 스텝 수 / 같은 성공 수에서의 PSNR↑ (Random 곡선과 대조하는 '의미 있는 국면' 판정) ---
    marks = [m for m in STEP_MARKS if m <= MAX_STEPS]
    if marks:
        print(f"\n  스텝 구간별 PSNR↑ (이미지 평균):")
        print(f"    {'':<22}" + "".join(f"{('step ' + str(m)):>12}" for m in marks))
        for r in all_results:
            print(f"    {os.path.basename(r['checkpoint']):<22}" + "".join(f"{r['at_step'].get(m, float('nan')):>+12.4f}" for m in marks))
    print(f"\n  성공 횟수 도달 시 PSNR↑ (이미지 평균, 괄호 = 도달 이미지 수):")
    print(f"    {'':<22}" + "".join(f"{('succ ' + str(m)):>16}" for m in SUCCESS_MARKS))
    for r in all_results:
        print(f"    {os.path.basename(r['checkpoint']):<22}" + "".join(
            f"{r['at_success'][m][0]:>+11.4f}({r['at_success'][m][1]:>2})" for m in SUCCESS_MARKS))

    # --- 결과 CSV 저장 ---
    result_dir = f"./eval_results/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/"
    os.makedirs(result_dir, exist_ok=True)

    csv_path = os.path.join(result_dir, "checkpoint_comparison.csv")
    with open(csv_path, "w") as f:
        std_h = ",psnr_diff_std" if EVAL_SEEDS is not None else ""   # 시드 반복을 켰을 때만 열이 늘어난다 (기본 출력은 그대로)
        f.write(f"episode,checkpoint,avg_psnr_diff{std_h},avg_success_ratio,avg_flip_count,time\n")
        for r in all_results:
            std_v = f",{r['psnr_diff_std']:.6f}" if EVAL_SEEDS is not None else ""
            f.write(f"{r['episode']},{os.path.basename(r['checkpoint'])},"
                    f"{r['avg_psnr_diff']:.6f}{std_v},{r['avg_success_ratio']:.6f},"
                    f"{r['avg_flip_count']:.1f},{r['time']:.2f}\n")

    detail_path = os.path.join(result_dir, "per_image_detail.csv")
    with open(detail_path, "w") as f:
        f.write("episode,image,initial_psnr,final_psnr,psnr_diff,flip_count,success_ratio\n")
        for r in all_results:
            for i, img_r in enumerate(r["per_image"]):
                f.write(f"{r['episode']},{eval_images[i]['file_name']},"
                        f"{img_r['initial_psnr']:.6f},{img_r['final_psnr']:.6f},"
                        f"{img_r['psnr_diff']:.6f},{img_r['flip_count']},"
                        f"{img_r['success_ratio']:.6f}\n")

    with open(os.path.join(result_dir, "trajectory_marks.csv"), "w") as f:
        f.write("episode,checkpoint," + ",".join(f"step{m}" for m in STEP_MARKS) + ","
                + ",".join(f"succ{m},succ{m}_n" for m in SUCCESS_MARKS) + "\n")
        for r in all_results:
            f.write(f"{r['episode']},{os.path.basename(r['checkpoint'])},"
                    + ",".join(f"{r['at_step'].get(m, float('nan')):.6f}" for m in STEP_MARKS) + ","
                    + ",".join(f"{r['at_success'][m][0]:.6f},{r['at_success'][m][1]}" for m in SUCCESS_MARKS) + "\n")
    print(f"\n  Results saved to: {result_dir}")
    print(f"  - trajectory_marks.csv       (스텝/성공 수 구간별 PSNR↑)")
    print(f"  - checkpoint_comparison.csv  (체크포인트별 요약)")
    print(f"  - per_image_detail.csv       (이미지별 상세)")
