"""
GRPO vs Random DBS 비교 테스트
동일한 초기 홀로그램에서 시작하여 두 방법의 PSNR 향상을 비교.
"""

import sys
import logging
from datetime import datetime
import os
from utils.logger import setup_logger

log_file = setup_logger()
logging.info("GRPO vs Random DBS Test Initialized")

import glob
import copy
import time
import warnings

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision

import torchOptics.optics as tt
import torchOptics.metrics as tm

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
# GRPO Policy (train_grpo.py 와 동일 구조)
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

    def get_distribution(self, x):
        return torch.distributions.Categorical(logits=self.forward(x))


# ============================================================
# DBS 엔진 (환경 없이 직접 시뮬레이션)
# ============================================================
def simulate_psnr(state, target_image, z=2e-3):
    """현재 binary state → 시뮬레이션 → PSNR 반환"""
    binary = torch.tensor(state, dtype=torch.float32).cuda()
    binary = tt.Tensor(binary, meta={'dx': (7.56e-6, 7.56e-6), 'wl': 515e-9})
    with torch.no_grad():
        sim = tt.simulate(binary, z).abs() ** 2
        result = torch.mean(sim, dim=1, keepdim=True)
        psnr = tt.relativeLoss(result, target_image, tm.get_PSNR)
    return float(psnr), result


def run_dbs(state, pre_model, target_image, target_image_np,
            max_steps, select_action_fn, label=""):
    """
    DBS 루프 실행.
    select_action_fn(obs_dict) → action (int) 를 받아서 방법만 교체.
    """
    state = state.copy()
    state_record = np.zeros_like(state)
    num_pixels = CH * IPS * IPS

    initial_psnr, recon = simulate_psnr(state, target_image)
    current_psnr = initial_psnr
    recon_np = recon.cpu().numpy()

    flip_count = 0
    psnr_history = [initial_psnr]

    t_start = time.time()

    for step in range(1, max_steps + 1):
        obs = {
            "state": state,
            "state_record": state_record,
            "pre_model": pre_model,
            "recon_image": recon_np,
            "target_image": target_image_np,
        }

        action = select_action_fn(obs)
        ch, px = divmod(action, IPS * IPS)
        row, col = divmod(px, IPS)

        # 플립
        state[0, ch, row, col] = 1 - state[0, ch, row, col]

        psnr_after, recon_after = simulate_psnr(state, target_image)

        if psnr_after > current_psnr:
            current_psnr = psnr_after
            recon_np = recon_after.cpu().numpy()
            state_record[0, ch, row, col] += 1
            flip_count += 1
        else:
            state[0, ch, row, col] = 1 - state[0, ch, row, col]

        psnr_history.append(current_psnr)

    elapsed = time.time() - t_start
    psnr_diff = current_psnr - initial_psnr
    success_ratio = flip_count / max_steps if max_steps > 0 else 0

    return {
        "label": label,
        "initial_psnr": initial_psnr,
        "final_psnr": current_psnr,
        "psnr_diff": psnr_diff,
        "flip_count": flip_count,
        "success_ratio": success_ratio,
        "time": elapsed,
        "psnr_history": psnr_history,
    }


# ============================================================
# 액션 선택 함수
# ============================================================
def make_random_action_fn():
    """랜덤 DBS: 무작위 픽셀 선택"""
    num_pixels = CH * IPS * IPS
    def select(obs):
        return np.random.randint(num_pixels)
    return select


def make_grpo_action_fn(policy, device='cuda', temperature=1.0):
    """
    GRPO 정책 기반 액션 선택.
    확률적 샘플링 + 실패 마스킹으로 같은 나쁜 픽셀 반복 방지.
    """
    policy.eval()
    num_pixels = CH * IPS * IPS
    failed_mask = torch.zeros(num_pixels, device=device)
    prev_state_hash = [None]

    def select(obs):
        state_bytes = obs['state'].tobytes()
        if state_bytes != prev_state_hash[0]:
            failed_mask.zero_()
            prev_state_hash[0] = state_bytes

        parts = []
        for key in ('state', 'state_record', 'pre_model', 'recon_image', 'target_image'):
            parts.append(torch.as_tensor(obs[key], dtype=torch.float32))
        x = torch.cat(parts, dim=1).to(device)

        with torch.no_grad():
            logits = policy(x).squeeze(0)  # (CH*IPS*IPS,)
            logits = logits / temperature
            logits[failed_mask.bool()] = -1e9
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample().item()

        failed_mask[action] = 1.0
        return action

    return select


# ============================================================
# 결과 출력
# ============================================================
def print_result(r):
    print(
        f"  [{r['label']}] "
        f"Initial: {r['initial_psnr']:.4f} → Final: {r['final_psnr']:.4f}  "
        f"(+{r['psnr_diff']:.4f})  "
        f"Flips: {r['flip_count']}  "
        f"Success: {r['success_ratio']:.2%}  "
        f"Time: {r['time']:.1f}s"
    )


def print_comparison(results_grpo, results_random, dataset_name=""):
    title = f"GRPO vs Random DBS 종합 비교 [{dataset_name}]" if dataset_name else "GRPO vs Random DBS 종합 비교"
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

    g = results_grpo
    r = results_random
    n = len(g)

    avg = lambda lst, key: np.mean([x[key] for x in lst])
    std = lambda lst, key: np.std([x[key] for x in lst])

    metrics = [
        ("PSNR 향상 (dB)", "psnr_diff"),
        ("최종 PSNR (dB)", "final_psnr"),
        ("성공 Flip 수", "flip_count"),
        ("성공률", "success_ratio"),
        ("소요 시간 (s)", "time"),
    ]

    fmt_pct = lambda key: "success_ratio" == key

    print(f"\n  {'지표':<20} {'GRPO (mean±std)':>25}   {'Random (mean±std)':>25}   {'차이':>12}")
    print("  " + "-" * 90)

    for name, key in metrics:
        g_mean, g_std = avg(g, key), std(g, key)
        r_mean, r_std = avg(r, key), std(r, key)
        diff = g_mean - r_mean

        if key == "success_ratio":
            g_str = f"{g_mean:.2%} ± {g_std:.2%}"
            r_str = f"{r_mean:.2%} ± {r_std:.2%}"
            d_str = f"{diff:+.2%}"
        elif key == "flip_count":
            g_str = f"{g_mean:.1f} ± {g_std:.1f}"
            r_str = f"{r_mean:.1f} ± {r_std:.1f}"
            d_str = f"{diff:+.1f}"
        elif key == "time":
            g_str = f"{g_mean:.1f} ± {g_std:.1f}"
            r_str = f"{r_mean:.1f} ± {r_std:.1f}"
            d_str = f"{diff:+.1f}"
        else:
            g_str = f"{g_mean:.4f} ± {g_std:.4f}"
            r_str = f"{r_mean:.4f} ± {r_std:.4f}"
            d_str = f"{diff:+.4f}"

        print(f"  {name:<20} {g_str:>25}   {r_str:>25}   {d_str:>12}")

    # 이미지별 승패 — PSNR 향상 기준
    psnr_grpo_wins = sum(1 for gi, ri in zip(g, r) if gi['psnr_diff'] > ri['psnr_diff'])
    psnr_random_wins = sum(1 for gi, ri in zip(g, r) if gi['psnr_diff'] < ri['psnr_diff'])
    psnr_ties = n - psnr_grpo_wins - psnr_random_wins

    # 이미지별 승패 — 성공률 기준
    sr_grpo_wins = sum(1 for gi, ri in zip(g, r) if gi['success_ratio'] > ri['success_ratio'])
    sr_random_wins = sum(1 for gi, ri in zip(g, r) if gi['success_ratio'] < ri['success_ratio'])
    sr_ties = n - sr_grpo_wins - sr_random_wins

    print(f"\n  [PSNR 향상 기준]  GRPO {psnr_grpo_wins}승  Random {psnr_random_wins}승  무승부 {psnr_ties}  (총 {n}장)")
    print(f"  [성공률 기준]     GRPO {sr_grpo_wins}승  Random {sr_random_wins}승  무승부 {sr_ties}  (총 {n}장)")
    print("=" * 80)


# ============================================================
# 메인
# ============================================================
def run_test_on_dataset(dataset_name, data_loader, hologram_model,
                        grpo_policy, max_steps, num_images, result_dir,
                        temperature=1.0):
    """단일 데이터셋에 대해 GRPO vs Random DBS 비교 실행"""
    results_grpo = []
    results_random = []
    data_iter = iter(data_loader)

    ds_dir = os.path.join(result_dir, dataset_name)
    os.makedirs(ds_dir, exist_ok=True)

    print(f"\n{'━' * 70}")
    print(f"  데이터셋: {dataset_name}  ({min(num_images, len(data_loader.dataset))}장, {max_steps} steps/image)")
    print(f"{'━' * 70}")

    for img_idx in range(num_images):
        try:
            target_image, file_path = next(data_iter)
        except StopIteration:
            break

        target_image = target_image.cuda()
        target_image_np = target_image.cpu().numpy()
        file_name = os.path.basename(file_path[0])

        print(f"\n{'─' * 60}")
        print(f"[{dataset_name} {img_idx+1}/{num_images}] {file_name}")
        print(f"{'─' * 60}")

        with torch.no_grad():
            model_output = hologram_model(target_image)
        pre_model = model_output.cpu().numpy()
        initial_state = (pre_model >= 0.5).astype(np.int8)

        # 이미지마다 새 액션 함수 생성 (실패 마스크 초기화)
        grpo_action_fn = make_grpo_action_fn(grpo_policy, temperature=temperature)
        random_action_fn = make_random_action_fn()

        # GRPO DBS
        torch.cuda.empty_cache()
        r_grpo = run_dbs(
            state=initial_state, pre_model=pre_model,
            target_image=target_image, target_image_np=target_image_np,
            max_steps=max_steps, select_action_fn=grpo_action_fn, label="GRPO",
        )
        print_result(r_grpo)

        # Random DBS
        torch.cuda.empty_cache()
        r_random = run_dbs(
            state=initial_state, pre_model=pre_model,
            target_image=target_image, target_image_np=target_image_np,
            max_steps=max_steps, select_action_fn=random_action_fn, label="Random",
        )
        print_result(r_random)

        results_grpo.append(r_grpo)
        results_random.append(r_random)

        with open(os.path.join(ds_dir, f"{file_name}.txt"), "w") as f:
            f.write(f"Image: {file_name}\nDataset: {dataset_name}\nMax Steps: {max_steps}\n\n")
            for r in [r_grpo, r_random]:
                f.write(f"[{r['label']}]\n")
                f.write(f"  Initial PSNR: {r['initial_psnr']:.6f}\n")
                f.write(f"  Final PSNR:   {r['final_psnr']:.6f}\n")
                f.write(f"  PSNR Diff:    {r['psnr_diff']:.6f}\n")
                f.write(f"  Flip Count:   {r['flip_count']}\n")
                f.write(f"  Success Ratio:{r['success_ratio']:.4f}\n")
                f.write(f"  Time:         {r['time']:.2f}s\n\n")

    if results_grpo:
        print_comparison(results_grpo, results_random, dataset_name)

        csv_path = os.path.join(ds_dir, f"summary_{dataset_name}.csv")
        with open(csv_path, "w") as f:
            f.write("dataset,image,method,initial_psnr,final_psnr,psnr_diff,flip_count,success_ratio,time\n")
            for rg, rr in zip(results_grpo, results_random):
                for r in [rg, rr]:
                    f.write(f"{dataset_name},{r['label']},{r['label']},"
                            f"{r['initial_psnr']:.6f},{r['final_psnr']:.6f},"
                            f"{r['psnr_diff']:.6f},{r['flip_count']},"
                            f"{r['success_ratio']:.6f},{r['time']:.2f}\n")

    return results_grpo, results_random


if __name__ == '__main__':
    # --- 설정 ---
    MAX_STEPS = 1000                # 이미지당 DBS 스텝 수
    GRPO_CHECKPOINT = "./grpo_models/grpo_latest.pt"

    target_dir = '/nfs/dataset/DIV2K/DIV2K_train_HR/DIV2K_train_HR/'
    valid_dir = '/nfs/dataset/DIV2K/DIV2K_valid_HR/DIV2K_valid_HR/'
    meta = {'wl': 515e-9, 'dx': (7.56e-6, 7.56e-6)}
    padding = 0

    # --- 데이터 ---
    train_dataset = Dataset512(target_dir=target_dir, meta=meta, isTrain=False, padding=padding)
    valid_dataset = Dataset512(target_dir=valid_dir, meta=meta, isTrain=False, padding=padding)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
    print(f"Train images: {len(train_dataset)},  Validation images: {len(valid_dataset)}")

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

    # --- GRPO 정책 로드 ---
    grpo_policy = GRPOPolicy(num_channels=CH, img_size=IPS, mid_channels=64).cuda()
    ckpt = torch.load(GRPO_CHECKPOINT, map_location='cuda')
    grpo_policy.load_state_dict(ckpt['policy_state_dict'])
    grpo_policy.eval()
    print(f"GRPO checkpoint loaded: {GRPO_CHECKPOINT} (episode {ckpt['episode_count']})")

    result_dir = f"./test_results/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/"
    os.makedirs(result_dir, exist_ok=True)

    # ============================================================
    # 1) Train 데이터셋 테스트
    # ============================================================
    train_grpo, train_random = run_test_on_dataset(
        dataset_name="Train",
        data_loader=train_loader,
        hologram_model=hologram_model,
        grpo_policy=grpo_policy,
        max_steps=MAX_STEPS,
        num_images=len(train_dataset),
        result_dir=result_dir,
    )

    # ============================================================
    # 2) Validation 데이터셋 테스트
    # ============================================================
    valid_grpo, valid_random = run_test_on_dataset(
        dataset_name="Validation",
        data_loader=valid_loader,
        hologram_model=hologram_model,
        grpo_policy=grpo_policy,
        max_steps=MAX_STEPS,
        num_images=len(valid_dataset),
        result_dir=result_dir,
    )

    # ============================================================
    # 3) Train + Validation 전체 종합 비교
    # ============================================================
    all_grpo = train_grpo + valid_grpo
    all_random = train_random + valid_random
    if all_grpo:
        print_comparison(all_grpo, all_random, "Train + Validation 전체")

    print(f"\nAll results saved to: {result_dir}")
