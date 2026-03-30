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
        sim = tt.simulate(binary, z).abs() ** 2
        result = torch.mean(sim, dim=1, keepdim=True)
        psnr = tt.relativeLoss(result, target_image, tm.get_PSNR)
    return float(psnr), result


def make_grpo_action_fn(policy, device='cuda'):
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
            logits = policy(x).squeeze(0)
            logits[failed_mask.bool()] = -1e9
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample().item()

        failed_mask[action] = 1.0
        return action

    return select


def run_dbs(state, pre_model, target_image, target_image_np,
            max_steps, select_action_fn):
    state = state.copy()
    state_record = np.zeros_like(state)

    initial_psnr, recon = simulate_psnr(state, target_image)
    current_psnr = initial_psnr
    recon_np = recon.cpu().numpy()

    flip_count = 0

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

        state[0, ch, row, col] = 1 - state[0, ch, row, col]
        psnr_after, recon_after = simulate_psnr(state, target_image)

        if psnr_after > current_psnr:
            current_psnr = psnr_after
            recon_np = recon_after.cpu().numpy()
            state_record[0, ch, row, col] += 1
            flip_count += 1
        else:
            state[0, ch, row, col] = 1 - state[0, ch, row, col]

    return {
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
    """grpo_ep*.pt 파일을 에피소드 순으로 정렬하여 반환"""
    pattern = os.path.join(model_dir, "grpo_ep*.pt")
    files = sorted(glob.glob(pattern))

    checkpoints = []
    for f in files:
        basename = os.path.basename(f)
        match = re.search(r'grpo_ep(\d+)\.pt', basename)
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
    MODEL_DIR       = "./grpo_models/"                  # 체크포인트 폴더 경로
    MAX_STEPS       = 500                               # 이미지당 DBS 스텝 수 (작을수록 빠름)
    NUM_EVAL_IMAGES = 10                                # 평가에 사용할 이미지 수 (0 = 전체)
    EVAL_DIR        = '/nfs/dataset/DIV2K/DIV2K_valid_HR/DIV2K_valid_HR/'  # 평가 데이터셋 경로
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
    print(f"Eval images loaded: {len(eval_images)}")

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
    print(f"Found {len(checkpoints)} checkpoints: ep{checkpoints[0][0]} ~ ep{checkpoints[-1][0]}")

    # --- 정책 네트워크 (재사용) ---
    grpo_policy = GRPOPolicy(num_channels=CH, img_size=IPS, mid_channels=64).cuda()

    # --- 체크포인트별 평가 ---
    all_results = []

    print(f"\n{'━' * 90}")
    print(f"  {'Checkpoint':<20} {'Episode':>8}  {'Avg PSNR↑':>10}  {'Avg Success%':>13}  "
          f"{'Avg Flips':>10}  {'Time':>8}")
    print(f"{'━' * 90}")

    for ep_num, ckpt_path in checkpoints:
        # 체크포인트 로드
        ckpt = torch.load(ckpt_path, map_location='cuda')
        grpo_policy.load_state_dict(ckpt['policy_state_dict'])
        grpo_policy.eval()

        ep_results = []
        t_start = time.time()

        for img in eval_images:
            torch.cuda.empty_cache()
            action_fn = make_grpo_action_fn(grpo_policy)

            result = run_dbs(
                state=img["initial_state"],
                pre_model=img["pre_model"],
                target_image=img["target_image"],
                target_image_np=img["target_image_np"],
                max_steps=MAX_STEPS,
                select_action_fn=action_fn,
            )
            ep_results.append(result)

        elapsed = time.time() - t_start

        avg_psnr_diff = np.mean([r["psnr_diff"] for r in ep_results])
        avg_success = np.mean([r["success_ratio"] for r in ep_results])
        avg_flips = np.mean([r["flip_count"] for r in ep_results])

        all_results.append({
            "episode": ep_num,
            "checkpoint": ckpt_path,
            "avg_psnr_diff": avg_psnr_diff,
            "avg_success_ratio": avg_success,
            "avg_flip_count": avg_flips,
            "time": elapsed,
            "per_image": ep_results,
        })

        print(f"  {os.path.basename(ckpt_path):<20} {ep_num:>8}  "
              f"{avg_psnr_diff:>+10.4f}  {avg_success:>12.2%}  "
              f"{avg_flips:>10.1f}  {elapsed:>7.1f}s")

    # --- 최고 성능 체크포인트 ---
    print(f"\n{'━' * 90}")

    best_psnr = max(all_results, key=lambda x: x["avg_psnr_diff"])
    best_success = max(all_results, key=lambda x: x["avg_success_ratio"])

    print(f"\n  🏆 PSNR 향상 최고:  ep{best_psnr['episode']}  "
          f"(+{best_psnr['avg_psnr_diff']:.4f} dB)  "
          f"→ {best_psnr['checkpoint']}")
    print(f"  🏆 성공률 최고:     ep{best_success['episode']}  "
          f"({best_success['avg_success_ratio']:.2%})  "
          f"→ {best_success['checkpoint']}")

    # --- 결과 CSV 저장 ---
    result_dir = f"./eval_results/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/"
    os.makedirs(result_dir, exist_ok=True)

    csv_path = os.path.join(result_dir, "checkpoint_comparison.csv")
    with open(csv_path, "w") as f:
        f.write("episode,checkpoint,avg_psnr_diff,avg_success_ratio,avg_flip_count,time\n")
        for r in all_results:
            f.write(f"{r['episode']},{os.path.basename(r['checkpoint'])},"
                    f"{r['avg_psnr_diff']:.6f},{r['avg_success_ratio']:.6f},"
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

    print(f"\n  Results saved to: {result_dir}")
    print(f"  - checkpoint_comparison.csv  (체크포인트별 요약)")
    print(f"  - per_image_detail.csv       (이미지별 상세)")
