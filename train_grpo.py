import sys
import logging
from datetime import datetime
import os
from utils.logger import setup_logger

log_file = setup_logger()
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

IPS = 256
CH = 8
warnings.filterwarnings('ignore')

current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
torch.backends.cudnn.enabled = False


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

        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self.episode_count = 0

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
    def evaluate_group_rewards(self, actions, z=2e-3):
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
            sub_batch = tt.Tensor(sub_batch, meta={'dx': (7.56e-6, 7.56e-6), 'wl': 515e-9})

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

        # 레퍼런스 log prob (KL 계산용)
        ref_dist = self.ref_policy.get_distribution(obs_tensor)
        ref_log_probs = ref_dist.log_prob(actions_gpu).squeeze(-1)

        # 4) 정책 업데이트 (다중 에폭, KL early stopping 포함)
        total_loss = 0.0
        actual_epochs = 0
        max_kl = 0.1  # KL이 이 값을 넘으면 조기 중단

        for _ in range(self.update_epochs):
            dist_new = self.policy.get_distribution(obs_tensor)
            new_log_probs = dist_new.log_prob(actions_gpu).squeeze(-1)

            # NaN 감지 → 해당 에폭 스킵
            if torch.isnan(new_log_probs).any():
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

            # 에피소드 시작 시 마스크 초기화
            self.failed_mask.zero_()
            self._prev_state_hash = None

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
            print(
                f"\033[41mEpisode {self.episode_count}: "
                f"Reward={episode_reward:.2f}, Steps={step_count}, "
                f"GRPO Updates={grpo_updates}, Loss={last_loss:.4f}, "
                f"Time={elapsed:.1f}s\033[0m"
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
    # --- 데이터 ---
    batch_size = 1
    target_dir = '/nfs/dataset/DIV2K/DIV2K_train_HR/DIV2K_train_HR/'
    valid_dir = '/nfs/dataset/DIV2K/DIV2K_valid_HR/DIV2K_valid_HR/'
    meta = {'wl': 515e-9, 'dx': (7.56e-6, 7.56e-6)}
    padding = 0

    train_dataset = Dataset512(target_dir=target_dir, meta=meta, isTrain=True, padding=padding)
    valid_dataset = Dataset512(target_dir=valid_dir, meta=meta, isTrain=False, padding=padding)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # --- Pre-trained BinaryNet 로드 ---
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

    # --- 환경 ---
    env = BinaryHologramEnv(
        target_function=hologram_model,
        trainloader=train_loader,
        max_steps=10000,
        T_PSNR=30,
        T_steps=1,
        T_PSNR_DIFF=1/4,
        num_samples=10000,
    )

    # --- GRPO 정책 & 트레이너 ---
    grpo_policy = GRPOPolicy(num_channels=CH, img_size=IPS, mid_channels=64)

    save_dir = "./grpo_models/"
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, "grpo_latest.pt")
    resume_training = True

    trainer = GRPOTrainer(
        policy=grpo_policy,
        env=env,
        group_size=16,          # 그룹 내 샘플 수 G
        sim_batch_size=4,       # 시뮬레이션 서브 배치 크기 (GPU 메모리에 맞게 조절)
        lr=1e-4,
        clip_range=0.2,         # PPO-style 클리핑
        kl_coef=0.04,           # KL(π_θ || π_ref) 계수
        update_epochs=4,        # GRPO 업데이트당 epoch 수
        max_grad_norm=0.5,
        ref_update_interval=10, # π_ref 업데이트 주기 (에피소드)
        grpo_update_interval=1, # 스텝당 GRPO 업데이트 빈도
    )

    if resume_training and os.path.exists(checkpoint_path):
        trainer.load_checkpoint(checkpoint_path)
    else:
        if resume_training:
            print(f"Warning: No checkpoint at {checkpoint_path}. Training from scratch.")

    # --- 학습 시작 ---
    trainer.train(
        num_episodes=8000,
        save_dir=save_dir,
        save_interval=100,
    )
