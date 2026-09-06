"""v2 파이프라인 스모크 테스트 — 작은 크기로 전체 경로(오라클 → 특징 → 정책 3종(+상태 게이트) → 트레이너 몇 반복 → 체크포인트) 를 돌린다.
서버(GPU) 는 물론 CPU 에서도 돈다 (torch + torchOptics 필요). 데이터셋·사전학습 모델 불필요(합성 목표 + 무작위 홀로그램).

    python grpo/smoke_v2.py          # 저장소 루트에서. 1~2분.

무엇을 확인하나: 형상·dtype·loss 유한성·검증 지표 출력·체크포인트 저장/로드·거절 시 상태 복원. 학습 성능은 보지 않는다.
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.getcwd())

from utils.torchoptics_pin import assert_torchoptics_pinned
assert_torchoptics_pinned()

import numpy as np
import torch

from optics_constants import OPTICS_META, PROP_Z
from grpo.oracle import FlipOracle
from grpo.features import build_features, num_channels
from grpo.policies import make_policy, count_params, POLICY_KINDS
from grpo.dbs_state import DBSImage
from grpo.trainer_v2 import GRPOTrainerV2

IPS, CH = 64, 4      # 작은 크기 (실제는 256, 8)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={device}")

oracle = FlipOracle(OPTICS_META, PROP_Z, IPS, CH, padding=16, device=device)
g = torch.Generator().manual_seed(0)


def synth():
    yy, xx = torch.meshgrid(torch.linspace(0, 1, IPS), torch.linspace(0, 1, IPS), indexing="ij")
    T = (0.5 + 0.3 * torch.sin(6.28 * xx * 2) * torch.cos(6.28 * yy * 3)).clamp(0, 1)
    pre = torch.rand(CH, IPS, IPS, generator=g)
    h0 = (pre >= 0.5).float()
    return T.to(device), h0.to(device), pre.to(device), "synthetic"


# 1. 오라클 + 상태 (채택/거절/복원)
T, h0, pre, _ = synth()
img = DBSImage(oracle, T, h0)
img.pre_model = pre
R = img.reward_map()
print(f"oracle map {tuple(R.shape)} psnr={img.psnr:.4f} P_unif={float((R > 0).float().mean()):.2%} top1={float(R.max()):+.3e}")
ok, p2 = img.apply(int(torch.argmax(R)))
print(f"apply(argmax): accepted={ok} psnr {img.initial_psnr:.4f}->{p2:.4f} (oracle 예측 +{float(R.max()):.3e})")
assert ok, "오라클 최선 액션이 채택되지 않음 — 오라클/상태 불일치"
before = (img.psnr, img.reward_map().sum().item())
ok2, _ = img.apply(int(torch.argmin(img.reward_map())))
assert not ok2 and abs(img.psnr - before[0]) == 0 and abs(img.reward_map().sum().item() - before[1]) == 0, "거절 시 상태·지도 복원 실패"
print("apply(argmin): rejected and state/map restored")
img.random_accept_steps(5, np.random.default_rng(0))
print(f"random_accept_steps(5): steps={img.steps} flips={img.flips} gain={img.gain:+.4f}")

# 1b. 실제 데이터 경로처럼 tt.Tensor(meta 포함) 서브클래스가 들어와도 하류는 plain 텐서여야 한다 (Tensor.__format__ 크래시 방지)
import torchOptics.optics as tt
img_tt = DBSImage(oracle, tt.Tensor(T.clone(), meta=OPTICS_META), tt.Tensor(h0.clone(), meta=OPTICS_META))
assert type(img_tt.T) is torch.Tensor and type(img_tt.h) is torch.Tensor and type(img_tt.reward_map()) is torch.Tensor
print(f"tt.Tensor input stripped to plain torch.Tensor: map mean {float(img_tt.reward_map().mean()):+.3e}")

# 2. 특징 + 정책 3종 × 게이트
for spec in ("legacy", "field"):
    feats = build_features(spec, state=img.h, pre_model=img.pre_model, target=img.T, I=img.I, U=img.U, c=img.c)
    assert feats.shape == (1, num_channels(spec, CH), IPS, IPS), feats.shape
    for kind in POLICY_KINDS:
        for gate in (True, False):
            if kind == "fno" and (spec != "field" or not gate):
                continue
            pol = make_policy(kind, feats.shape[1], CH, IPS, feature_spec=spec, state_gate=gate, unet_base=8, fno_hidden=4).to(device)
            out = pol(feats)
            assert out.shape == (1, CH * IPS * IPS) and torch.isfinite(out).all(), (kind, out.shape)
            print(f"policy {kind:<4} spec {spec:<6} gate={gate!s:<5}: in_ch={feats.shape[1]} params={count_params(pol):,} sat={pol.saturation(feats):.3f}")

# 3. 트레이너 몇 반복: (grpo, sample, best), (grpo, policy, sample), (exact, uniform, best)
cfg = dict(policy_kind="unet", feature_spec="field", state_gate=True, objective="grpo", reward_transform="relu",
           adv_baseline="sample", images_per_batch=2, group_size=16, steps_per_image=5, update_epochs=1,
           minibatch_states=2, lr=1e-3, clip_range=0.2, kl_coef=0.04, max_grad_norm=0.5, ref_update_iters=2,
           entropy_coef=0.0, nonfinite_limit=20, adv_std_floor_rel=0.0, advance="best", num_iters=4, val_images=2, val_advance_steps=3, val_every=2,
           save_every=4, startup_check=False, unet_base=8, fno_hidden=4)
tmp = tempfile.mkdtemp(prefix="grpo_v2_smoke_")
try:
    for objective, baseline, advance, transform in (("grpo", "sample", "best", "relu"), ("grpo", "policy", "sample", "raw"),
                                                    ("exact", "uniform", "best", "relu")):
        cfg.update(objective=objective, adv_baseline=baseline, advance=advance, reward_transform=transform)
        feats_ch = num_channels(cfg["feature_spec"], CH)
        pol = make_policy(cfg["policy_kind"], feats_ch, CH, IPS, feature_spec=cfg["feature_spec"],
                          state_gate=cfg["state_gate"], unet_base=cfg["unet_base"])
        vals = []
        rng = np.random.default_rng(1)
        for i in range(cfg["val_images"]):
            Tv, hv, pv, _ = synth()
            s = DBSImage(oracle, Tv, hv)
            s.pre_model = pv
            if i >= cfg["val_images"] // 2:
                s.random_accept_steps(cfg["val_advance_steps"], rng)
            vals.append(s)
        tr = GRPOTrainerV2(pol, oracle, cfg, feature_spec=cfg["feature_spec"], new_image_fn=synth,
                           val_states=vals, device=device, log_dir=tmp)
        tr.extra_meta = {"policy_kind": cfg["policy_kind"], "feature_spec": cfg["feature_spec"],
                         "in_channels": feats_ch, "state_gate": cfg["state_gate"]}
        tr.train(num_iters=cfg["num_iters"], save_dir=tmp, save_every=cfg["save_every"], val_every=cfg["val_every"])
        ck = torch.load(os.path.join(tmp, "grpo_v2_latest.pt"), map_location=device)
        assert ck["policy_kind"] == cfg["policy_kind"] and ck["iteration"] == cfg["num_iters"] and ck["state_gate"] is True
        print(f"[{objective}/{baseline}/{advance}/{transform}] 4 iterations ok, checkpoint ok, val.jsonl lines="
              f"{sum(1 for _ in open(os.path.join(tmp, 'val.jsonl'), encoding='utf-8'))}")
    try:
        GRPOTrainerV2(pol, oracle, dict(cfg, objective="exact", adv_baseline="sample"), feature_spec="field",
                      new_image_fn=synth, val_states=vals, device=device, log_dir=tmp)
        raise AssertionError("exact + sample 조합이 막히지 않음")
    except ValueError:
        print("exact + adv_baseline=sample 은 ValueError 로 막힘 (의도)")
    print("SMOKE PASS")
finally:
    shutil.rmtree(tmp, ignore_errors=True)
