"""오라클 자체 검증 (서버에서 실행: torch + torchOptics 8e50d6a 필요). PLAN.md 2.1절의 게이트.

    python grpo/oracle_selftest.py            # 저장소 루트에서

검사 (합성 케이스 + 실제 케이스 각각):
  1. FlipOracle.forward(h) == tt.simulate(h, z)  — 같은 H 면 FFT 반올림 차이(~1e-6)만 남는다 → 상대 오차 < 1e-5
  2. <A h, g> == <h, A^H g>  (수반 일치)
  3. 오라클 현재 PSNR == env 경로(float32 kornia) PSNR  (|차| < 1e-4 dB) 및 == float64 재계산 (|차| < 1e-6 dB)
  4. 무작위 (채널, 픽셀) NUM_FLIPS 개 + 오라클 최선/최악 각 8개: 오라클 ΔPSNR vs 브루트포스(실제 플립 후 tt.simulate, PSNR 은 float64 로 조립)
       - 오차: |오라클 − 브루트| ≤ ABS_TOL + REL_TOL·|브루트|  (모든 표본)
       - 부호: |브루트| > SIGN_FLOOR 인 표본에서 100% 일치 (잡음 바닥 근처 표본은 제외하고 개수를 출력; 제외가 과반이면 FAIL)
       - 순서: 피어슨·스피어만 상관 > 0.999
  5. 속도: flip_psnr_map 1회 시간
하나라도 임계를 넘으면 FAIL 로 종료 코드 1. 통과 전에는 v2 학습을 돌리지 않는다. 폴백은 없다 — 실패하면 오라클을 고친다.
"""
import os
import sys
import time

sys.path.insert(0, os.getcwd())   # 저장소 루트에서 실행한다는 전제 (utils/env 와 같은 규칙)

from utils.logger import setup_logger
log_file = setup_logger()
from utils.torchoptics_pin import assert_torchoptics_pinned
assert_torchoptics_pinned()       # 고정 커밋이 아니면 여기서 죽는다 — 다른 torchOptics 에 맞춘 오라클이 통과하면 안 된다

import numpy as np
import torch

from optics_constants import OPTICS_META, PROP_Z
from grpo.oracle import FlipOracle, env_style_psnr

# ── 설정 (여기만 수정) ─────────────────────────────────────────────
IPS, CH = 256, 8
NUM_FLIPS = 200         # 무작위 브루트포스 대조 픽셀 수 (+ 최선/최악 각 8개)
SEED = 0
TOL_FIELD = 1e-5        # forward vs tt.simulate 상대 오차 (같은 H 라면 FFT 반올림만 남는다)
TOL_ADJOINT = 1e-4
TOL_PSNR32 = 1e-4       # 오라클 PSNR vs env 경로(float32 kornia) (dB)
TOL_PSNR64 = 1e-6       # 오라클 PSNR vs 같은 시뮬레이션 결과의 float64 PSNR (dB)
ABS_TOL = 5e-6          # ΔPSNR 절대 허용오차 (dB): float32 상관의 상쇄 한계 ~1e-6 의 몇 배
REL_TOL = 0.01          # ΔPSNR 상대 허용오차
SIGN_FLOOR = 1e-5       # 이보다 작은 |ΔPSNR| 은 부호 검사에서 제외 (float32 잡음 바닥 ~1e-6 의 10배)
MIN_CORR = 0.999
PRETRAINED = ('result_v/2024-12-19 20:37:52.499731_pre_reinforce_8_0.002/'
              '2024-12-19 20:37:52.499731_pre_reinforce_8_0.002')   # 있으면 실제 초기 홀로그램으로도 검사
VALID_DIR = '/nfs/dataset/DIV2K/DIV2K_valid_HR/DIV2K_valid_HR/'
# ─────────────────────────────────────────────────────────────────

fails = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        fails.append(msg)


def synthetic_case(device):
    """의존성 없는 합성 케이스: 저주파 목표 + 50% 무작위 홀로그램."""
    g = torch.Generator(device="cpu").manual_seed(SEED)
    yy, xx = torch.meshgrid(torch.linspace(0, 1, IPS), torch.linspace(0, 1, IPS), indexing="ij")
    T = (0.5 + 0.3 * torch.sin(6.28 * xx * 3) * torch.cos(6.28 * yy * 2)).clamp(0, 1)
    h = (torch.rand(CH, IPS, IPS, generator=g) < 0.5).float()
    return h.to(device), T.to(device)


def real_case(device):
    """사전학습 BinaryNet + 검증 이미지 1장으로 실제 초기 홀로그램 (파일이 없으면 None)."""
    if not os.path.exists(PRETRAINED) or not os.path.isdir(VALID_DIR):
        return None
    from train_grpo import BinaryNet, Dataset512     # import 만 (학습은 __main__ 에서만 돈다)
    ds = Dataset512(target_dir=VALID_DIR, meta=OPTICS_META, isTrain=False, padding=0)
    T, _ = ds[0]
    T = T.unsqueeze(0).to(device)                       # (1,1,n,n)
    net = BinaryNet(num_hologram=CH, in_planes=1, convReLU=False, convBN=False,
                    poolReLU=False, poolBN=False, deconvReLU=False, deconvBN=False).to(device)
    net.load_state_dict(torch.load(PRETRAINED, map_location=device))
    net.eval()
    with torch.no_grad():
        h = (net(T) >= 0.5).float()[0]                   # (C,n,n)
    return h, T[0, 0]


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def run_case(name, h, T, oracle, device):
    print(f"== {name} ==")
    rng = np.random.default_rng(SEED)
    T4 = T.reshape(1, 1, IPS, IPS)
    import torchOptics.optics as tt
    # 1. forward == simulate
    b = tt.Tensor(h.unsqueeze(0).clone(), meta=OPTICS_META)
    U_ref = tt.simulate(b, PROP_Z)[0]                          # (C,n,n) complex
    U = oracle.forward(h)
    rel = ((U - U_ref).abs().max() / U_ref.abs().max()).item()
    check(rel < TOL_FIELD, f"forward == tt.simulate: 상대 최대 오차 {rel:.2e} (< {TOL_FIELD})")
    # 2. adjoint
    g = torch.randn(CH, IPS, IPS, dtype=torch.complex64, device=device)
    lhs = (oracle.forward(h) * g.conj()).sum()
    rhs = (h.to(torch.complex64) * oracle.adjoint(g).conj()).sum()
    rel_adj = (abs(lhs - rhs) / abs(lhs)).item()
    check(rel_adj < TOL_ADJOINT, f"<A h, g> == <h, A^H g>: 상대 오차 {rel_adj:.2e} (< {TOL_ADJOINT})")
    # 3. PSNR: float32 env 경로 및 float64 재계산과 대조
    out = oracle.flip_psnr_map(h, T)
    psnr32, _, base_result = env_style_psnr(h.unsqueeze(0), T4)
    psnr64, _ = oracle.psnr_from(base_result[0, 0], T)
    check(abs(out["psnr"] - psnr32) < TOL_PSNR32, f"오라클 PSNR {out['psnr']:.6f} vs env 경로(float32) {psnr32:.6f}: |차| {abs(out['psnr'] - psnr32):.2e} < {TOL_PSNR32}")
    check(abs(out["psnr"] - psnr64) < TOL_PSNR64, f"오라클 PSNR vs 같은 시뮬레이션의 float64 PSNR {psnr64:.6f}: |차| {abs(out['psnr'] - psnr64):.2e} < {TOL_PSNR64}")
    # 4. 브루트포스 대조 (기준값은 float64 로 조립 — float32 PSNR 두 값의 차는 ~2e-6 dB 양자화 잡음을 갖는다)
    dp = out["dpsnr"].cpu().numpy()
    flat = dp.reshape(-1)
    picks = [int(rng.integers(dp.size)) for _ in range(NUM_FLIPS)]
    picks += [int(i) for i in np.argsort(flat)[-8:]] + [int(i) for i in np.argsort(flat)[:8]]
    bf, orc = [], []
    for a in picks:
        ci, rest = divmod(a, IPS * IPS)
        r, cc = divmod(rest, IPS)
        h2 = h.clone()
        h2[ci, r, cc] = 1.0 - h2[ci, r, cc]
        _, _, res2 = env_style_psnr(h2.unsqueeze(0), T4)
        p2, _ = oracle.psnr_from(res2[0, 0], T)
        bf.append(p2 - psnr64)
        orc.append(float(flat[a]))
    bf, orc = np.array(bf), np.array(orc)
    err = np.abs(bf - orc)
    tol = ABS_TOL + REL_TOL * np.abs(bf)
    big = np.abs(bf) > SIGN_FLOOR
    sign_ok = float(np.mean(np.sign(bf[big]) == np.sign(orc[big]))) if big.any() else float("nan")
    pear = float(np.corrcoef(bf, orc)[0, 1])
    spear = spearman(bf, orc)
    print(f"          ΔPSNR 브루트포스(float64): 중앙값 |{np.median(np.abs(bf)):.3e}|, 최소 {bf.min():+.3e}, 최대 {bf.max():+.3e} dB; 양수 비율(무작위 표본) {np.mean(bf[:NUM_FLIPS] > 0):.2%}")
    print(f"          오라클과 차이: 최대 {err.max():.2e} dB (허용 {tol[np.argmax(err)]:.2e}), 중앙값 {np.median(err):.2e}; "
          f"부호 검사 표본 {int(big.sum())}/{len(bf)} (|ΔPSNR|>{SIGN_FLOOR:g} 만); 피어슨 {pear:.6f}, 스피어만 {spear:.6f}")
    check(bool(np.all(err <= tol)), f"ΔPSNR 오차 ≤ {ABS_TOL:g} + {REL_TOL:g}·|브루트| (전 표본)")
    check(big.mean() > 0.5, f"부호 검사 대상이 과반 ({big.mean():.1%})")
    check(sign_ok == 1.0, f"부호 일치 100% (실제 {sign_ok:.1%})")
    check(pear > MIN_CORR and spear > MIN_CORR, f"피어슨·스피어만 상관 > {MIN_CORR}")
    # 5. 속도
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(5):
        oracle.flip_psnr_map(h, T)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / 5
    print(f"          flip_psnr_map 1회: {dt * 1000:.1f} ms  (브루트포스 1픽셀 ≈ 시뮬레이션 1회)")
    print(f"          전체 지도: P_unif(ΔPSNR>0) = {np.mean(dp > 0):.2%}, E_unif = {dp.mean():+.3e} dB, E_unif[R+] = {np.clip(dp, 0, None).mean():+.3e}, top-1 = {dp.max():+.3e} dB")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}, torch {torch.__version__}, log = {log_file}")
    oracle = FlipOracle(OPTICS_META, PROP_Z, IPS, CH, device=device)
    h, T = synthetic_case(device)
    run_case("합성 케이스 (무작위 홀로그램 + 저주파 목표)", h, T, oracle, device)
    rc = real_case(device)
    if rc is None:
        print("== 실제 케이스: 사전학습 가중치 또는 검증 데이터가 없어 건너뜀 (서버에서는 둘 다 있어야 한다) ==")
        fails.append("실제 케이스 미실행")
    else:
        run_case("실제 케이스 (BinaryNet 초기 홀로그램 + DIV2K 검증 1장)", rc[0], rc[1], oracle, device)
    print()
    if fails:
        print(f"FAIL {len(fails)}건:")
        for f in fails:
            print("  - " + f)
        sys.exit(1)
    print("ALL PASS - 오라클을 학습 보상으로 써도 된다")


if __name__ == "__main__":
    main()
