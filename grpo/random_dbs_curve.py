"""Random DBS 가 '의미 있는' PSNR 상승에 이르려면 몇 번 시도하고 몇 번 성공해야 하는지 — 오라클로 정확히, 시뮬레이션 없이 잰다.

원리: Random DBS 에서 거절된 시도는 상태를 바꾸지 않는다. 오라클 지도(현재 상태에서 모든 픽셀의 ΔPSNR)가 있으면 어느 시도가
거절될지 미리 아므로 거절은 공짜로 세고, 채택될 플립만 실제로 적용해(FFT 1회 + 지도 재계산 ≈ 6~7 ms) 상태를 갱신한다.
결과는 실제 Random DBS 와 분포가 같다 ("sweep" 은 순열까지 같아 같은 시드면 궤적도 같다). 비용은 채택 수에만 비례한다.

두 변형:
  "sweep"       — DBS.py 의 고전 DBS: 패스마다 픽셀 순열을 새로 뽑아 비복원으로 한 번씩 시도 (1 패스 = 524,288 시도)
  "replacement" — test_grpo.py / eval_checkpoints.py 의 Random 기준선: 매 시도 복원 추출 (거절 수를 기하분포로 뽑는다)

    python grpo/random_dbs_curve.py          # 저장소 루트에서. 이미지당 패스당 수 분 (채택 플립 수 × ≈7 ms)

출력: 이미지별로 (시도, 채택, PSNR, 상승, 개선 픽셀 비율) 곡선 CSV 와, 상승 임계(0.1/0.5/1/2/3/5 dB)에 처음 닿은 시도·채택 수 표,
패스별 상승, 후반 한계 이득. include_oracle_greedy 를 켜면 같은 이미지의 오라클 탐욕 곡선(상한)도 같이 잰다.
"""
import os
import sys
import time

sys.path.insert(0, os.getcwd())

from utils.logger import setup_logger
log_file = setup_logger()
from utils.torchoptics_pin import assert_torchoptics_pinned
assert_torchoptics_pinned()

import numpy as np
import torch

from optics_constants import OPTICS_META, PROP_Z
from grpo.oracle import FlipOracle
from grpo.dbs_state import DBSImage
from grpo.data_prep import load_initial_states

# ── 설정 (여기만 수정) ─────────────────────────────────────────────
CONFIG = {
    "num_images": 3,             # 검증 폴더 앞 N 장 (예: 1, 3, 10)
    "variant": "sweep",          # "sweep"(DBS.py 고전 DBS, 패스마다 순열·비복원) | "replacement"(평가 스크립트의 Random 기준선, 복원 추출)
    "max_trials": 2 * 524288,    # 총 시도 수 상한 (예: 524288 = 1 패스, 5 * 524288 = 5 패스)
    "seed": 0,
    "accept_floor": 1e-6,        # 오라클 ΔPSNR 이 이보다 커야 "채택될 시도" 로 본다 (dB). 실제 DBS 의 채택 판정(재계산)은 float32 잡음 ≈1e-6 dB 를
                                 # 가져서 그 아래 플립은 동전 던지기이고 PSNR 에는 기여하지 않는다. 0 이면 오라클 부호 그대로 (거절이 늘어 느려진다)
    "thresholds_db": (0.1, 0.5, 1.0, 2.0, 3.0, 5.0),   # 이 상승에 처음 닿은 시도·채택 수를 표로
    "log_every_accepted": 5000,  # 진행 출력 주기 (채택 플립 수)
    "csv_every_accepted": 100,   # CSV 기록 주기 (채택 플립 수)
    "out_dir": "./eval_results/random_dbs_curve/",
    "include_oracle_greedy": False,   # True: 같은 이미지에서 오라클 탐욕(매 스텝 최선 픽셀) 곡선도 greedy_steps 만큼 (상한 참고, 스텝당 ≈7 ms)
    "greedy_steps": 20000,
}
# ─────────────────────────────────────────────────────────────────

IPS, CH = 256, 8


def _fmt_hit(hit, th):
    return f"{hit[th][0]:>9,} / {hit[th][1]:>7,}" if th in hit else f"{'-':>9} / {'-':>7}"


def random_curve(img, variant, max_trials, rng, thresholds, log_every, csv_every, label, floor):
    """반환 rows [(trials, accepted, psnr, gain, pos_frac)], hit {threshold: (trials, accepted)}, pass_gains [gain at end of pass],
    mismatches (오라클은 채택이라 했는데 재계산이 거절한 횟수 — 잡음 바닥 근처 플립)."""
    N = img.h.numel()
    rows, hit, pass_gains = [], {}, []
    trials = accepted = mismatches = 0
    R = img.reward_map().reshape(-1)
    pos = R > floor

    def record():
        rows.append((trials, accepted, img.psnr, img.gain, float(pos.float().mean())))

    record()
    t0, last_log = time.time(), 0
    perm, i = None, N
    while trials < max_trials:
        n_pos = int(pos.sum())
        if n_pos == 0:
            print(f"    [{label}] 개선 픽셀 0개 - 지역 최적 (시도 {trials:,}, 채택 {accepted:,})")
            break
        if variant == "sweep":
            if i >= N:
                perm = torch.as_tensor(rng.permutation(N), device=img.h.device)
                i = 0
            nz = torch.nonzero(pos[perm[i:]], as_tuple=False)
            if nz.numel() == 0:                       # 이 패스의 남은 픽셀은 전부 거절
                trials += N - i
                i = N
                pass_gains.append(img.gain)
                record()
                continue
            j = int(nz[0])
            trials += j + 1
            a = int(perm[i + j])
            i += j + 1
            end_of_pass = i >= N
        else:
            k = int(rng.geometric(n_pos / N))         # 성공까지의 시도 수 (성공 포함)
            if trials + k > max_trials:
                trials = max_trials
                break
            trials += k
            cand = torch.nonzero(pos, as_tuple=False).reshape(-1)
            a = int(cand[int(rng.integers(n_pos))])
            end_of_pass = False
        r_pred = float(R[a])
        ok, _ = img.apply(a)
        if not ok:
            # 오라클 ΔPSNR 은 1e-7 dB 까지 맞지만 채택 판정은 재계산(float32 FFT) 이라 ≈1e-6 dB 잡음이 있다. 실제 DBS 도 이런 플립은
            # 거절할 수 있으므로 "거절된 시도" 로 센다 (상태는 apply 가 복원했다). 큰 R 에서 거절되면 진짜 불일치 → 아래 상한에서 죽는다.
            mismatches += 1
            if mismatches <= 3 or r_pred > 1e-4:
                print(f"    [{label}] 오라클 R={r_pred:+.2e} dB 인 픽셀 {a} 가 재계산에서 거절됨 (누적 {mismatches}회)")
            if mismatches > 10 + 0.01 * accepted:
                raise RuntimeError(f"오라클 채택 예측과 재계산 거절이 {mismatches}회 (채택 {accepted}회) - 잡음 바닥이 아니라 불일치, "
                                   "grpo/oracle_selftest.py 를 볼 것")
            continue
        accepted += 1
        R = img.reward_map().reshape(-1)
        pos = R > floor
        for th in thresholds:
            if th not in hit and img.gain >= th:
                hit[th] = (trials, accepted)
        if end_of_pass:
            pass_gains.append(img.gain)
        if accepted % csv_every == 0 or end_of_pass:
            record()
        if accepted - last_log >= log_every:
            last_log = accepted
            el = time.time() - t0
            rate = accepted / el
            remaining = "?"
            if variant == "sweep":
                # 남은 시도 × 현재 개선 비율 ≈ 남은 채택 수 (개선 비율은 줄어드니 상한)
                remaining = f"≤{(max_trials - trials) * float(pos.float().mean()) / rate / 60:.0f}분"
            print(f"    [{label}] 채택 {accepted:>7,}  시도 {trials:>9,}  PSNR {img.psnr:.4f} ({img.gain:+.4f} dB)  "
                  f"개선비율 {float(pos.float().mean()):6.2%}  {rate:5.0f} 채택/s  경과 {el / 60:.1f}분 남은 {remaining}")
    record()
    if mismatches:
        print(f"    [{label}] 잡음 바닥 근처 거절 {mismatches}회 (채택 {accepted:,}회의 {mismatches / max(accepted, 1):.2%})")
    return rows, hit, pass_gains, mismatches


def greedy_curve(img, steps, thresholds, csv_every, label, floor):
    rows, hit = [], {}
    R = img.reward_map().reshape(-1)
    rows.append((0, 0, img.psnr, img.gain, float((R > floor).float().mean())))
    t0 = time.time()
    for s in range(1, steps + 1):
        a = int(torch.argmax(R))
        if float(R[a]) <= floor:
            print(f"    [{label}] 개선 픽셀 0개 - 지역 최적 (스텝 {s - 1:,})")
            break
        ok, _ = img.apply(a)
        if not ok:
            raise RuntimeError(f"오라클 최선 픽셀 {a} 가 실제로 거절됨 - 오라클/상태 불일치")
        R = img.reward_map().reshape(-1)
        for th in thresholds:
            if th not in hit and img.gain >= th:
                hit[th] = (s, s)
        if s % csv_every == 0:
            rows.append((s, s, img.psnr, img.gain, float((R > floor).float().mean())))
        if s % 5000 == 0:
            print(f"    [{label}] 스텝 {s:>7,}  PSNR {img.psnr:.4f} ({img.gain:+.4f} dB)  개선비율 {float((R > floor).float().mean()):6.2%}  "
                  f"{s / (time.time() - t0):5.0f} 스텝/s")
    rows.append((s, s, img.psnr, img.gain, float((R > floor).float().mean())))
    return rows, hit


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write("trials,accepted,psnr,gain,pos_frac\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]:.6f},{r[3]:.6f},{r[4]:.6f}\n")


def main():
    cfg = CONFIG
    if cfg["variant"] not in ("sweep", "replacement"):
        raise ValueError(f"variant 는 sweep | replacement: {cfg['variant']!r}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg["out_dir"], exist_ok=True)
    N = CH * IPS * IPS
    print(f"device={device}  variant={cfg['variant']}  max_trials={cfg['max_trials']:,} (= {cfg['max_trials'] / N:.2f} 패스)  "
          f"images={cfg['num_images']}  seed={cfg['seed']}  log={log_file}")
    oracle = FlipOracle(OPTICS_META, PROP_Z, IPS, CH, device=device)
    states = load_initial_states(cfg["num_images"], device)
    ths = cfg["thresholds_db"]
    summary = []
    for T, h0, pre, name in states:
        print(f"\n== {name} ==")
        rng = np.random.default_rng(cfg["seed"])
        img = DBSImage(oracle, T, h0, name=name)
        R0 = img.reward_map().reshape(-1)
        print(f"    초기 PSNR {img.psnr:.4f} dB, 개선 픽셀(R>{cfg['accept_floor']:g}) {float((R0 > cfg['accept_floor']).float().mean()):.2%}, "
              f"무작위 성공 1회당 기대 이득 {float(R0[R0 > 0].mean()):.3e} dB, top-1 {float(R0.max()):.3e} dB")
        t0 = time.time()
        rows, hit, pass_gains, mism = random_curve(img, cfg["variant"], cfg["max_trials"], rng, ths,
                                                   cfg["log_every_accepted"], cfg["csv_every_accepted"], f"{cfg['variant']}",
                                                   cfg["accept_floor"])
        el = time.time() - t0
        csv_path = os.path.join(cfg["out_dir"], f"{cfg['variant']}_{os.path.splitext(name)[0]}.csv")
        write_csv(csv_path, rows)
        last = rows[-1]
        # 후반 한계 이득: 마지막 10% 시도 구간의 상승 vs 처음 10%
        cut_lo = next((r for r in rows if r[0] >= 0.1 * last[0]), last)
        cut_hi = next((r for r in rows if r[0] >= 0.9 * last[0]), last)
        print(f"    끝: 시도 {last[0]:,}, 채택 {last[1]:,} ({last[1] / max(last[0], 1):.2%}), PSNR {last[2]:.4f} ({last[3]:+.4f} dB), "
              f"개선 픽셀 {last[4]:.2%}, {el / 60:.1f}분, CSV {csv_path}")
        print(f"    처음 10% 시도의 상승 {cut_lo[3]:+.4f} dB  vs  마지막 10% 시도의 상승 {last[3] - cut_hi[3]:+.4f} dB")
        if pass_gains:
            print("    패스별 누적 상승: " + ", ".join(f"{k + 1}패스 {g:+.4f}" for k, g in enumerate(pass_gains)))
        print(f"    {'상승 임계':>8}  {'시도':>9} / {'채택':>7}")
        for th in ths:
            print(f"    {th:>7.1f} dB  {_fmt_hit(hit, th)}")
        row = {"name": name, "variant": cfg["variant"], "trials": last[0], "accepted": last[1], "gain": last[3], "hit": hit,
               "mismatches": mism}
        if cfg["include_oracle_greedy"]:
            g_img = DBSImage(oracle, T, h0, name=name)
            g_rows, g_hit = greedy_curve(g_img, cfg["greedy_steps"], ths, cfg["csv_every_accepted"], "oracle_greedy", cfg["accept_floor"])
            write_csv(os.path.join(cfg["out_dir"], f"oracle_greedy_{os.path.splitext(name)[0]}.csv"), g_rows)
            gl = g_rows[-1]
            print(f"    오라클 탐욕 {gl[0]:,}스텝: PSNR {gl[2]:.4f} ({gl[3]:+.4f} dB), 개선 픽셀 {gl[4]:.2%}")
            for th in ths:
                print(f"    {th:>7.1f} dB  {_fmt_hit(g_hit, th)}   (오라클 탐욕: 스텝 = 채택)")
            row["greedy_gain"], row["greedy_hit"] = gl[3], g_hit
        summary.append(row)

    print(f"\n== 요약 ({cfg['variant']}, {len(summary)}장, 시도 {cfg['max_trials']:,}) ==")
    print(f"    평균 상승 {np.mean([r['gain'] for r in summary]):+.4f} dB, 평균 채택 {np.mean([r['accepted'] for r in summary]):,.0f}")
    print(f"    {'상승 임계':>8}  {'도달 이미지':>6}  {'평균 시도':>10} / {'평균 채택':>8}")
    for th in ths:
        got = [r["hit"][th] for r in summary if th in r["hit"]]
        if got:
            print(f"    {th:>7.1f} dB  {len(got):>3}/{len(summary):<3}  {np.mean([g[0] for g in got]):>10,.0f} / {np.mean([g[1] for g in got]):>8,.0f}")
        else:
            print(f"    {th:>7.1f} dB  {0:>3}/{len(summary):<3}  {'미도달':>10}")
    print("    읽는 법: '의미 있는' 상승을 어느 임계로 볼지 정하면, 그 행의 시도 수가 Random DBS 에 필요한 예산이고 채택 수가 학습·평가가 다뤄야 할 성공 횟수다.")


if __name__ == "__main__":
    main()
