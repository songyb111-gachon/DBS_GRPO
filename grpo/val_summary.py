# -*- coding: utf-8 -*-
"""v2 학습의 검증 궤적(val.jsonl)을 한 표로 찍는다 (의존성 없음, 서버·로컬 어디서나).

    python grpo/val_summary.py                # 아래 CONFIG 의 폴더
    python grpo/val_summary.py <폴더 또는 val.jsonl 경로>

한 줄 = val_every 반복마다의 고정 검증 상태 지표. 판정(PLAN.md §0): E_pi_relu 가 E_unif_relu 를 넘어 오르고,
recovery ≥ 0.1 (0 = 균등, 1 = 오라클 탐욕), P_pi > P_unif. v2.1 부터는 깊이별(recovery_d<깊이>) 표도 같이 찍는다.
"""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CONFIG = {
    "path": "./grpo_models_v2_unet_r2/val.jsonl",   # 학습 save_dir 의 val.jsonl (폴더만 줘도 된다). 1차 런: ./grpo_models_v2_unet/val.jsonl
    "every": 1,                                      # n 번째 줄마다 (예: 1, 5, 10). 처음·마지막·최고 recovery 줄은 항상 찍는다
}


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else CONFIG["path"]
    if os.path.isdir(path):
        path = os.path.join(path, "val.jsonl")
    if not os.path.exists(path):
        raise SystemExit(f"없음: {path}")
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if not rows:
        raise SystemExit(f"비어 있음: {path}")
    best = max(range(len(rows)), key=lambda i: rows[i].get("recovery", float("-inf")))
    shown = [i for i in range(len(rows)) if i % CONFIG["every"] == 0 or i in (0, len(rows) - 1, best)]
    print(f"{path}: {len(rows)}줄 (iter {rows[0]['iter']} ~ {rows[-1]['iter']})")
    print(f"{'iter':>7} {'E_pi_relu':>10} {'E_unif_relu':>11} {'top1':>10} {'recov':>6} {'P_pi':>7} {'P_unif':>7} "
          f"{'E_pi_raw':>10} {'H':>5} {'eff':>7} {'sat':>5}")
    for i in shown:
        r = rows[i]
        mark = " <- best recovery" if i == best else ""
        print(f"{r['iter']:>7} {r['E_pi_relu']:>+10.3e} {r['E_unif_relu']:>+11.3e} {r['top1']:>+10.3e} {r['recovery']:>6.3f} "
              f"{r['P_pi']:>7.2%} {r['P_unif']:>7.2%} {r['E_pi_raw']:>+10.3e} {r['entropy']:>5.2f} {r['eff_support']:>7.0f} "
              f"{r['saturation']:>5.3f}{mark}")
    depths = sorted(int(k[len("recovery_d"):]) for k in rows[-1] if k.startswith("recovery_d"))
    if depths:
        print(f"\n깊이별 (recovery / P_pi / eff):")
        print(f"{'iter':>7} " + " ".join(f"{('d' + str(d)):>22}" for d in depths))
        for i in shown:
            r = rows[i]
            cells = []
            for d in depths:
                rec, p, eff = r.get(f"recovery_d{d}"), r.get(f"P_pi_d{d}"), r.get(f"eff_support_d{d}")
                cells.append(f"{rec:>6.3f} {p:>7.1%} {eff:>7.0f}" if rec is not None else f"{'-':>22}")
            print(f"{r['iter']:>7} " + " ".join(cells))
    dbs_rows = [r for r in rows if "dbs_gain" in r]
    if dbs_rows:
        print(f"\n학습 중 DBS 판정 (깊이 0 검증 이미지, 정책 vs Random, 같은 스텝):")
        print(f"{'iter':>7} {'policy':>9} {'random':>9} {'ratio':>6} {'acc':>6}")
        for r in dbs_rows:
            ratio = r['dbs_gain'] / r['dbs_random_gain'] if r['dbs_random_gain'] else float('nan')
            print(f"{r['iter']:>7} {r['dbs_gain']:>+9.4f} {r['dbs_random_gain']:>+9.4f} {ratio:>6.2f} {r['dbs_acc']:>6.0f}")
    last = rows[-1]
    verdict = (last["E_pi_relu"] > last["E_unif_relu"] and last["recovery"] >= 0.1 and last["P_pi"] > last["P_unif"])
    print(f"\n마지막 줄 판정(PLAN §0 '학습됨', 전체 평균): {'충족' if verdict else '미충족'} "
          f"(E_pi_relu/E_unif_relu = {last['E_pi_relu'] / last['E_unif_relu'] if last['E_unif_relu'] else float('nan'):.1f}x, "
          f"recovery {last['recovery']:.3f}, P_pi {last['P_pi']:.1%} vs P_unif {last['P_unif']:.1%})")


if __name__ == "__main__":
    main()
