"""검증 상태: 검증 이미지 몇 장을 오라클 탐욕 DBS 로 정해진 깊이(채택 플립 수)까지 진행시킨 고정 상태. 디스크에 캐시한다.

깊이 0 은 초기 홀로그램. 오라클 탐욕은 결정론적이라 같은 (이미지, 초기 홀로그램, 깊이) 면 같은 상태이고,
캐시 키에 초기 홀로그램 해시를 넣어 사전학습 모델·전처리가 바뀌면 자동으로 다시 만든다.
깊이는 오름차순으로 증분 진행하므로 총 비용은 최대 깊이 × 약 7.5 ms (이미지당, 1회)."""
import hashlib
import os
import time

import torch

from grpo.dbs_state import DBSImage


def build_val_states(oracle, images, depths, cache_dir, verbose=True):
    """images: [(T (n,n), h0 (C,n,n) 0/1, pre (C,n,n), name)], depths: 정수 목록. 반환 [DBSImage] (이미지 순 × 깊이 오름차순),
    각 상태에 .depth(깊이), .flips(=깊이), .pre_model 이 채워져 있다."""
    depths = sorted(int(d) for d in depths)
    if not depths:
        raise ValueError("val_depths 가 비어 있다")
    os.makedirs(cache_dir, exist_ok=True)
    out = []
    for T, h0, pre, name in images:
        key = hashlib.sha1(h0.to(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:8]
        stem = os.path.splitext(os.path.basename(str(name)))[0]
        cur = None                                   # 증분 진행 중인 상태
        for d in depths:
            path = os.path.join(cache_dir, f"{stem}_{key}_d{d}.pt")
            if os.path.exists(path):
                ck = torch.load(path, map_location="cpu")
                h = ck["h"].float().to(h0.device)
                cur = DBSImage(oracle, T, h, name=name)
                cur.flips = d
            else:
                if cur is None:
                    cur = DBSImage(oracle, T, h0, name=name)
                t0 = time.time()
                done = cur.greedy_steps(d - cur.flips)
                if cur.flips != d:
                    print(f"  [val] {stem}: 깊이 {d} 에 못 미침 (개선 픽셀 소진, {cur.flips} 에서 멈춤) - 그 상태를 쓴다")
                torch.save({"h": cur.h.to(torch.uint8).cpu(), "depth": d, "reached": cur.flips, "psnr": cur.psnr,
                            "name": str(name)}, path)
                if verbose:
                    print(f"  [val] {stem} 깊이 {d}: 오라클 탐욕 {done} 스텝, psnr {cur.psnr:.3f} ({time.time() - t0:.0f}s) -> {path}")
            img = DBSImage(oracle, T, cur.h.clone(), name=name)
            img.flips = d
            img.depth = d
            img.pre_model = pre
            out.append(img)
    return out
