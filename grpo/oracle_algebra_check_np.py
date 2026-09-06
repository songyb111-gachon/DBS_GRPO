# -*- coding: utf-8 -*-
"""오라클 대수·인덱스 규약의 로컬 검증 (numpy 만, torch 불필요, 수 초).

grpo/oracle.py 의 flip_psnr_map 조립식(창 내 합 → 상관 → ΔMSE 차이 형태 → ΔPSNR)을 numpy 로 그대로 옮기고,
임의의 복소 승수 H (실제 ASM 커널이 아니어도 '패딩 격자 위 순환 컨볼루션' 이면 대수는 같다) 위에서
모든 픽셀의 오라클 ΔPSNR 을 브루트포스(실제로 뒤집고 다시 전파해 PSNR 계산, float64)와 대조한다.
tt.simulate 와의 일치는 보장하지 않는다(그건 서버의 oracle_selftest.py 몫) — 여기서는 수식·부호·roll/flip·크롭 오프셋만 본다.

    python grpo/oracle_algebra_check_np.py

★ 이 파일의 조립식은 oracle.py:flip_psnr_map 의 사본이다. 한쪽을 고치면 다른 쪽도 고칠 것 (check_conventions 가 대조하지 않는다).
"""
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

rng = np.random.default_rng(0)
IPS, PAD, C = 12, 5, 3          # 작은 격자. N = 22 (짝수 아니어도 대수는 같지만 ifftshift 규약을 위해 짝수로)
N = IPS + 2 * PAD


def rev(x):
    """q_rev(y) = q((−y) mod N): torch.roll(torch.flip(x), 1) 과 동일."""
    return np.roll(np.flip(x, axis=(-2, -1)), shift=(1, 1), axis=(-2, -1))


def pad(x):
    out = np.zeros((*x.shape[:-2], N, N), dtype=np.complex128)
    out[..., PAD:PAD + IPS, PAD:PAD + IPS] = x
    return out


def crop(x):
    return x[..., PAD:-PAD, PAD:-PAD]


# 임의 승수 H: 통과대역 마스크 × 무작위 위상 (ASM 과 같은 구조: |H| ∈ {0,1}, 순환 컨볼루션)
fy, fx = np.meshgrid(np.fft.fftfreq(N), np.fft.fftfreq(N), indexing="ij")
mask = (np.abs(fx) < 0.42) & (np.abs(fy) < 0.42)
H = mask * np.exp(1j * rng.uniform(0, 2 * np.pi, (N, N)))
k = np.fft.ifft2(H)
ak2 = np.abs(k) ** 2
S_k, S_k2 = np.fft.fft2(rev(k)), np.fft.fft2(rev(k * k))
S_ak2, S_kak2, S_ak4 = np.fft.fft2(rev(ak2)), np.fft.fft2(rev(k * ak2)), np.fft.fft2(rev(ak2 * ak2))


def forward(h):
    return crop(np.fft.ifft2(np.fft.fft2(pad(h)) * H))


def spec(g):
    return np.fft.fft2(pad(g))


def corr(G, S):
    return crop(np.fft.ifft2(G * S))


def psnr_mse(I, T):
    c = T.mean() / I.mean()
    mse = ((c * I - T) ** 2).mean()
    return 10 * np.log10(np.abs(T).max() ** 2 / mse), mse


def flip_psnr_map(h, T):
    """oracle.py:flip_psnr_map 의 numpy 사본 (같은 기호)."""
    Npix = float(IPS * IPS)
    U = forward(h)
    I = (np.abs(U) ** 2).mean(axis=0)
    mI, mT = I.mean(), T.mean()
    c = mT / mI
    SI2, SIT, STT = (I * I).mean(), (I * T).mean(), (T * T).mean()
    MSE = c * c * SI2 - 2 * c * SIT + STT
    F_I, F_T, F_1 = spec(I), spec(T), spec(np.ones((IPS, IPS)))
    R_I_ak2, R_T_ak2 = corr(F_I, S_ak2).real, corr(F_T, S_ak2).real
    R_1_ak2, R_1_ak4 = corr(F_1, S_ak2).real, corr(F_1, S_ak4).real
    s = 1.0 - 2.0 * h
    out = np.empty((C, IPS, IPS))
    for ci in range(C):
        Uc = U[ci]
        Ucc = np.conj(Uc)
        F_U = spec(Ucc)
        A_IU = corr(spec(I * Ucc), S_k).real
        A_TU = corr(spec(T * Ucc), S_k).real
        A_U = corr(F_U, S_k).real
        A_U2 = corr(spec(Ucc * Ucc), S_k2).real
        A_absU2 = corr(spec(np.abs(Uc) ** 2), S_ak2).real
        A_Ukak2 = corr(F_U, S_kak2).real
        sc = s[ci]
        SID = (2 * sc * A_IU + R_I_ak2) / (C * Npix)
        SDT = (2 * sc * A_TU + R_T_ak2) / (C * Npix)
        mD = (2 * sc * A_U + R_1_ak2) / (C * Npix)
        SD2 = (2 * (A_absU2 + A_U2) + 4 * sc * A_Ukak2 + R_1_ak4) / (C * C * Npix)
        cp = mT / (mI + mD)
        delta = cp - c
        dMSE = delta * (cp + c) * SI2 - 2 * delta * SIT + cp * cp * (2 * SID + SD2) - 2 * cp * SDT
        out[ci] = -10.0 * np.log10(1.0 + dMSE / MSE)
    return out, psnr_mse(I, T)[0]


def main():
    h = (rng.random((C, IPS, IPS)) < 0.5).astype(float)
    T = np.clip(0.5 + 0.3 * rng.standard_normal((IPS, IPS)), 0, 1)
    dp, base = flip_psnr_map(h, T)
    # 브루트포스: 모든 (c, p) 를 실제로 뒤집어 재전파
    bf = np.empty_like(dp)
    for ci in range(C):
        for r in range(IPS):
            for cc in range(IPS):
                h2 = h.copy()
                h2[ci, r, cc] = 1 - h2[ci, r, cc]
                I2 = (np.abs(forward(h2)) ** 2).mean(axis=0)
                bf[ci, r, cc] = psnr_mse(I2, T)[0] - base
    err = np.abs(dp - bf)
    print(f"격자 IPS={IPS} PAD={PAD} C={C}: 플립 {dp.size}개 전수 대조")
    print(f"  |ΔPSNR| 중앙값 {np.median(np.abs(bf)):.3e} dB, 최대 {np.abs(bf).max():.3e} dB, 양수 비율 {np.mean(bf > 0):.1%}")
    print(f"  오라클 vs 브루트포스 최대 오차 {err.max():.3e} dB, 상대 최대 {np.max(err / np.maximum(np.abs(bf), 1e-12)):.3e}")
    print(f"  부호 일치 {np.mean(np.sign(dp) == np.sign(bf)):.1%}")
    ok = err.max() < 1e-9
    print("  (참고) 검사 자체의 민감도: 위 대조가 1e-9 이하면 roll/flip·크롭·부호·2차 항이 전부 맞다는 뜻")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
