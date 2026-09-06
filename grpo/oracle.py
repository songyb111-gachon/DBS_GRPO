"""단일 픽셀 플립의 PSNR 변화를 **모든 픽셀에 대해 한 번에, 정확히** 계산하는 오라클 (PLAN.md 2.1절).

원리: tt.simulate(h, z) 는 (100 패딩 → 중심화 FFT → 사각 마스크 M × 대역제한 ASM 커널 K → IFFT → 크롭) 이므로
패딩 격자(N = IPS + 2·pad) 위의 순환 컨볼루션 A 다. 채널 c 의 필드 U_c = A h_c, 픽셀 p 를 뒤집으면(부호 s = 1 − 2h)
U_c' = U_c + s·a_p (a_p(x) = k((x − p) mod N), k = A 의 임펄스 응답). 세기 I = (1/C) Σ_c |U_c|² 의 변화는
    D_p = (1/C) [ 2 s Re(conj(U_c) a_p) + |a_p|² ]
로 a_p 에 대해 정확히 2차이고, relativeLoss(평균 정규화) 후 MSE 의 변화는 창 내 합 ΣI·D, ΣT·D, ΣD, ΣD² 로 닫힌다.
각 합은 k, |k|², k², k|k|², |k|⁴ 와의 순환 상관이라 FFT 로 모든 p 에 대해 동시에 나온다. 근사 없음.

★ 커널 K 와 마스크 M 은 반드시 torchOptics 의 함수(get_ASM_kernel, get_mask_by_range)로 만든다 — 재유도·재정의 금지.
  이유: (1) ASM 위상 2π z/λ ≈ 2.4e4 rad 을 torchOptics 가 float32 로 계산하므로 float64 로 '더 정확히' 만들면 빈마다
  ~1e-3 rad 어긋나 tt.simulate 와 1e-3 수준으로 달라진다. (2) 마스크는 FFT 주파수 격자가 아니라 linspace(-0.5,0.5,N)
  격자 위의 heaviside 라 '|f| < 0.45 f_s' 로 다시 쓰면 가장자리 한 빈이 달라진다. (3) 이 파라미터에서 대역제한·파장
  마스크는 전부 1 이라 유효 전달함수는 0.9 사각 마스크 × ASM 위상뿐이고, PSF 의 sinc 꼬리가 100 px 패딩을 넘어
  순환 되감김이 0 이 아니다 — 그래서 격자를 456² 순환 그대로 두어야 한다(512 격자·선형 상관으로 '최적화' 금지).

수치: 큰 수끼리 빼지 않도록 ΔMSE 를 차이 형태로 직접 조립하고, 스칼라 조합은 float64 로 한다.
       ΔPSNR = −10 log10(1 + ΔMSE / MSE).  도달 가능한 일치 한계는 float32 상관의 상쇄(cp·SID − SDT) 때문에 ~1e-6 dB.

검증: grpo/oracle_algebra_check_np.py (로컬, numpy 만 — 대수·인덱스 규약) 와 grpo/oracle_selftest.py (서버 — tt.simulate 대조).
      둘 다 통과 전에는 이 오라클을 학습 보상으로 쓰지 않는다. 폴백은 없다 — 실패하면 오라클을 고친다.
"""
import torch
import torch.nn.functional as F

import torchOptics.optics as tt

from optics_constants import OPTICS_META, PROP_Z   # env_style_psnr 는 학습 env 와 같은 상수로 시뮬레이션해야 한다


class FlipOracle:
    """flip_psnr_map(h, T) → 모든 (채널, 픽셀) 플립의 ΔPSNR (dB). forward/adjoint 는 검증·특징용."""

    def __init__(self, meta, z, ips, ch, padding=100, filter_width=(0.9, 0.9), device="cuda"):
        if padding <= 0:
            raise ValueError("padding 은 양수여야 한다 (_crop 이 [pad:-pad] 슬라이스를 쓴다)")
        self.ips, self.C, self.pad, self.z, self.device = ips, ch, padding, z, torch.device(device)
        self.N = ips + 2 * padding
        dx, wl = tuple(meta["dx"]), meta["wl"]
        shape = (1, 1, self.N, self.N)
        # simulate → propASM → prop_ASM 이 쓰는 함수·인자 그대로 (angle=None → (0,0), fft_shift=True 중심화 커널)
        K = tt.get_ASM_kernel(shape, z, dx=dx, wl=wl, device=self.device, angle=None, fft_shift=True)
        M = tt.get_mask_by_range((self.N, self.N), (0.0, 0.0), tuple(filter_width), device=self.device)
        H_centered = (K * M).reshape(self.N, self.N).to(torch.complex64)
        # prop_ASM: ifftn(ifftshift(fftshift(fftn x)·M·K)) = ifftn(fftn(x)·ifftshift(M·K)) → 표준 순서 승수 H
        self.H = tt.ifftshift(H_centered)
        k = torch.fft.ifft2(self.H)                # 임펄스 응답 (패딩 격자, 원점 index 0)
        self.k = k
        ak2 = k.abs() ** 2
        # corr(g, q)(p) = Σ_x g(x) q(x − p) = ifft2( fft2(g) · fft2(q_rev) ),  q_rev(y) = q(−y mod N)
        self.S_k = torch.fft.fft2(self._rev(k))
        self.S_k2 = torch.fft.fft2(self._rev(k * k))
        self.S_ak2 = torch.fft.fft2(self._rev(ak2).to(torch.complex64))
        self.S_kak2 = torch.fft.fft2(self._rev(k * ak2))
        self.S_ak4 = torch.fft.fft2(self._rev(ak2 * ak2).to(torch.complex64))

    @staticmethod
    def _rev(x):
        """q_rev(y) = q((−y) mod N) for the last two axes."""
        return torch.roll(torch.flip(x, dims=(-2, -1)), shifts=(1, 1), dims=(-2, -1))

    def _pad(self, x):
        x = x if x.is_complex() else x.to(torch.complex64)
        return F.pad(x, [self.pad] * 4)

    def _crop(self, x):
        return x[..., self.pad:-self.pad, self.pad:-self.pad]

    # ---------------------------------------------------------------- 선형 연산자
    def forward(self, h):
        """A h — tt.simulate(h, z) 와 같아야 한다 (selftest 가 확인). h: (..., ips, ips) 실수/복소."""
        return self._crop(torch.fft.ifft2(torch.fft.fft2(self._pad(h)) * self.H))

    def adjoint(self, g):
        """A^H g — <A h, g> = <h, A^H g>."""
        return self._crop(torch.fft.ifft2(torch.fft.fft2(self._pad(g)) * self.H.conj()))

    def _corr(self, G_spec, S):
        return self._crop(torch.fft.ifft2(G_spec * S))

    def _spec(self, g):
        return torch.fft.fft2(self._pad(g))

    # ---------------------------------------------------------------- PSNR (env/test 와 같은 정의, float64)
    @staticmethod
    def psnr_from(I, T):
        """relativeLoss(result, target, get_PSNR) 과 같은 정의를 float64 로: c = mean(T)/mean(I), MSE = mean((cI − T)²),
        PSNR = 10 log10(max(T)² / MSE). 반환 (psnr, mse)."""
        I64, T64 = I.double(), T.double()
        c = T64.mean() / I64.mean()
        mse = ((c * I64 - T64) ** 2).mean()
        return (10.0 * torch.log10(T64.abs().max() ** 2 / mse)).item(), mse.item()

    # ---------------------------------------------------------------- 오라클
    @torch.no_grad()
    def flip_psnr_map(self, h, T, U=None):
        """
        h: (C, ips, ips) **정확히 0/1**, T: (ips, ips) 또는 (1, ips, ips) 목표 영상 (0~1). 둘 다 self.device 로 옮겨 계산.
        U: (C, ips, ips) 복소 필드가 이미 있으면 재사용.
        반환 dict:
          dpsnr (C, ips, ips) float32 — 각 플립 후 PSNR − 현재 PSNR (dB)
          psnr, mse — 현재 값 (float);  U (C, ips, ips) complex64;  I (ips, ips) float32;  c — 정규화 계수
        """
        C, n, Npix = self.C, self.ips, float(self.ips * self.ips)
        h = h.to(self.device).float()
        if tuple(h.shape) != (C, n, n):
            raise ValueError(f"h 형상 {tuple(h.shape)} != ({C}, {n}, {n})")
        if not bool(torch.all((h == 0) | (h == 1))):
            raise ValueError("h 는 정확히 0/1 이어야 한다 (플립 부호 s = 1 − 2h). 확률 맵이 아니라 이진화된 상태를 넘길 것")
        T = T.to(self.device).float().reshape(n, n)
        if U is None:
            U = self.forward(h)                                   # (C, n, n) complex
        I = (U.abs() ** 2).mean(dim=0)                            # (n, n)

        I64, T64 = I.double(), T.double()
        mI, mT = I64.mean(), T64.mean()
        c = mT / mI
        SI2, SIT, STT = (I64 * I64).mean(), (I64 * T64).mean(), (T64 * T64).mean()
        MSE = c * c * SI2 - 2 * c * SIT + STT                     # = mean((cI − T)²)

        # 채널 무관 항: Σ_x g(x) |a_p(x)|² 와 Σ |a_p|⁴
        F_I, F_T, F_1 = self._spec(I), self._spec(T), self._spec(torch.ones(n, n, device=self.device))
        R_I_ak2 = self._corr(F_I, self.S_ak2).real
        R_T_ak2 = self._corr(F_T, self.S_ak2).real
        R_1_ak2 = self._corr(F_1, self.S_ak2).real
        R_1_ak4 = self._corr(F_1, self.S_ak4).real

        s = 1.0 - 2.0 * h                                         # (C, n, n) 플립 부호
        dpsnr = torch.empty(C, n, n, dtype=torch.float64, device=self.device)
        for ci in range(C):
            Uc = U[ci]
            Ucc = Uc.conj()
            F_U = self._spec(Ucc)
            A_IU = self._corr(self._spec(I * Ucc), self.S_k).real       # Re Σ I conj(U) a_p
            A_TU = self._corr(self._spec(T * Ucc), self.S_k).real       # Re Σ T conj(U) a_p
            A_U = self._corr(F_U, self.S_k).real                        # Re Σ conj(U) a_p
            A_U2 = self._corr(self._spec(Ucc * Ucc), self.S_k2).real    # Re Σ conj(U)² a_p²
            A_absU2 = self._corr(self._spec(Uc.abs() ** 2), self.S_ak2).real   # Σ |U|² |a_p|²
            A_Ukak2 = self._corr(F_U, self.S_kak2).real                 # Re Σ conj(U) a_p |a_p|²
            sc = s[ci]
            # 창 평균 (Σ/Npix). D = (1/C)[2s Re(conj U a) + |a|²]
            SID = (2 * sc * A_IU + R_I_ak2).double() / (C * Npix)
            SDT = (2 * sc * A_TU + R_T_ak2).double() / (C * Npix)
            mD = (2 * sc * A_U + R_1_ak2).double() / (C * Npix)
            # ΣD² = (1/C²)[4 Σ Re(conj U a)² + 4 s Σ Re(conj U a)|a|² + Σ|a|⁴],  Σ Re(conj U a)² = ½[Σ|U|²|a|² + Re Σ conj(U)² a²]
            SD2 = (2 * (A_absU2 + A_U2) + 4 * sc * A_Ukak2 + R_1_ak4).double() / (C * C * Npix)
            # 정규화 계수 변화까지 포함한 ΔMSE (차이 형태 — 큰 수끼리 빼지 않는다)
            cp = mT / (mI + mD)
            delta = cp - c
            dMSE = delta * (cp + c) * SI2 - 2 * delta * SIT + cp * cp * (2 * SID + SD2) - 2 * cp * SDT
            dpsnr[ci] = -10.0 * torch.log10(1.0 + dMSE / MSE)

        psnr = (10.0 * torch.log10(T64.abs().max() ** 2 / MSE)).item()
        return {"dpsnr": dpsnr.float(), "psnr": psnr, "mse": MSE.item(), "U": U, "I": I, "c": c.item()}


def env_style_psnr(binary, target):
    """env.py / test_grpo.py 와 같은 경로(tt.Tensor → tt.simulate → 채널 평균 → relativeLoss(get_PSNR))로 PSNR 을 계산한다.
    검증·기준선용. binary: (1, C, n, n) float, target: (1, 1, n, n). 반환 (psnr_float32경로, sim, result)."""
    import torchOptics.metrics as tm
    b = tt.Tensor(binary, meta=OPTICS_META)
    sim = tt.simulate(b, z).abs() ** 2
    result = torch.mean(sim, dim=1, keepdim=True)
    return float(tt.relativeLoss(result, target, tm.get_PSNR)), sim, result


def startup_consistency_check(oracle, h, T, n_flips=8, tol_abs=5e-6, tol_rel=0.01):
    """학습 시작 시 첫 상태에서 오라클 vs 실제 시뮬레이션을 몇 픽셀 대조한다 (값싼 안전장치: 커널 캐시가 다른 shape 로
    바뀌는 등의 드리프트를 잡는다). 불일치면 RuntimeError — 조용히 틀린 보상으로 학습하지 않는다."""
    import numpy as np
    out = oracle.flip_psnr_map(h, T)
    n = oracle.ips
    T4 = T.reshape(1, 1, n, n)
    _, _, base_result = env_style_psnr(h.unsqueeze(0), T4)
    base64, _ = oracle.psnr_from(base_result[0, 0], T.reshape(n, n))
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(n_flips):
        ci, r, cc = int(rng.integers(oracle.C)), int(rng.integers(n)), int(rng.integers(n))
        h2 = h.clone()
        h2[ci, r, cc] = 1.0 - h2[ci, r, cc]
        _, _, res2 = env_style_psnr(h2.unsqueeze(0), T4)
        p2, _ = oracle.psnr_from(res2[0, 0], T.reshape(n, n))
        bf = p2 - base64
        orc = float(out["dpsnr"][ci, r, cc])
        err = abs(bf - orc)
        worst = max(worst, err)
        if err > tol_abs + tol_rel * abs(bf):
            raise RuntimeError(f"[oracle] 시작 검사 실패: ({ci},{r},{cc}) 브루트포스 {bf:+.3e} vs 오라클 {orc:+.3e} dB (오차 {err:.2e}). "
                               "grpo/oracle_selftest.py 를 다시 돌려 원인을 찾을 것. 폴백 없음.")
    print(f"[oracle] 시작 검사 통과: {n_flips}픽셀 최대 오차 {worst:.2e} dB")
