"""정책 입력 특징 (PLAN.md 2.2절).

두 사양:
  "legacy" — v1 과 같은 26채널: state 8, state_record 8, pre_model 8, recon(I, 정규화 전) 1, target 1. v1 체크포인트 호환용.
  "field"  — 51채널: state 8, pre_model 8, target 1, recon(c·I) 1, error(c·I − T) 1,
             Re/Im(Ũ_c) 16, Re/Im(E·conj Ũ_c) 16.  Ũ = U·sqrt(c) 로 스케일을 맞춘다 (mean|Ũ|² = mean(T)).
오라클 지도(플립 효과) 자체는 절대 넣지 않는다 — 그것이 답이다. 곱 특징 E·conj(U) 는 관측(오차 × 필드)의 국소 곱일 뿐이고,
답이 되려면 PSF 와의 전역 상관이 더 필요하다. 그 상관을 배우는 것이 정책의 몫이다.
"""
import torch

FEATURE_SPECS = ("legacy", "field")


def num_channels(spec, C):
    if spec == "legacy":
        return 3 * C + 2
    if spec == "field":
        return 2 * C + 3 + 2 * C + 2 * C
    raise ValueError(f"모르는 feature_spec {spec!r} (허용: {FEATURE_SPECS})")


def build_features(spec, *, state, pre_model, target, I, U=None, c=None, state_record=None):
    """
    모두 같은 device 의 텐서. state (C,n,n) 0/1, pre_model (C,n,n), target (n,n), I (n,n) 정규화 전 세기,
    U (C,n,n) complex (field 사양에 필요), c 정규화 계수 mean(T)/mean(I) (field 사양에 필요), state_record (C,n,n) (legacy).
    반환 (1, ch, n, n) float32.
    """
    if spec == "legacy":
        if state_record is None:
            state_record = torch.zeros_like(state)
        parts = [state.float(), state_record.float(), pre_model.float(), I.float().unsqueeze(0), target.float().unsqueeze(0)]
        return torch.cat(parts, dim=0).unsqueeze(0)
    if spec == "field":
        if U is None or c is None:
            raise ValueError("feature_spec='field' 에는 U 와 c 가 필요하다")
        Tn = target.float()
        recon = (c * I).float()
        E = recon - Tn                                           # (n,n)
        Ut = U * (c ** 0.5)                                      # (C,n,n) complex, 스케일 맞춤
        prod = E.unsqueeze(0) * Ut.conj()                        # (C,n,n) complex
        parts = [state.float(), pre_model.float(), Tn.unsqueeze(0), recon.unsqueeze(0), E.unsqueeze(0),
                 Ut.real, Ut.imag, prod.real, prod.imag]
        return torch.cat(parts, dim=0).unsqueeze(0)
    raise ValueError(f"모르는 feature_spec {spec!r}")
