"""정책 네트워크 3종 + 상태 게이트 헤드 (PLAN.md 2.2절). 모두 입력 (B, in_ch, n, n) → 로짓 (B, C·n·n), ±20 클램프(v1 과 동일).

  "fcn"  — v1 의 GRPOPolicy 와 같은 5층 3×3. 수용 영역 9px (PSF 에너지의 약 25% 만 본다 — 비교용).
  "unet" — 4단 U-Net(GroupNorm). 수용 영역 수백 px. 기본 후보.
  "fno"  — 곱 특징 채널에 공유 스펙트럼 필터(학습 가능한 복소 승수) 1개를 적용한 지도 + 얕은 conv 트렁크.
           오라클 1차 항이 "같은 고정 커널 k 를 곱 특징에 적용한 전역 상관 × 상태 부호" 라 이 귀납 편향이 가장 빨리 배운다.
           ★ 그래서 fno 의 학습은 'PSF 회귀' 에 가깝다 — 결과 보고 시 unet 과 같은 반복 수로 줄세우지 말 것(학습 속도 비교임을 명시).

상태 게이트(state_gate=True): 망이 2C 채널 L0/L1 을 내고 logits_c = h_c·L1_c + (1 − h_c)·L0_c 로 상태 채널이 고른다.
  오라클 1차 항은 상관 결과에 부호 s_p = 1 − 2h_c(p) 가 곱해지므로, 곱셈을 ReLU 망이 근사하게 두는 대신 관측 h 로 직접 게이트한다.
  fno 에는 필수(선형 헤드가 부호 곱을 만들 수 없다), unet/fcn 에는 최적화 보조. 관측 h 를 쓰므로 답 누설이 아니다.
체크포인트에는 policy_kind / feature_spec / in_channels / state_gate 를 같이 저장해 평가 스크립트가 같은 구성을 만든다.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

POLICY_KINDS = ("fcn", "unet", "fno")
LOGIT_CLAMP = 20.0


class FCNTrunk(nn.Module):
    """v1 GRPOPolicy 와 같은 5층. raw(x) → (B, out_ch, n, n)."""

    def __init__(self, in_channels, out_channels, mid_channels=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(mid_channels, mid_channels * 2, 3, padding=1), nn.ReLU(),
            nn.Conv2d(mid_channels * 2, mid_channels * 2, 3, padding=1), nn.ReLU(),
            nn.Conv2d(mid_channels * 2, mid_channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, 1),
        )

    def raw(self, x):
        return self.features(x)


def _block(cin, cout):
    g = 8 if cout % 8 == 0 else 1
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(g, cout), nn.GELU(),
        nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(g, cout), nn.GELU(),
    )


class UNetTrunk(nn.Module):
    """4단 U-Net. base=32 → 32/64/128/256(+mid 512), 파라미터 약 7.8M. n 은 2^depth 의 배수여야 한다(256, 64 OK)."""

    def __init__(self, in_channels, out_channels, base=32, depth=4):
        super().__init__()
        chs = [base * (2 ** i) for i in range(depth)]
        self.enc = nn.ModuleList()
        cin = in_channels
        for c in chs:
            self.enc.append(_block(cin, c))
            cin = c
        self.mid = _block(chs[-1], chs[-1] * 2)
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        cin = chs[-1] * 2
        for c in reversed(chs):
            self.up.append(nn.ConvTranspose2d(cin, c, 2, stride=2))
            self.dec.append(_block(c * 2, c))
            cin = c
        self.head = nn.Conv2d(chs[0], out_channels, 1)

    def raw(self, x):
        skips = []
        h = x
        for blk in self.enc:
            h = blk(h)
            skips.append(h)
            h = F.max_pool2d(h, 2)
        h = self.mid(h)
        for up, blk, s in zip(self.up, self.dec, reversed(skips)):
            h = up(h)
            h = blk(torch.cat([h, s], dim=1))
        return self.head(h)


class SpectralFilter(nn.Module):
    """실수 입력 (B, G, n, n) 에 공유 복소 승수 W(f) 를 곱한다: y = Re IFFT2( FFT2(x) · W ). 파라미터 2·n·n."""

    def __init__(self, n):
        super().__init__()
        self.w_re = nn.Parameter(torch.randn(n, n) * 0.02)
        self.w_im = nn.Parameter(torch.randn(n, n) * 0.02)

    def forward(self, x):
        W = torch.complex(self.w_re, self.w_im)
        X = torch.fft.fft2(x.to(torch.complex64))
        return torch.fft.ifft2(X * W).real


class FNOTrunk(nn.Module):
    """곱 특징(Re/Im 쌍)에 스펙트럼 필터 두 개(W1 은 실수부에, W2 는 허수부에)를 적용해 합친다.
    Re[(pr + i·pi) ⋆ W] = Re[pr ⋆ W] − Im[pi ⋆ W] 는 W2 = −i·W1 일 때의 특수형이고, 독립 W1/W2 는 그것을 포함하는 더 넓은 집합이다.
    prod_slice 는 features.py 'field' 사양의 곱 특징 구간(Re C개 + Im C개)."""

    def __init__(self, in_channels, out_channels, n, num_channels, prod_slice, hidden=16):
        super().__init__()
        self.C = num_channels
        self.prod_slice = prod_slice
        self.spec_re = SpectralFilter(n)
        self.spec_im = SpectralFilter(n)
        self.trunk = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
        )
        self.head = nn.Conv2d(hidden + num_channels, out_channels, 1)

    def raw(self, x):
        s, e = self.prod_slice
        pr, pi = x[:, s:s + self.C], x[:, s + self.C:e]
        corr = self.spec_re(pr) - self.spec_im(pi)               # (B, C, n, n)
        t = self.trunk(x)
        return self.head(torch.cat([corr, t], dim=1))


class Policy(nn.Module):
    """trunk.raw(x) → (B, C or 2C, n, n) → [상태 게이트] → (B, C·n·n) 로짓, ±20 클램프."""

    def __init__(self, trunk, num_channels, state_gate):
        super().__init__()
        self.trunk = trunk
        self.C = num_channels
        self.state_gate = state_gate

    def forward(self, x):
        out = self.trunk.raw(x)
        if self.state_gate:
            h = x[:, :self.C]                                     # 특징 사양 둘 다 첫 C 채널이 state (0/1)
            L0, L1 = out[:, :self.C], out[:, self.C:2 * self.C]
            out = h * L1 + (1.0 - h) * L0
        logits = out.reshape(x.size(0), -1)
        return torch.clamp(logits, -LOGIT_CLAMP, LOGIT_CLAMP)

    @torch.no_grad()
    def saturation(self, x):
        """|로짓| 이 클램프 경계에 닿은 비율 (진단)."""
        out = self.trunk.raw(x)
        if self.state_gate:
            h = x[:, :self.C]
            out = h * out[:, self.C:2 * self.C] + (1.0 - h) * out[:, :self.C]
        return float((out.abs() >= LOGIT_CLAMP).float().mean())


def make_policy(kind, in_channels, num_channels, n, *, feature_spec="field", state_gate=True,
                mid_channels=64, unet_base=32, fno_hidden=16):
    out_ch = 2 * num_channels if state_gate else num_channels
    if kind == "fcn":
        trunk = FCNTrunk(in_channels, out_ch, mid_channels=mid_channels)
    elif kind == "unet":
        trunk = UNetTrunk(in_channels, out_ch, base=unet_base)
    elif kind == "fno":
        if feature_spec != "field":
            raise ValueError("fno 정책은 feature_spec='field' (곱 특징 포함) 에서만 쓴다")
        if not state_gate:
            raise ValueError("fno 정책은 state_gate=True 여야 한다 (선형 헤드는 플립 부호 s=1−2h 를 만들 수 없다)")
        C = num_channels
        prod_start = 2 * C + 3 + 2 * C          # state, pre_model, target, recon, error, Re/Im Ũ 뒤
        trunk = FNOTrunk(in_channels, out_ch, n, C, prod_slice=(prod_start, prod_start + 2 * C), hidden=fno_hidden)
    else:
        raise ValueError(f"모르는 policy_kind {kind!r} (허용: {POLICY_KINDS})")
    return Policy(trunk, num_channels, state_gate)


def count_params(model):
    return sum(p.numel() for p in model.parameters())
