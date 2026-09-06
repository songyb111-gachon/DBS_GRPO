"""평가 스크립트(test_grpo.py, eval_checkpoints.py) 공용: 체크포인트에서 정책 구성 복원, 정책 입력 구성, 오라클 탐욕 기준선.

v1 체크포인트(policy_kind 없음)는 스크립트가 가진 레거시 GRPOPolicy 로 예전과 똑같이 읽는다 — v1 평가 결과는 바뀌지 않는다.
v2 체크포인트는 policy_kind / feature_spec / in_channels / state_gate 와 config 를 읽어 grpo.policies.make_policy 로 만든다.
env.py 는 import 하지 않는다 (gymnasium/SB3 의존을 평가 스크립트에 끌어오지 않기 위해; 광학 상수는 optics_constants).
"""
import torch

from grpo.features import build_features


def load_policy_checkpoint(path, legacy_policy_cls, CH, IPS, device="cuda"):
    """반환 (policy(eval), feature_spec, info dict). info: kind, step, feature_spec, v2(bool)."""
    ck = torch.load(path, map_location=device)
    if "policy_kind" in ck:
        from grpo.policies import make_policy
        kind, spec, in_ch = ck["policy_kind"], ck["feature_spec"], ck["in_channels"]
        cfg = ck.get("config") or {}
        v2 = cfg.get("v2", {})
        policy = make_policy(kind, in_ch, CH, IPS, feature_spec=spec,
                             state_gate=ck.get("state_gate", v2.get("state_gate", True)),
                             mid_channels=cfg.get("mid_channels", 64),
                             unet_base=v2.get("unet_base", 32), fno_hidden=v2.get("fno_hidden", 16))
        info = {"kind": kind, "step": ck.get("iteration"), "feature_spec": spec, "v2": True}
    else:
        print(f"[eval] {path}: policy_kind 없음 -> v1 체크포인트로 해석 (레거시 GRPOPolicy, 26채널 legacy 특징)")
        kind, spec = "fcn(v1)", "legacy"
        policy = legacy_policy_cls(num_channels=CH, img_size=IPS, mid_channels=64)
        info = {"kind": kind, "step": ck.get("episode_count"), "feature_spec": spec, "v2": False}
    policy = policy.to(device)
    policy.load_state_dict(ck["policy_state_dict"])
    policy.eval()
    return policy, spec, info


def policy_input(obs, spec, device="cuda", v2=False):
    """run_dbs 의 obs dict → 정책 입력 (1, ch, n, n).
    legacy: v1 과 바이트 단위로 같은 순서/방식 (state, state_record, pre_model, recon_image, target_image 를 float32 로 concat).
            단 v2 로 학습한 legacy 정책(v2=True)은 학습 때 state_record 채널이 항상 0 이었으므로 평가에서도 0 을 넣는다.
    field : obs 의 'field'(1,C,n,n complex GPU), 'recon_t'(1,1,n,n GPU), 'target_t'(1,1,n,n GPU) 로 grpo.features 구성."""
    if spec == "legacy":
        parts = []
        for key in ('state', 'state_record', 'pre_model', 'recon_image', 'target_image'):
            t = torch.as_tensor(obs[key], dtype=torch.float32)
            if v2 and key == 'state_record':
                t = torch.zeros_like(t)
            parts.append(t)
        return torch.cat(parts, dim=1).to(device)
    state = torch.as_tensor(obs["state"]).to(device)[0]
    pre = torch.as_tensor(obs["pre_model"]).to(device)[0]
    # run_dbs 의 텐서는 tt.Tensor 서브클래스일 수 있다 -> 학습과 같은 plain 텐서로 벗긴다
    T = obs["target_t"][0, 0].as_subclass(torch.Tensor)
    I = obs["recon_t"][0, 0].as_subclass(torch.Tensor)
    U = obs["field"][0].as_subclass(torch.Tensor)
    c = (T.double().mean() / I.double().mean()).item()
    return build_features(spec, state=state, pre_model=pre, target=T, I=I, U=U, c=c)


_ORACLE = {}


def make_oracle_action_fn(CH, IPS, device="cuda"):
    """오라클 탐욕 DBS: 매 스텝 실제 최선 픽셀(ΔPSNR 최대)을 고른다. 정책이 도달할 수 있는 상한 기준선(스텝당 목적의 상한)."""
    from optics_constants import OPTICS_META, PROP_Z
    from grpo.oracle import FlipOracle
    key = (CH, IPS, str(device))
    if key not in _ORACLE:
        _ORACLE[key] = FlipOracle(OPTICS_META, PROP_Z, IPS, CH, device=device)
    oracle = _ORACLE[key]

    def select(obs):
        h = torch.as_tensor(obs["state"]).to(device)[0].float()
        R = oracle.flip_psnr_map(h, obs["target_t"][0, 0].as_subclass(torch.Tensor),
                                 U=obs["field"][0].as_subclass(torch.Tensor))["dpsnr"]
        return int(torch.argmax(R.reshape(-1)))

    return select
