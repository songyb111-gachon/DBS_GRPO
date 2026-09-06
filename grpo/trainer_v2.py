"""배치 GRPO 트레이너 v2 (PLAN.md 2.3~2.4절). 보상은 오라클(모든 액션의 ΔPSNR 지도)에서만 온다 — 폴백 없음.

한 반복(iteration):
  1. K 장의 이미지 각각에서 오라클 지도 R(a) 와 정책 로짓을 계산하고 π_old 에서 G 개 액션을 샘플.
     학습 보상 R' = reward_transform(R):  "relu" = max(R,0) — DBS 가 실제로 실현하는 스텝 이득(평가 지표와 같은 함수), 기본.
                                          "raw"  = R 그대로 (음수 영역의 순서 정보 유지, 비교 arm).
     어드밴티지 A = (R'(a) − m) / s  의 기준(adv_baseline):
       "sample"  — G 개 표본의 평균·표준편차 (DeepSeek GRPO 그대로, 기본. 그룹이 한 액션으로 뭉치면 A=0 → 자기제한)
       "policy"  — π_old 가중 모집단 통계 E_π[R'], std_π[R'] (G→∞ 극한, 바닥값 0.1·std_unif)
       "uniform" — 모든 액션의 균등 통계 (정책 무관 고정 목표, 검증 지표 E_unif 와 일관)
  2. K×G 표본으로 E epoch 미니배치 업데이트: clipped surrogate + kl_coef·KL(π‖π_ref) (k3 추정) [+ entropy_coef·H(π)].
  3. 각 이미지의 상태를 전진(advance): "best" 는 G 개 중 R>0 인 최고 액션을 실행, "sample" 은 정책 샘플 1개를 실행.
     실행은 DBSImage.apply (실제 PSNR 이 오르면 채택). steps_per_image 에 도달하면 새 이미지로 교체.
  4. val_every 반복마다 고정 검증 상태 V 개에서 시뮬레이션 없이 정확한 지표 (PLAN.md 2.4):
     E_pi_relu = Σ π(a)R⁺(a) (1차 판정 지표), P_pi = Σ π(a)1[R>0], top1 = max R, E_unif_relu, P_unif,
     recovery = (E_pi_relu − E_unif_relu)/(top1 − E_unif_relu), E_pi(raw), entropy, exp(entropy), 로짓 포화 비율.

objective="exact" 는 진단용: 표본 대신 Σ_a π(a)A(a) 를 직접 최대화(표본 분산 0). GRPO 서로게이트의 G→∞·E=1 극한과 같은
gradient 이며 GRPO 가 아니므로 그 결과를 GRPO 결과로 보고하지 않는다. adv_baseline="sample" 과는 조합 불가(모집단 통계 필요).
"""
import copy
import json
import os
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F

from grpo.features import build_features
from grpo.dbs_state import DBSImage

REWARD_TRANSFORMS = ("relu", "raw")
ADV_BASELINES = ("sample", "policy", "uniform")
OBJECTIVES = ("grpo", "exact")
ADVANCES = ("best", "sample")


class GRPOTrainerV2:
    def __init__(self, policy, oracle, cfg, *, feature_spec, new_image_fn, val_states, device, log_dir):
        """
        policy: grpo.policies.Policy, oracle: FlipOracle, cfg: CONFIG["v2"] dict,
        new_image_fn(): → (target (n,n), h0 (C,n,n), pre_model (C,n,n), name) — 학습 이미지 공급자,
        val_states: [DBSImage] 고정 검증 상태 (pre_model 채워진 것, 1개 이상), log_dir: val.jsonl 기록 위치.
        """
        for key, allowed in (("reward_transform", REWARD_TRANSFORMS), ("adv_baseline", ADV_BASELINES),
                             ("objective", OBJECTIVES), ("advance", ADVANCES)):
            if cfg[key] not in allowed:
                raise ValueError(f"v2.{key}={cfg[key]!r} 는 허용값 {allowed} 가 아니다")
        if cfg["objective"] == "exact" and cfg["adv_baseline"] == "sample":
            raise ValueError("objective='exact' 는 모집단 통계가 필요하다: adv_baseline 을 'policy' 또는 'uniform' 으로")
        if not val_states:
            raise ValueError("검증 상태가 비어 있다 (v2.val_images >= 1)")
        self.policy = policy.to(device)
        self.ref_policy = copy.deepcopy(policy).to(device).eval()
        for p in self.ref_policy.parameters():
            p.requires_grad = False
        self.oracle = oracle
        self.cfg = cfg
        self.spec = feature_spec
        self.new_image_fn = new_image_fn
        self.val_states = val_states
        self.device = device
        self.log_dir = log_dir
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=cfg["lr"])
        self.iteration = 0
        self.extra_meta = {}      # policy_kind / feature_spec / in_channels / state_gate / config — 호출측이 채운다
        self.images = [self._spawn() for _ in range(cfg["images_per_batch"])]
        # 검증 상태의 오라클 지도는 상태가 고정이라 한 번만
        self.val_maps = [s.reward_map().reshape(-1) for s in self.val_states]
        self.val_feats = [self._features(s) for s in self.val_states]

    # ---------------------------------------------------------------- 상태/특징
    def _spawn(self):
        T, h0, pre, name = self.new_image_fn()
        img = DBSImage(self.oracle, T, h0, name=name)
        img.pre_model = pre.to(self.device).float()
        return img

    def _features(self, img):
        return build_features(self.spec, state=img.h, pre_model=img.pre_model, target=img.T,
                              I=img.I, U=img.U, c=img.c)

    # ---------------------------------------------------------------- 보상/어드밴티지
    def _transform(self, R):
        return torch.clamp(R, min=0.0) if self.cfg["reward_transform"] == "relu" else R

    def _baseline(self, Rp_all, Rp_samples, probs):
        """(m, s): 어드밴티지 기준. Rp_all (N,) 변환된 보상 지도, Rp_samples (G,), probs (N,) π_old."""
        kind = self.cfg["adv_baseline"]
        if kind == "sample":
            return Rp_samples.mean(), Rp_samples.std(unbiased=False)
        std_unif = Rp_all.std(unbiased=False)
        if kind == "uniform":
            return Rp_all.mean(), std_unif
        m = (probs * Rp_all).sum()
        s = torch.sqrt((probs * (Rp_all - m) ** 2).sum())
        return m, torch.maximum(s, 0.1 * std_unif)

    # ---------------------------------------------------------------- 한 반복
    def collect(self):
        """K 장 각각에서 G 개 샘플과 어드밴티지를 모은다. 반환 list of dict."""
        G = self.cfg["group_size"]
        batch = []
        self.policy.eval()
        for img in self.images:
            feats = self._features(img)
            with torch.no_grad():
                logits = self.policy(feats).squeeze(0)               # (N,)
                dist = torch.distributions.Categorical(logits=logits)
                actions = dist.sample((G,))                          # (G,)
                old_logp = dist.log_prob(actions)
                R_all = img.reward_map().reshape(-1)                 # (N,) 원시 ΔPSNR
                Rp_all = self._transform(R_all)
                R = R_all[actions]
                Rp = Rp_all[actions]
                m, s = self._baseline(Rp_all, Rp, dist.probs)
                adv = (Rp - m) / (s + 1e-12)
                adv_map = (Rp_all - m) / (s + 1e-12) if self.cfg["objective"] == "exact" else None
            batch.append({"feats": feats, "actions": actions, "old_logp": old_logp, "adv": adv,
                          "R": R, "adv_map": adv_map, "img": img})
        self.policy.train()
        return batch

    def update(self, batch):
        cfg = self.cfg
        E, M = cfg["update_epochs"], cfg["minibatch_states"]
        stats = {"loss": [], "pg": [], "kl": [], "clipfrac": [], "entropy": []}
        idx = list(range(len(batch)))
        for _ in range(E):
            np.random.shuffle(idx)
            for s in range(0, len(idx), M):
                items = [batch[i] for i in idx[s:s + M]]
                feats = torch.cat([it["feats"] for it in items], dim=0)          # (M, ch, n, n)
                logits = self.policy(feats)                                      # (M, N)
                logp_all = F.log_softmax(logits, dim=1)
                with torch.no_grad():
                    ref_logp_all = F.log_softmax(self.ref_policy(feats), dim=1)
                pg_terms, kl_terms, clip_terms, ent_terms = [], [], [], []
                for j, it in enumerate(items):
                    a = it["actions"]
                    new_lp, ref_lp = logp_all[j][a], ref_logp_all[j][a]
                    if cfg["objective"] == "exact":
                        probs = logp_all[j].exp()
                        pg = -(probs * it["adv_map"]).sum()
                        clipfrac = torch.zeros((), device=self.device)
                    else:
                        ratio = torch.exp(new_lp - it["old_logp"])
                        ratio = torch.clamp(ratio, 0.0, 10.0)
                        surr1 = ratio * it["adv"]
                        surr2 = torch.clamp(ratio, 1 - cfg["clip_range"], 1 + cfg["clip_range"]) * it["adv"]
                        pg = -torch.min(surr1, surr2).mean()
                        clipfrac = ((ratio - 1).abs() > cfg["clip_range"]).float().mean()
                    lr_ref = torch.clamp(ref_lp - new_lp, -10.0, 10.0)
                    kl = (torch.exp(lr_ref) - lr_ref - 1.0).mean()               # k3 추정 (샘플 액션에서)
                    ent = -(logp_all[j].exp() * logp_all[j]).sum()
                    pg_terms.append(pg); kl_terms.append(kl); clip_terms.append(clipfrac); ent_terms.append(ent)
                pg = torch.stack(pg_terms).mean()
                kl = torch.stack(kl_terms).mean()
                ent = torch.stack(ent_terms).mean()
                loss = pg + cfg["kl_coef"] * kl - cfg["entropy_coef"] * ent
                if not torch.isfinite(loss):
                    print(f"  [v2] non-finite loss at iter {self.iteration}: pg={pg.item()} kl={kl.item()} - 이 미니배치 건너뜀")
                    continue
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg["max_grad_norm"])
                self.optimizer.step()
                stats["loss"].append(loss.item()); stats["pg"].append(pg.item()); stats["kl"].append(kl.item())
                stats["clipfrac"].append(torch.stack(clip_terms).mean().item()); stats["entropy"].append(ent.item())
        return {k: (float(np.mean(v)) if v else float("nan")) for k, v in stats.items()}

    def advance(self, batch):
        """각 이미지의 DBS 상태를 한 스텝 전진. 반환 (채택 수, 교체 수)."""
        accepted, replaced = 0, 0
        for i, it in enumerate(batch):
            img = it["img"]
            if self.cfg["advance"] == "best":
                j = int(torch.argmax(it["R"]))
                action = int(it["actions"][j]) if float(it["R"][j]) > 0 else int(it["actions"][np.random.randint(len(it["actions"]))])
            else:
                action = int(it["actions"][np.random.randint(len(it["actions"]))])
            ok, _ = img.apply(action)
            accepted += int(ok)
            if img.steps >= self.cfg["steps_per_image"]:
                self.images[i] = self._spawn()
                replaced += 1
        return accepted, replaced

    # ---------------------------------------------------------------- 검증
    @torch.no_grad()
    def validate(self):
        """고정 검증 상태에서 정확한 지표. 시뮬레이션 없음."""
        self.policy.eval()
        rows = []
        for feats, R in zip(self.val_feats, self.val_maps):
            logits = self.policy(feats).squeeze(0)
            p = torch.softmax(logits, dim=0)
            Rp = torch.clamp(R, min=0.0)
            E_pi_relu, E_unif_relu, top1 = float((p * Rp).sum()), float(Rp.mean()), float(R.max())
            ent = float(-(p * torch.log(p + 1e-30)).sum())
            rows.append({
                "E_pi_relu": E_pi_relu, "E_unif_relu": E_unif_relu, "top1": top1,
                "recovery": (E_pi_relu - E_unif_relu) / (top1 - E_unif_relu) if top1 > E_unif_relu else float("nan"),
                "P_pi": float((p * (R > 0).float()).sum()), "P_unif": float((R > 0).float().mean()),
                "E_pi_raw": float((p * R).sum()), "E_unif_raw": float(R.mean()),
                "entropy": ent, "eff_support": float(np.exp(ent)),
                "R_argmax": float(R[int(torch.argmax(logits))]),
                "saturation": self.policy.saturation(feats),
            })
        self.policy.train()
        return {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}

    # ---------------------------------------------------------------- 루프
    def train(self, num_iters, save_dir, save_every, val_every):
        """num_iters 는 이번 실행에서 추가로 도는 반복 수 (재개 시 누적이 아님)."""
        os.makedirs(save_dir, exist_ok=True)
        val_path = os.path.join(self.log_dir, "val.jsonl")
        for _ in range(num_iters):
            t0 = time.time()
            batch = self.collect()
            st = self.update(batch)
            acc, rep = self.advance(batch)
            self.iteration += 1
            R_cat = torch.cat([it["R"] for it in batch])
            print(f"[v2 it {self.iteration}] loss={st['loss']:+.4f} pg={st['pg']:+.4f} kl={st['kl']:.4f} "
                  f"clip={st['clipfrac']:.2f} H={st['entropy']:.2f} | sampled R: mean={float(R_cat.mean()):+.2e} "
                  f"P>0={float((R_cat > 0).float().mean()):.2%} | advance acc={acc}/{len(batch)} new_img={rep} "
                  f"| psnr_gain(avg)={np.mean([it['img'].gain for it in batch]):+.4f} | {time.time() - t0:.1f}s")
            if self.cfg["ref_update_iters"] > 0 and self.iteration % self.cfg["ref_update_iters"] == 0:
                self.ref_policy.load_state_dict(self.policy.state_dict())
                print(f"  [v2] pi_ref <- pi at iter {self.iteration}")
            if self.iteration % val_every == 0:
                v = self.validate()
                print(f"  [v2 val {self.iteration}] E_pi_relu={v['E_pi_relu']:+.3e} E_unif_relu={v['E_unif_relu']:+.3e} "
                      f"top1={v['top1']:+.3e} recovery={v['recovery']:.3f} | P_pi={v['P_pi']:.2%} P_unif={v['P_unif']:.2%} "
                      f"| E_pi_raw={v['E_pi_raw']:+.3e} E_unif_raw={v['E_unif_raw']:+.3e} argmax_R={v['R_argmax']:+.3e} "
                      f"| H={v['entropy']:.2f} eff={v['eff_support']:.0f} sat={v['saturation']:.3f}")
                with open(val_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"iter": self.iteration, **v}) + "\n")
            if self.iteration % save_every == 0:
                self.save_checkpoint(os.path.join(save_dir, f"grpo_v2_it{self.iteration}.pt"))
        self.save_checkpoint(os.path.join(save_dir, "grpo_v2_final.pt"))

    # ---------------------------------------------------------------- 저장/로드
    def checkpoint_dict(self):
        d = {"policy_state_dict": self.policy.state_dict(),
             "ref_policy_state_dict": self.ref_policy.state_dict(),
             "optimizer_state_dict": self.optimizer.state_dict(),
             "iteration": self.iteration, "trainer": "v2"}
        d.update(self.extra_meta)
        return d

    def save_checkpoint(self, path):
        torch.save(self.checkpoint_dict(), path)
        shutil.copyfile(path, os.path.join(os.path.dirname(path), "grpo_v2_latest.pt"))
        print(f"  [v2] checkpoint saved: {path}")

    def load_checkpoint(self, path):
        ck = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ck["policy_state_dict"])
        self.ref_policy.load_state_dict(ck["ref_policy_state_dict"])
        self.optimizer.load_state_dict(ck["optimizer_state_dict"])
        self.iteration = ck["iteration"]
        print(f"  [v2] loaded checkpoint {path} (iteration {self.iteration})")
