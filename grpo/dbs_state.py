"""이미지 하나의 DBS 상태 (PLAN.md 2.3절). 채택 규칙은 test_grpo.py/eval_checkpoints.py 의 run_dbs 와 같다:
플립 후 PSNR 이 **엄격히** 오르면 채택, 아니면 되돌린다. env.py 의 에피소드/임계 정의는 쓰지 않는다(학습 전용 루프).

채택 판정은 오라클 지도가 아니라 실제 재계산(oracle.forward + float64 PSNR)으로 한다 — 오라클 오차에 의존하지 않기 위해.
평가 경로(tt.simulate + float32 kornia PSNR)와의 차이는 ≈1e-6 dB 급이라 1e-6 dB 이하의 플립에서만 채택 여부가 갈릴 수 있다.
거절 시에는 저장해 둔 U·I·psnr·오라클 지도를 그대로 복원한다 (상태가 안 바뀌었으므로 재계산이 필요 없다)."""
import torch


class DBSImage:
    def __init__(self, oracle, target, h0, name=""):
        """oracle: FlipOracle, target (n,n) 0~1, h0 (C,n,n) 0/1 초기 홀로그램 (모두 oracle.device)."""
        self.oracle = oracle
        # tt.Tensor 같은 서브클래스는 여기서 벗긴다 (as_subclass 는 __torch_function__ 이 다시 감싸지 않는다).
        # 안 벗기면 인덱싱·연산을 타고 특징·액션·보상까지 번지고, 0-차원 서브클래스 텐서는 f-string 포맷에서 TypeError 를 낸다.
        self.T = target.to(oracle.device).float().as_subclass(torch.Tensor)
        self.h = h0.to(oracle.device).float().clone().as_subclass(torch.Tensor)
        self.pre_model = None      # (C,n,n) 사전학습 모델 확률 (특징용), 호출측이 채운다
        self.name = name
        self.steps = 0
        self.flips = 0
        self._map = None           # 마지막 오라클 결과 (dict)
        self.refresh()
        self.initial_psnr = self.psnr

    def refresh(self):
        self.U = self.oracle.forward(self.h)
        self.I = (self.U.abs() ** 2).mean(dim=0)
        self.psnr, self.mse = self.oracle.psnr_from(self.I, self.T)
        self.c = (self.T.double().mean() / self.I.double().mean()).item()
        self._map = None

    def reward_map(self):
        """모든 (채널, 픽셀) 플립의 ΔPSNR (C,n,n). 상태가 바뀌기 전까지 캐시."""
        if self._map is None:
            self._map = self.oracle.flip_psnr_map(self.h, self.T, U=self.U)
        return self._map["dpsnr"]

    def apply(self, action):
        """action: int in [0, C·n·n). 플립 → 실제 PSNR 재계산 → 오르면 채택, 아니면 되돌림(캐시 복원). 반환 (accepted, psnr_after)."""
        n = self.h.shape[-1]
        ci, rest = divmod(int(action), n * n)
        r, cc = divmod(rest, n)
        snapshot = (self.U, self.I, self.psnr, self.mse, self.c, self._map)
        before = self.psnr
        self.h[ci, r, cc] = 1.0 - self.h[ci, r, cc]
        self.refresh()
        self.steps += 1
        if self.psnr > before:
            self.flips += 1
            return True, self.psnr
        self.h[ci, r, cc] = 1.0 - self.h[ci, r, cc]
        self.U, self.I, self.psnr, self.mse, self.c, self._map = snapshot
        return False, self.psnr

    def random_accept_steps(self, k, rng):
        """무작위 픽셀을 뽑아 오라클 지도에서 R>0 이면 채택하는 DBS 를 k 스텝 진행 (검증용 '진행 상태' 를 만들 때 쓴다)."""
        n = self.h.shape[-1]
        N = self.h.shape[0] * n * n
        for _ in range(k):
            a = int(rng.integers(N))
            if float(self.reward_map().reshape(-1)[a]) > 0:
                self.apply(a)
            else:
                self.steps += 1

    @property
    def gain(self):
        return self.psnr - self.initial_psnr
