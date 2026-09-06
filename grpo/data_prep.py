"""검증 이미지 → (target, 초기 홀로그램, BinaryNet 확률, 이름). train_grpo.py 의 데이터·사전학습 경로를 그대로 쓴다(한 곳).

train_grpo 를 import 하면 그 모듈의 진입부(setup_logger, 고정 커밋 검사)가 한 번 더 돌지만 학습은 __main__ 에서만 시작한다.
env.py(gymnasium) 는 v1 경로에서만 import 되므로 여기서는 필요 없다."""
import os

import torch


def load_initial_states(num_images, device, start=0):
    """검증 폴더의 정렬 순 [start, start+num_images) 이미지. 반환 [(T (n,n), h0 (C,n,n) 0/1, pre (C,n,n), name)] — 모두 plain 텐서."""
    from train_grpo import BinaryNet, Dataset512, CONFIG, CH
    from optics_constants import OPTICS_META
    ds = Dataset512(target_dir=CONFIG["valid_dir"], meta=OPTICS_META, isTrain=False, padding=0)
    if len(ds) < start + num_images:
        raise ValueError(f"검증 이미지 {len(ds)}장 < 요청 {start}+{num_images} ({CONFIG['valid_dir']})")
    net = BinaryNet(num_hologram=CH, in_planes=1, convReLU=False, convBN=False,
                    poolReLU=False, poolBN=False, deconvReLU=False, deconvBN=False).to(device)
    net.load_state_dict(torch.load(CONFIG["pretrained_path"], map_location=device))
    net.eval()
    out = []
    for i in range(start, start + num_images):
        T, path = ds[i]
        T4 = T.unsqueeze(0).to(device).as_subclass(torch.Tensor)          # (1,1,n,n); tt.Tensor 서브클래스를 벗긴다
        with torch.no_grad():
            pre = net(T4)[0]                                                # (C,n,n)
        out.append((T4[0, 0], (pre >= 0.5).float(), pre, os.path.basename(path)))
    return out
