<div align="center">

# Direct Binary Search Reinforcement Learning

<img src="https://img.shields.io/badge/Python-3.8.10-blue?logo=python&logoColor=white" alt="Python 3.8.10">
<img src="https://img.shields.io/badge/Stable--Baselines3-2.4.1-green?logo=python&logoColor=white" alt="Stable-Baselines3 2.4.1">
<img src="https://img.shields.io/badge/License-AGPL_v3.0-red?logo=gnu&logoColor=white" alt="AGPL v3.0 License">

</div>

터미널에 아래 명령어를 입력 후 코드를 주피터 노트북에 복붙해서 사용.

```
git clone git@github.com:DHLabRepo/Direct-Binary-Search-Reinforcement-Learning.git

cd Direct-Binary-Search-Reinforcement-Learning

mkdir -p result_v

cp -r "/home/songyb111/20260116/Direct-Binary-Search-Reinforcement-Learning/result_v/2024-12-19 20:37:52.499731_pre_reinforce_8_0.002" result_v/

cp -r "/home/songyb111/20260116/Direct-Binary-Search-Reinforcement-Learning/dataset6" .

git submodule update --init torchOptics
```

`torchOptics` 는 DHLabRepo/torchOptics 커밋 `8e50d6a` 에 고정된 서브모듈이다. 최신 torchOptics 에서는 사전학습 U-Net(BinaryNet)이 동작하지 않으므로 업데이트하지 말 것. 모든 실행 스크립트는 진입부에서 `utils/torchoptics_pin.py` 로 이 커밋을 검사하고, 다르면 즉시 에러로 멈춘다.

코드를 고친 뒤에는 `python check_conventions.py` 를 돌린다 (수 초, torch 불필요). 손복사본 일치·고정 커밋·광학 상수·설정 키를 검사한다.
GRPO 학습 설정은 `train_grpo.py` 상단의 `CONFIG` 딕셔너리 한 곳에서만 바꾼다.
스윕/자동화는 파일을 고치지 않고 **앞 셀에서 전역 변수로** 준다: `grpo__lr = 3e-5`, `grpo__env__T_PSNR_DIFF = 0.5` 처럼 `grpo__` 접두사 + 키 경로(점 대신 `__`)를 두고, 같은 셀 흐름에서 `train_grpo.py` 를 실행한다(셀에 붙여넣기 또는 `%run -i train_grpo.py`; `-i` 없는 `%run` 은 앞 셀이 안 보이므로 주입이 있으면 에러로 멈춘다). 안 읽히는 이름·종류가 다른 값(int 자리에 16.5 등)은 즉시 에러, 파일 값을 덮으면 크게 출력, `pretrained_path` 와 광학 상수(`env.py` 의 `OPTICS_META`/`PROP_Z`)는 덮을 수 없다. 주입이 있으면 산출물은 `grpo_models/sweep_<축>_<시각>_j<잡ID 또는 _r무작위>/` 새 폴더에 쌓인다(이미 있으면 에러). 주입 유무와 무관하게 `save_dir` 에 최종 설정이 `overrides.json`(최신)과 `overrides_history.jsonl`(누적)로 남고, 같은 폴더의 체크포인트를 다른 설정으로 이어받으려 하면 멈춘다. 한 커널에서 스크립트를 여러 번 돌리면 로그 파일은 첫 호출 이름 하나에 이어 붙는다(구분은 셀 출력의 `[GRPO] save_dir` 줄과 `overrides.json` 의 `log_file`). 평가는 `eval_checkpoints.py`(체크포인트 비교, `INCLUDE_RANDOM_BASELINE=True` 로 Random 기준선 포함) 와 `test_grpo.py`(GRPO vs Random 전체 데이터셋) 를 쓴다.

학습 진행 결과를 보려면 아래를 터미널에 입력해서 설치한 뒤,
```
pip install tensorflow
```

`log_py/tensorboard.ipynb` 파일을 복사해서 
`ppo_MultiInputPolicy/` 에서 원하는 로그 폴더에 붙여넣고 코드 돌리면 됨.