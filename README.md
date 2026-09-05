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

.
```

`torchOptics` 는 DHLabRepo/torchOptics 커밋 `8e50d6a` 에 고정된 서브모듈이다. 최신 torchOptics 에서는 사전학습 U-Net(BinaryNet)이 동작하지 않으므로 업데이트하지 말 것. 모든 실행 스크립트는 진입부에서 `utils/torchoptics_pin.py` 로 이 커밋을 검사하고, 다르면 즉시 에러로 멈춘다.

학습 진행 결과를 보려면 아래를 터미널에 입력해서 설치한 뒤,
```
pip install tensorflow
```

`log_py/tensorboard.ipynd` 파일을 복사해서 
`ppo_MultiInputPolicy/` 에서 원하는 로그 폴더에 붙여넣고 코드 돌리면 됨.