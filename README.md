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

git clone git@github.com:DHLabRepo/torchOptics.git

cd torchOptics

git reset --hard 8e50d6a

.
```

학습 진행 결과를 보려면 아래를 터미널에 입력해서 설치한 뒤,
```
pip install tensorflow
```

`log_py/tensorboard.ipynd` 파일을 복사해서 
`ppo_MultiInputPolicy/` 에서 원하는 로그 폴더에 붙여넣고 코드 돌리면 됨.