"""광학 도메인 상수 — 물리 장치가 정하는 값이라 튜닝 대상이 아니다.

의존성 없는 모듈로 분리한 이유: env.py 는 gymnasium/stable_baselines3 를 import 하므로, 평가 스크립트·grpo 패키지가
상수만 필요할 때 env.py 를 끌어오면 그 패키지들이 없는 환경에서 죽는다. env.py 도 여기서 import 한다.
모든 시뮬레이션 지점이 이 값을 쓰는지는 check_conventions.py 6번이 대조한다.
"""
OPTICS_META = {'dx': (7.56e-6, 7.56e-6), 'wl': 515e-9}   # 픽셀 피치 7.56 um, 파장 515 nm
PROP_Z = 2e-3                                            # 전파 거리 2 mm
