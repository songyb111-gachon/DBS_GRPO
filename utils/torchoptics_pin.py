"""torchOptics 고정 커밋 검사.

이 프로젝트의 사전학습 U-Net(BinaryNet, result_v/…_pre_reinforce_8_0.002)은 최신 torchOptics 에서 동작하지 않는다.
그래서 torchOptics 는 DHLabRepo/torchOptics 의 아래 커밋에 서브모듈로 고정돼 있고, 모든 실행 스크립트는 진입부에서
assert_torchoptics_pinned() 를 호출한다. 다른 커밋이면 즉시 RuntimeError 로 죽는다.
조용히 다른 버전으로 돌아가 그럴듯한 결과를 내는 것이 최악이기 때문에 폴백은 두지 않는다.

사용:
    from utils.torchoptics_pin import assert_torchoptics_pinned
    assert_torchoptics_pinned()
"""
import os
import subprocess

# 고정 커밋. 상위 저장소가 기록한 서브모듈 gitlink(`git ls-tree HEAD torchOptics`)와 같아야 한다.
# 어긋나면 검사가 실패한다 — 한쪽만 갱신하는 실수를 잡기 위한 사본 대조.
# 8e50d6a = DHLabRepo/torchOptics 'check_aliasing option added' (2025-05-29). 이후 버전은 BinaryNet 이 깨진다.
TORCHOPTICS_PINNED_COMMIT = "8e50d6a9b1a9f63bb70961ff27fe0a724bd31852"

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TORCHOPTICS_DIR = os.path.join(_REPO_ROOT, "torchOptics")
_FIX_HINT = (
    "저장소 루트에서 `git submodule update --init torchOptics` 를 실행하거나, "
    f"`git -C torchOptics checkout {TORCHOPTICS_PINNED_COMMIT[:7]}` 로 되돌릴 것."
)


def _git(args, cwd):
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    ).stdout.strip()


def assert_torchoptics_pinned():
    """torchOptics 체크아웃이 고정 커밋인지 확인한다. 아니면 RuntimeError."""
    if not os.path.isfile(os.path.join(_TORCHOPTICS_DIR, "optics.py")):
        raise RuntimeError(
            f"[torchOptics] {_TORCHOPTICS_DIR} 가 비어 있다(서브모듈 미초기화). " + _FIX_HINT
        )

    try:
        head = _git(["rev-parse", "HEAD"], _TORCHOPTICS_DIR)
    except (OSError, subprocess.CalledProcessError) as e:
        raise RuntimeError(
            f"[torchOptics] {_TORCHOPTICS_DIR} 의 git 커밋을 읽을 수 없다 ({e}). "
            "git 이 PATH 에 있고 torchOptics 가 git 체크아웃인지 확인할 것."
        ) from e

    if head != TORCHOPTICS_PINNED_COMMIT:
        raise RuntimeError(
            f"[torchOptics] HEAD 가 {head[:7]} 인데 고정 커밋은 {TORCHOPTICS_PINNED_COMMIT[:7]} 이다. "
            "최신 torchOptics 에서는 사전학습 BinaryNet 이 동작하지 않는다. " + _FIX_HINT
        )

    # 사본 대조: 상위 저장소가 기록한 서브모듈 커밋과 위 상수가 어긋나면 한쪽이 갱신되지 않은 것이다.
    try:
        gitlink = _git(["rev-parse", "HEAD:torchOptics"], _REPO_ROOT)
    except (OSError, subprocess.CalledProcessError):
        gitlink = None
    if gitlink is None:
        # 상위가 git 체크아웃이 아닌 파일 복사본이면 gitlink 를 읽을 수 없다. 이 경우만 대조를 생략하고 그 사실을 남긴다.
        print(
            "[torchOptics] 상위 저장소의 서브모듈 gitlink 를 읽을 수 없어 상수-gitlink 대조는 생략했다 "
            f"(torchOptics HEAD {head[:7]} 는 고정 커밋과 일치)."
        )
    elif gitlink != TORCHOPTICS_PINNED_COMMIT:
        raise RuntimeError(
            f"[torchOptics] utils/torchoptics_pin.py 의 TORCHOPTICS_PINNED_COMMIT({TORCHOPTICS_PINNED_COMMIT[:7]})와 "
            f"상위 저장소의 서브모듈 gitlink({gitlink[:7]})가 다르다. 둘 중 하나가 갱신되지 않았다. "
            "의도한 커밋으로 둘을 맞출 것."
        )
    else:
        print(f"[torchOptics] pinned commit OK: {head[:7]}")
