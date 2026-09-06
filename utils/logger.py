import sys
import logging
from datetime import datetime
import os

# 첫 setup_logger() 가 만든 로그 파일. 재호출(노트북 셀 재실행 등)은 새 파일을 만들지 않고 이 값을 돌려준다.
_ACTIVE_LOG_FILENAME = None


class Tee:
    """stdout 을 원본 스트림과 로그 파일에 동시에 쓴다. Jupyter 가 요구하는 속성은 원본에 위임한다."""

    def __init__(self, original, log_file):
        self._original = original
        self._log_file = log_file

    def write(self, data):
        self._original.write(data)
        self._original.flush()   # stdout 이 파일/파이프로 리다이렉트된 잡 출력도 실시간으로 보이게
        self._log_file.write(data)
        self._log_file.flush()   # 실시간 저장

    def flush(self):
        self._original.flush()
        self._log_file.flush()

    @property
    def encoding(self):
        return getattr(self._original, 'encoding', 'utf-8')

    def isatty(self):
        return getattr(self._original, 'isatty', lambda: False)()

    def fileno(self):
        return self._original.fileno()

    def readable(self):
        return False

    def writable(self):
        return True


def _caller_script_name():
    """실행 스크립트 이름(확장자 제거)과 폴백 이유. sys.argv[0] 기준이라
    `python x.py`, `python -m x`, IPython `%run x.py` 모두 'x' 가 나온다.
    노트북 셀(ipykernel_launcher), 인터프리터(''), `-c` 는 'interactive'.
    (inspect.stack() 방식은 Python<=3.10 의 -m/Jupyter 에서 'runpy' 를 돌려주는 함정이 있어 쓰지 않는다.
     예전 구현은 이 모듈 자신의 __file__ 을 봐서 어느 스크립트가 불러도 항상 'logger' 가 됐다.)"""
    argv0 = sys.argv[0] if sys.argv else ""
    name = os.path.splitext(os.path.basename(argv0))[0]
    if not name or name == "-c" or name.startswith("ipykernel"):
        return "interactive", f"sys.argv[0]={argv0!r} 은 스크립트가 아님(노트북/인터프리터)"
    return name, None


def setup_logger(log_dir="log"):
    global _ACTIVE_LOG_FILENAME
    if _ACTIVE_LOG_FILENAME is not None:
        print(f"[logger] 이미 초기화됨 - 핸들러/Tee 를 다시 만들지 않고 기존 로그 파일 {_ACTIVE_LOG_FILENAME} 을 계속 쓴다")
        return _ACTIVE_LOG_FILENAME

    os.makedirs(log_dir, exist_ok=True)  # 디렉토리가 없으면 생성

    current_file, fallback_reason = _caller_script_name()
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = os.path.join(log_dir, f"{current_file}_{current_datetime}.log")

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    if root_logger.handlers:
        # 다른 코드가 먼저 logging 을 설정한 경우. 조용히 넘기지 않고 남긴다.
        print(f"[logger] root 로거에 핸들러가 이미 있어 FileHandler 를 추가하지 않는다 - "
              f"logging.* 출력은 {log_filename} 에 남지 않는다 (print 출력은 Tee 로 기록)")
    else:
        root_logger.addHandler(logging.FileHandler(log_filename, encoding="utf-8"))  # Tee 와 같은 인코딩
        root_logger.addHandler(logging.StreamHandler())
    for h in root_logger.handlers:
        h.setFormatter(logging.Formatter('%(message)s'))

    # stdout 을 파일과 콘솔로 동시에 출력
    if not isinstance(sys.stdout, Tee):
        log_file = open(log_filename, "a", encoding="utf-8")
        sys.stdout = Tee(sys.stdout, log_file)

    _ACTIVE_LOG_FILENAME = log_filename
    if fallback_reason:
        # Tee 설치 뒤에 찍어야 로그 파일에도 남는다
        print(f"[logger] 호출 스크립트 이름을 정하지 못해 'interactive' 로 폴백 ({fallback_reason})")
    return log_filename  # 로그 파일 이름 반환
