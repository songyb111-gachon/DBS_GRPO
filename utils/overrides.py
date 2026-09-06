"""스윕/오버라이드 배관 — train_grpo.py 의 CONFIG(중첩 dict)에 바깥 값을 안전하게 얹는 조각들.

각 조각이 막는 사고 (전부 실제로 일어난 종류라, 이 가드를 지우면 같은 사고가 다시 난다):
  1. 스윕에 넣은 이름을 아무도 안 읽으면 모든 arm 이 기본값으로 돌고, 표는 "그 축은 효과 없음" 으로 읽힌다.
     → check_unread(): 접두사가 붙었는데 스키마에 없는 이름은 즉시 ValueError (경고는 로그에 묻힌다).
  2. 손으로 유지하는 이름 목록은 반드시 뒤처진다. → sweepable_names(): CONFIG 자체가 목록이다.
  3. 없는 키에 값을 꽂으면 조용히 무시된다. → apply_overrides(): 없는 키는 KeyError.
  4. 바깥 값이 파일 값을 덮은 사실을 모르면 "고쳤는데 왜 안 먹지" 사고가 난다.
     → collect_overrides() 가 shadowed 목록을 돌려주고 호출측이 크게 찍는다.
  5. JSON/numpy 에서 온 값(16.0, np.int64)이 그대로 꽂히면 학습 도중 torch 내부 문구로 늦게 죽거나 기록이 망가진다.
     → coerce(): 파일 값의 종류에 맞춰 검사하고 파이썬 기본형으로 정규화. 안 맞으면 주입 시점에 ValueError.
  6. 이름이 같으면 arm 끼리 산출물을 덮어쓴다. → axes_string()/run_stamp() 로 이름에 축·잡 번호·스탬프를 넣는다.

주입 소스는 서버 노트북의 앞 셀 전역(globals())이다.
층을 쌓는 순서(CONFIG ← 주입 ← 불변식)와 산출물 이름 규칙은 train_grpo.py 의 build_config()/__main__ 에 있다 —
"왜 이 순서인가" 가 그 자리에서 읽혀야 하므로 여기로 옮기지 않았다.
"""
import datetime as _dt
import difflib
import numbers
import os
import re
import secrets


def flatten(config, sep="."):
    """중첩 dict 를 점 표기 키로 편다. dict 만 내려가고 tuple/list 는 잎(값)으로 본다.
    예: {"env": {"max_steps": 10000}} → {"env.max_steps": 10000}"""
    out = {}

    def rec(d, prefix):
        for k, v in d.items():
            key = f"{prefix}{sep}{k}" if prefix else str(k)
            if isinstance(v, dict):
                rec(v, key)
            else:
                out[key] = v

    rec(config, "")
    return out


def sweepable_names(config, prefix, sep="__"):
    """{'주입 이름': '점 표기 키'}. 스키마(CONFIG)가 곧 목록이라 필드를 더하면 자동으로 닿는다.
    예: prefix='grpo__' → {'grpo__lr': 'lr', 'grpo__env__max_steps': 'env.max_steps', ...}"""
    return {prefix + key.replace(".", sep): key for key in flatten(config)}


def _is_bool(v):
    return isinstance(v, bool) or type(v).__name__ == "bool_"      # numpy.bool_ 포함


def _is_int(v):
    return isinstance(v, numbers.Integral) and not _is_bool(v)      # numpy 정수 포함, bool 제외


def _is_real(v):
    return isinstance(v, numbers.Real) and not _is_bool(v)          # numpy 실수/정수 포함, bool 제외


def coerce(file_value, new_value, opt):
    """파일 값의 '종류' 에 맞춰 주입 값을 검사하고 파이썬 기본형으로 정규화한다. 안 맞으면 ValueError.
      bool 자리 : True/False 만 (1/0 은 거부 — True 가 1 로 조용히 들어가는 것 방지)
      int 자리  : 정수(numpy 포함) 또는 16.0 같은 정수값 실수(JSON 출처) → int. 16.5·문자열은 거부
      float 자리: 숫자(정수 포함) → float
      str 자리  : 문자열만
      None 자리 : (예: seed) 정수값 숫자만 → int. 문자열·불리언은 거부
    정규화 덕에 cfg 와 overrides.json 에는 기본형만 남는다."""
    def fail(why):
        return ValueError(
            f"주입 값이 파일 값과 종류가 다릅니다: {opt}={new_value!r} ({type(new_value).__name__}) "
            f"vs 파일 {file_value!r} ({type(file_value).__name__}) - {why}"
        )

    if isinstance(file_value, bool):
        if _is_bool(new_value):
            return bool(new_value)
        raise fail("bool 자리에는 True/False 만")
    if isinstance(file_value, int):
        if _is_int(new_value):
            return int(new_value)
        if _is_real(new_value) and float(new_value).is_integer():
            return int(new_value)
        raise fail("int 자리에는 정수(또는 16.0 같은 정수값 실수)만")
    if isinstance(file_value, float):
        if _is_real(new_value):
            return float(new_value)
        raise fail("float 자리에는 숫자만")
    if isinstance(file_value, str):
        if isinstance(new_value, str):
            return new_value
        raise fail("str 자리에는 문자열만")
    if file_value is None:
        if _is_int(new_value) or (_is_real(new_value) and float(new_value).is_integer()):
            return int(new_value)
        raise fail("None(미설정) 자리에는 정수만 (예: seed=0)")
    if isinstance(new_value, type(file_value)):
        return new_value
    raise fail(f"{type(file_value).__name__} 자리")


def collect_overrides(source, names, file_values):
    """주입된 값을 모아 dict 로 돌려준다. **적용은 하지 않는다** — 불변식을 뒤에 얹고 출처를 남기기 위해.

    반환 (overrides, shadowed).
      overrides — {'lr': 3e-5, 'env.T_PSNR_DIFF': 0.5, ...} (coerce 로 정규화된 값)
      shadowed  — [(키, 파일값, 주입값)] 바깥 값이 파일 값을 덮은 목록
    주입 값이 None 이면 "안 준 것" 으로 본다."""
    overrides, shadowed = {}, []
    for opt, key in names.items():
        if opt not in source or source[opt] is None:
            continue
        file_value = file_values.get(key)
        value = coerce(file_value, source[opt], opt)
        if key in file_values and file_value != value:
            shadowed.append((key, file_value, value))
        overrides[key] = value
    return overrides, shadowed


def check_unread(source, names, prefix):
    """접두사가 붙었는데 아무도 안 읽는 이름이 있으면 **멈춘다.** 오타 하나로 예산 전체가 헛돈다."""
    unknown = [n for n in source if n.startswith(prefix) and n not in names and source[n] is not None]
    if not unknown:
        return
    lines = []
    for n in unknown:
        near = difflib.get_close_matches(n, sorted(names), n=2, cutoff=0.6)
        lines.append(f"    {n} = {source[n]!r}" + (f"   -> {' 또는 '.join(near)} 인가?" if near else ""))
    raise ValueError(
        f"주어졌는데 아무도 안 읽는 이름 {len(unknown)}개 - 오타면 그 설정 없이 끝까지 돕니다:\n"
        + "\n".join(lines)
        + "\n  일부러 둔 것이면 이름을 바꾸거나 지우세요."
    )


def apply_overrides(cfg, overrides, sep="."):
    """점 표기 키를 중첩 dict 에 꽂는다. **없는 키면 KeyError** — 파일 안 오타를 막는 유일한 방어."""
    for key, value in overrides.items():
        parts = key.split(sep)
        obj = cfg
        for p in parts[:-1]:
            if not isinstance(obj, dict) or p not in obj:
                raise KeyError(f"설정에 '{p}' 가 없습니다 (키='{key}')")
            obj = obj[p]
        if not isinstance(obj, dict) or parts[-1] not in obj:
            raise KeyError(f"설정에 '{parts[-1]}' 가 없습니다 (키='{key}')")
        obj[parts[-1]] = value


def _fmt(v):
    if isinstance(v, bool):
        return "T" if v else "F"
    if isinstance(v, float):
        return f"{v:.3g}"
    if isinstance(v, int):
        return str(v)
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", str(v)).strip("-")
    return s[:40]


def axes_string(overrides):
    """산출물 이름용: 흔든 축을 정렬해 'group_size32_lr3e-05' 처럼. 키의 점은 '-' 로.
    이름만 보고 어느 arm 인지 읽혀야 한다."""
    return "_".join(f"{k.replace('.', '-')}{_fmt(v)}" for k, v in sorted(overrides.items()))


# 배치 스케줄러가 내보내는 잡/태스크 번호. 순서대로 처음 있는 것을 쓴다.
_JOB_VARS = ("SLURM_JOB_ID", "PBS_JOBID", "LSB_JOBID", "JOB_ID")
_TASK_VARS = ("SLURM_ARRAY_TASK_ID", "PBS_ARRAY_INDEX", "LSB_JOBINDEX", "SGE_TASK_ID")


def run_stamp(now=None, environ=None):
    """'YYYYmmdd-HHMMSS' + 잡/태스크 번호(있으면). 잡 번호가 없으면 무작위 4자리를 붙인다 —
    같은 설정의 arm 이 같은 초에 떠서 폴더를 공유하고 서로 이어달리는 사고를 막기 위해."""
    now = now or _dt.datetime.now()
    environ = os.environ if environ is None else environ
    stamp = now.strftime("%Y%m%d-%H%M%S")
    job = next((environ[v] for v in _JOB_VARS if environ.get(v)), None)
    task = next((environ[v] for v in _TASK_VARS if environ.get(v)), None)
    if job:
        stamp += f"_j{re.sub(r'[^A-Za-z0-9]+', '', job.split('.')[0])}"   # PBS 의 '12345.server' 는 앞부분만
    if task:
        stamp += f"_t{re.sub(r'[^A-Za-z0-9]+', '', task)}"
    if not job:
        stamp += f"_r{secrets.token_hex(2)}"
    return stamp
