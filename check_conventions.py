"""프로젝트 규약/일관성 검사. torch 없이 수 초 안에 돈다.

코드를 고친 뒤 항상 실행한다:  python check_conventions.py
실패 항목이 하나라도 있으면 종료 코드 1. 새 규약을 추가하면 여기에도 항목을 추가한다.

검사 항목
  1. torchOptics 가 고정 커밋(8e50d6a)인지 (utils/torchoptics_pin.py)
  2. .gitignore 가 torchOptics/ 를 무시하지 않는지 (서브모듈이므로 추적돼야 한다)
  3. 실행 스크립트 8개가 torchOptics import 전에 setup_logger() 와 assert_torchoptics_pinned() 를 부르는지
  4. 손복사본 대조: 같은 클래스/함수의 사본들이 AST 수준에서 같은지 (허용된 변형 수를 넘으면 실패)
  5. 사전학습 가중치 경로 문자열이 모든 실행 스크립트에서 같은지
  6. 광학 도메인 상수(dx, wl, z)가 모든 파일·모든 시뮬레이션 지점에서 같은 값 하나인지 — AST 로 값을 모은다.
     meta dict 리터럴/OPTICS_META 정의(dict 리터럴·dict() 호출), 인자 z 기본값, PROP_Z 정의, tt.simulate* 의 z(위치·키워드),
     tt.Tensor(meta=...) 의 형태, 이름 z 의 재할당까지 본다. 시뮬레이션 호출이 없는 파일의 meta 리터럴도 대조한다.
  7. train_grpo.py CONFIG 최상위 키가 전부 코드에서 cfg[...] 로 참조되는지(AST 첨자, 주석은 안 셈)와 역방향,
     CONFIG["env"] 의 잎이 env.py BinaryHologramEnv.__init__ 인자와 정확히 같은지
  8. utils/logger.py 가 예전 버그('__file__' in globals())로 되돌아가지 않았는지
  9. log_py/tensorboard.ipynb 코드 셀이 log_py/tensorboard.py 와 같은지 (손복사본 쌍)
 10. 스윕 배관: CONFIG 의 모든 잎이 주입 이름으로 닿는지, 오타가 멈추는지, 타입 정규화, 불변식이 수집 뒤에 오고
     충돌 가드가 있는지, FORCED_KEYS 가 CONFIG 잎에 실재하는지, 축이 다르면 이름이 다른지, 주입이 없으면 이름을 안 바꾸는 분기
"""
import ast
import builtins
import json
import os
import re
import sys

# Windows 기본 콘솔(cp949)에서 라벨의 특수문자로 죽지 않게. 메시지 인코딩 문제로 검사기가 멈추면 안 된다.
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

ENTRY_SCRIPTS = [
    "DBS.py", "train.py", "train_dataset6.py", "valid.py",
    "optimize_hyperparameter.py", "train_grpo.py", "test_grpo.py", "eval_checkpoints.py",
]
GRPO_FILES = ["train_grpo.py", "test_grpo.py", "eval_checkpoints.py"]
GRPO_PKG = ["grpo/oracle.py", "grpo/features.py", "grpo/policies.py", "grpo/dbs_state.py", "grpo/trainer_v2.py",
            "grpo/eval_utils.py", "grpo/oracle_selftest.py", "grpo/smoke_v2.py"]
GRPO_SCRIPTS = ["grpo/oracle_selftest.py", "grpo/smoke_v2.py"]   # torchOptics 를 쓰는 실행 스크립트 — 고정 커밋 검사가 먼저 와야 한다
EVAL_FILES = ["test_grpo.py", "eval_checkpoints.py"]

PRETRAINED_NAME = "2024-12-19 20:37:52.499731_pre_reinforce_8_0.002"
# 광학 도메인 상수. 값 자체는 실험(물리 장치)이 정한다 — 튜닝 대상이 아니다.
# 여기서는 모든 파일·모든 시뮬레이션 지점이 정확히 이 값 하나만 쓰는지 본다.
EXPECTED_DX = (7.56e-6, 7.56e-6)
EXPECTED_WL = 515e-9
EXPECTED_Z = 2e-3
OPTICS_CONST_NAME = "OPTICS_META"     # tt.Tensor(meta=<이 이름>) 은 리터럴 대신 허용
Z_CONST_NAME = "PROP_Z"
SIM_FUNCS = {"simulate", "simulate_faster", "simulate_fresnel"}   # torchOptics 의 (tensor, z, ...) 시그니처 함수들

_failures = []
_passes = 0


def ok(msg):
    global _passes
    _passes += 1
    print(f"  PASS  {msg}")


def fail(msg):
    _failures.append(msg)
    print(f"  FAIL  {msg}")


def check(cond, msg):
    ok(msg) if cond else fail(msg)


def read(name):
    with open(os.path.join(ROOT, name), encoding="utf-8") as f:
        return f.read()


# ------------------------------------------------------------------ 4. 사본 대조
def _strip_docstrings(node):
    for n in ast.walk(node):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            if n.body and isinstance(n.body[0], ast.Expr) and isinstance(getattr(n.body[0], "value", None), ast.Constant) \
                    and isinstance(n.body[0].value.value, str):
                n.body = n.body[1:] or [ast.Pass()]
    return node


def find_def(tree_body, qualname):
    scope, node = tree_body, None
    for part in qualname.split("."):
        node = next((n for n in scope if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == part), None)
        if node is None:
            return None
        scope = node.body
    return node


def node_dump(src, qualname):
    """'Class.method' 또는 'func' 이름의 노드를 찾아 docstring 을 뺀 AST 덤프를 돌려준다. 없으면 None."""
    node = find_def(ast.parse(src).body, qualname)
    return None if node is None else ast.dump(_strip_docstrings(node), include_attributes=False)


def expect_variants(qualname, files, max_variants, note=""):
    """files 안의 qualname 사본을 AST 로 묶는다. 변형 수가 max_variants 를 넘으면 실패."""
    groups, missing = {}, []
    for f in files:
        d = node_dump(read(f), qualname)
        if d is None:
            missing.append(f)
        else:
            groups.setdefault(d, []).append(f)
    if missing:
        fail(f"{qualname}: 정의가 없는 파일 {missing}")
        return
    variants = list(groups.values())
    label = f"{qualname}: 사본 {len(files)}개, 변형 {len(variants)}개 (허용 {max_variants}){' - ' + note if note else ''}"
    check(len(variants) <= max_variants, label)
    if len(variants) > 1:
        for g in variants:
            print(f"          변형: {g}")


# ------------------------------------------------------------------ 6. 광학 상수 (AST)
def _const(node):
    """Constant / 상수 Tuple 을 파이썬 값으로. 아니면 None."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Tuple) and all(isinstance(e, ast.Constant) for e in node.elts):
        return tuple(e.value for e in node.elts)
    return None


def _dict_items(node):
    """dict 리터럴 또는 dict(k=v, ...) 호출을 [(키, 값노드)] 로. 아니면 None."""
    if isinstance(node, ast.Dict):
        return [(k.value, v) for k, v in zip(node.keys, node.values) if isinstance(k, ast.Constant)]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "dict" and not node.args:
        return [(kw.arg, kw.value) for kw in node.keywords if kw.arg]
    return None


def _z_from_node(a, where, z, bad):
    if isinstance(a, ast.Name):
        if a.id not in ("z", Z_CONST_NAME):
            bad.append(f"{where}: z 가 이름 '{a.id}' (z/{Z_CONST_NAME} 만 허용)")
    else:
        v = _const(a)
        if v is None:
            bad.append(f"{where}: z 가 상수도 z 도 아님")
        else:
            z.append(v)


def collect_optics(src):
    """파일에서 (dx 값들, wl 값들, z 값들, 위반 설명들) 을 모은다."""
    tree = ast.parse(src)
    dx, wl, z, bad = [], [], [], []
    for n in ast.walk(tree):
        items = _dict_items(n)
        if items is not None:
            keys = [k for k, _ in items]
            if "dx" in keys or "wl" in keys:
                for k, v in items:
                    if k == "dx":
                        dx.append(_const(v))
                    if k == "wl":
                        wl.append(_const(v))
        if isinstance(n, ast.FunctionDef):
            args = n.args
            names = [a.arg for a in args.args] + [a.arg for a in getattr(args, "posonlyargs", [])]
            defaults = dict(zip(names[-len(args.defaults):] if args.defaults else [], args.defaults))
            for a, d in zip(args.kwonlyargs, args.kw_defaults):
                if d is not None:
                    defaults[a.arg] = d
            if "z" in defaults:
                _z_from_node(defaults["z"], f"def {n.name} 기본값", z, bad)
        if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            for t in targets:
                if isinstance(t, ast.Name) and t.id == Z_CONST_NAME:
                    v = _const(n.value) if n.value is not None else None
                    if v is None:
                        bad.append(f"{Z_CONST_NAME} 정의가 상수가 아님")
                    else:
                        z.append(v)
                if isinstance(t, ast.Name) and t.id == "z":
                    # 이름 z 의 재할당 — 기본값 경로로만 값이 보증되므로 재할당은 위반으로 본다
                    bad.append(f"이름 z 재할당 (line {n.lineno})")
                if isinstance(t, ast.Name) and t.id == OPTICS_CONST_NAME and _dict_items(n.value) is None:
                    bad.append(f"{OPTICS_CONST_NAME} 정의가 dict 리터럴/dict() 호출이 아님")
        if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "z":
            bad.append(f"이름 z 를 for 변수로 사용 (line {n.lineno})")
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            if n.func.attr in SIM_FUNCS:
                a = n.args[1] if len(n.args) >= 2 else next((kw.value for kw in n.keywords if kw.arg == "z"), None)
                if a is None:
                    bad.append(f"tt.{n.func.attr} 호출에 z 인자가 없음 (line {n.lineno})")
                else:
                    _z_from_node(a, f"tt.{n.func.attr} line {n.lineno}", z, bad)
            if n.func.attr == "Tensor":
                for kw in n.keywords:
                    if kw.arg == "meta":
                        if isinstance(kw.value, ast.Name):
                            if kw.value.id != OPTICS_CONST_NAME:
                                bad.append(f"tt.Tensor meta 가 이름 '{kw.value.id}' ({OPTICS_CONST_NAME} 만 허용)")
                        elif _dict_items(kw.value) is None:
                            bad.append("tt.Tensor meta 가 dict 리터럴도 상수 이름도 아님")
    return dx, wl, z, bad


# ------------------------------------------------------------------ 7/10. CONFIG (AST)
def _eval_simple(node):
    """리터럴, 상수 튜플/리스트, 상수끼리의 사칙연산(1/4 등)만 값으로. 그 외는 None 잎."""
    try:
        return ast.literal_eval(node)
    except Exception:
        pass
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        left, right = _eval_simple(node.left), _eval_simple(node.right)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return {ast.Add: left + right, ast.Sub: left - right, ast.Mult: left * right, ast.Div: left / right}[type(node.op)]
    return None


def config_shape(node):
    """CONFIG 의 AST 를 중첩 dict 로. dict 리터럴과 dict(...) 호출을 내려가고, 잎은 평가된 값(불가하면 None)."""
    items = _dict_items(node)
    if items is not None:
        return {k: config_shape(v) for k, v in items}
    return _eval_simple(node)


def find_assign(tree, name):
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets):
            return n.value
    return None


def init_params(src, classname):
    """classname.__init__ 의 인자 이름(self 제외)."""
    node = find_def(ast.parse(src).body, f"{classname}.__init__")
    if node is None:
        return None
    return [a.arg for a in node.args.args[1:]] + [a.arg for a in node.args.kwonlyargs]


# ------------------------------------------------------------------ 11. 미정의 이름 (AST)
# py_compile 은 NameError 를 못 잡는다. 서버에서만 처음 도는 코드가 미정의 이름으로 죽어 왕복이 든 적이 있어
# (grpo/oracle.py env_style_psnr 의 z) 의존성 없이 스코프를 따라가는 검사를 둔다. 정밀도를 우선한다: 잡히면 진짜다.
_SCOPE_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef,
                ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
_BUILTIN_NAMES = set(dir(builtins)) | {"__file__", "__name__", "__doc__", "__builtins__", "__spec__", "__loader__",
                                       "__package__", "get_ipython", "display", "exit", "quit"}


def _name_targets(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _scope_bindings(node, is_root=True):
    """이 스코프에서 바인딩되는 이름. 중첩 스코프(함수/클래스/람다/컴프리헨션) 내부는 제외하되 그 이름 자체는 포함."""
    out = set()
    if is_root:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            a = node.args
            for arg in a.posonlyargs + a.args + a.kwonlyargs:
                out.add(arg.arg)
            if a.vararg:
                out.add(a.vararg.arg)
            if a.kwarg:
                out.add(a.kwarg.arg)
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            for gen in node.generators:
                out |= _name_targets(gen.target)
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(child.name)
            continue
        if isinstance(child, _SCOPE_NODES):
            continue
        if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)):
            out.add(child.id)
        elif isinstance(child, (ast.Import, ast.ImportFrom)):
            for alias in child.names:
                out.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(child, ast.ExceptHandler) and child.name:
            out.add(child.name)
        elif isinstance(child, (ast.Global, ast.Nonlocal)):
            out |= set(child.names)
        out |= _scope_bindings(child, is_root=False)
    return out


def _check_loads(node, chain, out):
    """chain: 안쪽부터 바깥쪽으로 (bound_set, is_class). 함수/컴프리헨션에서는 클래스 스코프가 안 보인다."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, _SCOPE_NODES):
            b = _scope_bindings(child)
            inner = [(b, True)] + chain if isinstance(child, ast.ClassDef) else [(b, False)] + [c for c in chain if not c[1]]
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                for d in child.decorator_list:                       # 데코레이터·기본값은 바깥 스코프에서 평가
                    _check_loads_expr(d, chain, out)
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                for d in child.args.defaults + [x for x in child.args.kw_defaults if x is not None]:
                    _check_loads_expr(d, chain, out)
            _check_loads(child, inner, out)
            continue
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
            if child.id not in _BUILTIN_NAMES and not any(child.id in b for b, _ in chain):
                out.append((child.lineno, child.id))
        _check_loads(child, chain, out)


def _check_loads_expr(expr, chain, out):
    if isinstance(expr, ast.Name) and isinstance(expr.ctx, ast.Load):
        if expr.id not in _BUILTIN_NAMES and not any(expr.id in b for b, _ in chain):
            out.append((expr.lineno, expr.id))
    _check_loads(expr, chain, out)


def undefined_names(src):
    """모듈 소스에서 어느 스코프에서도 바인딩되지 않은 채 읽히는 이름 [(줄, 이름)]."""
    tree = ast.parse(src)
    out = []
    _check_loads(tree, [(_scope_bindings(tree), False)], out)
    return sorted(set(out))


def main():
    print("== 1. torchOptics 고정 커밋 ==")
    try:
        from utils.torchoptics_pin import assert_torchoptics_pinned
        assert_torchoptics_pinned()
        ok("torchOptics HEAD == 고정 커밋")
    except Exception as e:  # noqa: BLE001 — 실패 이유를 그대로 보여준다
        fail(f"torchOptics 고정 검사 실패: {e}")

    print("== 2. .gitignore ==")
    gi = read(".gitignore").splitlines()
    check("torchOptics/" not in [l.strip() for l in gi], ".gitignore 에 torchOptics/ 없음 (서브모듈 추적)")

    print("== 3. 실행 스크립트 진입부 ==")
    for f in ENTRY_SCRIPTS:
        lines = read(f).splitlines()

        def idx(pred):
            return next((i for i, l in enumerate(lines) if pred(l)), None)

        i_logger = idx(lambda l: "setup_logger()" in l and "def " not in l)
        i_pin = idx(lambda l: l.strip().startswith("assert_torchoptics_pinned()"))
        i_to = idx(lambda l: l.startswith("import torchOptics"))
        cond = None not in (i_logger, i_pin, i_to) and i_logger < i_pin < i_to
        check(cond, f"{f}: setup_logger() -> assert_torchoptics_pinned() -> import torchOptics 순서")
    for f in GRPO_SCRIPTS:
        lines = read(f).splitlines()
        i_pin = next((i for i, l in enumerate(lines) if l.strip().startswith("assert_torchoptics_pinned()")), None)
        i_heavy = next((i for i, l in enumerate(lines)
                        if l.startswith(("import torch", "from grpo", "from env", "import torchOptics", "from train_grpo"))), None)
        check(i_pin is not None and (i_heavy is None or i_pin < i_heavy),
              f"{f}: assert_torchoptics_pinned() 가 torch/grpo/env import 보다 앞")

    print("== 4. 손복사본 대조 (AST) ==")
    expect_variants("BinaryNet.__init__", ENTRY_SCRIPTS, 2,
                    "GRPO 계열 3개는 CRB2d/TRB2d 의 리스트 생성·return 을 한 줄로 줄인 동치 표현 (알려진 변형)")
    expect_variants("BinaryNet.forward", ENTRY_SCRIPTS, 2,
                    "test_grpo/eval_checkpoints 는 중간 변수를 합친 동치 표현 (알려진 변형)")
    expect_variants("Dataset512.__init__", ENTRY_SCRIPTS, 2,
                    "test_grpo/eval_checkpoints 는 self.transform 을 안 둠 (알려진 변형)")
    expect_variants("Dataset512.__getitem__", ENTRY_SCRIPTS, 2,
                    "pad 를 분기 안/밖에 둔 동치 표현 (알려진 변형)")
    expect_variants("GRPOPolicy.__init__", GRPO_FILES, 1, "체크포인트 호환의 근거")
    expect_variants("GRPOPolicy.forward", GRPO_FILES, 1, "logit clamp 포함")
    expect_variants("simulate_psnr", EVAL_FILES, 1)
    expect_variants("make_random_action_fn", EVAL_FILES, 1)

    print("== 5. 사전학습 가중치 경로 ==")
    for f in ENTRY_SCRIPTS:
        check(read(f).count(PRETRAINED_NAME) >= 2, f"{f}: '{PRETRAINED_NAME}' 폴더/파일 경로 사용")

    print("== 6. 광학 도메인 상수 (AST, 모든 파일) ==")
    check("from optics_constants import OPTICS_META, PROP_Z" in read("env.py"),
          "env.py 가 optics_constants 에서 OPTICS_META/PROP_Z 를 가져옴 (정의는 optics_constants.py 한 곳)")
    all_dx, all_wl, all_z = set(), set(), set()
    for f in ENTRY_SCRIPTS + ["env.py", "optics_constants.py"] + GRPO_PKG:
        dx, wl, z, bad = collect_optics(read(f))
        all_dx.update(dx)
        all_wl.update(wl)
        all_z.update(z)
        check(not bad, f"{f}: meta/z 가 리터럴 또는 허용된 상수 이름, z 재할당 없음" + (f" - 위반 {bad}" if bad else ""))
        print(f"          {f}: dx {len(dx)}곳, wl {len(wl)}곳, z {len(z)}곳")
        if f == "optics_constants.py":
            check(len(dx) >= 1 and len(wl) >= 1 and len(z) >= 1,
                  f"optics_constants.py 에 {OPTICS_CONST_NAME}(dx, wl)·{Z_CONST_NAME} 정의가 잡힘")
    check(all_dx == {EXPECTED_DX}, f"모든 파일의 dx 값 집합 == {{{EXPECTED_DX}}} (실제 {all_dx})")
    check(all_wl == {EXPECTED_WL}, f"모든 파일의 wl 값 집합 == {{{EXPECTED_WL}}} (실제 {all_wl})")
    check(all_z == {EXPECTED_Z}, f"모든 파일의 z 값 집합 == {{{EXPECTED_Z}}} (실제 {all_z})")

    print("== 7. train_grpo.py CONFIG 키 참조 (AST) ==")
    src = read("train_grpo.py")
    tree = ast.parse(src)
    cfg_node = find_assign(tree, "CONFIG")
    shape = None
    if cfg_node is None or _dict_items(cfg_node) is None:
        fail("train_grpo.py: 모듈 상단에 CONFIG = {...} 딕셔너리가 없음")
    else:
        shape = config_shape(cfg_node)
        keys = set(shape)
        refs = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id in ("cfg", "CONFIG"):
                s = n.slice
                s = s.value if isinstance(s, getattr(ast, "Index", ())) else s   # py3.8 호환
                if isinstance(s, ast.Constant) and isinstance(s.value, str):
                    refs.add(s.value)
        unused = sorted(keys - refs)
        unknown = sorted(refs - keys)
        check(not unused, f"CONFIG 최상위 키 {len(keys)}개 모두 코드에서 cfg[...]/CONFIG[...] 로 참조됨" + (f" - 미참조 {unused}" if unused else ""))
        check(not unknown, "참조된 키가 전부 CONFIG 에 존재" + (f" - CONFIG 에 없는 키 참조 {unknown}" if unknown else ""))
        env_params = init_params(read("env.py"), "BinaryHologramEnv")
        env_kwargs = set(env_params or []) - {"target_function", "trainloader"}
        env_leaves = set(shape.get("env", {}) or {})
        check(env_params is not None and env_leaves == env_kwargs,
              f"CONFIG['env'] 잎 {sorted(env_leaves)} == BinaryHologramEnv.__init__ 인자 {sorted(env_kwargs)}")

    print("== 8. utils/logger.py 회귀 ==")
    check("'__file__' in globals()" not in read("utils/logger.py"),
          "utils/logger.py 가 자기 모듈 __file__ 을 보는 예전 버그로 돌아가지 않음")

    print("== 9. tensorboard.ipynb == tensorboard.py ==")
    nb = json.loads(read("log_py/tensorboard.ipynb"))
    nb_code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    norm = lambda s: "\n".join(l.rstrip() for l in s.strip().splitlines())  # noqa: E731
    check(norm(nb_code) == norm(read("log_py/tensorboard.py")),
          "log_py/tensorboard.ipynb 코드 셀 == log_py/tensorboard.py (줄 끝 공백·말미 개행 무시)")

    print("== 10. 스윕 배관 (utils/overrides.py + train_grpo.py) ==")
    from utils.overrides import sweepable_names, collect_overrides, flatten, axes_string, check_unread
    if shape is None:
        fail("CONFIG 를 읽지 못해 스윕 검사를 건너뜀")
    else:
        prefix_node = find_assign(tree, "SWEEP_PREFIX")
        prefix = prefix_node.value if isinstance(prefix_node, ast.Constant) else None
        check(isinstance(prefix, str) and prefix.endswith("__"), f"SWEEP_PREFIX 가 문자열이고 '__' 로 끝남 ({prefix!r})")
        prefix = prefix or "grpo__"
        names = sweepable_names(shape, prefix)
        file_values = flatten(shape)
        probe = {opt: file_values[key] for opt, key in names.items() if file_values[key] is not None}
        probe_none = [key for opt, key in names.items() if file_values[key] is None]
        reached, _ = collect_overrides(probe, names, file_values)
        missing = sorted(set(names.values()) - set(reached) - set(probe_none))
        check(not missing, f"CONFIG 잎 {len(names)}개가 주입 이름으로 닿음 (값이 None 인 잎 {probe_none} 은 검사 제외)" + (f" - 못 닿음 {missing}" if missing else ""))
        try:
            check_unread({prefix + "lrr": 1.0}, names, prefix)
            fail("오타 주입(lrr)이 멈추지 않음")
        except ValueError:
            ok("오타 주입(lrr)은 ValueError 로 멈춤")
        # 타입 정규화: int 자리에 16.0 → 16, int 자리에 16.5 → 멈춤
        int_key = next((k for k, v in file_values.items() if isinstance(v, int) and not isinstance(v, bool)), None)
        if int_key:
            opt = prefix + int_key.replace(".", "__")
            got, _ = collect_overrides({opt: float(file_values[int_key])}, names, file_values)
            check(type(got[int_key]) is int, f"int 자리({int_key})에 정수값 실수는 int 로 정규화")
            try:
                collect_overrides({opt: file_values[int_key] + 0.5}, names, file_values)
                fail(f"int 자리({int_key})에 비정수 실수가 멈추지 않음")
            except ValueError:
                ok(f"int 자리({int_key})에 비정수 실수는 ValueError 로 멈춤")
        forced_node = find_assign(tree, "FORCED_KEYS")
        forced = _eval_simple(forced_node) if forced_node is not None else None
        check(isinstance(forced, tuple) and all(k in file_values for k in forced),
              f"FORCED_KEYS {forced!r} 가 CONFIG 잎에 실재")
        # docstring 에도 FORCED_KEYS 가 적혀 있으므로, 수집 호출 '뒤' 에서 불변식 대조와 raise 를 찾는다
        bc = src[src.index("def build_config"):] if "def build_config" in src else ""
        i_collect = bc.find("collect_overrides(")
        i_forced = bc.find("FORCED_KEYS", i_collect) if i_collect >= 0 else -1
        i_raise = bc.find("raise ValueError", i_forced) if i_forced >= 0 else -1
        check(0 <= i_collect < i_forced < i_raise, "build_config: 수집 -> 불변식 대조 -> raise ValueError 순서")
        check("v1_only" in bc and "v2_keys" in bc, "build_config: 트레이너 전용 키 주입 가드(v1_only/v2_keys) 존재")
        a, b, c = axes_string({"lr": 1e-5}), axes_string({"lr": 3e-5}), axes_string({"lr": 1e-5, "seed": 0})
        check(len({a, b, c}) == 3, f"축이 다르면 산출물 이름이 다름 ({a} / {b} / {c})")
        has_branch = any(isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == "overrides"
                         for n in ast.walk(tree))
        check(has_branch, "주입 여부(overrides)로 save_dir 을 가르는 if 분기가 코드에 있음 (주입 없으면 예전 경로)")

    print("== 11. 미정의 이름 (AST, 모든 모듈) ==")
    for f in ENTRY_SCRIPTS + GRPO_PKG + ["env.py", "optics_constants.py", "grpo/oracle_algebra_check_np.py",
                                        "utils/logger.py", "utils/overrides.py", "utils/torchoptics_pin.py", "check_conventions.py"]:
        u = undefined_names(read(f))
        check(not u, f"{f}: 미정의 이름 없음" + (f" - 위반 {u}" if u else ""))

    print("== 12. CONFIG['v2'] 키 <-> trainer_v2/train_grpo 참조 <-> grpo/smoke_v2.py cfg (손복사본) ==")
    src = read("train_grpo.py")
    blk = src[src.index('"v2": dict('):]
    blk = blk[:blk.index("
    ),")]
    v2_keys = set(re.findall(r"^\s+([a-z_]+)=", blk, re.M))
    tr = read("grpo/trainer_v2.py")
    used = set(re.findall(r'cfg\["([a-z_]+)"\]', tr)) | set(re.findall(r'v2\["([a-z_]+)"\]', src))
    check(not (used - v2_keys), f"코드가 읽는 v2 키가 CONFIG['v2'] 에 전부 있음 (없는 키: {sorted(used - v2_keys)})")
    check(not (v2_keys - used), f"CONFIG['v2'] 키 {len(v2_keys)}개를 코드가 전부 읽음 (안 읽는 키: {sorted(v2_keys - used)})")
    sm = read("grpo/smoke_v2.py")
    missing = sorted(k for k in v2_keys if k + "=" not in sm)
    check(not missing, f"grpo/smoke_v2.py 의 cfg 가 CONFIG['v2'] 키를 전부 가짐 (빠진 키: {missing})")

    print()
    print(f"통과 {_passes}, 실패 {len(_failures)}")
    if _failures:
        print("실패 항목:")
        for m in _failures:
            print(f"  - {m}")
        sys.exit(1)


if __name__ == "__main__":
    main()
