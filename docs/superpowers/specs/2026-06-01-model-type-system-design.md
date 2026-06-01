# curvefit model 类型系统 + local 包 设计 (Design Spec)

- 日期: 2026-06-01
- 状态: 待实现
- 原则: 代码风格与 bayspec 严格一致；永不修改 bayspec

## 1. 目标

bayspec 的四种模型类型 `'add' / 'mul' / 'conv' / 'math'` 同样适用于 curvefit。为 curvefit 的 model 层引入类型系统（`Additive`/`Multiplicative`/`Mathematic` 基类 + `CompositeModel` 严格类型校验），并把单文件 `model/local.py` 重组为 `model/local/` 包（`additive.py` / `multiplicative.py` / `mathematic.py`），与 bayspec 结构一一对应。

纯增量改动：不触碰 data / infer / util；现有模型数学行为不变（仅基类与文件位置变化）。

## 2. `model/model.py` 类型系统（移植 bayspec）

- 基类 `Model`：`__init__` 末尾设默认 `self.type = 'add'`；保留 `_allowed_types = ('add', 'mul', 'conv', 'math')`。
- 新增三个基类，`type` 为只读 property + no-op setter（逐字仿 bayspec model.py 的 `Additive`/`Multiplicative`/`Mathematic`）：
  - `Additive` → `'add'`
  - `Multiplicative` → `'mul'`
  - `Mathematic` → `'math'`
- `FrozenConst` 改为继承 `Mathematic`（仿 bayspec）。`func(self, X)` 行为不变（返回常量标量 C）。
- `CompositeModel` 增加：
  - 类属性 `tdict`：`'<t1><op><t2>'` → 结果类型字符串，或 `False`（非法）。包含 `+ - * /` 在 `add/mul/math` 间的全部组合；`conv` 相关行保留作 bayspec parity（因无 conv 模型而不会触发）。
  - `type` property：构造 `type_op` 后查 `tdict`；为 `False` 或缺失则 raise `ValueError`。校验**惰性**（访问 `.type` 时触发），与 bayspec 一致。
  - `type_operation()` 方法：仿 bayspec，返回当前 `(m1, op, m2)` 的结果类型。
  - `func(self, X)` **不变**：仅处理 `+ - * /`。**不**移植 `()` 卷积算子（曲线拟合无卷积、func 签名为 `func(self, X)` 无 O 参数）。

### tdict（add/mul/math 部分，conv 行保留 parity）

`+`/`-`: `add+add→add`, `add+math→add`, `math+add→add`, `math+math→math`, `mul+mul→mul`, `mul+math→mul`, `math+mul→mul`；`add+mul→False` 等其它为 `False`。
`*`: `add*mul→add`, `mul*add→add`, `add*math→add`, `math*add→add`, `mul*mul→mul`, `mul*math→mul`, `math*mul→mul`, `math*math→math`；`add*add→False`。
`/`: 右操作数须为 `mul`/`math`：`add/mul→add`, `add/math→add`, `mul/mul→mul`, `mul/math→mul`, `math/math→math` 等；右操作数为 `add` → `False`。
（以上严格对齐 bayspec `model.py` 的 tdict / type_operation；实现时以 bayspec 源为准逐条移植，仅去掉 `()`/conv 触发路径。）

## 3. `model/local.py` → `model/local/` 包

删除 `model/local.py`，新建 `model/local/`：

- `additive.py`：`ln, pl, expd, bln, bpl, sbpl, spindown, psd`。
  - 基类 `Model` → `Additive`。
  - import 深度调整：`from .model import Model` → `from ..model import Additive`；`from ..util.X import Y` → `from ...util.X import Y`；`docs_path`/`solve_ivp` 等顶部依赖随之调整路径（`dirname` 层数 +1）。
  - 每个模型的 `func` 数学逐字不变。
- `mathematic.py`：新增 `const(Mathematic)`，`Par('C', unif(-10,10))`，`func(self, X)` 返回 `C * np.ones_like(X[:, 0])`。
- `multiplicative.py`：新增 `expcut(Multiplicative)`，`Par('logxc', unif(-10,10))`，`func(self, X)` 返回 `np.exp(-X[:, 0] / 10**logxc)`（无量纲乘性因子，值域 (0,1]）。
- `__init__.py`（仿 bayspec local/__init__）：
  - `from .additive import *  # noqa: F403`、`from .mathematic import *  # noqa: F403`、`from .multiplicative import *  # noqa: F403`、`from ..model import Model`。
  - `local_models = {name: cls for name, cls in globals().items() if isinstance(cls, type) and issubclass(cls, Model) and name not in ['Model', 'Additive', 'Multiplicative', 'Mathematic']}`。
  - `list_local_models()`、`__all__ = [*local_models, 'list_local_models', 'local_models']`。
- `model/__init__.py` 保持 `from .local import *  # noqa: F403`。

新模型代码风格与现有 curvefit local 模型一致（`super().__init__()`、`X[:, 0]` 取 x、`OrderedDict` 参数）。

## 4. 验证标准

- 现有 `tests/test_local_models.py`（golden 输出）守护 8 个迁移模型数学不变 —— 必须仍绿。
- 新增 `tests/test_model_types.py`：
  - `Additive().type == 'add'`、`Multiplicative().type == 'mul'`、`Mathematic().type == 'math'`。
  - `const`/`expcut` 数值正确（如 `const` 全等于 C；`expcut` 在 x=0 处为 1、随 x 增大递减）。
  - `(ln() + pl()).type == 'add'`（add+add）。
  - 非法组合：访问 `(ln() * pl()).type`（add*add）抛 `ValueError`。
- `from curvefit.model.local import ln, pl, const, expcut, list_local_models` 正常；`list_local_models()` 含全部 10 个模型。
- 全套 pytest 通过；`ruff check .` 与 `ruff format --check .` 通过。

## 5. 范围边界

- 仅改 `model/`。不改 data / infer / util / plot。
- 不移植 `()` 卷积算子、不新增 conv 基类或 conv 模型（`conv` 仅作为 `_allowed_types`/tdict 中的 parity 占位）。
- 不新增除 `const`/`expcut` 以外的业务模型。
