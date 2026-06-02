# curvefit infer 对齐 bayspec：(stat, residual) + MaxLikeFit 设计 (Design Spec)

- 日期: 2026-06-02
- 状态: 待实现
- 原则: 代码风格与 bayspec 严格一致；永不修改 bayspec；注释遵循 `docs/docstring-standard.md`

## 0. 背景与分期

curvefit 的 infer 层此前偏离了 bayspec：保留了用户给的六个曲线拟合统计量（仅返回 loglike），但未移植 bayspec 的 `(stat, residual)` 约定、`pseudo_residual` 流、`MaxLikeFit`(lmfit/iminuit) 与 `Bootstrap` 分析器。本设计把 infer 层对齐 bayspec。

分两期（各自可独立交付）：
- **Phase 1（本 spec）**：功能对齐 —— statistic `(stat, residual)` 重构、Pair pseudo_residual、Infer calc_*/clean labels、Bootstrap、MaxLikeFit。本期新增/重写的文件同时按 docstring 标准书写。
- **Phase 2（后续 spec）**：对整个 curvefit 包按 `docstring-standard.md` 全量补齐 docstring（以 bayspec 逐文件为参照）。

## 1. 已确认的设计决策

1. **归一化项保留**：`chi^2f`/`vdr`/`odr` 的 `log(2πσ²)` 项是 σ 自由时 MLE 所必需，保留；`chi^2`/`logchi^2`（σ 固定）不含该项。六个统计量的数学公式**不改**。
2. **lmfit 限纯 χ²**：`lmfit()` 仅接受 residual 良定义的统计量 `{chi^2, logchi^2}`；其余调用 lmfit 时报错并提示改用 `iminuit()`。`iminuit()` 最小化标量 stack，对全部六个统计量精确。
3. **residual 约定**：每个统计量返回 `(stat, residual)`。`residualᵢ = sign(y−myᵢ)·√clip(Sᵢ, 0, ∞)`（纯 χ² 统计 Sᵢ≥0 自然为正；σ 自由统计的 clip 仅用于诊断/绘图，lmfit 不接受这些统计量）。上限点：允许 `Sᵢ=0→residualᵢ=0`，违反 `Sᵢ=inf→residualᵢ=inf`。
4. **不用 numba**：`groth` 用 `Decimal` 无法 numba 化，曲线数据点数小，统一用纯 numpy（不建 `StatisticNB`）。

## 2. `infer/statistic.py` 重构

保持现有调用接口（`mo_func, params, x, y, x_err, y_err, w, up`），但：
- 每个统计量改为 `@staticmethod Name(**kwargs)`，内部计算 `my = mo_func(x, params)` 后委托给模块级 `_name_core(...)`，返回 `(stat, residual)`（仿 bayspec `Gstat`→`_gstat_core`）。
- `stat = ΣSᵢ`（即 −2·loglike；`groth` 为 −2·lnL）。
- `residual` 按决策 3 定义。

核心函数（纯 numpy）：
- `_chi_square_core(my, y, y_err, w, up)` → (stat, residual)
- `_chi_square_full_core(my, y, y_err, w, up, logv)`
- `_log_chi_square_core(my, y, y_err, w, up)`
- `_vdr_core(my, x, y, x_err, y_err, w, up, k, logv)`
- `_odr_core(my, x, y, x_err, y_err, w, up, k, logv)`
- `_groth_core(my, y, w, up)`

公式逐字沿用现有 `Statistic`（只把 `-0.5*ΣS`/`lnL` 改成返回 `stat=ΣS`(或 `-2lnL`) 与 per-point residual）。`vdr`/`odr` 仍取 `k=params[0]`、`logv=params[2]`（线性模型约定）；`chi^2f` 取 `logv=params[-1]`。

`_allowed_stats` 字典（`MappingProxyType`）：6 个名字 → 对应 `@staticmethod`。
新增模块级常量 `LMFIT_SAFE_STATS = frozenset({'chi^2', 'logchi^2'})`，供 `MaxLikeFit.lmfit()` 校验。

## 3. `infer/pair.py`

- `stat_func`（cached）：`...(**kwargs)[0]`，模型非有限时返回 `inf`。
- `pseudo_residual_func`（cached）：`...(**kwargs)[1]`，模型非有限时返回 `ones_like·inf`。
- `_stat_calculate` / `_pseudo_residual_calculate`：按 unit 映射。
- 属性：`stat_list`、`pseudo_residual_list`、`stat`(=Σ stat_list·weight_list)、`pseudo_residual`(=concatenate(pseudo_residual_list))、`loglike_list`(=−0.5·stat_list)、`loglike`(=−0.5·stat)、`npoint_list`/`npoint`。
- 每点权重 `w` 已在统计量内部，故 `weight_list` 仍为 `ones(nunits)`，`pseudo_residual` 不再额外乘 √weight。
- residual 诊断属性：`residual`（逐 unit `(y−my)/yerr` 列表，用于绘图/残差面板）。

## 4. `infer/infer.py`

新增（其余保持，含 emcee/multinest/minimize/BayesInfer）：
- `__init__` 增加 `self.loglike_func = None`、`self.logprior_func = None`。
- `clean_free_plabels`：`free_plabels` 去掉 `$ { } \`（lmfit/iminuit 参数名需合法）。
- `clean_free_indexed_plabels`：`f'p{key}({clean_label})'`。
- `calc_loglike(theta)`：`at_par` 后返回 `loglike`（或 `loglike_func` 覆盖）。
- `calc_stat(theta)`：`at_par` 后返回 `stat`。
- `calc_pseudo_residual(theta)`：`at_par` 后返回 `pseudo_residual`。
- 属性 `stat`（=Σ pair.stat）、`pseudo_residual`（=hstack(pair.pseudo_residual)）、`residual`（=逐 unit 列表）。
- 现有 `_loglike` 保留（emcee/multinest 用），`calc_loglike` 为对齐 bayspec 的公开入口。

## 5. `infer/posterior.py` → SampleAnalyzer + Posterior + Bootstrap

重构（仿 bayspec `analyzer.py`，保持 emcee/multinest 返回 `Posterior` 不变）：
- `SampleAnalyzer(Infer)`：吸收一个 infer（`__dict__.update`），从 `sample_attribute` 命名的属性加载 `(nsample, nfree+1)` 样本矩阵并校验；提供 per-free-param `Post`、点估计（mean/median/best/best_ci）、区间、AIC/AICc/BIC/lnZ、`free_par_info`/`stat_info`/`IC_info`/`__str__`。`sample_attribute=None`、`analyzer_type` 由子类设。
- `Posterior(SampleAnalyzer)`：`sample_attribute='posterior_sample'`，`analyzer_type='Posterior Results'`。行为与现状一致。
- `Bootstrap(SampleAnalyzer)`：`sample_attribute='bootstrap_sample'`，`analyzer_type='Bootstrap Results'`；best 取样本首行（best-fit）。

现有 `Posterior` 的方法整体迁入 `SampleAnalyzer`，确保 emcee/multinest（`return Posterior(self)`）行为不变。

## 6. `MaxLikeFit(Infer)`（infer.py）

仿 bayspec：
- `__init__`：`super().__init__(pairs)`；`self.inference_type = 'Maximum Likelihood Estimation'`。
- `lmfit(savepath=None)`：校验所有 unit 的 stat ∈ `LMFIT_SAFE_STATS`，否则 raise（提示用 iminuit）；用 `clean_free_plabels` 建 `lmfit.Parameters`（value/min/max）；`lmfit.minimize(self.lmfit_residual, params)`；提取 values/errors/covar；`_make_bootstrap_sample`；可选落盘；返回 `Bootstrap(self)`。
- `lmfit_residual(params)`：`calc_pseudo_residual([params[pl] for pl in clean_free_plabels])`。
- `iminuit(savepath=None)`：`iminuit.Minuit(self.iminuit_cost, *free_pvalues, name=clean_free_indexed_plabels)`；`errordef=2*LIKELIHOOD`；limits；migrad+hesse+minos；提取 values/errors/minos_errors/covar；bootstrap；返回 `Bootstrap(self)`。
- `iminuit_cost(*theta)`：`calc_stat(theta)`，非有限返回 `1e100`。
- `_make_bootstrap_sample(values, covar, errors, nsample=1000, random_seed=...)`：协方差（无效则用 errors 对角）→ eigh 正定化 → 范围内多元正态抽样 → 每个 theta 用 `calc_loglike` 打分 → `self.bootstrap_sample = hstack(params, loglike[:,None])`；`at_par(values)`。
- `_display_results(*objects)`：有 IPython 用 `display` 否则 `print`。
- `infer/__init__.py` 导出 `MaxLikeFit`。

## 7. 验证标准

- 现有 37 测试全绿（重构不破坏现状）。
- 新增 `tests/test_statistic_residual.py`：六个统计量均返回 `(stat, residual)`；`chi^2` 的 `Σresidual² == stat`、`residual==√w·(y−my)/yerr`；上限点 residual 行为（0 / inf）；`stat == -2*loglike` 一致。
- 新增 `tests/test_maxlikefit.py`：
  - 线性数据 `lmfit()`（stat=chi^2）恢复 k≈truth、b≈truth，返回 `Bootstrap`，`bootstrap_sample` 形状 `(N, nfree+1)`。
  - `iminuit()`（stat=chi^2 与 chi^2f）恢复参数，返回 `Bootstrap`。
  - 对 `vdr`/`odr`/`chi^2f`/`groth` 调 `lmfit()` 抛 `ValueError`（提示 iminuit）。
  - `Bootstrap` 的点估计/区间可用。
- `from curvefit import MaxLikeFit` 可用；`ruff check .`/`ruff format --check .` 通过。
- emcee 端到端（现有 quickstart 路径）仍工作，返回 `Posterior`。

## 8. 范围边界

- 本期不动 data/model/util（除 infer 外不改）。Phase 2 再做全包 docstring。
- 不引入 numba；不移植 bayspec 的能谱专用统计量（pgstat/cstat 等）。
- `Infer.minimize`（scipy）保留不变（与 MaxLikeFit 并存）。
