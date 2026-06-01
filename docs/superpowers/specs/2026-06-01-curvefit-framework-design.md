# curvefit 框架设计 (Design Spec)

- 日期: 2026-06-01
- 状态: 待实现
- 原则: **curvefit 为当前可修改项目，永不修改 bayspec**

## 1. 目标与背景

bayspec 已实现高能天体物理能谱的贝叶斯拟合，但其数据层针对能谱（src/bkg/响应/计数）定制。
curvefit 复用 bayspec 的拟合方法与代码风格，把数据层泛化为最广泛情形：`x, y, xerr, yerr`。

**代码风格要求**：与 bayspec 严格一致（`OrderedDict` 参数容器、property/setter 模式、`Info` 表格、`SuperDict` 1-based 索引、四层 `util/data/model/infer` 再导出）。没必要重写的地方尽量复用 bayspec 已有实现。

curvefit 目录已是 bayspec 的半成品移植：`model/`、`util/` 大部分已泛化可用；`data/`、`infer/`、`plot.py` 仍残留能谱假设或处于损坏状态。本设计描述把它补全/改写为通用曲线拟合框架。

## 2. 总体架构与文件复用 map

四大块结构与 bayspec 一致：`util / data / model / infer`。

| 文件 | 处理方式 |
|------|---------|
| `util/param.py, prior.py, post.py, info.py, tools.py, corner.py` | 原样复用（纯通用基础设施） |
| `util/significance.py` | 删除（能谱专用 `ppsig/pgsig`） |
| `util/plot.py` | 重写为通用曲线绘图（基础版） |
| `model/model.py` | 复用，仅修 `CompositeModel.func` 签名 |
| `model/local.py` | 原样复用（已是通用模型 ln/pl/expd/bln/bpl/sbpl/spindown/psd） |
| `data/data.py` | 重写为 x/y/xerr/yerr 数据层 |
| `data/__init__.py` | 重写（去掉 spectrum/response 导入） |
| `data/spectrum.py, data/response.py` | 不创建（curve 无响应/本底文件概念） |
| `infer/statistic.py` | 重写：集成用户提供的 `Stat` 类（6 个统计量公式逐字保留） |
| `infer/pair.py` | 重写 stat 计算路径（去卷积，走 `Stat`） |
| `infer/infer.py` | 改写 `calc_loglike` 与数据属性（x/y/model_y 取代 ctsrate/ctsspec） |
| `infer/posterior.py` | 基本复用（已通用） |

## 3. data 层

`DataUnit` = 一条数据集（一条曲线）；`Data` = 多条数据集的有序集合（联合拟合）。一一对应 bayspec 的 `DataUnit`/`Data`，沿用 `OrderedDict` 存储、`fit_with` 反向绑定、`info`/`__str__` 表格、`get_obj_name()` 取变量名等模式。

### 3.1 DataUnit 构造

```python
DataUnit(x, y, xerr=None, yerr=None, weight=1, up=None, stat='chi^2', name=None)
```

- `x`, `y`: 1D 数组（必填）。
- `xerr` / `yerr`: 支持标量、对称 1D、或非对称 `[low, high]`；内部统一规范成形如 `(low, high)` 两行结构，分别喂给 `Stat` 的 `x_err[0]/x_err[1]`、`y_err[0]/y_err[1]`。
- `weight`: 标量或逐点数组 → `Stat` 的 `w`。
- `up`: 逐点上限布尔标志（数组/列表）→ `Stat` 的 `up`；缺省全 `False`。
- `stat`: 该 unit 采用的统计量名（`Stat` 字典键之一），缺省 `'chi^2'`。
- 无 `yerr` 时默认 `yerr=1`（退化为最小二乘 / OLS）。
- `npoint`: x 的长度，供自由度统计。

约定数据点形态而非 bin 形态（curve 是点，不是能道）。

### 3.2 输入入口（5 种）

保持 bayspec 的构造/classmethod 风格：

1. 直接数组：`DataUnit(x, y, xerr, yerr)`。
2. `dict`：`DataUnit.from_dict({'x':..., 'y':..., 'xerr':..., 'yerr':..., 'up':..., 'weight':...})`。
3. `pandas.DataFrame`：`DataUnit.from_dataframe(df, x='x', y='y', ...)`（列名可指定，含合理默认）。
4. json 文件：`DataUnit.from_json(path)`（结构同 dict）。
5. csv 文件：`DataUnit.from_csv(path)`（经 DataFrame）。

非对称误差在文件/字典里的表示：`yerr` 为单列→对称；`yerr_low`/`yerr_high`（或二维）→非对称。具体列名约定在实现时定稿并写入 docstring。

### 3.3 Data 容器

- 接受 `None`、`list[(name, DataUnit)]`、或 `dict[name -> DataUnit]`（与 bayspec `Data` 同）。
- 聚合属性（供 infer/plot 消费）：各 unit 的 `x / y / xerr / yerr / weight / up / npoint / stat`，以 list 或拼接形式暴露。
- `info` property 输出表格（Name / Npoint / Statistic / Upperlimit 等列）。
- `fit_with` setter 与 `Model.fit_to` 双向绑定（沿用 bayspec 逻辑）。

## 4. model 层

- `model/local.py` 原样保留。模型已用 `func(self, X)`、`X[:, 0]` 取 x，且内置 `logv` 之类附加方差参数（对接 `chi^2f/vdr/odr`）。
- `model/model.py` 仅修一处：`CompositeModel.func(self, E, T=None, O=None)` → `func(self, X)`，内部改为 `m1.func(X)`、`m2.func(X)`，与基类 `Model.func(self, X)` 一致。其余（`mdicts/pdicts/fdicts/cfg/par`、`at_par`、`+ - * /` 组合、`par_best` 等）保持不变。

## 5. statistic 层（核心）

`infer/statistic.py` 重写为用户提供的 `Stat` 类。**6 个统计量的数学公式逐字保留，不改动**：

- `chi^2` — `chi_square`
- `chi^2f` — `chi_square_full`（含 `logv = params[-1]` 附加方差）
- `logchi^2` — `log_chi_square`
- `vdr` — 有效方差法（线性模型，`params=[k,b,logv]`）
- `odr` — 正交距离回归（线性模型，`params=[k,b,logv]`）
- `groth` — Groth 精确（Poisson 型）似然

每个统计量签名固定为：
```python
func(mo_func, params, x, y, x_err, y_err, w, up) -> loglike
```
其中 `y_err`/`x_err` 为 `[low, high]`；按 `(y < model)` / `(y >= model)` 选择上/下误差；`up` 为上限标志，命中规则：`S[(y>=my)&up]=0`、`S[(y<my)&up]=inf`。函数直接返回 `-0.5·ΣS`（即 loglike，`groth` 返回 `lnL`）。

**外壳风格**（用户确认）：公式保留，外壳改成 bayspec 风格——用字典派发（如 `Stat._allowed_stats` 或 `make_dict` 等价的静态字典），去掉构造函数里的 `print`/交互式 `show()`。保留按 `expr` 取 `func` 的能力。

## 6. model ↔ statistic 桥（关键衔接）（方案 A，已确认）

`Stat` 需要 `mo_func(x, params)`，而 `Model` 是 `func(X)` 从 `self.params` 读值。在 `Pair` 内构造适配闭包，复用 `Model` 不改其设计：

```python
def mo_func(x, params):
    self.model.at_par(params)
    return self.model.func(np.asarray(x)[:, None])   # X[:, 0] = x
```

- `params` 取 `model` 的完整有序参数向量（`model.pvalues`），`logv` 为最后一个参数，正好匹配 `vdr/odr` 的 `params[2]`、`chi^2f` 的 `params[-1]`。
- `vdr/odr` 对线性模型 `ln(k, b, logv)` 的参数序约定天然成立。
- 模型 `func` 忽略它不用的尾部参数（如 `ln.func` 只读 `k,b`），因此 `chi^2f` 传 `params[:-1]`、`vdr` 传全量都得到一致的 `my`。

## 7. infer 层

### 7.1 Pair

绑定一个 `Data` + 一个 `Model`，逐 unit 调 `Stat`：

```python
loglike = Σ_unit  stat.func(mo_func, model.pvalues,
                            x_u, y_u, xerr_u, yerr_u, w_u, up_u)
```

- `stat` 按 unit 的 `stat` 名从字典取函数。
- 保留 `stat`、`loglike`、`stat_list`、`npoint` 等 property 接口形态（去掉能谱卷积 `_convolve`、`phtspec_at_rsp`、`cts_to_flux` 等）。

### 7.2 Infer / BayesInfer

保持 bayspec 接口：

- `Infer([(data, model), ...])`，`append`，`nfree`、`free_value`、`free_prior`。
- `calc_loglike(theta)`：写入自由参数后，按 Pair 汇总各 unit loglike。
- `calc_logprior(theta)`、`prior_transform(cube)`。
- `loglike_func / logprior_func / prior_transform_func` 用户覆盖钩子。
- `BayesInfer.emcee(...)`、`BayesInfer.multinest(...)` → 返回 `Posterior`。
- `infer.py` 内能谱属性（`data_ctsrate/ctsspec/model_ctsrate`…）替换为 `data_x / data_y / data_yerr / model_y`，供绘图与残差。

### 7.3 Posterior

基本复用（已通用）：后验样本、点估计、置信区间、`save`、表格。

## 8. plot 层（基础版）

`util/plot.py` 重写，提供：

- 数据点 + 非对称误差棒；上限点特殊标记（箭头/下三角）。
- 最佳拟合曲线 + 后验置信带（1/2/3σ）。
- 残差面板。
- `corner` 后验图（复用 `util/corner.py`）。

保持 bayspec `Plot.infer / Plot.model / Plot.post_corner` 调用风格与 plotly/matplotlib 后端选择。完整能谱多样式（CE/vFv 等）不在本轮范围。

## 9. 端到端示例

`examples/quickstart.py`：

```python
import numpy as np
from curvefit import BayesInfer, Data, DataUnit, Plot
from curvefit.model.local import ln

x = np.array([...]); y = np.array([...])
xerr = np.array([...]); yerr = np.array([...])

unit = DataUnit(x, y, xerr, yerr, stat='chi^2f')
data = Data([('d1', unit)])
print(data)

model = ln()
print(model)

infer = BayesInfer([(data, model)])
post = infer.emcee(nstep=..., savepath='./quickstart')
print(post)

fig = Plot.infer(post)
fig = Plot.post_corner(post)
```

## 10. 范围边界（YAGNI）

- 本轮：core（data/model/infer/util）端到端跑通 + 基础绘图。
- 不做：完整能谱样式迁移、响应/本底文件读取、能谱专用统计量（pgstat/cstat 等）。
- 删除 `util/significance.py`，不创建 `data/spectrum.py`、`data/response.py`。

## 11. 验证标准

- `import curvefit` 不报错（修复级联导入链）。
- quickstart 端到端跑通：构造数据 → 定义模型 → `BayesInfer` → `emcee` → `Posterior` → 绘图。
- 6 个统计量均可被 `DataUnit(stat=...)` 选中并产出有限 loglike。
- 非对称误差与上限点行为符合 `Stat` 公式。
