---
primary_layer: layer1
related_layers: [layer2]
page_type: tutorial
---

# 多目的最適化

`saealib` を使って、目的間のトレードオフがある多目的最適化問題を解きます。

`minimize()`は目的関数の数に応じたデフォルトのコンポーネントを使います。
アルゴリズム、代理モデル、評価戦略を個別に設定する場合は `Optimizer.set_*()` を使います。

このページでは、2つ以上の目的を持つ問題に固有の内容として、Comparatorの選択とParetoフロントの取得を扱います。

:::{admonition} このページでできるようになること
:class: tip

このページを終えると、多目的問題でComparatorを選び、得られたParetoフロントを取り出せます。
:::

単目的問題の基本手順は [単目的最適化](single_objective.md) を使い、独自の Comparator が必要な場合は [独自コンポーネント](custom_components.md) を参照してください。
このページでは、多目的 `Problem` を実行し、Comparatorを選択してParetoフロントを取得します。

## 問題を設定する

複数の目的関数がトレードオフの関係にあると、一方を改善すると他方が悪化する解が存在します。

この関係において、すべての目的で他の解に支配されない解の集合を **Paretoフロント** と呼びます。

ここでは例として、`saealib` に組み込まれたZDT1関数を最小化します。

```python
from saealib.benchmarks import zdt1

problem = zdt1(n_var=10)
```

`zdt1` は、凸型のParetoフロントを持つ2目的のベンチマーク問題を返す `Problem` インスタンスです。

## minimizeを実行する

`Problem` インスタンスを直接渡すと、そこから目的数が引き継がれます。

```python
from saealib import minimize

result = minimize(problem, max_fe=2000, seed=0)

print(result.x.shape)  # (n_pareto, dim)
print(result.f.shape)  # (n_pareto, n_obj)
```

単目的の場合に1点だった `result.x`/`result.f` は、多目的の場合にはParetoフロントを形成する複数の解になります。
`x`と`f`は既定の基準で選んだ非劣解で、main archiveの真評価された行から計算し直しています。
`result.best()`は1つの解を、`result.front()`は非劣解の集合を返します。
どちらも`criterion`に`Comparator`を受け取り、省略時は既定の基準を使います。既定の基準は実行可能解を優先し、実行可能性は制約ごとの違反量（不等式は`0.0`、等式はその`tolerance`）で判定したうえでPareto支配を使います。実行可能な解が1つもない場合は、`cv`が最小の行のうち非劣な解を使います。

## Optimizerを構成する

`Optimizer(problem)`では、各コンポーネントを独立して設定できます。
`set_*()`は連鎖呼び出しのために同じ `Optimizer` を返し、`run()`または `iterate()`で最適化を実行します。

```python
from saealib import Optimizer, Termination, max_fe

optimizer = Optimizer(problem, seed=0).set_termination(Termination(max_fe(2000)))
ctx = optimizer.run()
pareto_f = ctx.pareto_archive.get_array("f")
```

未設定のコンポーネントはデフォルトに解決されます。
多目的の順位付けは、次のComparatorの選択で設定します。

## Comparatorを選択する

多目的最適化では、`Comparator` が候補解同士の相対的な優劣を決めます。

comparatorを設定しない場合、目的数に基づいて自動選択されます（`n_obj == 1` では `SingleObjectiveComparator`、`n_obj > 1` では `NSGA2Comparator`）。

| クラス | 動作 |
|--------|------|
| `NSGA2Comparator` | 非劣ソートと混雑度距離による多様性維持（既定） |
| `SPEA2Comparator` | 支配の強さと近傍密度に基づく適応度 |
| `HypervolumeComparator` | ハイパーボリューム寄与で優劣を判定 |
| `EpsilonDominanceComparator` | ε支配で優劣を判定 |
| `NSGA3Comparator` | 参照点による多様性維持。`reference_points` が必要です |
| `RNSGA2Comparator` | 指定した参照点の近くに解を集中させます。`reference_points` が必要です |

comparatorは `Optimizer.set_comparator()` で設定します。
明示的に設定したcomparatorは既定より優先され、実行状態には `ctx.comparator` として現れます。

```python
from saealib import Optimizer, Termination, max_fe
from saealib.comparators import HypervolumeComparator

optimizer = (
    Optimizer(problem, seed=0)
    .set_comparator(HypervolumeComparator())
    .set_termination(Termination(max_fe(2000)))
)
ctx = optimizer.run()
```

`Problem(comparator=...)` も動作しますが非推奨です。`DeprecationWarning` が出て、その値はOptimizerの明示的なcomparatorとして扱われます。

## パレートフロントを抽出する

実行後、`result.x`と`result.f`は既定の基準で選んだ非劣解を保持します。

```python
front_x = result.x
front_f = result.f
```

探索側の`result.ctx.pareto_archive`は別の、逐次に刈り込まれるアーカイブです。内容は探索の設定に従うため、最終結果の比較や描画には`result.x`/`result.f`または`result.front()`を使い、Paretoアーカイブを直接読まないでください。多目的問題では、main archiveからの計算し直しがParetoアーカイブと異なる集合になることがあり、例えばε緩和を使う実行で起こります。

```python
from saealib.comparators import ParetoComparator

front_x, front_f = result.front()
relaxed_x, relaxed_f = result.front(
    ParetoComparator(direction=problem.direction, eps_cv=1e-3)
)
```

任意の目的値配列からParetoフロントを計算するには、`non_dominated_sort` を直接使えます。

```python
from saealib.comparators import non_dominated_sort

archive_f = result.ctx.archive.get_array("f")
ranks, fronts = non_dominated_sort(archive_f, direction=problem.direction)
front0_f = archive_f[fronts[0]]  # first non-dominated front
```

## 関連コンセプトと参考情報

- {py:func}`saealib.minimize`
- {py:class}`saealib.Problem`
- {py:class}`saealib.NSGA2Comparator` / {py:class}`saealib.SPEA2Comparator` / {py:class}`saealib.HypervolumeComparator` / {py:class}`saealib.EpsilonDominanceComparator` / {py:class}`saealib.NSGA3Comparator` / {py:class}`saealib.RNSGA2Comparator`
- {py:func}`saealib.non_dominated_sort`
