---
primary_layer: layer1
related_layers: [layer2, layer3]
page_type: tutorial
---

# 制約付き最適化

設計変数に制約がある問題を、`saealib`で解きます。

`minimize()`では、アルゴリズム、代理モデル、評価戦略に文字列またはインスタンスを指定できます。 このページの例では問題固有の設定だけを追加し、それ以外にはデフォルトを使います。

:::{admonition} このページでできるようになること
:class: tip

このページを終えると、`Problem`に制約を定義し、実行可能性を確認しながら`ConstraintHandler`を選べます。
:::

単純な目的関数の最適化には[単目的最適化](single_objective.md)を使い、独自の制約処理を実装する場合は[独自コンポーネント](custom_components.md)を参照してください。このページでは、`Problem`への制約定義、実行可能性の確認、`ConstraintHandler`の選択を扱います。

## Problemを準備する

目的関数に加えて、解が満たすべき不等式制約`g(x) <= 0`を持つ問題を想定します。

```python
import numpy as np


def expensive_func(x):
    return np.sum(x**2)


def g1(x):
    # require the sum of the design variables to be at least 1
    return 1.0 - np.sum(x)


DIM = 5
LB = [-5.0] * DIM
UB = [5.0] * DIM
```

`g1(x) <= 0`を満たす解だけが実行可能解です。

## 制約を定義する

制約は`InequalityConstraint(func, threshold=0.0)`で定義し、`Problem`の`constraints`引数に渡します。

```python
from saealib import InequalityConstraint, Problem, minimize

constraint = InequalityConstraint(g1, threshold=0.0)

problem = Problem(
    func=expensive_func,
    dim=DIM,
    n_obj=1,
    direction=np.array([-1.0]),
    lb=LB,
    ub=UB,
    constraints=[constraint],
)

result = minimize(problem, max_fe=1000, seed=0)
print(result.x, result.f)
print(constraint.violation(result.x))  # 0.0 means the constraint is satisfied
```

`g(x) >= threshold`の形で制約を課したい場合は、`func`の符号を反転させて渡します。

等式制約`h(x) = 0`は、`InequalityConstraint`を符号反転で組み合わせる代わりに`EqualityConstraint(func, tolerance=1e-4)`を使います。`1e-4`は等式の既定の許容幅で、CEC 2006とpymooの既定に合わせています。

```python
from saealib import EqualityConstraint


def h1(x):
    # require the sum of the design variables to be exactly 1
    return np.sum(x) - 1.0


equality = EqualityConstraint(h1, tolerance=1e-4)
```

`EqualityConstraint`は、`|h(x)| <= tolerance`を満たす解を実行可能解として扱います。

## 実行可能性を確認する

各制約が許容幅の内側にある解が実行可能解です。不等式は`g(x) <= threshold`、等式は`|h(x)| <= tolerance`で判定します。

```python
g = result.ctx.archive.get_array("g")
feasible = np.array(
    [
        all(
            c.violation_from_value(float(value)) == 0.0
            for c, value in zip(problem.constraints, row)
        )
        for row in g
    ]
)
print(f"feasible: {feasible.sum()} / {len(feasible)}")
```

同じ判定が`result.x`/`result.f`、履歴の`best_f`と`feasible_ratio`、学習データのラベル、Paretoアーカイブを駆動します。そのため、`cv`の集約方法や探索中のε緩和には左右されません。

## ConstraintHandlerで制約処理をカスタマイズする

`ConstraintHandler`は、`Problem(cv_aggregator=...)`が設定されていないときに`cv`を供給し、comparatorが使う探索側の緩和を保持します。

`Problem`の`handler`引数を省略すると`StaticToleranceHandler`が使われ、`sum_violation`と同じく制約ごとの違反量の和を`cv`として集約します。

集約だけを変えたい場合は、ハンドラを書く代わりに`Problem(cv_aggregator=...)`を渡します。集約の契約は「すべての違反が0のとき、かつそのときに限り`cv == 0`」で、役割は実行不可能な解どうしの順位付けだけです。実行可能性は常に制約定義から決まります。

```python
from saealib.problem import sum_violation


def squared_inequality(constraints, x, g):
    return sum(
        max(0.0, float(gi) - c.threshold) ** 2 for gi, c in zip(g, constraints)
    )


problem = Problem(
    func=expensive_func,
    dim=DIM,
    n_obj=1,
    direction=np.array([-1.0]),
    constraints=constraints,
    cv_aggregator=squared_inequality,
)
```

| クラス | 動作 |
|--------|------|
| `StaticToleranceHandler` | 制約ごとの違反量を合計し、固定の探索側の緩和を持つ（既定） |
| `EpsilonConstraintHandler` | 探索側の緩和を世代とともに0へ近づけ、徐々に実行可能領域へ解を追い込む |
| `GradientRepairHandler` | 制約の勾配を使って実行不可能解を修復する |

`EpsilonConstraintHandler`には、世代番号を受け取り許容誤差を返す関数を渡します。
その緩和はcomparatorにだけ適用され、報告される実行可能性は制約の許容幅を使い続けます。

```python
from saealib import EpsilonConstraintHandler


def schedule(gen):
    return max(0.0, 1.0 - gen * 0.05)


problem = Problem(
    func=expensive_func,
    dim=DIM,
    n_obj=1,
    direction=np.array([-1.0]),
    lb=LB,
    ub=UB,
    constraints=[constraint],
    handler=EpsilonConstraintHandler(schedule),
)
result = minimize(problem, max_fe=1000, seed=0)
```

## 関連するConceptとReference

- {py:class}`saealib.InequalityConstraint` / {py:class}`saealib.EqualityConstraint`
- {py:class}`saealib.Problem`
- {py:class}`saealib.ConstraintHandler` / {py:class}`saealib.StaticToleranceHandler` / {py:class}`saealib.EpsilonConstraintHandler` / {py:class}`saealib.GradientRepairHandler`
- {py:func}`saealib.minimize`
