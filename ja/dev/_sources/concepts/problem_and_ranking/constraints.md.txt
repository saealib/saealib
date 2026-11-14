---
primary_layer: layer3
page_type: concept
---

# ConstraintHandler

制約の基本的な定義方法（`InequalityConstraint`/`EqualityConstraint`）、`Problem`の`handler`引数の指定方法、組み込み3種のハンドラの選び方は[制約付き最適化](../../tutorials/constraints.md)で扱っています。
このページでは、それらを踏まえた上で独自の`ConstraintHandler`を書きたい場合に絞って説明します。

## ConstraintHandlerの役割

`ConstraintHandler`は、制約の処理方式を差し替え可能なライフサイクルフックの集合として公開しています。

```
Ask            -> [repair(x, constraints, lb, ub)]
               -> evaluate f, g
               -> [compute_cv(constraints, x, g)]        -> cv
               -> [augment_objective(f, constraints, x, g)] -> f'
Tell           -> Comparator(f', cv) with eps_cv = feasibility_threshold
                  （探索中の緩和であり、報告される実行可能性は
                  制約ごとの違反量で決まる）
Generation end -> [on_generation_end(gen, population)]
```

抽象メソッドは`compute_cv`だけであり、残りのフックはいずれも既定実装を持ちます。
独自の`ConstraintHandler`は、必要なフックだけをオーバーライドすればよいです。

**`repair(x, constraints, lb, ub, **kwargs)`**：交叉と突然変異の後、評価の前に呼ばれます。
既定は`np.clip(x, lb, ub)`で、境界にクリップするだけです。

**`compute_cv(constraints, x, g)`**（抽象）：制約群を単一の`cv`値へ集約します。

**`augment_objective(f, constraints, x, g)`**：目的関数値を制約情報で変換します。
既定は恒等関数（変換しない）で、ペナルティ関数法やaugmented Lagrangian法を実装する際にオーバーライドする場所です。
組み込み3種のハンドラはいずれもこのフックをオーバーライドしていないため、現状は将来の拡張点として空いています。

**`feasibility_threshold`**（プロパティ）：既定`0.0`。
この値は探索中の緩和であり、`Optimizer`実行中は毎世代`Comparator`の`eps_cv`へ書き写されます。報告される実行可能性を決めることはありません。

**`on_generation_end(gen, population)`**：世代の終わりに呼ばれます。
既定はno-opで、`EpsilonConstraintHandler`のようにε値を世代ごとに更新する内部状態を持つハンドラが使います。

## 制約違反と実行可能性

実行可能性は集約後の`cv`ではなく制約定義から決まります。制約ごとに`violation_from_value`が許容幅の内側で0を返すとき、つまり不等式なら`g(x) <= threshold`、等式なら`|h(x)| <= tolerance`のとき、その解は実行可能です。
`Result`、履歴の`best_f`と`feasible_ratio`、`FeasibilityClassificationSet`、Paretoアーカイブはすべてこの判定を使うため、探索中にcomparatorを緩和しても報告される実行可能性は変わりません。
`GenerationBasedStrategy`で`gen_ctrl > 0`の場合、アルゴリズムの`tell()`の実装によっては、最終個体群にプレースホルダー`g=0`を持つ予測行が含まれることがあります。たとえばGAでは予測行を保持できます。
Historyの`summary`にある`feasible_ratio`と`min_cv`は、`state.population`のうち真に評価された行だけから計算するため、これらの行は数えられません。真に評価された行が1つもないときは、どちらもNaNになります。
真の評価値はアーカイブに保持されます。
自分で行を分類する場合は、各行の`g`列に`violation_from_value`を適用します。すべての結果が0の行が実行可能です。

集約後の`cv`の役割は1つで、実行不可能な解どうしの順位付けです。
契約は「すべての制約違反が0のとき、かつそのときに限り`cv == 0`」と単調性です。
既定の集約は制約ごとの違反量の和である`sum_violation`で、`StaticToleranceHandler`が計算する値と同じです。
`Problem(cv_aggregator=...)`に`cv_aggregator(constraints, x, g) -> float`というcallable（通常の関数も可）を渡すと置き換えられます。`cv_aggregator`を渡さない場合、ハンドラの`compute_cv`が`cv`を供給します。

`EpsilonConstraintHandler`は集約契約に対する探索側の例外です。その式（等式は`|h(x)|`、不等式は`max(0, g(x) - threshold) ** 2`）は、等式の残差が厳密に0のときだけ0になるため、制約の許容幅より厳しくなります。
この式はε制約法のために維持され、その緩和は`feasibility_threshold`を通じてcomparatorに届きます。報告される実行可能性は常に上記の制約定義から決まります。

## 組み込みハンドラがオーバーライドするフック

| クラス | オーバーライドするフック |
|---|---|
| `StaticToleranceHandler` | `compute_cv` / `feasibility_threshold` |
| `EpsilonConstraintHandler` | `compute_cv` / `feasibility_threshold` / `on_generation_end` |
| `GradientRepairHandler` | `repair` / `compute_cv` |

`EpsilonConstraintHandler`{cite}`mezuramontes2011epsilon`は、`schedule(gen) -> float`という関数でεを世代ごとに更新します。
`linear_epsilon_schedule(eps0, n_gen)`/`exponential_epsilon_schedule(eps0, decay)`という組み込みのスケジュール生成関数も公開されています。

`GradientRepairHandler`{cite}`chootinan2006gradientrepair`は、現在違反している等式制約と不等式制約に対して、Moore-Penrose擬似逆行列による同時更新を行う`repair()`を実装します。
`gradient()`が`None`を返す制約は、スキップせず前進差分による数値近似へフォールバックします。

## InequalityConstraint/EqualityConstraintの拡張点

`InequalityConstraint`自体にも2つの拡張点があります。

**`gradient(x)`**：既定は`None`を返します。
オーバーライドして解析的な勾配ベクトルを返さない場合、`GradientRepairHandler`は数値近似へフォールバックします。

**`violation_from_value(g)`**：生の制約値`g(x)`から違反量への変換を定義します。
既定は`max(0, g - threshold)`です。
`EqualityConstraint`はこのメソッドだけをオーバーライドし、`max(0, |h(x)| - tolerance)`という独自の変換を定義しています。

## 独自ConstraintHandlerの実装方法

独自の制約処理戦略が必要な場合は、`ConstraintHandler`を継承して`compute_cv()`だけを実装すればよいです。
次の例は、制約違反量を目的関数値へ加算するペナルティ関数法です。

```python
from saealib import ConstraintHandler


class PenaltyHandler(ConstraintHandler):
    """Adds the violation amount to the objective value as a penalty."""

    def __init__(self, penalty_coeff: float = 1e3):
        self.penalty_coeff = penalty_coeff

    def compute_cv(self, constraints, x, g):
        return sum(max(0.0, gi) for gi in g)

    def augment_objective(self, f, constraints, x, g):
        cv = self.compute_cv(constraints, x, g)
        return f + self.penalty_coeff * cv
```

`augment_objective`をオーバーライドしてペナルティを目的関数値に加算する一方で、`compute_cv`は制約違反として`0`を返す実装にすれば、ペナルティ関数法だけで実行可能性を扱わない（全解を実行可能とみなす）構成にもできます。

## 関連コンポーネント

- [制約付き最適化](../../tutorials/constraints.md)：制約の定義と組み込みハンドラの基本的な使い方
- [Problem](problem.md)：`handler`引数と`cv_aggregator`引数の渡し方
- [Comparator](comparators.md)：`feasibility_threshold`が`eps_cv`として同期される先

## 参照

- {py:class}`saealib.ConstraintHandler`
- {py:class}`saealib.StaticToleranceHandler`
- {py:class}`saealib.EpsilonConstraintHandler`
- {py:class}`saealib.GradientRepairHandler`
- {py:class}`saealib.InequalityConstraint`
- {py:class}`saealib.EqualityConstraint`
- {py:func}`saealib.problem.constraint.sum_violation`
- {py:func}`saealib.linear_epsilon_schedule`
- {py:func}`saealib.exponential_epsilon_schedule`
