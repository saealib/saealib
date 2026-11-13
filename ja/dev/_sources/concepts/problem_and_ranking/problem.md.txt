---
primary_layer: layer1
related_layers: [layer2, layer3]
page_type: concept
---

# Problem

`Problem`は、目的関数、目的方向、制約、比較方法、SearchSpaceを一つにまとめます。`Optimizer`と各コンポーネントは、この`Problem`を使って同じ最適化対象を参照します。

ベクトル表現では、`lb`、`ub`、または`variables`から`VectorSpace`が作られます。非ベクトル表現では、`space`に`SearchSpace`を渡します。`SearchSpace`とGenomeの設計は[SearchSpace](search_space.md)で説明します。

## Problemの役割

`Problem` のコンストラクタは次の引数を受け取ります。

```python
Problem(
    func, dim, n_obj, direction, lb=None, ub=None,
    comparator=None, constraints=None, *,
    eps_cv=None, eps_obj=None, handler=None, cv_aggregator=None,
    variables=None, space=None, evaluation_adapter=None,
)
```

**`func`**：評価器が使う目的関数です。ベクトル問題では通常、設計変数の配列を受け取り、非ベクトル問題では評価アダプターが生成したペイロードを受け取ることがあります。**`dim`**：設計変数の次元数です。`space`が`dim`を提供する場合は省略できます。**`n_obj`**：目的関数の数です。

**`direction`**：目的ごとの最適化方向を示す`(n_obj,)`形状の配列です。各要素は`+1`（最大化）または`-1`（最小化）でなければなりません。それ以外の値を渡すと構築時に例外が発生します。

**`lb`**/**`ub`**：ベクトル設計空間の下限と上限です。`variables`も`space`も表現を提供しない場合に必要です。`variables`が指定されると、境界は各変数の範囲から導出されます。

**`comparator`**：非推奨です。 解を比較する[Comparator](comparators.md)は`Optimizer.set_comparator()`で設定してください。comparatorはProblemではなくOptimizerが持ちます。 ここに値を渡すと`DeprecationWarning`が出て、このProblemから作られる`Optimizer`の明示的なcomparatorとして扱われます。`set_comparator()`で設定したcomparatorが優先されます。 明示的なcomparatorがない場合、`Optimizer`は目的数から既定（`n_obj == 1`なら`SingleObjectiveComparator`、`n_obj > 1`なら`NSGA2Comparator`）を選び、その選択をframework defaultとして記録します。 `Problem`自身の`comparator`属性は、このProblemから導かれる既定として残ります。`OptimizationState.load()`を単体で使うときに使われる値であり、実行中に比較に使われるオブジェクトではありません。 `direction`が未設定（`None`）のcomparatorは、Optimizerの枠に入る時点でProblemの`direction`を受け取ります。

**`constraints`**：不等式制約（`InequalityConstraint`）のリストです。定義方法は[制約付き最適化](../../tutorials/constraints.md)で説明します。

**`eps_cv`**/**`eps_obj`**：いずれも非推奨の探索側の許容誤差です。
実行可能性は`eps_cv`ではなく制約定義で決まります。制約ごとの違反量が許容幅の内側ですべて0の解が実行可能であり、`Result`、履歴の`best_f`と`feasible_ratio`、学習データのラベル、Paretoアーカイブはすべてこの判定を使います。
`eps_cv`に値を渡すと`DeprecationWarning`が出て、既定のハンドラの探索側の緩和として設定されます。ランタイムはその値を毎世代comparatorの`eps_cv`へ書き写すため、探索が実行可能とみなす範囲が広がりますが、報告される実行可能性は変わりません。
省略時の既定は`0.0`で、制約による判定と一致します。実際に使われる緩和値は`handler.feasibility_threshold`です。
`eps_obj`も既定のcomparatorの許容誤差を決めるだけで、ここに値を渡すと`DeprecationWarning`が出ます。許容誤差は`Optimizer.set_comparator()`で設定するcomparatorに渡してください。

**`cv_aggregator`**：`cv_aggregator(constraints, x, g) -> float`というcallableで、`cv`を計算して`handler.compute_cv`を置き換えます。通常の関数も使えます。
既定の集約は制約ごとの違反量の和である`sum_violation`で、すべての違反が0のとき、かつそのときに限り`cv`が0になります。
集約の契約は[ConstraintHandler](constraints.md)を参照してください。

**`handler`**：制約違反の集約や目的関数の補正を担う[ConstraintHandler](constraints.md)。 省略時は`StaticToleranceHandler`が使われます。

**`variables`**：各設計変数の型を`Variable`のリストで指定します。連続変数だけの問題では省略でき、その場合は各次元が`ContinuousVariable`として扱われます。整数変数とカテゴリ変数を混在させる場合、ここに`IntegerVariable`/`CategoricalVariable`を含めると、[Crossover](../search_algorithms/crossover.md)/[Mutation](../search_algorithms/mutation.md)が変数型ごとに異なる演算子を自動的に割り当てます。

**`space`**：Genomeの表現と、サンプリング、検証、比較、距離計算などのサービスを提供する `SearchSpace` です。 `space` を渡し、`variables`、`lb`、`ub` を省略した場合は、`dim` をSearchSpaceから取得します。 非ベクトルのSearchSpaceでは、`lb` と `ub` が存在しない場合があります。

**`evaluation_adapter`**：`GenomeBatch`を目的関数が受け取れる入力へ変換する`EvaluationAdapter`です。表現と目的関数の入力形式が一致しない場合は、`Problem`に明示的に設定します。

```{note}
旧バージョンのチュートリアルには`weight=`という引数を使う例がありますが、現行の`Problem`にこの引数は存在しません。 `weight=`を渡すと`TypeError`になります。
```

## Problemが保持するもの

`direction`は、符号だけを表すsaealib全体で統一された`±1`配列です。一方、`WeightedSumComparator`や`DecompositionComparator`が受け取る`weights`は、複数の目的をスカラー値へ集約するための非負の重みであり、方向とは独立した別の概念です。

この役割分担のもとでは、目的の重要度そのもの（スケーリング）を`weights`で表現することはできません。 目的関数値の大きさを調整したい場合は`func`の内部でスケーリングします。 `direction`は符号だけを、`weights`は集約の重み配分だけを表す、という2軸に整理されています。

## 不変条件

`direction` は目的ごとに `+1` または `-1` でなければならず、`dim`、変数、探索空間の表現は互いに整合していなければなりません。これらが一致しないと、構築時の検証や評価時の候補処理が失敗します。

## Problemが持つもの

`constraints`は`violation_from_value`を通じて実行可能性を決めます。不等式は`g(x) <= threshold`、等式は`|h(x)| <= tolerance`で、既定はそれぞれ`0`と`1e-4`です。
この判定は`cv`の集約方法から独立しており、`Result`、履歴の`best_f`と`feasible_ratio`、学習データのラベル、Paretoアーカイブがすべて適用します。

## Problemの拡張

### 独自Variableの実装方法

`Variable`(ABC)は、`lb`/`ub`という2つのプロパティと`repair(x)`というメソッドだけを要求します。 組み込みの`ContinuousVariable`/`IntegerVariable`/`CategoricalVariable`は、いずれも自分の定義域へ値を射影するだけの薄い実装であり、これ以外の変数型（周期変数、対数スケール変数など）が必要な場合は`Variable`を直接継承すればよいです。

次の例は、値を切り詰めるのではなく境界で折り返す変数です。 角度のように、上限を超えた値が下限側から連続しているとみなしたい設計変数に使えます。

```python
import numpy as np
from saealib import Variable


class PeriodicVariable(Variable):
    def __init__(self, lb: float, ub: float):
        self._lb = float(lb)
        self._ub = float(ub)

    @property
    def lb(self) -> float:
        return self._lb

    @property
    def ub(self) -> float:
        return self._ub

    def repair(self, x):
        span = self._ub - self._lb
        return self._lb + np.mod(np.asarray(x, dtype=float) - self._lb, span)
```

`ContinuousVariable.repair()`が`np.clip`で範囲外の値を境界に留めるのに対し、この実装は`np.mod`で範囲外の値を反対側の境界から巻き戻します。 `Variable`が表す値は、`Population`配列上で扱われる「エンコード済みfloat空間」上の値である点に注意してください。 `CategoricalVariable`のようにカテゴリ値と内部インデックスの対応が必要な変数型では、`repair()`はインデックス空間上で完結させ、実際のカテゴリ値への変換は別のメソッドで行います。

## 外部ライブラリアダプタ

`PymooProblem(pymoo_problem, *, eq_tolerance=1e-4, **problem_kwargs)`は、構築済みの[pymoo](https://pymoo.org/)問題インスタンス（組み込みベンチマークや既存の研究コード）を`Problem`としてラップし、`problem_kwargs`（`comparator`、`handler`、`eps_cv`、`eps_obj`）は`Problem.__init__`へそのまま渡されます。`eq_tolerance`の既定は等式の許容幅`1e-4`であり、転送した`comparator`・`eps_cv`・`eps_obj`には上記の非推奨がそのまま当てはまります。 pymooの問題は常に最小化なので`direction`は常に全て`-1`になり、pymooの不等式制約（`G`）は`InequalityConstraint`へ、等式制約（`H`）は`EqualityConstraint`へ、いずれも符号を変えずにそのままマッピングされます。

pymooの問題はバッチ評価（`_evaluate(X, out)`）を前提としており、`PymooProblem.evaluate_batch`はラップした問題自身の`evaluate(X, return_as_dictionary=True)`をバッチ全体に対して1回だけ呼び出すことで、これを直接活用します。 `SerialEvaluator`は利用可能な場合これを自動的に使用します。 一方、`SerialEvaluator`を介さず`Problem.evaluate`/`evaluate_constraints`を直接呼んで個体単位で評価する呼び出し元に対しては、`x`をキーとする1スロットのキャッシュにより、候補ごとにpymoo呼び出しは正確に1回で済みます。`evaluate_constraints()`と`evaluate()`は同じ`x`に対して連続して呼ばれるため、キャッシュがなければ制約の数だけpymoo呼び出しが発生し、さらに目的関数の分も加わってしまいます。

`pymoo` extraのインストールについては[インストール](../../getting_started/installation.md)を参照してください。

## 関連コンポーネント

- [Comparator](comparators.md)：`Optimizer.set_comparator()`で解の比較方法を差し替える
- [ConstraintHandler](constraints.md)：`handler`引数で制約違反の扱い方を差し替える
- [Crossover](../search_algorithms/crossover.md) / [Mutation](../search_algorithms/mutation.md)：`variables`で定義した変数型ごとに適用される演算子
- [制約付き最適化](../../tutorials/constraints.md)：制約の定義方法と組み込みの`ConstraintHandler`の選び方

## 参照

- {py:class}`saealib.Problem`
- {py:class}`saealib.PymooProblem`
- {py:class}`saealib.Variable`
- {py:class}`saealib.ContinuousVariable`
- {py:class}`saealib.IntegerVariable`
- {py:class}`saealib.CategoricalVariable`
