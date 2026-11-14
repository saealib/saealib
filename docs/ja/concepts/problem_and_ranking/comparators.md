---
primary_layer: layer2
related_layers: [layer3]
page_type: concept
---

# Comparator

`Optimizer`は、解の相対的な優劣を判断する処理を、差し替え可能なトップレベルコンポーネントである`Comparator`に委ねています。
`Optimizer.set_comparator()`で渡します。

```python
import numpy as np
from saealib import Optimizer, Problem
from saealib.comparators import HypervolumeComparator

problem = Problem(
    lambda x: np.array([x[0], 1.0 - x[0]]),
    dim=1,
    n_obj=2,
    direction=np.array([-1.0, -1.0]),
    lb=[0.0],
    ub=[1.0],
)
optimizer = Optimizer(problem, seed=0).set_comparator(HypervolumeComparator())
ctx = optimizer.run()
print(ctx.comparator is optimizer.comparator)  # True
```

comparatorはProblemの一部ではなく、Optimizerの結びつけ枠です。
comparatorを設定しない場合、`Optimizer`は目的数から既定（`n_obj == 1`なら`SingleObjectiveComparator`、`n_obj > 1`なら`NSGA2Comparator`）を選び、その出どころをframework defaultとして記録します。明示的に設定したcomparatorは常に優先されます。
comparator自身の`direction`が未設定であれば、枠に入る時点でProblemの`direction`を受け取ります。
direction、目的ごとの重み、参照点がProblemの目的数と一致しないcomparatorは、`Optimizer.validate()` や `contract_diagnostics()` によって診断コード `direction_length_mismatch`、`objective_count_mismatch`、`reference_points_dimension_mismatch` として報告されます。
実行中、各コンポーネントは有効なcomparatorを`state.comparator`から読みます。そのため、制約ハンドラが毎世代書き込むεの閾値は、探索が解の順序付けに使うのと同じオブジェクトに届きます。

`Problem(comparator=...)`と`Problem(eps_obj=...)`は非推奨です。
`Problem`に渡したcomparatorは`DeprecationWarning`を出したうえでOptimizerの明示的なcomparatorとして扱われ、`eps_obj`は既定のcomparatorの許容誤差を決めるだけです。
`Problem`は目的数から導いた既定のcomparatorを引き続き公開しており、Optimizerを伴わずに状態を作成・読み込みした`OptimizationState`はそれを使います。実行を再開するときは`Optimizer.run_from()`または`Optimizer.iterate_from()`を使うと、そのOptimizerのcomparatorが使われます。

アイランドでは、既定のcomparatorは`Optimizer`ごとに作られるため、2つの島が1つを共有することはありません。`set_comparator()`に同じインスタンスを渡した場合は意図どおり共有されますが、ランタイムはそのインスタンスへハンドラの閾値を書き込むため、ハンドラの異なる島で1つを共有することは勧めません。

## Comparatorの役割

`Comparator`には`__init__`自身を含め、4つの抽象メソッドがあります。

- **`__init__(weights, eps_cv, eps_obj, direction=None)`**: `weights`/`eps_cv`/`eps_obj`/`direction`を保持します
- **`sort_population(population) -> np.ndarray`**: 個体群全体を最良から最悪へ並べるインデックス配列を返します
- **`compare_population(population, idx_a, idx_b) -> int`**: 個体群内の2個体を比較します（`-1`: aがより良い、`1`: bがより良い、`0`: 同等）
- **`compare(fa, cv_a, fb, cv_b) -> int`**: `Population`を経由せず、目的値と制約違反から2点を直接比較する軽量版です

`__init__`が抽象メソッドである点は、他のコンポーネントと異なります。
独自の`Comparator`を実装するサブクラスは、必ず`super().__init__(weights, eps_cv, eps_obj, direction=...)`を呼び出さなければなりません。

また、`f`/`cv`から毎回新しく導出するのではなく、選択をまたいで永続化すべきpopulation相対的なランキング状態のために、3つの具象メソッドを提供します。
`get_required_attrs`と`prepare_population`は既定で空/no-opですが、`rank_population`はこの2つを組み合わせるものであり、それ自体はno-opではありません。

- **`get_required_attrs(problem) -> list[PopulationAttribute]`**: Comparatorが`Population`/`Archive`に保持させる必要がある`PopulationAttribute`を宣言します。既定では空です
- **`prepare_population(population) -> None`**: 指定された個体群についてそれらの属性を新たに再計算し、書き込みます。既定ではno-opです
- **`rank_population(population) -> np.ndarray`**: `prepare_population`に続けて`sort_population`を実行します。新しくマージしたプールでの環境選択では、ランク付け前に永続化状態を新しい集合に対して再計算するため、`sort_population`を直接呼ばずにこれを呼び出します

現在ランク付け対象になっている個体群だけから完全なランキングを導出でき、population extractionをまたいで状態を持ち越す必要がない`Comparator`は、この3つのいずれも必要としません。
これは`SPEA2Comparator`を除く組み込みComparatorすべてに当てはまります。個体群全体に依存する`NSGA2Comparator`の混雑度距離でさえ、与えられた`Population`が現在保持する`f`/`cv`の値から毎回新しく再計算されます。
`SPEA2Comparator`は、ランキング状態が環境選択を生き延びる必要がある組み込みの例です。次世代の（より小さい）個体群における交配選択が、以前の統合プールで計算された適応度をゼロから再計算するのではなく再利用しなければならないためです。

## 組み込みComparator

| Class | When to use |
|---|---|
| `SingleObjectiveComparator` | 単一目的問題 |
| `WeightedSumComparator` | 重み付き線形結合によるスカラー化。単一目的と多目的の両方に対応 |
| `ParetoComparator` | 支配関係だけでランク付け（混雑度などの副次的指標なし） |
| `NSGA2Comparator` | Paretoランク + 混雑度距離 {cite}`deb2002nsga2` |
| `SPEA2Comparator` | 個体群全体に依存するSPEA2適応度 {cite}`zitzler2001spea2` |
| `HypervolumeComparator` | フロントランク + 排他的HV寄与度（SMS-EMOA方式） {cite}`beume2007smsemoa` |
| `EpsilonDominanceComparator` | ε-dominanceによるランキング {cite}`laumanns2002epsilon` |
| `NSGA3Comparator` | 参照点によるニッチの維持 {cite}`deb2014nsga3` |
| `RNSGA2Comparator` | ユーザー指定の参照点による選好誘導 {cite}`deb2006rnsga2` |

`SingleObjectiveComparator(direction=None, *, eps_cv=0.0, eps_obj=1e-6)`は`direction`を省略でき、その場合は最小化として扱います。
`WeightedSumComparator(direction, *, eps_cv=0.0, eps_obj=1e-6)`は`direction`が必須で、省略すると`TypeError`になります。
このクラスに限り、渡した`direction`がそのままスカラー化の重みとして使われます（`score = f @ direction`）。
符号だけでなく大きさも重みとして機能するため、[Problem](problem.md)で述べた「`direction`は符号のみ、重みの大きさは別の概念」という一般的な役割分担とは異なる、このクラス固有の扱いです。
組み込みのcomparatorは`eps_cv`の既定値を`0.0`にしており、`Problem`と`ParetoArchive`と同じ既定です。
そのため`cv == 0`が実行可能を表します。
`Optimizer`の実行中は毎世代`handler.feasibility_threshold`が`eps_cv`に書き写されます。
`EpsilonConstraintHandler`はこの仕組みで探索だけを緩和し、報告される実行可能性は変えません。
comparatorのコンストラクタで指定した`eps_cv`は実行準備時と各世代終了時に上書きされるため、実行中の比較には使われません。

`ParetoComparator(direction=None, *, eps_cv=0.0, eps_obj=1e-6, sorter=non_dominated_sort, dominator=None)`は、支配関係のみで個体群をランク付けします。
`NSGA2Comparator`/`HypervolumeComparator`/`NSGA3Comparator`/`RNSGA2Comparator`/`EpsilonDominanceComparator`の共通基底であり、具象クラスとして単独でも使えます。
`dominator`引数は[Dominator](dominance.md)、`sorter`引数は[NonDominatedSorter](nondominated_sorting.md)を注入する差し替え点で、互いに独立した軸です。
`NSGA2Comparator`は、`ParetoComparator`に混雑度距離による副次的な順位付けを加えます。
ソート結果は`Population`のキャッシュ（`get_cache`/`set_cache`）に保存され、個体群が変更されるまで世代内で使い回されます。

### 個体群相対的なComparator

`SPEA2Comparator`と`HypervolumeComparator`は、いずれも`compare()`を呼ぶと`NotImplementedError`を送出します。
SPEA2のfitnessも排他的HV寄与度も、個体群全体に依存する指標であり、2点だけからは計算できないためです。
これはバグではなく意図した設計であり、`is_population_relative=True`というクラス属性がその旨を示すマーカーになっています。
PSOのpbest更新や`PairwiseComparisonSet`のように、2点だけの比較（`compare()`）が必要な場面でこれらのComparatorは使えません。
そのような場面では代わりに`ParetoComparator`を使います。
`compare_population()`（個体群のインデックスを介した比較）はどちらのクラスでも定義されているため、トーナメント選択などはそのまま使えます。
`SPEA2Comparator`はさらに、前述の`get_required_attrs`/`prepare_population`の組を使います。`get_required_attrs`が永続的な`spea2_fitness`属性を宣言し、`prepare_population`が与えられた個体群に対して$S(i)$/$R(i)$/$D(i)$/$F(i)$を再計算してそこに書き込みます。
環境選択（`rank_population`を呼ぶ`TruncationSelection`）は、個体群と子個体を統合したプール全体に対して適応度を新しく再計算します。打ち切りを生き残った個体はその適応度の値を保持したまま次世代に持ち越されるため、次世代の交配選択における`compare_population()`呼び出しは、サイズの異なる集合に対して再計算するのではなく、永続化された値をそのまま再利用します。
`HypervolumeComparator`のHV計算は、フロントごとにO(N)回のleave-one-out評価を行います。
目的数が多い問題では、この計算コストが大きくなります。

```{note}
`HypervolumeComparator`の内部実装とは別に、公開関数`saealib.hypervolume(f, reference_point)`があります。
これは最適化後の結果を評価する性能指標として単独で使え、`HypervolumeComparator`とは関係ありません。
詳細は[Utils](../../api/utils.md)を参照してください。
```

### 参照点を使うComparator

`NSGA3Comparator(reference_points, direction=None, *, ...)`は、`reference_points`（`shape (n_ref, n_obj)`、単体シンプレックス上の点）が必須引数です。
通常は`saealib.utils.weight_vectors.uniform_weight_vectors(n_obj, n_divisions)`で一様に生成したものを渡します。
`rng`プロパティは遅延生成され、`Optimizer`実行時は`Runner`が`ctx.rng`からspawnした乱数生成器を注入します。
この内部rngはチェックポイントの保存対象に含まれず、再開時は新しくspawnし直されます。
`NSGA3Comparator`とは異なり、`RNSGA2Comparator(reference_points, epsilon=0.001, direction=None, *, ...)`は参照点が単体シンプレックス上にある必要はなく、ユーザーが望む目的関数値（aspiration point）をそのまま指定できます。
`epsilon`は、同じ参照点に近い解同士を間引くε-clearingの半径です。
`EpsilonDominanceComparator(eps, mode="additive", direction=None, *, ...)`は、`ParetoComparator`の`dominator`を[EpsilonDominator](dominance.md)に差し替えるだけの薄いラッパーです。
[DecompositionComparator](decomposition.md)は、MOEA/D風のスカラー化によるランク付けを行うComparatorです。
詳細はそちらのページで扱います。

## 独自Comparatorの実装方法

独自のランキング方式が必要な場合は、`Comparator`をサブクラス化し、4つの抽象メソッドすべてを実装します。
`__init__`は必ず`super().__init__(weights, eps_cv, eps_obj, direction=...)`を呼び出さなければなりません。

```python
import numpy as np
from saealib import Comparator


class RandomComparator(Comparator):
    """A simple example that always considers only feasibility."""

    def __init__(self, direction=None, *, eps_cv=0.0, eps_obj=1e-6):
        super().__init__(np.empty(0), eps_cv, eps_obj, direction=direction)

    def sort_population(self, population):
        cv = population.get_array("cv")
        return np.argsort(cv)

    def compare_population(self, population, idx_a, idx_b):
        cv = population.get_array("cv")
        return self.compare(None, cv[idx_a], None, cv[idx_b])

    def compare(self, fa, cv_a, fb, cv_b):
        if cv_a > self.eps_cv and cv_b <= self.eps_cv:
            return 1
        if cv_b > self.eps_cv and cv_a <= self.eps_cv:
            return -1
        return 0
```

クラス属性`is_population_relative = True`を設定し、指標のペア単位の順序を孤立した2つの目的値/CVタプルから本当に定義できない場合には、その理由を説明する`NotImplementedError`を`compare()`から送出させます。SPEA2のfitnessや排他的HV寄与度がこれに該当します。単に指標が個体群全体に依存するという理由だけでは該当しません（`NSGA2Comparator`の混雑度距離も個体群全体に依存しますが、`compare()`は通常のPareto支配によって定義できます）。
別途、その指標をより小さい個体群へ引き継ぐ必要もある場合（SPEA2のfitnessは、次世代の交配選択で有効であり続けるために打ち切り後も保持する必要があります）は、`get_required_attrs`をオーバーライドして保存先を宣言し、`prepare_population`をオーバーライドして新しく再計算します。そうしないと、後の読み取りでサイズの異なる個体群に対して計算された値を参照することになります。

## 関連コンポーネント

- [Dominator](dominance.md): `ParetoComparator`系の`dominator`引数
- [NonDominatedSorter](nondominated_sorting.md): `ParetoComparator`系の`sorter`引数
- [Decomposition](decomposition.md): `DecompositionComparator`が使うスカラー化関数
- [Problem](problem.md): Problemが公開する既定のcomparatorと非推奨の`comparator`引数
- [ParentSelection](../search_algorithms/parent_selection.md) / [SurvivorSelection](../search_algorithms/survivor_selection.md): `Comparator`を使って個体を選択するオペレーター
- [Population](../observation_and_state/population.md): `get_required_attrs`が拡張する`PopulationAttribute`スキーマ

## 参照

- {py:class}`saealib.Comparator`
- {py:class}`saealib.SingleObjectiveComparator`
- {py:class}`saealib.WeightedSumComparator`
- {py:class}`saealib.ParetoComparator`
- {py:class}`saealib.NSGA2Comparator`
- {py:class}`saealib.SPEA2Comparator`
- {py:class}`saealib.HypervolumeComparator`
- {py:class}`saealib.EpsilonDominanceComparator`
- {py:class}`saealib.NSGA3Comparator`
- {py:class}`saealib.RNSGA2Comparator`
- {py:func}`saealib.hypervolume`
- {py:func}`saealib.comparators.default_comparator`
