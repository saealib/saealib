---
primary_layer: layer3
page_type: concept
---

# ParentSelection

`GA`(`saealib.GA`)は、交叉に使う親個体を個体群から選ぶ処理を、`ParentSelection`という差し替え可能な演算子に委ねています。 選択圧（優れた個体をどれだけ優遇するか）を変えたいときは、`GA`本体ではなくこの`ParentSelection`だけを差し替えればよいです。

## ParentSelectionの役割

`ParentSelection`が実装する必要があるメソッドは`select(population, n_pair, n_parents, rng, *, ctx)`の1つだけです。`n_parents`個体からなる親のグループを`n_pair`組選び、そのインデックスを形状`(n_pair, n_parents)`の配列として返します。

## 組み込みParentSelection

| クラス | パラメータ | 特徴 |
|---|---|---|
| `TournamentSelection` | `tournament_size` | `tournament_size`個体をランダムに抽出し、最良個体を選ぶ操作を`n_pair * n_parents`回繰り返す{cite}`miller1995tournament` |
| `SequentialSelection` | なし | 比較を行わず、個体群のインデックスを順番に割り当てるだけ |
| `LinearRankSelection` | なし | 順位から導いた線形ランクベースの確率で選択する |

`TournamentSelection`は、`ctx.comparator.compare_population`による1対1比較を繰り返してトーナメント内の最良個体を選びます{cite}`blickle1996selection`。 `compare_population`は[SPEA2Comparator](../problem_and_ranking/comparators.md)や[HypervolumeComparator](../problem_and_ranking/comparators.md)のように2点だけの直接比較（`compare()`）が使えないComparatorでも定義されているため、どの`Comparator`と組み合わせても問題なく動作します。

`SequentialSelection`は比較を一切行わないため、選択圧という概念自体を持たない最も単純な選択方式です。 どの`Comparator`とも組み合わせられます。

`LinearRankSelection`は、生の適応度ではなく`ctx.comparator.sort_population`が返す順位を確率に変換します。 これにより、目的関数値が負の値を取ったりスケールが大きく異なったりする問題でも、数値的な問題を起こさずに選択確率を計算できます。

```{note}
`SequentialSelection`のみ`@register()`済みで、`TournamentSelection`/`LinearRankSelection`は現状Registry未登録です。 Registry経由でクラスを文字列から解決する使い方をする場合はこの違いに注意してください。
```

## 独自ParentSelectionの実装方法

独自の選択方式が必要な場合は、`ParentSelection`を継承して`select()`だけを実装すればよいです。 次の例は、比較を行わず親を完全にランダムに選ぶ選択方式です。

```python
import numpy as np
from saealib import OptimizationState, ParentSelection


class RandomPairSelection(ParentSelection):
    """A selection scheme that chooses parent individuals completely at random."""

    def select(
        self,
        population,
        n_pair,
        n_parents,
        rng: np.random.Generator,
        *,
        ctx: OptimizationState | None,
    ):
        n_pop = len(population)
        return rng.integers(0, n_pop, size=(n_pair, n_parents))
```

```{note}
`ctx`は末尾のキーワード専用・必須引数で、型は`OptimizationState | None`です。 この規則は文脈依存のフック全体（`SurvivorSelection`、`EvaluationPlanner`、`FeedbackBuilder`、`TrainingSet`、`AcquisitionFunction`、`SurrogateManager`）に共通します。 Selectionは比較と乱数を必要とするため、組み込み実装は`ctx=None`で呼ばれると`ValidationError`を送出します。 実行の外から単独で呼ぶときは、必ず`ctx=None`（または実際のstate）をキーワード引数として渡してください。
```

`ctx.comparator`を参照する実装にすれば、`TournamentSelection`のように選択圧を持つ独自方式にもできます。

## 関連コンポーネント

- [Algorithm](algorithm.md)：`GA`が`ParentSelection`をどう組み合わせるか
- [Crossover](crossover.md)：`ParentSelection`が選んだ親個体を受け取る演算子
- [Comparator](../problem_and_ranking/comparators.md)：`TournamentSelection`/`LinearRankSelection`が個体を比較する際に使う

## 参照

- {py:class}`saealib.ParentSelection`
- {py:class}`saealib.TournamentSelection`
- {py:class}`saealib.SequentialSelection`
- {py:class}`saealib.LinearRankSelection`
