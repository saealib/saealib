---
primary_layer: layer3
page_type: concept
---

# SurvivorSelection

`GA`(`saealib.GA`)は、次世代に残す個体を選択プールから選ぶ処理を、`SurvivorSelection`という差し替え可能な演算子に委ねています。 世代交代の方式を変えたいときは、`GA`本体ではなくこの`SurvivorSelection`だけを差し替えればよいです。

## SurvivorSelectionの役割

`SurvivorSelection`が実装を要求するメソッドは`select(pool, n_survivors, *, ctx) -> np.ndarray`の1つだけです。 [ParentSelection](parent_selection.md)と同様に、`ctx`は末尾のキーワード専用・必須引数（型は`OptimizationState | None`）で、Selectionは`ctx=None`を`ValidationError`で拒否します。 `pool`は、親個体群と子個体群など`Algorithm`側が構築した統合済みの`Population`で、その中から生き残る`n_survivors`件のインデックスを返します。

$(\mu+\lambda)$方式（親と子を合わせたプールから選ぶ）か$(\mu,\lambda)$方式（子だけのプールから選ぶ）かは、`pool`に何を含めるかという`Algorithm`側の構築方法で決まります。 `SurvivorSelection`のインターフェース自体はどちらの方式かを区別しません。

## 組み込みのSurvivorSelection

| クラス | パラメータ | 特徴 |
|---|---|---|
| `TruncationSelection` | `randomize_ties=False` | 打ち切り選択：`ctx.comparator.rank_population(pool)`でランク付けし、上位`n_survivors`個体を採用 |

`randomize_ties=True`を設定すると、打ち切り境界でタイになった個体（`compare_population`が`0`を返す場合）を、打ち切る前にシャッフルします。 既定の`False`では、`rank_population`が返す順序をそのまま使う決定的な打ち切りになります。 このタイブレークは`ctx.rng`を消費するため、`randomize_ties=True`の使用はチェックポイント再開時の乱数状態にも影響することに注意してください。

`TruncationSelection`は`@register()`済みです。

## 独自SurvivorSelectionの実装方法

独自の世代交代方式が必要な場合は、`SurvivorSelection`を継承して`select()`だけを実装すればよいです。 次の例は、最良個体1件を必ず残し、残りはランダムに選ぶ生存選択です。

```python
import numpy as np
from saealib import OptimizationState, SurvivorSelection


class ElitistSurvivorSelection(SurvivorSelection):
    """Always keeps the single best individual, choosing the rest at random."""

    def select(self, pool, n_survivors, *, ctx: OptimizationState | None):
        sorted_idx = ctx.comparator.rank_population(pool)
        best = sorted_idx[:1]
        rest_pool = sorted_idx[1:]
        rest = ctx.rng.choice(rest_pool, size=n_survivors - 1, replace=False)
        return np.concatenate([best, rest])
```

`pool`は呼び出しごとに`Algorithm`が新しく組み立てるため、独自の`select()`は`sort_population`を直接ではなく`rank_population`でランク付けすべきです——理由は[Comparator](../problem_and_ranking/comparators.md)を参照してください。 トーナメント方式の生存者選択や年齢ベースの入れ替えのように、そもそもランキングを前提としない方式も同じ`select()`シグネチャの中で実装できます。

## 関連コンポーネント

- [Algorithm](algorithm.md)：`GA.tell()`が`pool`をどう構築し`SurvivorSelection`をどう呼ぶか
- [Comparator](../problem_and_ranking/comparators.md)：`rank_population`/`compare_population`による個体のランク付け

## 参照

- {py:class}`saealib.SurvivorSelection`
- {py:class}`saealib.TruncationSelection`
