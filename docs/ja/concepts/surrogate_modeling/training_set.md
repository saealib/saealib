---
primary_layer: layer3
page_type: concept
---

# 学習データ集合

`GlobalSurrogateManager`/`LocalSurrogateManager`/`PairwiseSurrogateManager`は、交換可能なコンポーネントである`TrainingSet`に学習データの構築を委譲します。`training_set`引数で渡します。

## TrainingSetの役割

`TrainingSet`が実装する必要があるメソッドは`build(archive, population, candidate_x=None, *, ctx) -> TrainingData`の1つだけです。`candidate_x`は、`LocalSurrogateManager`が候補ごとのk-NN検索中心として渡す引数です。`GlobalSurrogateManager`は`None`を渡して呼び出します。

`ctx`に準備済みの`FeatureEncoder`がなく、空間にも明示登録がない場合、`ArchiveObjectiveSet`は最適化の実行準備と同じ対応済みの既定エンコーダを使い、Pythonの`UserWarning`を出します。準備済みまたは明示登録されたエンコーダは優先され、既定を使ったという警告は出ません。

`TrainingData`は、`train_x`（通常は形状`(n_train, dim)`で、`PairwiseComparisonSet`だけが`(n_train, 2*dim)`を使います）と、回帰では`(n_train, n_obj)`、分類またはランキングでは`(n_train,)`となる`train_y`を保持するデータクラスです。

組み込みの8クラスは、直交する2つの軸に沿って整理されています。

- **データソース軸**：学習データの取得元 — アーカイブ / 個体群 / k近傍 / ペア / 参照点
- **ラベル付け軸**：値の割り当て方 — 生の目的値（回帰） / 二値分類 / 多段階ランキング / ペア比較

## 文献パターンとの対応

| パターン | 出典 | 対応クラス |
|---|---|---|
| P1 CA-LLSO | {cite}`wei2021callso` | `LevelBasedSet` |
| P2 CPS-MOEA | {cite}`zhang2018cpsmoea` | `TopKBipartitionSet` |
| P3 Pairwise SAEA | {cite}`hao2024pairwise` | `PairwiseComparisonSet` |
| P4 SAPSO pbest | （一般） | `ReferencePointComparisonSet` |
| P5 CSEA / 事前選択 | （一般） | `KNNObjectiveSet`, `ArchiveObjectiveSet` |
| P6 制約BO | {cite}`regis2005cors,letham2019constraintbo` | `ConstraintObjectiveSet`, `KNNConstraintObjectiveSet` |

## 組み込みのTrainingSet

| クラス | パラメータ | 説明 |
|---|---|---|
| `ArchiveObjectiveSet` | なし | アーカイブ全体を生の目的値とともに使います。`GlobalSurrogateManager`の既定値です。 |
| `KNNObjectiveSet` | `n_neighbors=50` | `candidate_x`に最も近いアーカイブ上のk点。`LocalSurrogateManager`の既定値。 |
| `ConstraintObjectiveSet` | なし | アーカイブ全体を生の制約値とともに使います `g`。 |
| `KNNConstraintObjectiveSet` | `n_neighbors=50` | `ConstraintObjectiveSet`のk-NN版 |
| `FeasibilityClassificationSet` | `source="archive"` | 制約ごとの実行可能性判定による二値分類ラベル。 |
| `TopKBipartitionSet` | `source="archive", top_ratio=0.5` | ソート後、上位`floor(n * top_ratio)`件にlabel=1、残りにlabel=0を付けます。 |
| `LevelBasedSet` | `source="population", n_levels=5` | ソート後、`n_levels`個の等分されたグループに複数レベルのラベルを割り当てます。 |
| `PairwiseComparisonSet` | `source="archive", n_pairs=None, rng=None` | 2点をペアにし、比較の勝敗をラベル付けします |
| `ReferencePointComparisonSet` | `ref_source="population_best"` | アーカイブ上の点が参照点を支配するかどうかの二値ラベル。 |

`ConstraintObjectiveSet`/`KNNConstraintObjectiveSet`は、問題に制約がない場合（`archive.g`の列数が0）に`ValueError`を送出します。

`FeasibilityClassificationSet`は、`g`の制約ごとの違反量が許容幅の内側ですべて0のサンプルを実行可能とラベル付けします。`Result`とsummary履歴が使うのと同じ判定です。`ctx`が無い場合は制約が分からないため`cv <= 0`にフォールバックします。

`source`引数を持つクラス（`FeasibilityClassificationSet`/`TopKBipartitionSet`/`LevelBasedSet`/`PairwiseComparisonSet`）は、`source="population"`を指定したときに`population=None`であれば、いずれも`ValueError`を送出します。

`PairwiseComparisonSet`は、各ペア`(a, b)`について`train_x = [x_a, x_b]`を連結し、形状`(n_pairs, 2*dim)`の配列を構築します。`comparator.compare(f_a, cv_a, f_b, cv_b) <= 0`（aがbに勝つか同点）の場合は`1`、それ以外は`0`とラベル付けします。`n_pairs=None`の場合は、すべてのペア`n*(n-1)/2`個を使います。

```{warning}
`PairwiseComparisonSet`の`train_x`は`(n_pairs, 2*dim)`という特殊な形状であり、`RBFSurrogate`のような標準的な回帰サロゲートとは形状が非互換です。 [ComparisonSurrogate](surrogate.md)系のペア比較専用サロゲートと組み合わせる必要があります。
```

`PairwiseComparisonSet`と異なり、`ReferencePointComparisonSet`の`train_x`は`(n_archive, dim)`だけなので、`GlobalSurrogateManager`/`LocalSurrogateManager`と互換性があります。

## 独自TrainingSetの実装方法

独自の学習データ抽出方式が必要な場合は、`TrainingSet`をサブクラス化し、`build()`だけを実装します。次の例は、直近に追加された`k`件だけを学習データとして使う実装です。

```python
from saealib import OptimizationState, TrainingData, TrainingSet


class RecentKSet(TrainingSet):
    """Uses only the most recently added k entries as training data."""

    def __init__(self, k: int = 20):
        self.k = k

    def build(
        self,
        archive,
        population,
        candidate_x=None,
        *,
        ctx: OptimizationState | None,
    ):
        x = archive.get_array("x")[-self.k :]
        y = archive.get_array("f")[-self.k :]
        return TrainingData(train_x=x, train_y=y)
```

## 関連コンポーネント

- [SurrogateManager](surrogate_manager.md): managers と a `training_set` argument。
- [Surrogate](surrogate.md)：`TrainingData`の受け渡し先です。`PairwiseComparisonSet`は`ComparisonSurrogate`系と組み合わせる必要があります。
- [Comparator](../problem_and_ranking/comparators.md)：`TopKBipartitionSet`/`LevelBasedSet`/`PairwiseComparisonSet`/`ReferencePointComparisonSet`で使うソートと比較。

## 参照

- {py:class}`saealib.TrainingSet`
- {py:class}`saealib.TrainingData`
- {py:class}`saealib.ArchiveObjectiveSet`
- {py:class}`saealib.KNNObjectiveSet`
- {py:class}`saealib.ConstraintObjectiveSet`
- {py:class}`saealib.KNNConstraintObjectiveSet`
- {py:class}`saealib.FeasibilityClassificationSet`
- {py:class}`saealib.TopKBipartitionSet`
- {py:class}`saealib.LevelBasedSet`
- {py:class}`saealib.PairwiseComparisonSet`
- {py:class}`saealib.ReferencePointComparisonSet`
