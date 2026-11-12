---
primary_layer: layer3
page_type: concept
---

# サロゲート管理

[Surrogate](surrogate.md)がfit/predictだけを扱うのに対し、`SurrogateManager`はモデルのフィットとバッチ予測を協調させます。オプティマイザに設定した獲得関数が、返された予測をスコア化します。

`Optimizer.set_surrogate_manager()`は、`Optimizer.set_surrogate()`（[Surrogate](surrogate.md)を`LocalSurrogateManager`でラップする簡易版）とは別のトップレベル差し替え点です。

## SurrogateManagerの役割

`SurrogateManager`の抽象メソッドは`predict()`です。フィットと世代フックには既定実装があります。

**`predict(candidates_x, archive, *, refit=True, ctx) -> SurrogatePrediction`**（抽象メソッド）：候補を予測します。`refit=True`（既定値）の場合は、予測前にサロゲートを再学習します。

**`fit(archive, *, ctx) -> None`**：既定では何もしません。アーカイブが変化しない場合に、`predict(..., refit=False)`を連続して呼び出す前に1回だけ呼ぶ事前フィット用フックです。

**`last_accuracy: SurrogateAccuracy | None`**（クラス属性）：直近の`fit`が計算した精度指標。詳細は[サロゲート精度評価と動的切り替え](surrogate_switching.md)で扱います。

**`on_generation_end(gen, archive, ctx=None)`** / **`with_on_generation_end(fn)`**：世代末フック。同じくコピー＋チェーン方式で拡張できます。

二つの境界で受け取る値と返す値は次のとおりです。

| 境界 | 受け取る値 | 返す値 | 主な利用者 |
|---|---|---|---|
| Stage互換境界 | `archive`、`candidates_x: np.ndarray`、`ctx`、`refit` | `SurrogatePrediction`、fit の完了状態 | 互換Stage、逐次互換ランタイム |
| graph-native境界 | `GenomeBatch`、`FeatureEncoder`、`FeatureBatch`、宣言済みサービス | Surrogateがフィット・予測できる特徴と予測結果 | コンポーネント、Compiler、構造化ランタイム |

Stage互換経路の現行vector形式のManager APIでは、`fit(archive, *, ctx)`と`predict(candidates_x: np.ndarray, archive, *, refit=True, ctx) -> SurrogatePrediction`を使います。現行APIには、`GlobalSurrogateManager(surrogate, training_set=None, accuracy_evaluator=None)`、`LocalSurrogateManager(surrogate, training_set=None, accuracy_evaluator=None)`、`PairwiseSurrogateManager(surrogate, training_set=None, n_ref=10)`などのコンストラクタも含まれます。
構造化ランタイムでは、Surrogateの入力境界は`GenomeBatch → FeatureEncoder → FeatureBatch → Surrogate`です。`FeatureEncoder`は意味変換を行い、Surrogateが学習できる特徴を決めます。
実行準備は、選択した戦略が`FeatureEncoder`を使う場合にエンコーダを選びます。`VectorSpace`は恒等エンコーディングを既定にし、選択元と理由をINFOで報告します。`PermutationSpace`と`SequenceSpace`は組み込みの既定エンコーダを使い、エンコーディングがサロゲートの学習対象を決めることを示して、明示選択を勧めるWARNINGを報告します。このWARNINGは実行を妨げません。
`SearchSpace.services`に明示されたエンコーダは既定より優先され、既定選択の診断は出ません。`ObjectSpace`と独自の空間はエンコーダを自動で補わないため、必要な戦略を使う場合は空間が提供します。
診断ストリームは`optimizer.contract_diagnostics()`で確認できます。`optimizer.validate()`の後は`optimizer.last_contract_diagnostics`にも診断が残ります。`Optimizer.validate()`が返すのはERRORだけなので、INFOとWARNINGは診断ストリームで確認してください。

## 組み込みSurrogateManager

| クラス | 方式 |
|---|---|
| `GlobalSurrogateManager` | アーカイブ全体で1回グローバルにフィットし、全候補を一括予測する |
| `LocalSurrogateManager` | 候補ごとにk近傍でローカルフィット |
| `CompositeSurrogateManager` | 名前付きの予測チャンネルを組み合わせる |
| `PairwiseSurrogateManager` | ペア比較サロゲートで勝率を予測する |

`GlobalSurrogateManager(surrogate, training_set=None, accuracy_evaluator=None)`は、`training_set`省略時に`ArchiveObjectiveSet()`が使われます。

`LocalSurrogateManager(surrogate, training_set=None, accuracy_evaluator=None)`は、`training_set`省略時に`KNNObjectiveSet(n_neighbors=50)`が使われます。`n_neighbors`は`LocalSurrogateManager`自体のコンストラクタ引数ではなく、既定の`training_set`が持つパラメータです。候補間で同一の`surrogate`インスタンスを使い回して再フィットする実装のため、スレッドセーフではありません。

`CompositeSurrogateManager(managers)`は各名前付きマネージャーの`predict()`を呼び出し、複数チャンネルの`SurrogatePrediction`を返します。`CompositeAcquisition`を使うとチャンネルごとに取得関数を評価し、結果のスコア配列を`product_combine`または`rank_weighted_combine(weights=None)`で組み合わせられます。

`PairwiseSurrogateManager(surrogate, training_set=None, n_ref=10)`は、`training_set`省略時に`PairwiseComparisonSet()`が使われます。

各マネージャーの`training_set`/`accuracy_evaluator`引数の詳細は、それぞれ[TrainingSet](training_set.md)と[サロゲート精度評価と動的切り替え](surrogate_switching.md)を参照してください。

### アーカイブベースの取得関数

これらの取得関数は候補の幾何学的な特徴をアーカイブと直接比較してスコア化するため、サロゲートの予測を必要としません。

| クラス | パラメータ | スコアの意味 |
|---|---|---|
| `NoveltyAcquisition` | `k=1` | アーカイブへのk近傍平均距離が大きいほど良い |
| `InverseDensityAcquisition` | `eps=1.0` | ε近傍密度の逆数 |
| `MaximinDistanceAcquisition` | なし | 候補間の最小距離とアーカイブまでの最小距離の合計 |

## SurrogateManagerとアーカイブベースの取得関数の拡張

独自の予測方式が必要な場合は`SurrogateManager`を継承して`predict()`を実装します。独自のアーカイブベース基準が必要な場合は`AcquisitionFunction`を継承して評価契約を実装します。

```python
import numpy as np
from saealib import AcquisitionFunction, AcquisitionResult, OptimizationState


class ConstantAcquisition(AcquisitionFunction):
    """A minimal acquisition that assigns every candidate the same score."""

    def evaluate(
        self,
        candidates_x,
        prediction,
        archive,
        *,
        prepared=None,
        ctx: OptimizationState | None,
    ):
        return AcquisitionResult(scores=np.ones(len(candidates_x)))
```

## 関連コンポーネント

- [Surrogate](surrogate.md)：`SurrogateManager`が協調させるfit/predictの実体
- [TrainingSet](training_set.md)：各`SurrogateManager`が学習データの抽出に使う
- [AcquisitionFunction](acquisition_functions.md)：`AcquisitionStage`で予測をスコア化する
- [サロゲート精度評価と動的切り替え](surrogate_switching.md)：`accuracy_evaluator`/`last_accuracy`の詳細
- [strategies](../execution_and_evaluation/strategies.md)：`SurrogatePredictStage`と`AcquisitionStage`を組み立てる

## 参照

- {py:class}`saealib.SurrogateManager`
- {py:class}`saealib.GlobalSurrogateManager`
- {py:class}`saealib.LocalSurrogateManager`
- {py:class}`saealib.CompositeSurrogateManager`
- {py:class}`saealib.PairwiseSurrogateManager`
- {py:func}`saealib.product_combine`
- {py:func}`saealib.rank_weighted_combine`
- {py:class}`saealib.NoveltyAcquisition`
- {py:class}`saealib.InverseDensityAcquisition`
- {py:class}`saealib.MaximinDistanceAcquisition`
