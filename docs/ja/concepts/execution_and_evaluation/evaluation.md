---
primary_layer: layer2
related_layers: [layer3, layer4]
page_type: concept
---

# Evaluator

`OptimizationStrategy` は `EvaluationPlan` を作り、Evaluatorへ評価を委譲します。

## Evaluatorの役割

同期・非同期の二つの評価境界で受け取る値と返す値は次のとおりです。

| 境界 | 受け取る値 | 返す値 | 主な利用者 |
|---|---|---|---|
| graph-native境界 | 評価プランナーの`EvaluationRequestBatch`で選ばれた`GenomeBatch`、候補ID、提案関係、メタデータ | 部分的・順不同・反復を含む `ObservationBatch` | Evaluator、非同期Scheduler、構造化ランタイム |
| Stage互換境界 | `x`、`Problem`、`EvaluationRequest` | `EvaluationResult`（`f`、`g`、`cv`、任意の `candidate_ids`） | 互換Stage、sequential compatibility ランタイム |

Evaluatorまたは非同期Schedulerへ渡されるRequestは、候補IDと評価対象を識別する情報を持ちます。
評価が同期か非同期かは、パイプライン側の候補生成契約を変えずにEvaluatorとランタイムプロバイダーで切り替えられます。
`EvaluationRequestBatch`は、評価プランナーの出力ポートに宣言された`DataSpec`のkindです。
Evaluatorは選択された`GenomeBatch`を受け取ります。`Problem.evaluation_adapter`が設定されている場合は`EvaluationAdapter.transform()`の戻り値を`Problem.evaluate_batch()`に渡し、未設定で入力が`DenseVectorBatch`の場合は`SerialEvaluator`が`DenseVectorBatch.array`をpayloadとして使います。
Evaluatorの出力は`ObservationBatch`で、候補ID、提案関係、順序、状態、出所、完了の意味論で観測を対応付け、行位置を前提にしません。

Stage互換経路では`Evaluator.evaluate_batch(...) -> EvaluationResult`を使います。`EvaluationRequest`はRequest ID、候補ID、GenomeBatch、メタデータを保持します。互換性境界でのみ、`EvaluationResult`は`f`、`g`、`cv`の形状と行数の規則を満たす必要があります。`candidate_ids`を指定した場合は一意で、結果の行数と一致しなければなりません。`Evaluator.submit`が`candidate_ids`を返す場合、それらはリクエストの候補IDと完全に一致しなければなりません。外部の`ObservationBatch`に行順を使わせる必要はありません。

`EvaluationResult` は、評価結果の配列と候補IDを保持するデータクラスです。

- **`f`**：目的値。形状は `(n, n_obj)`。
- **`g`**：生の制約値。形状は `(n, n_constraints)`。制約がない場合は `(n, 0)`。
- **`cv`**：候補ごとの集約制約違反。形状は `(n,)`。制約がない場合はすべて `0`。
- **`candidate_ids`**：結果の各行に対応する候補ID。

## 組み込みEvaluator

| クラス | パラメータ | 特徴 |
|---|---|---|
| `SerialEvaluator` | なし | 候補を1件ずつ逐次評価する。既定値 |
| `JoblibEvaluator` | `n_jobs=-1, backend="loky", **joblib_kwargs` | `joblib.Parallel`経由で候補を並列評価する |
| `ThreadPoolEvaluator` | Executorのオプション | スレッドプールを使った同期バッチ評価 |
| `AsyncEvaluator` | アダプターとスケジューラーのオプション | リクエストを送信し、ノンブロッキングなライフサイクル操作を公開します |

`JoblibEvaluator`を使うには`parallel`追加機能（`pip install saealib[parallel]`）が必要で、未インストールなら構築時に`ImportError`が発生します。`backend`の既定値`"loky"`はcloudpickleでシリアライズするプロセスプールですが、1つのパラメータ変更で`"dask"`/`"ray"`などのサードパーティ製バックエンドに切り替えられます（対応パッケージとクラスタは別途必要です）。複数の`JoblibEvaluator`を同時に使う構成（アイランドモデルなど）ではCPUコアを過剰予約することがあります。各アイランドの`n_jobs`を`1`に制限してアイランド間の並列性で全体を制御するか、`joblib.parallel_backend`をコンテキストマネージャとして使って内部ワーカー数を制限してください。完了時刻が独立している場合は、`AsyncEvaluator`、`SteadyStateStrategy`、`AsyncEvaluationScheduler`、`Optimizer.set_async_evaluation_scheduler()`を組み合わせます。各候補は保留中のRequestとして管理され、Schedulerは容量、予算、Requestの状態を管理します。`collect(wait=False)`は完了を待たずに進捗を返し、PopulationとArchiveの更新順序はRuntimeが管理します。

同期Evaluatorは`Optimizer.set_evaluator(evaluator)`で差し替えます。

`EvaluationRequest.metadata`は、明示的な実行プランナーを公開するEvaluatorが消費します。 `RepeatedEvaluation`は、安定した候補IDを持つリクエストを複製ごとに1つずつ含む`EvaluationPlan`を返します。 Schedulerが容量、予算、リクエストのライフサイクルを管理します。

## 独自Evaluatorの実装方法

次の例は、`evaluate_batch()`を実装するStage互換の同期Evaluatorです。ベクトルでないGenomeでは、評価関数に渡すペイロードを`EvaluationAdapter`で定義します。

```python
import numpy as np
from saealib import EvaluationResult, Evaluator


class ReversedOrderEvaluator(Evaluator):
    """An Evaluator that evaluates candidates in reverse order, from the end."""

    def evaluate_batch(self, x, problem):
        x = np.atleast_2d(np.asarray(x, dtype=float))
        n = len(x)
        f = np.empty((n, problem.n_obj), dtype=float)
        g = np.empty((n, problem.n_constraints), dtype=float)
        cv = np.zeros(n, dtype=float)
        for i in reversed(range(n)):
            g_i, cv_i = problem.evaluate_constraints(x[i])
            f[i] = problem.evaluate(x[i], g_i)
            g[i] = g_i
            cv[i] = cv_i
        return EvaluationResult(f=f, g=g, cv=cv)
```

`problem.evaluate_constraints(xi)` を `problem.evaluate(xi, g_i)` より先に呼び出す順序を維持します。[ConstraintHandler](../problem_and_ranking/constraints.md) は制約値を使って目的値を補正するため、この順序を逆にすると補正が正しく適用されません。

## 関連コンポーネント

- [Problem](../problem_and_ranking/problem.md)：`evaluate_batch` が評価する目的関数と制約関数の定義
- [ConstraintHandler](../problem_and_ranking/constraints.md)：`problem.evaluate`内で制約違反を集約し、目的値を補正します
- [Initializer](initialization.md)：初期個体群の評価に`Evaluator`を使う
- [strategies](strategies.md)：各世代の候補評価に`Evaluator`を使う
- [Feedback](../observation_and_state/feedback.md)：評価結果をAlgorithmへ渡すFeedbackの構成

## 参照

- {py:class}`saealib.Evaluator`
- {py:class}`saealib.SerialEvaluator`
- {py:class}`saealib.JoblibEvaluator`
- {py:class}`saealib.AsyncEvaluator`
- {py:class}`saealib.AsyncEvaluationScheduler`
- {py:class}`saealib.EvaluationResult`
