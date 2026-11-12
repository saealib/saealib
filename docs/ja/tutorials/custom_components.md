---
primary_layer: layer3
related_layers: [layer2, layer4]
page_type: guide
---

# 既存の契約に独自Componentを追加する

前提は、既存のComponentの契約に収まる差し替えを実装することです。
:::{admonition} このページでできるようになること
:class: tip

このページを終えると、拡張点を選び、最小の契約を確認し、登録してテストできます。
:::

[拡張ガイドライン](../concepts/extension_guidelines.md)で拡張点の分類を確認します。契約や実行フレームワーク自体を変更する場合は、[フレームワーク拡張](../framework/extensions.md)へ進みます。

importの方針は [Canonical Imports](../api/imports.md) にまとめています。

## 拡張点を選ぶ

ビルトインコンポーネントの設定変更で足りるなら、[ビルトインコンポーネントの差し替え](component_swap.md)または[Hook](interface_hooks.md)を使います。新しい探索、予測、評価、生成ロジックが必要な場合は、対応する契約を持つコンポーネントを選びます。

| 変更 | 拡張点 | 最小実装境界 |
|---|---|---|
| 候補を生成し、評価結果を消費する | `Algorithm` | `ask()`と`tell()`、必須属性と契約 |
| 予測モデル | `Surrogate` | `fit()`と`predict()` |
| 予測を候補スコアに変換する | `AcquisitionFunction` | `evaluate()`、または対応するPointwiseの`compute_reference()`と`score()` |
| graph-native経路で評価バックエンドを拡張する | `EvaluationAdapter` / `Evaluator` | `GenomeBatch → EvaluationAdapter → EvaluationPayload → Evaluator → ObservationBatch` |
| Stage互換経路で同期Evaluatorを差し替える | `Evaluator` | `evaluate_batch()`と`EvaluationResult` |
| 世代実行ポリシー | `Strategy` | `Pipeline` 宣言を渡した `TemplateStrategy` |
| 互換実行単位 | `Stage` | `execute(state) -> state` |

各契約で必要なメソッドは、[Algorithmリファレンス](../api/algorithms.md)、[Surrogateリファレンス](../api/surrogate.md)、[Acquisitionリファレンス](../api/acquisition.md)、[Stageリファレンス](../api/stages.md)で確認します。似た名前でも契約が同じとは限らないため、境界を確認せずに別コンポーネントのメソッドを再利用してはいけません。

## 最小の契約を確認する

実装前に、既存の抽象基底クラスの抽象メソッド、`contract()`、実行時に参照される属性を確認します。`Algorithm.ask()`と`Algorithm.tell()`は`StateView`を読み、`ProposalBatch`または`StatePatch`を返します。`Surrogate`は配列を受け取り、`SurrogatePrediction`を返します。`Stage`は`OptimizationState`を受け取り、更新後の状態を返します。

## graph-native経路のComponent拡張

`Algorithm`などのgraph-nativeコンポーネントは、`ask(...) -> ProposalBatch`と`tell(FeedbackBatch, StateView) -> StatePatch`の境界で拡張します。評価を追加する場合は、`GenomeBatch → EvaluationAdapter → EvaluationPayload → Evaluator → ObservationBatch`の経路に接続します。

## Stage互換経路の同期Evaluator拡張

この例では、Stage互換経路の評価処理だけを`Evaluator`に置き換えます。実装境界は`evaluate_batch(x, problem) -> EvaluationResult`で、`EvaluationResult`は1次元または2次元の`f`、`g`、`cv`配列を受け取ります。

```python
import numpy as np
from saealib import EvaluationResult, Evaluator


class BatchSphereEvaluator(Evaluator):
    def evaluate_batch(self, x, problem):
        x = np.atleast_2d(np.asarray(x, dtype=float))
        values = np.sum(x**2, axis=1)
        return EvaluationResult(
            f=values[:, None],
            g=np.empty((len(x), 0)),
            cv=np.zeros(len(x)),
        )
```

独自Evaluatorを`Optimizer.set_evaluator()`に渡すと、組み込みの同期評価アダプターが呼び出します。`EvaluationResult`は行数と配列形状を検証します。

## 登録または組み込む

直接使う場合は、公開メソッド`Optimizer.set_evaluator(evaluator)`にインスタンスを渡します。設定ファイルやプリセットが名前でクラスを構築する場合だけ、`@register()`を追加してRegistryに登録します。

```python
import numpy as np
from saealib import Optimizer, Problem, register


@register()
class NamedBatchSphereEvaluator(BatchSphereEvaluator):
    pass


problem = Problem(
    lambda x: np.sum(x**2),
    dim=3,
    n_obj=1,
    direction=np.array([-1.0]),
    lb=[-5.0] * 3,
    ub=[5.0] * 3,
)
optimizer = Optimizer(problem).set_evaluator(NamedBatchSphereEvaluator())
```

`Problem` の評価関数を常にこのEvaluatorで置き換えるとは限らないため、`set_evaluator()` を呼んだ構成を実際に使って検証します。

## 最小テストを置く

まず `evaluate_batch()` の形状と値を固定した小さな単体テストを書きます。次に `Optimizer(problem).set_evaluator(...)` を短い `max_fe` で実行し、契約検証と実行経路を確認します。抽象基底クラスの一覧化や実行フレームワークの変更はこのガイドの範囲外です。対応するAPIリファレンスまたはフレームワーク拡張のドキュメントを参照してください。

(port-operators-to-native-saealib-code)=
## 外部OperatorをnativeなComponentへ移植する

[pymooアダプター](external_libraries.md)で外部演算子をラップすると、そのライブラリへのランタイム依存が残ります。pymooアダプターは`GA`の既定バッチモードで個体群単位のベクトル化を維持し、ラップした演算子をバッチ全体またはゲート済み部分集合に対して一度だけ呼び出すため、要素ごとの呼び出しオーバーヘッドをなくす目的だけなら通常は移植不要です。ランタイム依存をなくしたい場合や、ソフトウェア論文向けにロジックをsaealibネイティブコードとして直接監査・引用可能にしたい場合に演算子を移植します。中核のロジックは通常変わらず、主な変更は必要に応じた配列形状の調整とsaealibのRNGの使用です。

移植時に間違えやすい点は、個体レベルの確率ゲートを置く場所です。saealibの`GA`は`Crossover.prob`を自ら判定し、ゲートを通過した親グループだけを`crossover_batch()`へ渡します。一方、`Mutation.mutate_batch()`は行ごとに1つの`self.prob`ゲートを抽選し、通過しなかった行を変更せずに残す必要があります。pymooの組み込みMutationもDEAPの`toolbox.mutate()`（`algorithms.varAnd`の`mutpb`によって外部でゲートされます）もゲートを演算子の外に置くため、移植後のMutationには元のコードになかったゲート配列が通常必要です。

### pymooから移植する

独自のpymoo `Mutation` は通常、batch全体の`X`（形状`(n_individuals, dim)`）に対してベクトル化します：

```python
import numpy as np
from pymoo.core.mutation import Mutation as PymooMutationBase


class MyPymooMutation(PymooMutationBase):
    def __init__(self, sigma=1.0):
        super().__init__()
        self.sigma = sigma

    def _do(self, problem, X, random_state=None, **kwargs):
        rng = random_state if random_state is not None else np.random.default_rng()
        Xp = X.copy()
        mask = rng.random(Xp.shape) < 0.5
        Xp[mask] += rng.normal(0, self.sigma, size=Xp.shape)[mask]
        return Xp
```

nativeなsaealib `Mutation`へ移植しても、batch軸は維持されます。`_do()`本体は`mutate_batch()`へほぼそのまま対応し、主な追加はsaealibの行ごとの`prob`ゲートです：

```python
import numpy as np
from saealib.operators import Mutation


class MyMutation(Mutation):
    def __init__(self, prob=1.0, *, sigma=1.0, prob_var=0.5):
        super().__init__()
        self.prob = prob
        self.sigma = sigma
        self.prob_var = prob_var

    def mutate_batch(self, candidates_batch, mutate_range, rng=np.random.default_rng()):
        candidates_batch = np.asarray(candidates_batch, dtype=float)
        n, dim = candidates_batch.shape
        gate = rng.random(n) < self.prob
        result = candidates_batch.copy()
        if not np.any(gate):
            return result

        selected = result[gate]
        mask = rng.random((len(selected), dim)) < self.prob_var
        noise = rng.normal(0, self.sigma, size=selected.shape)
        selected[mask] += noise[mask]
        result[gate] = selected
        return result
```

### DEAPから移植する

DEAPのOperatorは一度に1個体または1組の親を処理するため、nativeなsaealibへの移植では先頭にbatch軸を追加し、その軸に沿ってベクトル化する必要があります。独自の交叉は次のようになります：

```python
import random


def my_cx(ind1, ind2, swap_rate=0.5):
    for i in range(len(ind1)):
        if random.random() < swap_rate:
            ind1[i], ind2[i] = ind2[i], ind1[i]
    return ind1, ind2
```

親の組と次元の両方に対してベクトル化することで、nativeな`Crossover`になります。また、`random.random()`をsaealibから渡される`rng`に置き換え、DEAPが暗黙に持つグローバルRNG状態を、実行ごとのsaealibの`np.random.Generator`（`minimize(..., seed=...)`で再現可能）に置き換えます：

```python
import numpy as np
from saealib.operators import Crossover


class MyCrossover(Crossover):
    def __init__(self, prob, swap_rate=0.5):
        super().__init__()
        self.prob = prob
        self.swap_rate = swap_rate

    def crossover_batch(self, parents_batch, bounds=None, rng=np.random.default_rng()):
        n_pair, _, dim = parents_batch.shape
        p1, p2 = parents_batch[:, 0, :], parents_batch[:, 1, :]
        mask = rng.random((n_pair, dim)) < self.swap_rate
        c1 = np.where(mask, p2, p1)
        c2 = np.where(mask, p1, p2)
        return np.stack((c1, c2), axis=1)
```

`toolbox.mate`は`varAnd`の`cxpb`によって外部からゲートされ、saealibの`GA`と一致するため、突然変異とは異なり、ここで追加のゲートは不要です。

### AlgorithmとProblemの移植

検索アルゴリズム全体（生存選択、アーカイブ管理、DEのようなインデックスに結び付いた状態）の移植は、同じ意味で機械的な書き換えではありません。`Algorithm.ask()`/`tell()`をゼロから再実装することになります。pymooでは、[外部ライブラリとの連携](external_libraries.md)で説明しているエンジンモードの`PymooAlgorithm`アダプターが書き換えなしで対応します。他のライブラリについては、ネイティブな`Algorithm`サブクラスに必要な実装を[Algorithm](../concepts/search_algorithms/algorithm.md)で確認してください。

## コンポーネントの責務を分ける

既存の実行契約を保ったまま拡張する場合は、担当する責務と状態の境界を先に決めます。

| Component | 主な責務 | 状態境界 |
|---|---|---|
| `Algorithm` | 候補を提案しフィードバックを消費する | `ProposalBatch`、`FeedbackBatch`、`StatePatch` |
| Operator | 交叉、突然変異、親選択、生存選択 | Algorithmから渡される候補とRNG |
| `Surrogate` | 予測モデルでfit/predictする | Archiveからの学習データと予測 |
| `OptimizationStrategy` | 評価計画、候補選択、フィードバックの流れ | `Pipeline` 宣言を渡した `TemplateStrategy` |
| `Stage` | 既存の逐次経路で1つの操作 | `OptimizationState`と`replace()` |

各コンポーネントを実装する前に、対応するConceptページで契約と組み込み実装を確認します。`Algorithm.ask()`と`Algorithm.tell()`、戦略が宣言する`Pipeline`、`Stage.execute()`は異なる境界です。

## 実装順とStageの移行

1. `Problem`と`SearchSpace`が返すデータ形式を確認します。
2. 公開名前空間から依存コンポーネントを取得します。
3. コンポーネント単体の契約と状態アクセスを実装します。
4. `Optimizer`の差し替えAPIで組み合わせます。
5. Feedbackの候補ID、評価の出所、チェックポイント動作を確認します。

既存Stageを構造化パイプラインへ接続する場合は`stage_component(stage)`を明示します。新しいgraph-nativeコンポーネントではStageの`OptimizationState`境界を再利用せず、`ComponentContract`、`StateView`、`StatePatch`を実装します。

詳細は[Algorithm](../concepts/search_algorithms/algorithm.md)、[OptimizationStrategy](../concepts/execution_and_evaluation/strategies.md)、[Stage](../concepts/observation_and_state/stage.md)を参照してください。

## 関連するConceptとReference

- [Evaluatorの概念](../concepts/execution_and_evaluation/evaluation.md)
- [拡張ガイドライン](../concepts/extension_guidelines.md)
- [Evaluationリファレンス](../api/evaluation.md)
- [Coreリファレンス](../api/core.md)
- {py:class}`saealib.Crossover`
- {py:class}`saealib.Mutation`
