---
primary_layer: layer2
related_layers: [layer3]
page_type: concept
---

# Stage

組み込みStrategyは、世代処理を `Stage` という単位に分けて実行します。
Stageは既存AlgorithmとカスタムStageを支えるsequential compatibility surfaceです。

`Pipeline` は構造化DSLです。
graph-native component、入れ子のPipeline、`Repeat`、`Loop`、`Branch`を記述しますが、`OptimizationState` を直接実行しません。
Pipelineを実行可能な計画へ変換する処理は、Optimizerが内部でCompilerへ委譲します。
通常の利用者がCompilerを直接呼び出す必要はありません。

このページでは、Stageの互換性用契約、組み込みStage、カスタムStageの実装方法を説明します。
graph-native componentの実装方法は [Framework](../../framework/index.md) を参照してください。

## Stageの役割

`Stage` が実装する境界は `execute(state: OptimizationState) -> OptimizationState` です。
Stageは `OptimizationState` を受け取り、一つの処理を行って更新後のStateを返します。
この境界はgraph-native componentの `execute(StateView) -> StatePatch` とは異なります。

クラス属性は3つあります。

- **`name`**: `pipeline["name"]`で検索する機械可読識別子
- **`label`**: 人間が読める説明
- **`notation`**: `to_pseudocode()`で使うLaTeX表記

## パイプライン

`Pipeline`は構造化されたシーケンスです。要素にはグラフネイティブなコンポーネント、入れ子の`Pipeline`、構造化された制御値を指定できます。Loweringは入れ子の名前をグラフ領域として保持し、実行時にrepeat/loopの進行を再開できるようにします。

| 操作 | 説明 |
|---|---|
| `pipeline["name"]` | `name`で構造エントリを検索 |
| `pipeline.replace(name, entry)` | トップレベルの構造エントリを置換 |
| `pipeline.find(name, *, recursive=False)` | `recursive=True`では、入れ子のパイプラインと制御本体を検索 |

## 組み込みStage

次の表は、各運用Stageの本番用`contract()`（`reads`、`writes`、`exports`）から生成されます。次のコマンドで更新します: `python scripts/generate_stage_docs.py`。

<!-- BEGIN GENERATED STAGE CONTRACTS: do not edit -->

| クラス | 名前 | ラベル | 表記 | 読み取り | 書き込み | エクスポート |
|---|---|---|---|---|---|---|
| CountGenerationStage | count_generation | Count generation | $gen \leftarrow gen + 1$ | `runtime.generation`, `evaluations.pending` | `runtime.generation` | — |
| AskStage | ask | Generate offspring | $\mathcal{Q} \leftarrow \text{ask}(P, n)$ | `proposals.offspring`, `proposals.proposal_id_allocator`, `evaluations.plan`, `evaluations.plan_state`, `runtime.candidate_id_allocator` | `proposals.offspring`, `proposals.current`, `runtime.candidate_id_allocator`, `evaluations.evaluated_offspring` | — |
| SurrogatePredictStage | surrogate_predict | Surrogate prediction | $\hat{y} \leftarrow \text{predict}(\mathcal{Q}, \mathcal{A})$ | `proposals.offspring`, `populations.main`, `archives.main` | `proposals.offspring`, `surrogates.predictions` | — |
| PendingEvaluationContextStage | pending_evaluation_context | Pending evaluation context | $C \leftarrow \text{pending}(C)$ | — | — | — |
| AcquisitionStage | acquisition | Acquisition scoring | $\mathbf{s} \leftarrow \text{acquire}(\mathcal{Q}, \hat{y}, \mathcal{A})$ | `proposals.offspring`, `surrogates.predictions`, `archives.main`, `runtime.generation`, `runtime.decision_count`, `runtime.rng` | `evaluations.scores`, `evaluations.acquisition_result` | — |
| SurrogateFitStage | surrogate_fit | Fit surrogate | $\hat{f} \leftarrow \text{fit}(\mathcal{A})$ | `populations.main`, `archives.main` | — | — |
| TopKSelectionStage | top_k_selection | Top-k pre-selection | $\mathcal{Q} \leftarrow \text{top-}k(\mathcal{Q}, \mathbf{s})$ | `proposals.offspring`, `evaluations.scores` | `proposals.offspring` | — |
| SortByScoreStage | sort_by_score | Sort offspring by score | $\mathcal{Q} \leftarrow \text{sort\_desc}(\mathcal{Q},\,\mathbf{s})$ | `proposals.offspring`, `evaluations.scores` | `proposals.offspring`, `evaluations.scores` | — |
| EvaluationPlanStage | evaluation_plan | Plan evaluation | $R \leftarrow \text{plan}(Q)$ | `proposals.offspring`, `evaluations.pending`, `evaluations.request`, `evaluations.plan`, `evaluations.plan_state`, `evaluations.handles`, `evaluations.owners`, `evaluations.acquisition_result`, `evaluations.scores`, `surrogates.predictions`, `runtime.request_id_allocator`, `runtime.decision_count` | `evaluations.request`, `evaluations.plan`, `evaluations.plan_state`, `evaluations.plan_updates`, `evaluations.pending`, `evaluations.updates`, `evaluations.update_new_ids`, `evaluations.new_ids`, `evaluations.handles`, `evaluations.owners`, `runtime.request_id_allocator`, `runtime.decision_count` | — |
| AsyncEvaluationSubmitStage | async_evaluation_submit | Submit asynchronous evaluation |  | `proposals.offspring`, `evaluations.acquisition_result`, `evaluations.scores`, `evaluations.plan`, `evaluations.plan_state`, `evaluations.pending`, `evaluations.handles`, `runtime.request_id_allocator`, `runtime.decision_count` | `evaluations.request`, `evaluations.plan`, `evaluations.plan_state`, `evaluations.plan_updates`, `evaluations.pending`, `runtime.request_id_allocator`, `runtime.decision_count` | — |
| EvaluationSubmitStage | evaluation_submit | Submit evaluation | $H \leftarrow \text{submit}(R)$ | `evaluations.request`, `evaluations.plan`, `evaluations.plan_state`, `evaluations.pending`, `evaluations.handles`, `evaluations.owners` | `evaluations.pending`, `evaluations.handles`, `evaluations.plan_state` | — |
| EvaluationCollectStage | evaluation_collect | Collect evaluation | $U \leftarrow \text{collect}(H)$ | `evaluations.plan`, `evaluations.plan_state`, `evaluations.pending`, `evaluations.handles`, `evaluations.plan_updates`, `evaluations.request` | `evaluations.updates`, `evaluations.request`, `evaluations.plan`, `evaluations.plan_state`, `evaluations.plan_updates`, `evaluations.pending`, `evaluations.update_new_ids`, `evaluations.new_ids` | — |
| EvaluationApplyStage | evaluation_apply | Apply evaluation | $Q \leftarrow \text{apply}(U)$ | `proposals.offspring`, `evaluations.request`, `evaluations.updates`, `evaluations.plan`, `evaluations.plan_state`, `evaluations.plan_updates`, `evaluations.pending` | `proposals.offspring`, `evaluations.evaluated_offspring`, `evaluations.new_ids`, `evaluations.update_new_ids`, `evaluations.pending` | — |
| EvaluationAcknowledgeStage | evaluation_acknowledge | Acknowledge evaluation | $H \leftarrow \text{ack}(U)$ | `proposals.offspring`, `evaluations.request`, `evaluations.plan`, `evaluations.plan_state`, `evaluations.pending`, `evaluations.handles`, `evaluations.updates`, `evaluations.update_new_ids`, `evaluations.plan_updates`, `evaluations.count` | `evaluations.plan`, `evaluations.plan_state`, `evaluations.plan_updates`, `evaluations.pending`, `evaluations.handles`, `evaluations.count` | — |
| TrueEvaluationStage | true_evaluation | True objective evaluation | $\mathcal{Q}_{eval} \leftarrow \text{eval}(\mathcal{Q})$ | `proposals.offspring` | `proposals.offspring`, `evaluations.evaluated_offspring`, `evaluations.count` | — |
| ArchiveUpdateStage | archive_update | Archive update | $\mathcal{A} \leftarrow \mathcal{A} \cup \mathcal{Q}_{eval}$ | `evaluations.evaluated_offspring`, `archives.main`, `archives.pareto`, `evaluations.plan`, `evaluations.plan_state` | `archives.main`, `archives.pareto`, `evaluations.evaluated_offspring` | — |
| FeedbackStage | feedback | Apply feedback | $\mathcal{Q} \leftarrow \mathrm{feedback}(\mathcal{Q})$ | `proposals.offspring`, `evaluations.evaluated_offspring`, `evaluations.new_ids`, `surrogates.predictions`, `evaluations.plan`, `evaluations.plan_state` | `proposals.offspring`, `feedback.result` | — |
| TellStage | tell | Update population | $P \leftarrow \text{tell}(P, \mathcal{Q})$ | `proposals.offspring`, `proposals.current`, `feedback.result`, `evaluations.evaluated_offspring`, `evaluations.plan`, `evaluations.plan_state`, `evaluations.updates` | — | — |
| SurrogateOnlyLoopStage | surrogate_only_loop | Surrogate-only generations | $\text{for}\;i=1\dots gen\_ctrl$: $P \leftarrow \mathrm{tell}(P,\,\mathrm{acquire}(\mathrm{predict}(\mathrm{ask}(P))))$ | `archives.main` | — | — |
| InitializationStage | initialization | Initialize population | $\mathcal{A}_0,\,P_0 \leftarrow \mathrm{init}(n_{\mathrm{init}})$ | — | — | — |
<!-- END GENERATED STAGE CONTRACTS -->

`TellStage`は、状態契約がStage境界を記述するため`writes=()`を宣言します。レガシーアルゴリズムを使う場合、レガシーアダプターは実行中にアルゴリズムの個体群を変更できますが、そのランタイム動作はStage契約に追加されません。

契約表には、非同期評価や入れ子のサロゲートのみのループで使われるStageを含む、すべての運用Stageが含まれます。

`AskStage`は`algorithm.ask()`を呼び出し、`state.offspring`に書き込み、`cbmanager`を介してPostCrossover/PostMutation/PostAskEventを発火します。

`SurrogatePredictStage`は予測データを取得し、`AcquisitionStage`はスコアを計算して計画用に完全な`AcquisitionResult`を保持します。

`SurrogateFitStage`は、アーカイブが変化しない内部ループに先立ってサロゲートを一度だけ事前適合するために使います。
内部ループで適合結果を再利用できる場合は、サロゲート予測Stageの前に使います。

`SurrogateOnlyLoopStage`はステージベースの互換性戦略で引き続き利用できます。`GenerationBasedStrategy`は、代わりに構造化された`Repeat`領域で同じ制御フローを表します。

```{note}
`InitializationStage`の`execute()`に渡された`state`引数は無視され、常に初期化から新しい状態を作ります。
ユーザー定義パイプラインの先頭で、初期化自体をパイプラインの一部として扱いたい場合に使います。
```

組み込み戦略がこれらのStageと構造化領域をどう組み合わせるかは、[strategies](../execution_and_evaluation/strategies.md)と[Components overview](../index.md)のパイプライン図を参照してください。
このページでは各Stage単独の契約を扱います。

## 独自Stageの実装方法

sequential compatibility経路にカスタムStageを追加するときは、`Stage`を継承して`execute()`を実装します。
`state.replace(...)`で新しいStateを返し、StageからCompilerの`StatePatch`値を直接操作しないようにします。

```python
from saealib import Stage


class LogGenerationStage(Stage):
    """A custom stage that just prints the generation number to stdout."""

    name = "log_generation"
    label = "Log generation number"

    def execute(self, state):
        print(f"generation {state.gen}")
        return state
```

互換性用Stageで追加の値を渡す場合は、`OptimizationState.data` に保存します。
値を追加するときは、`state.replace(data={**state.data, "key": value})` のように新しい辞書を渡します。
graph-native componentで同じことを行う場合は、宣言した `StateKey` と `StatePatch` を使います。

## `to_pseudocode`

`to_pseudocode(expand=False, indent=0)`は、各Stageの`notation`を論文用の擬似コード（LaTeX algorithmic表記）として出力する仕組みです。
`AskStage`/`TellStage`/`SurrogateOnlyLoopStage`には、`expand=True`のときに`Algorithm.ask_notation`/`tell_notation`を展開する専用実装があります。

## 関連コンポーネント

- [Extension guidelines](../extension_guidelines.md): `pipeline.replace`/`find`によるStageの並べ替え方
- [strategies](../execution_and_evaluation/strategies.md): 組み込みStrategyによるStageの組み合わせ方
- [OptimizationState](optimization_state.md): `execute()`が読み書きする状態オブジェクト
- [Components overview](../index.md): パイプライン全体の構造図

## 参照

- {py:class}`saealib.Stage`
- {py:class}`saealib.Pipeline`
- {py:class}`saealib.CountGenerationStage`
- {py:class}`saealib.AskStage`
- {py:class}`saealib.SurrogatePredictStage`
- {py:class}`saealib.SurrogateFitStage`
- {py:class}`saealib.ArchiveUpdateStage`
- {py:class}`saealib.TellStage`
- {py:class}`saealib.SurrogateOnlyLoopStage`
- {py:class}`saealib.InitializationStage`
