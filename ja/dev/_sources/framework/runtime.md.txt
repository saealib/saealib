---
primary_layer: layer4
related_layers: [layer3]
page_type: concept
---

# Runtime

ランタイムは、コンパイル済みの計画と状態を受け取り、観測可能な状態境界まで処理を進めます。 計画を作るコンパイラと、計画を進めるランタイムは別の責務を持ちます。

## ExecutionRuntimeの役割

`ExecutionRuntime`は`ExecutablePlan`を実行位置と状態に結び付け、`StatePatch`値、イベント、待機を順序付けます。契約の互換性とサービスの解決はCompilerの責務であり、Runtimeは未検証のGraphを修正しません。`RuntimeSession`は現在の状態と実行位置を所有し、`StateStore`は永続値を所有します。

## ExecutablePlanとExecutionRuntime

`ExecutablePlan` は `ComponentGraph` の検証済み表現であり、必要なランタイム能力と診断情報を持ちます。 `ExecutionRuntime` は計画を初期化し、状態を一段進め、終了や再コンパイルなどの要求を処理するプロトコルです。 ランタイムはノードの結果に含まれる `StatePatch`、イベント、コマンドを順に適用します。

## RuntimeSessionとRuntimeStep

`RuntimeSession` は計画、現在の状態、実行位置、完了状態をまとめた再開可能なセッションです。 `RuntimeStep` は一回の進行結果であり、次のセッション、状態、観測可能性を返します。 この分離により、チェックポイントや非同期評価の待機を、同じ実行モデルで扱えます。

StatePatchの適用、イベントの配送、コマンドの処理が完了した境界だけを次のStepとして公開します。

| 値 | 保持する情報 |
|---|---|
| `RuntimeSession` | `ExecutablePlan`、現在の`OptimizationState`、実行位置、完了状態、構造化領域のFrame |
| `RuntimeStep` | 更新後の状態、実行したノード、`NodeResult`、待機状態、次のSession |

`ExecutionRuntime.initialize(plan, state)`はセッションを作り、`advance(session)`は1つの`RuntimeStep`を返します。アプリケーションはstepの`finished`、`observable`、`session`フィールドを使って、次の操作または再開位置を選びます。

## PipelineRuntimeと非同期待機

`PipelineRuntime` は通常のパイプラインを順に実行し、`AsyncPipelineRuntime` は非同期評価の提出と回収を含む待機状態を扱います。 非同期評価では、提出した要求がすぐに完了するとは限らないため、ランタイムは状態を保持したまま次のポーリングへ戻ります。 評価の完了時に得た更新をパッチとして適用してから、後続ステージを再開します。

`RuntimeSession` はこの待機位置を保持し、`RuntimeStep` は進捗があったかどうかと次のセッションを返します。 非同期評価の実装は `AsyncEvaluator` とスケジューラーが担い、`AsyncPipelineRuntime` はそれを計画の実行境界へ接続します。

### ランナー

`Runner` は `Optimizer` から計画と初期状態を取得し、ランタイムを反復して状態の境界だけを公開する薄い内部実装です。 アプリケーションが依存する安定した構成APIではなく、実行ランタイムを駆動するための内部コンポーネントとして扱います。 通常の利用では `minimize()`、`maximize()`、`Optimizer.run()`、`Optimizer.iterate()` を使います。

## 生成時点と利用時点

`ExecutionRuntime.initialize(plan, state)`はセッションを作り、`advance(session)`は1つの`RuntimeStep`を返します。`RuntimeSession`は不変の実行スナップショットであり、Runtimeは次のセッションを返して再開位置を明示します。

## 不変条件と診断

Runtimeは、環境がPlanに必要な能力を提供すること、`NodeResult`のStatePatchが宣言された状態効果の範囲内にあること、待機要求が二重適用されないことを検証します。ERRORレベルの計画診断、未宣言の状態書き込み、無効な実行位置、回収できない非同期要求は実行を停止し、Diagnosticsまたは失敗結果として報告します。INFOとWARNINGの計画診断では実行を停止しません。ランタイム能力の不足はCompilerの`missing_runtime_capability`診断になります。

## 拡張点

実行能力または待機状態を追加する場合は、`RuntimeSession`の遷移とともに`ComponentContract`の`execution`契約を定義します。既存のStageを統合する場合は、その`OptimizationState`境界をアダプターで接続し、graph-native Componentの`StateView`境界とは別に保ちます。

## 関連ページ

実行計画の構成は[Compiler](compiler.md)を、状態の所有と変更は[OptimizationState](../concepts/observation_and_state/optimization_state.md)を参照してください。 拡張手順は[フレームワーク拡張](extensions.md)を、具体的なRuntime型の公開import経路は[APIリファレンス](../api/index.md)で確認してください。

## 参照

- {py:class}`saealib.core.ExecutionRuntime`・クラス
- {py:class}`saealib.core.ExecutablePlan`・クラス
- {py:class}`saealib.core.StatePatch`・クラス
