---
primary_layer: layer2
related_layers: [layer3, layer4]
page_type: reference
---

# パイプライン

`Pipeline`はコンポーネントと構造化された制御領域を記述するDSLです。
`OptimizationState`を直接実行するものではありません。Optimizerは実行計画の構築時に入力として使います。

```python
from saealib import Branch, Loop, Pipeline, Repeat

pipeline = Pipeline(
    name="generation",
    steps=[
        Repeat(
            Pipeline(steps=[ask, acquire, tell], name="surrogate_generation"),
            count=10,
            name="surrogate_generations",
        ),
        Loop(evaluation, until=budget_reached, name="evaluation_loop"),
        Branch(route, then=fast_path, else_=safe_path, name="route"),
    ],
)
```

この例はPipelineの構築方法だけを示します。
通常は`Optimizer`がStrategyの生成したGraphとPipelineをコンパイルするため、ユーザーがCompilerを直接呼び出す必要はありません。

ネストしたPipelineと制御値は、名前付きの構造化された領域として保持されます。
`Repeat`は固定回数の反復を表し、`Loop`と`Branch`は宣言された状態契約を通じて条件を評価します。
領域は通常のグラフサイクルへ縮約されないため、Runtimeは状態への影響を保持し、フレームを再開できます。

構造化Pipelineに配置するコンポーネントは、グラフネイティブな`contract()`と`execute(StateView)`の境界を提供します。
従来の`Stage`実行面は、互換Graph Builderへの接続点として残ります。
既存のStageを構造化Pipelineに配置する場合は、`stage_component(stage)`で明示的にラップします。

コンパイル時、必須入力ポートは制御順序に従って上流の互換出力と照合されます。
一意に一致した接続は`DataEdge`になり、一致しない場合や候補が複数ある場合はCompiler診断が生成されます。
あいまいな接続を解決する必要がある場合は、明示的なGraphを使います。

`Stage.flow_summary()`はコンポーネント呼び出しとデータ接続を宣言します。
戻り値が`None`の場合、Stageのflowは不明です。
空のsummaryは状態だけを扱い、コンポーネント間のデータ接続を持たないflowを表します。
Compilerは次の変換名と入出力のデータ型を検証します。

- **`encode_features`**：`Population`から`FeatureBatch`へ変換します。
- **`extract_population_genomes`**：`Population`から`GenomeBatch`を取り出します。
- **`apply_observations`**：`ObservationBatch`を`Population`へ適用します。
- **`extract_evaluation_observations`**：`Population`から`ObservationBatch`を取り出します。
- **`select_requested_candidates`**：`EvaluationRequestBatch`から`CandidateSelection`を作ります。
- **`extract_genomes`**：`CandidateSelection`から`GenomeBatch`を取り出します。

Stageからグラフを作る独自Strategyでは、`StructuredGraph`を`decompose_structured_graph()`へ渡すと、Stageのflow接続がCompilerの検証対象に加わります。

`stage_component(stage)`はStageの実行を`StateView`上のトランザクションへ適合させます。
トランザクションプロキシ経由の状態書き込みは`StatePatch`になり、ServicesまたはContext機能として公開された可変オブジェクトはStageが所有します。
グラフネイティブなコンポーネントでは、永続的な可変状態を宣言済みの`StateKey`とパッチで管理します。

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.Stage
   saealib.Pipeline
   saealib.Repeat
   saealib.Loop
   saealib.Branch
   saealib.pipeline.ComponentRef
```
