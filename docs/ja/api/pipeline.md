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

`stage_component(stage)`は移行用アダプターです。
トランザクションプロキシ経由の状態書き込みは`StatePatch`値になりますが、アダプターはServicesまたはContext機能として公開された可変オブジェクトの所有権を保持します。
新しいグラフネイティブコンポーネントでは、永続的な可変状態を宣言済みの`StateKey`値とパッチの背後に保持します。

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
