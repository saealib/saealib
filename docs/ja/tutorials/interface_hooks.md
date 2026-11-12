---
primary_layer: cross
related_layers: []
page_type: guide
---

# Hook、Stage、Callbackを選ぶ

前提は、既存の実行経路に観測や後処理を追加したいことです。
:::{admonition} このページでできるようになること
:class: tip

このページを終えると、データを変更するHook、実行単位を置き換えるStage、観測するCallbackを区別できます。
:::

構成要素の索引は [最適化の構成要素](../concepts/index.md) で、独自の契約が必要なら [独自Component](custom_components.md) に進みます。

## 境界を比較する

| 手段 | 目的 | 実行時点 | 状態変更の可否 | 適した用途 |
|---|---|---|---|---|
| `with_post` | 既存Operatorの出力を後処理する | CrossoverまたはMutationの後 | 戻り値で候補を変更できる | 修復、丸め、範囲内への変換 |
| `with_post_fit` | Surrogateのfit後に処理する | `fit()` の後 | Hookの処理でモデルや外部状態を更新できる | fit記録、モデル後処理 |
| `Stage` | 互換実行単位を実装または置換する | Pipeline内のStage位置 | `OptimizationState` を返して状態を更新できる | 世代処理の追加、Stage差替え |
| `CallbackManager` | イベントを観測する | 登録したEventの発火時 | `event.ctx` は読み取り用で、Pipeline入力の置換には使わない | ログ、履歴、条件判断 |

## Hookでデータを変更する

Operatorの候補配列を実際に変えるなら、`CallbackManager` ではなく `with_post()` を使います。 `with_post()` は元のインスタンスを変更せず、Hookを追加したコピーを返します。

```python
import numpy as np
from saealib.operators import MutationUniform


def snap_to_grid(offspring, parents, rng, ctx):
    return np.round(offspring * 10.0) / 10.0


mutation = MutationUniform(0.3).with_post(snap_to_grid)
```

Surrogateのfit後処理は、`fn(train_x, train_y, ctx) -> None` を `with_post_fit()` に渡します。 このHookを使う具体的な契約は [Surrogateの概念](../concepts/surrogate_modeling/surrogate.md) を参照してください。

コンポーネント側の文脈依存フック（`ParentSelection.select`、`SurvivorSelection.select`、`EvaluationPlanner.plan`、`FeedbackBuilder.build`、`TrainingSet.build`、`AcquisitionFunction.prepare`/`evaluate`、`SurrogateManager.fit`/`predict`）は、末尾のキーワード専用・必須引数 `ctx`（型は `OptimizationState | None`）という共通の引数規則に従います。 規則の詳細と独自実装での宣言方法は [拡張ガイドライン](../concepts/extension_guidelines.md) を参照してください。

## Stageで実行単位を置き換える

`Stage` は `execute(state) -> state` の互換性用境界です。 状態を更新するときは `state.replace()` で新しいStateを返します。 構造化FrameworkのComponentやCompilerを直接実装する境界ではないため、それが必要なら [フレームワーク拡張](../framework/extensions.md) を参照します。

```python
from saealib import Stage


class LogGenerationStage(Stage):
    name = "log_generation"

    def execute(self, state):
        print(state.gen)
        return state
```

Pipelineの構成とStageの契約は [Stageの概念](../concepts/observation_and_state/stage.md) に委ねます。

## Callbackで観測する

`CallbackManager.register()` は、Event型と `event` を受け取る関数を登録します。 ハンドラから読む `event.ctx` は、世代や評価回数などの公開された読み取り境界です。

```python
import numpy as np
from saealib import GenerationEndEvent
from saealib import Optimizer, Problem


def objective(x):
    return np.sum(x**2)


problem = Problem(
    objective,
    dim=3,
    n_obj=1,
    direction=np.array([-1.0]),
    lb=[-5.0] * 3,
    ub=[5.0] * 3,
)
optimizer = Optimizer(problem, seed=0)

history = []


def record(event):
    history.append((event.ctx.gen, event.ctx.fe))


optimizer.cbmanager.register(GenerationEndEvent, record)
```

候補配列を差し替える、または状態を書き換える処理はCallbackの責務ではありません。 Callbackはイベントの観測、ログ、履歴の記録に使います。

独自Componentの契約を実装する必要があるときは [独自Component](custom_components.md) を使い、契約やRuntimeそのものを変更するときは [フレームワーク拡張](../framework/extensions.md) を参照してください。

## 関連するConceptとReference

- [Callbackの概念](../concepts/observation_and_state/callbacks.md)
- [拡張方針](../concepts/extension_guidelines.md)
- [Stageリファレンス](../api/stages.md)
- [Callbackリファレンス](../api/callbacks.md)

