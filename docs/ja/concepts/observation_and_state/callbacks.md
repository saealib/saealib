---
primary_layer: layer2
related_layers: [layer3]
page_type: concept
---

# CallbackManager

`Optimizer`は、実行経過を外部へ通知する処理を`CallbackManager`に委譲します。
CallbackManagerは、ログ記録、収束履歴の収集、条件判断などの観測をサポートします。
Eventは実行境界を観測するものであり、Eventのフィールドを変更してもPipelineの入力は差し替わりません。

## CallbackManagerの役割

`CallbackManager`は、イベント型ごとにハンドラのリストを保持します。

| Method | Description |
|---|---|
| `register(event_type, func)` | イベント型`event_type`のイベントが発生するたびに呼び出す`func(event)`を登録します |
| `dispatch(event)` | 登録順に、登録されたすべてのハンドラを呼び出します |
| `unregister(event_type, func)` | 登録されたハンドラを削除します |
| `replace(event_type, old, new)` | 登録されたハンドラを別のハンドラに置き換えます |

Eventの`ctx`は、`archive`、`comparator`、`n_obj`、`gen`、`fe`などの値を読み取るための`_EventContext` Protocolです。
通常のsequential実行では実体が`OptimizationState`ですが、Eventの公開契約はそれらを読み取る能力に限定されます。
`ctx`は読み取り専用として扱い、値を変更するときは`with_post`、Component、Stageなどの正式な境界を使います。

## 利用可能なイベント一覧

| Event | Fired when | Main fields |
|---|---|---|
| `RunStartEvent` | 実行開始時に1回 | — |
| `RunEndEvent` | 実行終了時に1回 | — |
| `GenerationStartEvent` | 各世代の開始時 | — |
| `GenerationEndEvent` | 各世代の終了時（stateがyieldされる前） | — |
| `SurrogateStartEvent` / `SurrogateEndEvent` | Surrogateベースのスコアリングの前後 | `offspring` |
| `AcquisitionStartEvent` / `AcquisitionEndEvent` | acquisitionスコアリングの前後 | 開始時は`offspring`、終了時は`offspring`と`result` |
| `PostCrossoverEvent` | crossoverとrepairの後 | `candidates` |
| `PostMutationEvent` | mutationとrepairの後 | `candidates` |
| `PostAskEvent` | `ask()`（crossoverとmutation）のすべての処理の後 | `candidates` |
| `PostSurrogateFitEvent` | surrogateのfit後 | `surrogate`、`train_x`と`train_f`はoptional |
| `PostEvaluationEvent` | 評価計画ごとに1回、選択された候補のtrue evaluation後。コミット点でtransitionが要求した評価では発火しない（`PostEvaluationRequestEvent`を使う） | `offspring` |
| `InitialEvaluationStartEvent` | 初期サンプリング後、初期評価前 | `candidates_x` |
| `InitialEvaluationEndEvent` | 初期評価後、archiveのソート前 | `archive` |

`PostSurrogateFitEvent`は、Surrogateのfit後に組み込みStageから発火します。
組み込みStageが発火するイベントでは、`train_x`と`train_f`が設定されない場合があります。
`PostEvaluationEvent`には、評価済みの`offspring`に加えて`request_id`、`candidate_ids`、`status`が含まれる場合があります。
これは計画単位のイベントなので、複数の評価要求を持つ計画では1回だけ発火し、部品の差し替えのコミット点でtransitionが要求した評価では代わりに`PostEvaluationRequestEvent`が要求ごとに1回発火します。

## 既定のログ出力

`Optimizer`は構築時に`logging_generation`を`GenerationStartEvent`へ自動登録します。
標準ライブラリの`logging`モジュールを設定すると、世代ごとの進捗（評価回数、最良目的値、または多目的の場合は第一フロントのサイズと範囲）がログに出力されます。

## カスタムハンドラの登録と収束履歴の記録

`cbmanager.register(EventType, handler)`で任意のハンドラを登録します。
収束履歴を記録するには、クロージャで蓄積するリストを保持するハンドラを登録します。

```python
import numpy as np
from saealib import GenerationEndEvent, Optimizer, Problem, Termination, max_fe

problem = Problem(
    lambda x: np.array([x[0], 1.0 - x[0]]),
    dim=2,
    n_obj=2,
    direction=np.array([-1.0, -1.0]),
    lb=[0.0, 0.0],
    ub=[1.0, 1.0],
)
optimizer = Optimizer(problem, seed=0).set_termination(Termination(max_fe(20)))

history = []


def record_best(event):
    f = event.ctx.archive.get_array("f")
    history.append(float(f[:, 0].min()))


optimizer.cbmanager.register(GenerationEndEvent, record_best)
ctx = optimizer.run()
print(history)
```

## ハイパーボリュームの追跡

`logging_generation_hv(reference_point)`は、指定したreference pointに対する第一フロントのhypervolumeを世代ごとにログへ出力するハンドラを返します。

```python
import numpy as np
from saealib import (
    GenerationStartEvent,
    Optimizer,
    Problem,
    Termination,
    logging_generation_hv,
    max_fe,
)

problem = Problem(
    lambda x: np.array([x[0], 1.0 - x[0]]),
    dim=1,
    n_obj=2,
    direction=np.array([-1.0, -1.0]),
    lb=[0.0],
    ub=[1.0],
)
optimizer = Optimizer(problem, seed=0).set_termination(Termination(max_fe(20)))

optimizer.cbmanager.register(
    GenerationStartEvent,
    logging_generation_hv(reference_point=np.array([1.1, 1.1])),
)
```

## 既定ハンドラの差し替え

自動登録された`logging_generation`は`unregister(event_type, func)`で削除でき、`replace(event_type, old, new)`で別のハンドラに置き換えられます。

```python
import numpy as np
from saealib import (
    GenerationStartEvent,
    Optimizer,
    Problem,
    Termination,
    logging_generation,
    max_fe,
)

problem = Problem(
    lambda x: np.array([x[0], 1.0 - x[0]]),
    dim=1,
    n_obj=2,
    direction=np.array([-1.0, -1.0]),
    lb=[0.0],
    ub=[1.0],
)
optimizer = Optimizer(problem, seed=0).set_termination(Termination(max_fe(20)))

optimizer.cbmanager.unregister(GenerationStartEvent, logging_generation)
```

## candidatesフィールドは観測専用

`PostCrossoverEvent`/`PostMutationEvent`/`PostAskEvent`の`candidates`フィールドは観測用です。ハンドラ内で再代入しても（`event.candidates = new_array`）、Pipelineの出力には影響しません。
Event発火後も`GA`は自身のローカルな配列参照を使い続けるため、インプレースの変更（`event.candidates[:] = ...`）は`GA`に反映されます。
一方、`PSO`は`Population.extend()`が候補をコピーし終えた後にこのイベントを発火するため、インプレースの変更でも遅すぎてまったく影響しません。

候補配列そのものを差し替えたい場合は、`CallbackManager`ではなく[Crossover](../search_algorithms/crossover.md)/[Mutation](../search_algorithms/mutation.md)の`with_post(fn)`を使ってください。
`CallbackManager`は設計上、観測（ログ、記録、条件付きの分岐判断）のための仕組みであり、Pipelineデータを書き換える手段ではありません。

## 実行時の構成切り替え

`Event`は観測用であり、Callbackクロージャから`Optimizer`の内部を直接変更するのは標準的な手順ではありません。
データフローを変更するには`with_post()`または`with_post_fit()`を使い、実行構成は`iterate()`または`run()`のステップ境界で切り替えます。
ステップ境界で`optimizer.set_*()`またはコンポーネント属性が変更されると、実行環境が変更を検出してプランを再コンパイルし、次の世代から適用します。
この手順はStage互換パスとgraph-nativeパスの両方で機能します。
Componentが要求する再コンパイル経路については、[OptimizationStrategy](../execution_and_evaluation/strategies.md)の「実行時の差し替えの動作」を参照してください。
Callbackハンドラは`RuntimeCommand`を返せません。
Callbackクロージャで`optimizer.set_*()`を呼び出した変更は上述の経路で適用されますが、構成変更は`iterate()`側の手順として文書化されています。
コールバック内でセッターを呼ぶ場合は、`Optimizer`をキャプチャしたクロージャをハンドラとして登録します。

## CallbackManagerとiterate()の使い分け

| Aspect | CallbackManager | `iterate()` |
|---|---|---|
| 呼び出しの粒度 | 特定のイベントが発生したとき | 世代ごと（ホスト側の`for`ループ） |
| 主な用途 | ログ、観測、条件付きの副作用 | ループ構造そのものへの介入（例：surrogateの精度に基づくコンポーネントの切り替え） |
| `run()`との関係 | `run()`と`iterate()`のどちらでも動作 | `run()`の代わりに使用 |

[Surrogate accuracy evaluation and dynamic switching](../surrogate_modeling/surrogate_switching.md)のswitcherクラスは、`iterate()`ループ内で使うことを想定しています。

## チェックポイントコールバック

`CheckpointCallback`はCallback実装の例です。
npz形式では`OptimizationState.save()`を使い、pickle形式では`Optimizer.save_pickle()`を使います。
形式によって保存対象の値と再開条件が異なるため、詳細は[Checkpointing](../../tutorials/checkpoint.md)を参照してください。
このCallbackをカスタムCallback登録の参考にしてください。

## 関連コンポーネント

- [Extension guidelines](../extension_guidelines.md): `with_post`形式のフックを使う場面
- [Crossover](../search_algorithms/crossover.md) / [Mutation](../search_algorithms/mutation.md): 候補配列を差し替える実際の手段
- [strategies](../execution_and_evaluation/strategies.md): 実行時のコンポーネント差し替えが有効になる時点
- [Surrogate accuracy evaluation and dynamic switching](../surrogate_modeling/surrogate_switching.md): `iterate()`ループ内での動的な切り替え
- [Checkpointing](../../tutorials/checkpoint.md): `CheckpointCallback`の使い方

## 参照

- {py:class}`saealib.CallbackManager`
- {py:class}`saealib.RunStartEvent`
- {py:class}`saealib.RunEndEvent`
- {py:class}`saealib.GenerationStartEvent`
- {py:class}`saealib.GenerationEndEvent`
- {py:class}`saealib.AcquisitionStartEvent`
- {py:class}`saealib.AcquisitionEndEvent`
- {py:class}`saealib.PostSurrogateFitEvent`
- {py:class}`saealib.PostEvaluationEvent`
- {py:class}`saealib.InitialEvaluationStartEvent`
- {py:class}`saealib.InitialEvaluationEndEvent`
- {py:func}`saealib.logging_generation`
- {py:func}`saealib.logging_generation_hv`
- {py:class}`saealib.CheckpointCallback`
