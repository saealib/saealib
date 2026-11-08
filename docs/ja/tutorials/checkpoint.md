---
primary_layer: layer2
page_type: guide
---

# 再現性とチェックポイント

長時間実行する最適化を再現可能にし、途中から再開できるようにします。

チェックポイント機能は低レベルの `Optimizer` APIで使います。 `Optimizer(problem)`を構築し、`set_*()`を連鎖させてコンポーネントを設定し、`run()`で実行します。 保存した状態からは `run_from()`で再開できます。

:::{admonition} このページでできるようになること
:class: tip

このページを終えると、乱数シードで再現性を確保し、チェックポイントを保存・読み込みして最適化を再開できます。
:::

高レベルAPIでの一回実行だけで足りる場合は [高レベルAPI](highlevel_api.md) を使い、実行状態を細かく構成する場合は [低レベルAPI](lowlevel_api.md) を参照してください。 このページでは、低レベルAPIで再現可能な実行を保存し、途中から再開する手順を扱います。

## 乱数シードで実行を再現する

同じ `seed` を `Optimizer(problem, seed=...)` に渡すと、乱数を使うすべての処理が同じ順序で初期化され、同一の結果が得られます。

```python
import numpy as np
from saealib import (
    Problem,
    Optimizer,
    GA,
    CrossoverBLXAlpha,
    MutationUniform,
    SequentialSelection,
    TruncationSelection,
    RBFSurrogate,
    GaussianKernel,
    LocalSurrogateManager,
    MeanPrediction,
    IndividualBasedStrategy,
    LHSInitializer,
    Termination,
    max_fe,
)


def expensive_func(x):
    return np.sum(x**2)


DIM = 10
SEED = 0

problem = Problem(
    func=expensive_func,
    dim=DIM,
    n_obj=1,
    direction=np.array([-1.0]),
    lb=[-5.0] * DIM,
    ub=[5.0] * DIM,
)


def build_optimizer(max_fe_value):
    return (
        Optimizer(problem, seed=SEED)
        .set_initializer(
            LHSInitializer(n_init_archive=5 * DIM, n_init_population=4 * DIM, seed=SEED)
        )
        .set_algorithm(
            GA(
                crossover=CrossoverBLXAlpha(0.7, 0.4),
                mutation=MutationUniform(0.3),
                parent_selection=SequentialSelection(),
                survivor_selection=TruncationSelection(),
            )
        )
        .set_surrogate_manager(
            LocalSurrogateManager(RBFSurrogate(kernel=GaussianKernel()))
        )
        .set_acquisition(MeanPrediction())
        .set_strategy(IndividualBasedStrategy(evaluation_ratio=0.1))
        .set_termination(Termination(max_fe(max_fe_value)))
    )


ctx1 = build_optimizer(300).run()
ctx2 = build_optimizer(300).run()

print(np.allclose(ctx1.archive.get_array("f"), ctx2.archive.get_array("f")))  # True
```

以降の節でも、同じコンポーネント構成で `Optimizer` を再構築するために `build_optimizer` を使います。

再現性を保つには、SurrogateManagerとAcquisitionFunctionを同じ構成で再生成します。 `LocalSurrogateManager`の引数にAcquisitionFunctionを渡す構成は現行APIでは使いません。

## チェックポイントを保存して再開する

`run()` が返す `ctx` は、`ctx.save(path)` によって1つのnpzファイルに保存できます。

```python
ctx = build_optimizer(200).run()
ctx.save("checkpoint.npz")
```

保存できるのはstep境界です。`iterate()` が返すstate、`GenerationEndEvent`または`RunEndEvent`のcallback（`CheckpointCallback`を含む）、および`run()`の終了後が対象です。`run()`はregionが開いた状態で終了することがありますが、再開可能なstateなら保存できます。非同期評価では、`GenerationBasedStrategy`のrunが世代の途中で実行中の評価を残したまま終了条件に達することがあります。このとき`run()`が返したstateから再開しても同じ結果にならない場合があります。`GenerationEndEvent`で保存するか、`RunEndEvent` callbackから保存してください。非同期評価を待っている間に、その位置から実行を続けられる停止点も保存できます。step実行中に`Repeat`などのregionが開いている間や、それ以外のcallback内での保存は`CheckpointError`となります。投入待ちの要求がない`RunEndEvent` callbackからは保存できます。投入待ちの要求が残る場合は拒否されるため、`GenerationEndEvent`で保存してください。`run()`が世代境界で終了した場合は、返されたstateを保存できます。非同期の保留要求は、すべての要求を再接続でき、feedback accumulatorが要求と整合する場合に限り保存できます。callbackまたはfeedback更新が開始済みの要求や、`set_*()`または属性代入によるコンポーネント差し替えが保留中の場合も保存できません。pickle保存時にEvaluatorをシリアライズできない場合は、その型名を示して失敗します。portable形式にはEvaluatorを含めません。

保存したチェックポイントは `OptimizationState.load(path, problem)` で読み込めます。これを `Optimizer.run_from(ctx)` に渡すと、中断した箇所から再開します。

```python
from saealib import OptimizationState

loaded_ctx = OptimizationState.load("checkpoint.npz", problem)

resumed_ctx = build_optimizer(600).run_from(loaded_ctx)
print(resumed_ctx.fe)  # includes the evaluations from before saving
print(resumed_ctx.data["resumed"])  # True
```

`ctx.data["resumed"]` は、`run_from()` で再開したコンテキストだけで `True` になるフラグです。

`RunStartEvent` などのコールバックからは、`event.ctx.data["resumed"]` として参照できます。

## チェックポイントを自動保存する

`run()`/`iterate()` に `checkpoint_path` を渡すと、`checkpoint_interval` 世代ごとに自動保存します。

`checkpoint_path` は単一ファイルではなくディレクトリとして扱われ、世代ごとに `checkpoint_{gen:06d}.npz` という名前のスナップショットが作成されます。

```python
ctx = build_optimizer(300).run(checkpoint_path="checkpoints", checkpoint_interval=5)
```

再開するには、ディレクトリ内の最新のスナップショットを読み込みます。

```python
from pathlib import Path

latest = sorted(Path("checkpoints").glob("checkpoint_*.npz"))[-1]
loaded_ctx = OptimizationState.load(latest, problem)
```

実行成功後にスナップショットを残したくない場合は、`checkpoint_delete_on_success=True` を指定します（ディレクトリ自体は残り、その中のファイルだけが削除されます）。

```python
ctx = build_optimizer(300).run(
    checkpoint_path="checkpoints",
    checkpoint_interval=5,
    checkpoint_delete_on_success=True,
)
```

## pickle形式で保存する

npz形式では `ctx` だけを保存しますが、pickle形式では学習済みサロゲートのパラメータを含む `Optimizer` 全体を保存できます。

```python
optimizer = build_optimizer(200)
ctx = optimizer.run()
optimizer.save_pickle(ctx, "checkpoint.pkl")

loaded_optimizer, loaded_ctx = Optimizer.load_pickle("checkpoint.pkl")
```

実行時にPythonまたはライブラリのバージョンについての `UserWarning` が表示されることがあります。

`Termination` で使うlambdaなど、標準の `pickle` でシリアライズできないオブジェクトを含む `Optimizer` はpickle形式で保存できません。

## CheckpointCallbackを直接使う

`run()` の `checkpoint_path` 引数は、内部で `CheckpointCallback` を登録するだけです。

同じ動作を明示的に接続するには、`cbmanager` に `CheckpointCallback` を登録します。

```python
from saealib import CheckpointCallback

optimizer = build_optimizer(300)
callback = CheckpointCallback("checkpoints", interval=5, optimizer=optimizer)
callback.register(optimizer.cbmanager)

ctx = optimizer.run()
```

`format="pickle"` または `format="both"` を指定する場合は、`optimizer` 引数が必要です。

## 関連するConceptとReference

- {py:class}`saealib.Optimizer`
- {py:class}`saealib.CheckpointCallback`
- {py:func}`saealib.minimize`
