---
primary_layer: layer3
page_type: concept
---

# AccuracyBasedSurrogateSwitcher

サロゲートの精度に応じて`SurrogateManager`や`OptimizationStrategy`を動的に切り替える仕組みは、3つの独立した関心事の組み合わせでできています。

- **精度指標**（`SurrogateAccuracyMetric`）：1回のfit/predictペアからスカラー値を計算する
- **精度評価方法**（`AccuracyEvaluator`）：精度指標を使って、サロゲートの汎化性能をどう測るか（交差検証、held-outなど）を決める
- **切り替え判断**（`AccuracyBasedSurrogateSwitcher`）：評価結果を受けて、次に使うコンポーネントやパラメータを決める

精度評価方法は`GlobalSurrogateManager`などの`accuracy_evaluator`引数として注入され、その結果は`surrogate_manager.last_accuracy`プロパティを経由して切り替え判断へ渡されます。 この一連の仕組みは`iterate()`ループでの利用を前提としています。 高レベルAPI（`minimize`/`maximize`）は実行中のコンポーネント差し替え手段を提供しないため、これらのSwitcherは使えません。

## SurrogateSwitcherの役割

この仕組みは、精度指標、精度評価、切り替え判断を分離して組み合わせ、次に使う`SurrogateManager`または`OptimizationStrategy`の構成を決めます。

## 精度指標：SurrogateAccuracyMetric

`SurrogateAccuracyMetric`が実装を要求するのは`name`（プロパティ）と`compute(y_true, y_pred) -> float`の2つです。

| クラス | 範囲 | 意味 |
|---|---|---|
| `SpearmanCorrelation` | `[-1, 1]`、高いほど良い | 目的ごとのSpearman順位相関の平均。EAではランクの保存性が重要という観点{cite}`yu2019spearman`に基づく |
| `RMSE` | `[0, ∞)`、低いほど良い | 二乗平均平方根誤差 |
| `R2Score` | `(-∞, 1]`、高いほど良い | 決定係数 |

`AccuracyEvaluator`のコンストラクタで`metrics`を省略すると、この3指標全てが既定として使われます。

## 精度評価方法：AccuracyEvaluator

`AccuracyEvaluator`が実装を要求するのは`evaluate(surrogate, train_x, train_y) -> SurrogateAccuracy`の1つだけです。

| クラス | パラメータ | 評価方法 |
|---|---|---|
| `KFoldAccuracyEvaluator` | `metrics=None, n_splits=5` | サロゲートをfoldごとに複製して再学習するk分割交差検証 |
| `LOOAccuracyEvaluator` | `metrics=None` | `n_splits=n_samples`の`KFoldAccuracyEvaluator`と等価（1件抜き交差検証） |
| `HeldOutAccuracyEvaluator` | `held_x, held_y, metrics=None` | 既にfit済みのサロゲートを再学習せず、指定したheld-outデータで評価する |

`HeldOutAccuracyEvaluator`は、直近の真の評価点との比較用途{cite}`hanawa2025switching`で使います。

`SurrogateAccuracy`（`metrics: dict[str, float]`, `n_samples: int`）は、評価結果を保持する単純なコンテナで、`get(name, default=nan)`で指標名から値を取り出します。

## 切り替え判断：AccuracyBasedSurrogateSwitcher

`AccuracyBasedSurrogateSwitcher`が実装する必要があるメソッドは`switch(accuracy: SurrogateAccuracy | None) -> T`の1つだけです。精度結果から次のコンポーネントまたはパラメータを選びます。その選択をRuntimeの計画へ適用する操作は別です。

次の例では、`iterate()`と`optimizer.set_*()`で構成を変更します。実行環境が変更を検出して計画を再コンパイルし、次の世代から変更を適用します。この手順は、反復中に構成を変更する他のケースにも使えます。

```python
import numpy as np
from saealib import Optimizer, Problem, Termination, max_fe
from saealib.surrogate import (
    ArchiveObjectiveSet,
    GlobalSurrogateManager,
    KFoldAccuracyEvaluator,
    ManagerSwitcher,
    SklearnGPRSurrogate,
)

problem = Problem(
    lambda x: np.sum(x**2),
    dim=2,
    n_obj=1,
    direction=np.array([-1.0]),
    lb=[-5.0, -5.0],
    ub=[5.0, 5.0],
)
evaluator = KFoldAccuracyEvaluator(n_splits=3)
primary = GlobalSurrogateManager(
    SklearnGPRSurrogate(), ArchiveObjectiveSet(), accuracy_evaluator=evaluator
)
fallback = GlobalSurrogateManager(SklearnGPRSurrogate(), ArchiveObjectiveSet())
switcher = ManagerSwitcher(primary, fallback)
optimizer = (
    Optimizer(problem, seed=0)
    .set_surrogate_manager(primary)
    .set_termination(Termination(max_fe(30)))
)

for ctx in optimizer.iterate():
    optimizer.set_surrogate_manager(
        switcher.switch(optimizer.surrogate_manager.last_accuracy)
    )
```

| クラス | パラメータ | 切り替える対象 |
|---|---|---|
| `ManagerSwitcher` | `primary, fallback, *, metric="spearman", threshold=0.5` | `SurrogateManager` |
| `StrategySwitcher` | `primary, fallback, *, metric="spearman", threshold=0.56, mid=None, mid_threshold=None` | `OptimizationStrategy` |
| `GenCtrlSwitcher` | `*, gm_max=5, gm_min=0, update_rate=0.5, metric="spearman", initial_error=0.5` | `gen_ctrl`（整数） |

`ManagerSwitcher`は、指定した指標が閾値以上なら`primary`、そうでなければ`fallback`を返します。`StrategySwitcher`もこの二分岐で動作し、`mid`と`mid_threshold`を両方指定すると、`threshold`未満では`fallback`、`threshold`以上`mid_threshold`未満では`mid`、それ以上では`primary`を返します。
三分岐を使う場合は`mid`と`mid_threshold`を両方指定し、片方だけ指定すると`ValidationError`になります。
`StrategySwitcher`の既定閾値`0.56`は、{cite}`hanawa2025switching`によるPS/IB-GB切り替えの設定値に基づきます。

`GenCtrlSwitcher`は、二値切り替えではなく`gen_ctrl`を指数平滑化で連続的に調整します{cite}`repicky2017genctrl`。 公開属性`smoothed_error`に平滑化済みの誤差推定値を保持する状態を持つため、1回の`run`につき1つのインスタンスを使います。 `GenCtrlSwitcher.switch()`が返す`int`は、そのまま[GenerationBasedStrategy](../execution_and_evaluation/strategies.md)の`gen_ctrl`引数に渡す想定です。

コンポーネント側から再コンパイルを要求する経路については、[OptimizationStrategy](../execution_and_evaluation/strategies.md)の「ランタイム差し替えの動作」を参照してください。

## 独自Switcherの実装方法

独自の切り替えロジックが必要な場合は、`AccuracyBasedSurrogateSwitcher`を継承して`switch()`だけを実装すればよいです。

```python
from saealib import AccuracyBasedSurrogateSwitcher


class ThresholdIntSwitcher(AccuracyBasedSurrogateSwitcher):
    """A custom integer-parameter switch: increase n_neighbors if the Spearman correlation is at or above 0.7."""

    def switch(self, accuracy):
        if accuracy is None:
            return 20
        return 50 if accuracy.get("spearman") >= 0.7 else 20
```

## 関連コンポーネント

- [SurrogateManager](surrogate_manager.md)：`accuracy_evaluator`引数と`last_accuracy`プロパティ
- [strategies](../execution_and_evaluation/strategies.md)：`StrategySwitcher`/`GenCtrlSwitcher`が切り替える対象

## 参照

- {py:class}`saealib.AccuracyBasedSurrogateSwitcher`
- {py:class}`saealib.ManagerSwitcher`
- {py:class}`saealib.StrategySwitcher`
- {py:class}`saealib.GenCtrlSwitcher`
- {py:class}`saealib.SurrogateAccuracyMetric`
- {py:class}`saealib.SpearmanCorrelation`
- {py:class}`saealib.RMSE`
- {py:class}`saealib.R2Score`
- {py:class}`saealib.AccuracyEvaluator`
- {py:class}`saealib.KFoldAccuracyEvaluator`
- {py:class}`saealib.LOOAccuracyEvaluator`
- {py:class}`saealib.HeldOutAccuracyEvaluator`
- {py:class}`saealib.SurrogateAccuracy`
