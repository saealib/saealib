---
primary_layer: layer3
page_type: concept
---

# AcquisitionFunction

`SurrogatePredictStage`は[Surrogate](surrogate.md)の予測を`AcquisitionStage`へ渡し、スカラー化を`AcquisitionFunction`へ委譲します。
`AcquisitionFunction`は候補、予測、アーカイブ、コンテキスト、準備済みデータを受け取りますが、Surrogateの内部実装や予測を生成したアルゴリズムは知りません。

## AcquisitionFunctionの役割

`AcquisitionFunction`が実装するメソッドは`evaluate(candidates_x, prediction, archive, *, prepared=..., ctx) -> AcquisitionResult`です。
点ごとの基準には、便利な基底クラス`PointwiseAcquisition`がこの処理を2つのメソッドに分けます。

**`compute_reference(archive, rng=None) -> Any`**：`PointwiseAcquisition`で、スコアリングに使う参照値（現在の最良値など）をアーカイブから計算します。
参照値を使わない獲得関数は`None`を返してよいです。

**`score(prediction, reference, rng=None) -> np.ndarray`**：`PointwiseAcquisition`で、`SurrogatePrediction`と参照値からスコアを計算します。
saealib全体の規約どおり、スコアは高いほど良いです。

`compute_reference()`は`ctx`を受け取らないため、実行時状態が必要なサブクラス（`ctx.decision_count`を読む`LowerConfidenceBound`の `beta_schedule`など）は代わりに`prepare(archive, *, ctx)`を直接オーバーライドします。その戻り値は`reference`として`score()`にそのまま渡されます。

クラス属性`requires_uncertainty: bool`は、この獲得関数が`SurrogatePrediction.std`（不確実性）を必要とするかを示します。
`direction_sensitive: bool`（既定`True`）は、`Optimizer`が実行開始時に`problem.direction`をこの獲得関数へ自動注入する対象かどうかを示します。
実行可能性確率のように目的の方向という概念を持たない獲得関数は、`direction_sensitive = False`にしてこの自動注入を無効化します。

## 組み込みAcquisitionFunction

| クラス | 特徴 | `requires_uncertainty` |
|---|---|---|
| `MeanPrediction` | 予測平均のみを使う、最も単純な獲得関数 | `False` |
| `ExpectedImprovement` | 期待改善量（EI）{cite}`jones1998ego` | `True` |
| `BatchExpectedImprovement` | バッチの多様性を考慮したqEI風選択用の同時EI | `True` |
| `LowerConfidenceBound` | 信頼下限（LCB）{cite}`srinivas2012gpucb` | `True` |
| `MaxUncertainty` | 予測の不確実性が大きいほど良い（探索寄り） | `True` |
| `EHVIAcquisition` | 期待ハイパーボリューム改善量{cite}`emmerich2006ehvi,hupkens2015ehvi,daulton2020ehvi`。多目的向け | `True` |
| `SMSEGOAcquisition` | SMS-EMOA風のハイパーボリューム指標（Ponweiser et al., 2008が提案するSMS-EGO）。多目的向け | `True` |
| `ParEGOAcquisition` | ランダムスカラー化によるParEGO{cite}`knowles2006parego,chugh2020scalarizing`。多目的向け | `True` |
| `ProbabilityOfFeasibility` | 単一制約の実行可能性確率{cite}`schonlau1997pof,gelbart2014pof` | `True` |
| `ProductOfFeasibility` | 複数制約の実行可能性確率の積{cite}`gelbart2014pof` | `True` |
| `NoveltyAcquisition` | アーカイブの最近傍点までの平均距離 | `False` |
| `InverseDensityAcquisition` | アーカイブ近傍密度の逆数 | `False` |
| `MaximinDistanceAcquisition` | 候補とアーカイブの多様性基準 | `False` |

表の不確実性ベースの獲得関数は`requires_uncertainty=True`です。
アーカイブベースの獲得関数（`NoveltyAcquisition`、`InverseDensityAcquisition`、`MaximinDistanceAcquisition`）は予測の不確実性を必要としません。
不確実性ベースの獲得関数を使うには、組み合わせる`Surrogate`が`std`を返す（`provides_uncertainty=True`）必要があります。
バッチ獲得関数には、候補の共分散など明示的な同時事後分布の表現も必要です。
この契約の決定論的な実装例はサンプルを参照してください。組み込みSurrogateでは`SklearnGPRSurrogate`だけがこれを満たします。詳細は[Surrogate](surrogate.md)の不確実性対応表を参照してください。

`MaxUncertainty`/`ProbabilityOfFeasibility`/`ProductOfFeasibility`は`direction_sensitive = False`です。
不確実性の大きさや実行可能性確率は、目的の最大化と最小化という方向の概念を持たないためです。

`ProbabilityOfFeasibility`/`ProductOfFeasibility`は、制約値`g`を予測する分類サロゲートや回帰サロゲートと組み合わせて使います。
[TrainingSet](training_set.md)の`ConstraintObjectiveSet`で学習データを抽出し、目的側の獲得関数（EIなど）と`CompositeAcquisition`の`product_combine`で組み合わせるのが典型的な使い方です。

```python
ei_manager = GlobalSurrogateManager(gp_surrogate, ArchiveObjectiveSet())
pof_manager = GlobalSurrogateManager(
    PerObjectiveSurrogate([gp_g1, gp_g2]),
    ConstraintObjectiveSet(),
)
optimizer.set_surrogate_manager(
    CompositeSurrogateManager({"objective": ei_manager, "feasibility": pof_manager})
)
optimizer.set_acquisition(
    CompositeAcquisition(
        {"objective": ExpectedImprovement(), "feasibility": ProductOfFeasibility()},
        product_combine,
    )
)
```

## weights/direction引数の意味

`MeanPrediction`など一部の獲得関数は、`weights`引数で多目的予測をスカラー化できます。
`weights=np.array([-1.0])`のように、最小化したい目的には負の重みを使います。

`direction`引数を指定すると、大きさを持たない符号だけのスカラー化になり、`weights`よりも優先されます。
`direction`を明示しない場合、実行開始時に`problem.direction`が自動的に注入されます（`direction_sensitive`が`True`の獲得関数のみ）。

## 独自AcquisitionFunctionの実装方法

独自の点ごとのスコアリング方式が必要な場合は、`PointwiseAcquisition`を継承して`compute_reference()`/`score()`を実装します。
次の例は、予測平均が閾値を下回る候補ほど高いスコアを与える単純な獲得関数です。

```python
from saealib import PointwiseAcquisition


class ThresholdAcquisition(PointwiseAcquisition):
    """Gives a higher score to candidates whose predictive mean falls further below a threshold (assumes minimization)."""

    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold

    def compute_reference(self, archive, rng=None):
        return None

    def score(self, prediction, reference=None, rng=None):
        m = prediction.value[:, 0]
        return self.threshold - m
```

`Optimizer.set_acquisition()`へ渡して組み込みます。

```{note}
`Optimizer.validate()`は予測チャネルごとに不確実性の対応を確認します。獲得関数が不確実性を必要とし、対応するサロゲートが不確実性を提供しない場合、エラー診断を報告します。
```

## 関連コンポーネント

- [Surrogate](surrogate.md)：`SurrogatePrediction`の提供元。不確実性対応表もこちらを参照する
- [SurrogateManager](surrogate_manager.md)：`AcquisitionFunction`へ予測を提供する
- [TrainingSet](training_set.md)：`ProbabilityOfFeasibility`/`ProductOfFeasibility`と組み合わせる制約用データ抽出

## 参照

- {py:class}`saealib.AcquisitionFunction`
- {py:class}`saealib.MeanPrediction`
- {py:class}`saealib.ExpectedImprovement`
- {py:class}`saealib.BatchExpectedImprovement`
- {py:class}`saealib.LowerConfidenceBound`
- {py:class}`saealib.MaxUncertainty`
- {py:class}`saealib.EHVIAcquisition`
- {py:class}`saealib.SMSEGOAcquisition`
- {py:class}`saealib.ParEGOAcquisition`
- {py:class}`saealib.ProbabilityOfFeasibility`
- {py:class}`saealib.ProductOfFeasibility`
- {py:func}`saealib.gp_ucb_beta_schedule`
