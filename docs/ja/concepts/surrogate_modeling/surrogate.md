---
primary_layer: layer3
page_type: concept
---

# Surrogate

`saealib`は、目的関数を近似する予測モデルの責務を`Surrogate`という差し替え可能なコンポーネントに限定しています。
`Surrogate`はfit/predictだけを知り、予測値をどうスコアに変換するか（[AcquisitionFunction](acquisition_functions.md)）や、学習データをどこから集めるか（[TrainingSet](training_set.md)、[SurrogateManager](surrogate_manager.md)経由）は一切知りません。

## Surrogateの役割

`Surrogate`が実装を要求するメソッドは2つあります。

**`fit(train_x, train_y) -> None`**：形状`(n_samples, n_features)`の入力と、形状`(n_samples, n_obj)`（単目的では`(n_samples,)`）の出力でモデルを学習します。

**`predict(test_x) -> SurrogatePrediction`**：形状`(n_samples, n_features)`の入力に対する予測を返します。

クラス属性`provides_uncertainty: bool = False`は、予測が不確実性（標準偏差）を伴うかどうかを示します。
既定は`False`で、ガウス過程の実装だけが`True`にオーバーライドします。

`Surrogate`と`predict()`の間には2つのマーカー基底クラスがあります。

**`RegressionSurrogate`**：`train_y`が実数値の目的関数出力である回帰サロゲート向けのマーカー。

**`ComparisonSurrogate`**：`train_y`が`{0, 1}`の二値比較ラベルである比較サロゲート向け。
`predict_proba(test_x) -> SurrogatePrediction`（値は`[0, 1]`の勝率）が主インターフェースで、`predict()`は既定で`predict_proba()`に委譲します。

## SurrogatePrediction

`predict()`の戻り値は`SurrogatePrediction`という統一データクラスです。

| フィールド | 内容 |
|---|---|
| `value` | 予測値。形状 `(n_samples, n_obj)` |
| `std` | 不確実性（標準偏差）。提供しないサロゲートでは`None` |
| `label` | 分類モデルのみが持つクラスラベル |
| `metadata` | SHAP値など実装固有の付加情報を格納する`dict` |

`value`と`std`は目的関数チャンネル向けの便利なプロパティです。
サロゲートの予測が目的関数値ではない量（新規性スコアなど）を表す場合、この仕組みによってpbestのような値が汚染されるのを防ぎます。
`NoveltyAcquisition`などのアーカイブベースの基準は、候補とアーカイブの設計点を直接受け取ります。

## 組み込みSurrogate

**`RBFSurrogate(kernel, polynomial_degree="auto", solver="lstsq", alpha=1e-8)`**：RBF補間によるサロゲートです{cite}`gutmann2001rbf,regis2005cors,rasmussen2006gpml`（RBF補間自体の起源はHardy, 1971）。
`kernel`は既定値を持たない必須の`RBFKernel`オブジェクトです（組み込みカーネルは以下に示します）。カスタムカーネルは`evaluate(r)`を実装するサブクラスとして作ります。
`predict()`は`std=None`を明示的に返します（RBF補間は不確実性を提供しません）。

| パラメータ | 値 | 役割 |
|---|---|---|
| `polynomial_degree` | `"auto"`（既定） / `None` / `0` / `1` | 補間系を多項式項で拡張します（`kernel.auto_polynomial_degree`から決定 / なし / 定数 / 線形）。`ThinPlateSplineKernel`のような条件付き正定値カーネルには多項式項が必要です。`GaussianKernel`のような狭義正定値カーネルには不要ですが、`"auto"`は定数項に解決されます。目的関数値すべてに同じ定数を足しても予測が変わらないようにするためです。 |
| `solver` | `"lstsq"`（既定） / `"solve"` / `"tikhonov"` | 線形系（多項式項で拡張されている場合を含む）の解法：最小二乗法（ランク落ちした系にも対応） / 直接法 / `alpha`によるリッジ正則化。 |
| `alpha` | `1e-8`（既定） | リッジ正則化の強さ。`solver="tikhonov"`の場合だけ使われます。 |

`kernel`と`polynomial_degree`は設定した値そのものです。`resolved_kernel`と`resolved_polynomial_degree`は、`fit()`が実際に解決して使う値を公開し、設定した`RBFKernel`インスタンス自体は不変に保ちます。
`kernel`/`polynomial_degree`/`solver`/`alpha`を再設定すると、fit済みの状態は無効になり、次の`fit()`まで`predict()`は`RuntimeError`を送出します。

`length_scale`フィールドを持つカーネル（入力データと同じ距離単位）は、`None`のままにすると訓練点間のペアワイズ距離の中央値へ自動的に解決されます。

各カーネルは独立した2つの次数を宣言します。`min_polynomial_degree`はそのカーネルが受け付ける最小の次数で、これを下回る値（`None`を含む）を渡すと`ValidationError`になります。`auto_polynomial_degree`は`polynomial_degree="auto"`の解決先で、最小次数より高いことがあります。

| カーネル | 式 | `min_polynomial_degree` | `"auto"`の解決先 |
|---|---|---|---|
| `GaussianKernel` | $\exp\left(-\dfrac{r^2}{2\,\ell^2}\right)$ | なし | `0`（定数） |
| `ThinPlateSplineKernel` | $r^2 \log r$（`length_scale`は未使用） | `1`（線形） | `1`（線形） |
| `LinearKernel` | $r$（`length_scale`は未使用） | `0`（定数） | `0`（定数） |
| `CubicKernel` | $r^3$（`length_scale`は未使用） | `1`（線形） | `1`（線形） |
| `MultiquadricKernel` | $\sqrt{r^2 + \ell^2}$ | `0`（定数） | `0`（定数） |
| `MaternKernel`（`nu=0.5/1.5/2.5`） | Matérn共分散。`length_scale` = $\ell$ | なし | `0`（定数） |

6種類のカーネルと3種類の`solver`はいずれもサポート対象です。それぞれテストで検証されており、そのまま利用できます。`GaussianKernel`や`MaternKernel`に`polynomial_degree=None`を渡すこともサポートしており、`"auto"`が追加する定数項を落とせます。

**`PerObjectiveSurrogate(surrogates)`**：`RegressionSurrogate`のサブクラスで、目的ごとに異なるサロゲートを割り当てる合成クラス。
`fit`時に`train_y`の列数と`len(surrogates)`が一致しないと`ValueError`になります。
`provides_uncertainty`は、構成する全サロゲートが`True`の場合のみ`True`を返す複合判定になっています。

### 外部ライブラリアダプタ

scikit-learn互換API経由の回帰サロゲートには`Sklearn`という接頭辞が付きます。

| クラス | モデル |
|---|---|
| `SklearnGPRSurrogate` | ガウス過程{cite}`sacks1989dace,rasmussen2006gpml`。`provides_uncertainty=True`の唯一の実装 |
| `SklearnRFRSurrogate` | ランダムフォレスト回帰 |
| `SklearnSVMSurrogate` | SVM |
| `SklearnNNSurrogate` | MLP |
| `SklearnXGBSurrogate` | XGBoost（`xgboost` 追加機能） |
| `SklearnLGBMSurrogate` | LightGBM（`lightgbm` 追加機能） |
| `TorchSurrogate` | PyTorchベースのモデル（`torch` 追加機能） |

`SklearnGPRSurrogate`は`return_std=True`でGPカーネルから標準偏差を計算し、`provides_uncertainty=True`を返します。

実行可能性予測向けの分類サロゲートには次のクラスがあります。

| クラス | モデル |
|---|---|
| `SklearnClassificationSurrogate` | scikit-learn互換の分類モデル全般 |
| `SklearnRFCClassificationSurrogate` | ランダムフォレスト分類 |
| `SklearnSVCClassificationSurrogate` | SVM分類 |

これらの分類サロゲートの学習データ抽出方法は[TrainingSet](training_set.md)の`FeasibilityClassificationSet`を参照してください。
ペア比較には、これらの分類サロゲートではなく`ComparisonSurrogate`系の専用実装を使い、[SurrogateManager](surrogate_manager.md)の`PairwiseSurrogateManager`および`PairwiseComparisonSet`と組み合わせます。

各追加機能のインストール方法は[インストール](../../getting_started/installation.md)を参照してください。

```{note}
scikit-learn/XGBoost/LightGBM/PyTorch以外のBoTorch/SMTサロゲート用アダプターは、現時点で
`saealib` に実装されていません。`pyproject.toml` に対応する追加機能もありません。pymooにはアダプターがありますが、[Problem](../problem_and_ranking/problem.md)/[Crossover](../search_algorithms/crossover.md)/[Mutation](../search_algorithms/mutation.md)/[Algorithm](../search_algorithms/algorithm.md)
のレベルであり、`Surrogate` としてではありません。pymooはサロゲートモデルを提供しません。
```

## 拡張フック

フィット後の後処理だけを足したい場合、サブクラスを新設せずに`with_post_fit(fn)`で既存の`Surrogate`インスタンスへ処理を追加できます。
`with_post_fit`は元のインスタンスを変更せず、`fn`を追加したコピーを返します。

```python
from saealib import GaussianKernel, RBFSurrogate


def log_fit(train_x, train_y, ctx=None):
    print(f"fit on {len(train_x)} samples")


base = RBFSurrogate(kernel=GaussianKernel())
logged = base.with_post_fit(log_fit)
```

`fn`のシグネチャは`fn(train_x, train_y, ctx) -> None`です。

## 独自Surrogateの実装方法

独自の予測モデルが必要な場合は、`Surrogate`を継承して`fit()`/`predict()`を実装します。
回帰なら`RegressionSurrogate`、比較なら`ComparisonSurrogate`（`predict_proba()`を実装）を継承先に選びます。

次の例は、最近傍点の目的関数値をそのまま予測値として返す単純な回帰サロゲートです。

```python
import numpy as np
from saealib import RegressionSurrogate, SurrogatePrediction


class NearestNeighborSurrogate(RegressionSurrogate):
    """A simple surrogate that returns the nearest point's objective value directly as the prediction."""

    def fit(self, train_x, train_y):
        self.train_x = np.asarray(train_x, dtype=float)
        self.train_y = np.asarray(train_y, dtype=float)

    def predict(self, test_x):
        test_x = np.atleast_2d(test_x)
        dists = np.linalg.norm(self.train_x[None, :, :] - test_x[:, None, :], axis=2)
        nearest = dists.argmin(axis=1)
        value = self.train_y[nearest]
        return SurrogatePrediction(value=value)
```

## 不確実性対応表

不確実性ベースの[AcquisitionFunction](acquisition_functions.md)を使うには、`Surrogate`が`std`を返す必要があります。

| クラス | `provides_uncertainty` |
|---|---|
| `SklearnGPRSurrogate` | `True` |
| `RBFSurrogate` / `SklearnRFRSurrogate` / `SklearnSVMSurrogate` / `SklearnNNSurrogate` / `SklearnXGBSurrogate` / `SklearnLGBMSurrogate` / `TorchSurrogate` | `False` |
| `PerObjectiveSurrogate` | 構成する全サロゲートが`True`の場合のみ`True` |

`Optimizer.validate()`は予測チャネルごとに不確実性の対応を確認します。獲得関数が不確実性を必要とし、対応するサロゲートが不確実性を提供しない場合、エラー診断を報告します。

## 関連コンポーネント

- [SurrogateManager](surrogate_manager.md)：`Surrogate`のfit/predictを協調させる
- [TrainingSet](training_set.md)：`Surrogate`に渡す学習データの抽出方法
- [AcquisitionFunction](acquisition_functions.md)：`predict()`の結果をスコアへ変換する
- [サロゲート精度評価と動的切り替え](surrogate_switching.md)：サロゲートの汎化性能の評価
- [インストール](../../getting_started/installation.md)：各追加機能のインストール方法

## 参照

- {py:class}`saealib.Surrogate`
- {py:class}`saealib.RegressionSurrogate`
- {py:class}`saealib.ComparisonSurrogate`
- {py:class}`saealib.SurrogatePrediction`
- {py:class}`saealib.RBFSurrogate`
- {py:class}`saealib.RBFKernel`
- {py:class}`saealib.GaussianKernel`
- {py:class}`saealib.ThinPlateSplineKernel`
- {py:class}`saealib.LinearKernel`
- {py:class}`saealib.CubicKernel`
- {py:class}`saealib.MultiquadricKernel`
- {py:class}`saealib.MaternKernel`
- {py:class}`saealib.PerObjectiveSurrogate`
- {py:class}`saealib.SklearnGPRSurrogate`
- {py:class}`saealib.SklearnRFRSurrogate`
- {py:class}`saealib.SklearnSVMSurrogate`
- {py:class}`saealib.SklearnNNSurrogate`
- {py:class}`saealib.SklearnXGBSurrogate`
- {py:class}`saealib.SklearnLGBMSurrogate`
- {py:class}`saealib.TorchSurrogate`
- {py:class}`saealib.SklearnClassificationSurrogate`
- {py:class}`saealib.SklearnRFCClassificationSurrogate`
- {py:class}`saealib.SklearnSVCClassificationSurrogate`
