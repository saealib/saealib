---
primary_layer: layer3
page_type: guide
---

# 外部ライブラリの統合

`saealib`は、外部の機械学習ライブラリおよび進化計算ライブラリを、独自の抽象基底クラスの背後に薄くラップしたアダプタを提供します。

アダプタが翻訳するのは`Problem`/`Population`/`ctx`といった`saealib`側のデータ表現だけで、学習アルゴリズムや探索演算子そのものは外部ライブラリの実装をそのまま使います。

サロゲートモデル用にscikit-learn、XGBoost、LightGBM、PyTorchのアダプターを、進化計算ライブラリ用にpymooとDEAPのアダプターを実装しています。
DEAPアダプターは`DeapCrossover`、`DeapMutation`、`DeapGenerateUpdateAlgorithm`で、`deap` extraに含まれます。
外部の進化計算ライブラリへのランタイム依存を取り除きたい場合は、{ref}`saealibネイティブなコードへの演算子の移植 <port-operators-to-native-saealib-code>`で、pymooやDEAP風の演算子をネイティブな`Crossover`/`Mutation`サブクラスとして書き直す方法を扱います。

:::{admonition} このページでできるようになること
:class: tip

このページを終えると、外部ライブラリの依存関係を導入し、saealibのアダプター経由でモデルや演算子を利用できます。
:::

ビルトインのモデルや演算子で足りる場合は[高レベルAPI](highlevel_api.md)を使い、独自実装へ移植する場合は[独自コンポーネント](custom_components.md)を参照してください。

## アダプター追加機能のインストール

各アダプタは、対応する`extra`を指定してインストールしたときだけ使えます。

```bash
pip install "saealib[sklearn]"
```

インストール方法とextra一覧の詳細は[インストール](../getting_started/installation.md)を参照してください。

対応する`extra`をインストールしていない状態でアダプタをインポートすると、`ImportError`になります。

## サロゲートアダプターを使う

各アダプタは`saealib`の`Surrogate`基底クラスを実装しており、組み込みの`RBFSurrogate`と同じように`surrogate`引数へ渡せます。

| クラス | 対応する`extra` | ラップするモデル |
|--------|--------|--------|
| `SklearnGPRSurrogate` | `sklearn` | ガウス過程回帰 |
| `SklearnRFRSurrogate` | `sklearn` | ランダムフォレスト回帰 |
| `SklearnSVMSurrogate` | `sklearn` | サポートベクター回帰 |
| `SklearnNNSurrogate` | `sklearn` | 多層パーセプトロン |
| `SklearnXGBSurrogate` | `xgboost` | XGBoost回帰 |
| `SklearnLGBMSurrogate` | `lightgbm` | LightGBM回帰 |
| `TorchSurrogate` | `torch` | 任意のPyTorch `nn.Module` |

コンストラクタへのキーワード引数は、そのまま対応するライブラリのモデルへ渡されます。

```python
import numpy as np
from saealib import minimize, SklearnGPRSurrogate


def expensive_func(x):
    return np.sum(x**2)


DIM = 10

result = minimize(
    expensive_func,
    dim=DIM,
    lb=[-5.0] * DIM,
    ub=[5.0] * DIM,
    surrogate=SklearnGPRSurrogate(),
    max_fe=300,
    seed=0,
)
```

`Surrogate`インスタンスを `minimize(..., surrogate=...)` に渡すと、内部で `LocalSurrogateManager` にラップされます。

分類問題向けのアダプタ（実行可能性分類など）や、各アダプタの詳細な引数は[Surrogate](../concepts/surrogate_modeling/surrogate.md)を参照してください。

## pymooアダプターを使う

各アダプタは構築済みのpymooオブジェクトをラップし、対応する`saealib`の基底クラスを実装しているため、その基底クラスが期待される場所ならどこにでも渡せます。

| Class | ラップする対象 |
|--------|--------|
| `PymooCrossover` | pymooの`Crossover`（例：`SBX()`） |
| `PymooMutation` | pymooの`Mutation`（例：`PM()`） |
| `PymooAlgorithm` | pymooの`Algorithm`（例：`NSGA2()`、`DE()`） |
| `PymooProblem` | pymooの`Problem`（ベンチマークや既存の研究コード） |

`PymooCrossover`/`PymooMutation`は、そのまま`GA`に組み込めます。

```python
import numpy as np
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from saealib import GA, TournamentSelection, TruncationSelection, minimize
from saealib.operators import PymooCrossover, PymooMutation


def expensive_func(x):
    return np.sum(x**2)


DIM = 10

result = minimize(
    expensive_func,
    dim=DIM,
    lb=[-5.0] * DIM,
    ub=[5.0] * DIM,
    algorithm=GA(
        crossover=PymooCrossover(SBX(eta=15)),
        mutation=PymooMutation(PM(eta=20)),
        parent_selection=TournamentSelection(2),
        survivor_selection=TruncationSelection(),
    ),
    surrogate="rbf",
    max_fe=300,
    seed=0,
)
```

`PymooAlgorithm`はpymooアルゴリズム全体の探索・生存選択ロジックをそのまま再利用し、`PymooProblem`は既存のpymoo問題定義をそのまま再利用します。

```python
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from saealib import minimize, PymooProblem
from saealib.algorithms import PymooAlgorithm
from saealib.strategies import DirectStrategy

problem = PymooProblem(get_problem("zdt1"))
result = minimize(
    problem,
    algorithm=PymooAlgorithm(NSGA2(pop_size=20)),
    surrogate="rbf",
    strategy=DirectStrategy(),
    max_fe=200,
    pop_size=20,
    seed=0,
)
```

`PymooAlgorithm`は「エンジンモード」で動作します。
ラップしたpymooアルゴリズムが自身の個体群と内部の生存選択状態を保持し、`ctx.population`は基準となるデータではなく、各世代後にそこから更新されます。実際の意味（チェックポイントと再開には非対応、`n_offspring`は無視される、`PreSelectionStrategy`の部分的な`tell()`には明示的なオプトインが必要）については[Algorithm](../concepts/search_algorithms/algorithm.md)を参照してください。

## DEAPアダプターを使う

`DeapCrossover`と`DeapMutation`は、DEAPの交叉関数と突然変異関数を`Crossover`と`Mutation`として利用します。
`DeapGenerateUpdateAlgorithm`は、`generate`と`update`を持つDEAPのStrategy（`deap.cma.Strategy`など）をアルゴリズムとして利用します。
`DeapGenerateUpdateAlgorithm`は制約なしの単目的問題のみをサポートし、`DeapCrossover`と`DeapMutation`は固定長の数値列に対応します。
これらのアダプターには`deap` extraが必要です。

## 関連コンセプトと参考情報

- {py:class}`saealib.Surrogate`
- {py:class}`saealib.SklearnGPRSurrogate` / {py:class}`saealib.SklearnRFRSurrogate` / {py:class}`saealib.SklearnSVMSurrogate` / {py:class}`saealib.SklearnNNSurrogate`
- {py:class}`saealib.SklearnXGBSurrogate` / {py:class}`saealib.SklearnLGBMSurrogate`
- {py:class}`saealib.TorchSurrogate`
- {py:class}`saealib.PymooCrossover` / {py:class}`saealib.PymooMutation`
- {py:class}`saealib.PymooAlgorithm`
- {py:class}`saealib.PymooProblem`
- {py:class}`saealib.operators.DeapCrossover` / {py:class}`saealib.operators.DeapMutation`
- {py:class}`saealib.algorithms.DeapGenerateUpdateAlgorithm`
- {py:func}`saealib.minimize`
