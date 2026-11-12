---
primary_layer: layer2
related_layers: []
page_type: concept
---

# CORS-RBF（Constrained Optimization using Response Surfaces）

CORS-RBFは、評価コストの高い目的関数を対象に、RBF(Radial Basis Function)補間によるサロゲートモデルを使って次の評価点を1点ずつ選ぶ逐次最適化の手法です。 Regis & Shoemaker (2005)が提案した枠組みCORS(Constrained Optimization using Response Surfaces)を、RBFサロゲートモデルで具体化した実装がCORS-RBFです。

## 概要

RBF補間は、学習点を通る滑らかな曲面を再構成するだけで、GP回帰のような予測分散を持ちません。 そのため、サロゲートモデルの予測値をそのまま最小化して次の評価点を選ぶ素朴な方法は、既に良い値が観測された点の周辺だけを繰り返し探索してしまい、真の関数の局所最小値ですらない点に収束しかねません。

CORSは、候補点の選択そのものに**距離制約**を組み込むことでこの問題を回避します。各反復で解く補助問題は制約付き最適化であり、サロゲートモデル$\hat f_i(x)$を最小化するだけでなく、次の候補点が既存の評価済み点すべてから少なくとも$\beta_i \Delta_i$離れていることを要求します（$\Delta_i$は既存点集合からの最小距離の最大値です）。$\beta_i$は反復ごとに数列（**探索パターン**）として与えられ、1に近い値（大域探索寄り）から0（局所探索寄り、つまりサロゲートモデルの単純な最小化）まで循環します。この距離制約が、GPの予測分散が担うはずの探索の役割を引き受けます。

この距離制約は副産物ではなく、CORSの核心です。 search patternに0でない値が1つでも含まれていれば、サロゲートモデルの種類や初期評価点の選び方によらず、任意の連続関数の大域最小値に収束することが証明されています。

出典は{cite}`regis2005cors`。具体的な手順は次の擬似コードに示します。

## 擬似コード

```{prf:algorithm} CORS-RBF
:label: alg-rbf-cors

**Inputs** 目的関数 $f$、探索領域 $\mathcal{D}$、初期評価点集合 $S_1 = \{x_1, \ldots, x_k\}$、距離パラメータの周期列(search pattern) $\langle \beta_1, \ldots, \beta_{N+1}=0 \rangle$
**Output** 最良解 $x^*$

$S_1$ を真の関数 $f$ で評価し、$i := 1$ とする
これまでの評価済みデータ $D_i = \{(x, f(x)) \mid x \in S_i\}$ にRBFサロゲートモデル $\hat f_i$ をフィットする
制約付き最小化問題 $\min_{x \in \mathcal{D}} \hat f_i(x) \ \mathrm{s.t.} \ \|x - x_j\| \geqslant \beta_i \Delta_i \ (j=1,\ldots,|S_i|)$ を解いて候補点 $x_{k+i}$ を求める（$\Delta_i$は既存評価点集合からの最大最小距離）
$x_{k+i}$ を真の関数で評価し、$S_{i+1} := S_i \cup \{x_{k+i}\}$ に追加する
終了条件に達するまで、周期列に従い $\beta_i$ を更新して $i := i+1$ とし2へ戻る
```

## フローチャート

```{mermaid}
flowchart TD
    INIT["初期化器<br/>初期個体群をサンプリング<br/>→ 真の評価<br/>(L1)"] --> ASK
    subgraph GEN["1世代 (PreSelectionStrategy.step)"]
        direction TB
        ASK["GA.ask()<br/>候補点を生成"] --> SCORE["SurrogateManager<br/>RBFをフィット (L2)<br/>→ CORSDistanceでスコア化<br/>βᵢΔᵢ距離制約を適用<br/>(L3)"]
        SCORE --> SORT["上位1候補を選択<br/>CORSDistanceスコア順"]
        SORT --> EVAL["真の評価 →<br/>アーカイブに追加<br/>(L4)"]
        EVAL --> TELL["GA.tell()<br/>個体群を更新"]
    end
    GEN --> TERM{"評価予算N<br/>に達したか？"}
    TERM -- "未達 (L5)" --> ASK
    TERM -- "到達" --> RESULT(["最良解x*"])
```

## saealibでの構成

|役割|saealibでの実装|対応ステップ|
|---|---|---|
|探索アルゴリズム本体|`GA`（交叉、突然変異、選択の組み合わせ自体はCORSの定義に含まれない）|手順 L3|
|サロゲートモデル|`RBFSurrogate`（RBF補間。この例では`GaussianKernel()`を使い、既定の`polynomial_degree="auto"`が定数項を追加します。`kernel`は必須引数であり、任意の`RBFKernel`と`polynomial_degree`を注入できます。詳しくは「原法との差分」を参照してください）|手順 L2|
|獲得関数|`CORSDistance`（予測平均に$\beta_i\Delta_i$の距離制約を適用）|手順 L3|
|サロゲート管理|`GlobalSurrogateManager`（アーカイブ全体でRBFをフィットする）|「L2-3」|
|評価戦略|`PreSelectionStrategy(n_select=1)`（原法に忠実な設定。CORSの判断1回につき真の評価を1点だけ行います）|「L3-4」|

```python
import numpy as np
from saealib import (
    GA,
    GaussianKernel,
    Optimizer,
    Problem,
    PreSelectionStrategy,
    RBFSurrogate,
)
from saealib.acquisition import CORSDistance
from saealib.operators.crossover import CrossoverBLXAlpha
from saealib.operators.mutation import MutationUniform
from saealib.operators.selection import SequentialSelection, TruncationSelection
from saealib.surrogate import GlobalSurrogateManager
from saealib.termination import Termination, max_fe


def sphere(x: np.ndarray) -> float:
    return np.sum(x**2)


lb = np.asarray([-5.0] * 5)
ub = np.asarray([5.0] * 5)
problem = Problem(sphere, dim=5, lb=lb, ub=ub, n_obj=1, direction=[-1])

algorithm = GA(
    CrossoverBLXAlpha(prob=0.7, alpha=0.4),
    MutationUniform(prob_var=0.3),
    SequentialSelection(),
    TruncationSelection(),
)
surrogate_manager = GlobalSurrogateManager(RBFSurrogate(kernel=GaussianKernel()))
# delta=None uses the candidate-pool maximin approximation
acquisition = CORSDistance(delta=None, direction=problem.direction)
# n_select=1: source-faithful one-candidate-per-decision cadence
strategy = PreSelectionStrategy(n_candidates=10, n_select=1)

opt = (
    Optimizer(problem)
    .set_algorithm(algorithm)
    .set_surrogate_manager(surrogate_manager)
    .set_acquisition(acquisition)
    .set_strategy(strategy)
    .set_termination(Termination(max_fe(200)))
)
ctx = opt.run()
```

## 原法との差分

CORSの原法は、制約付き予測を最小化して候補点を逐次選びます。一方saealibは、`GA`が生成した候補プールから`CORSDistance`が最大の候補を選び、この最小化を近似します。上記の例では`PreSelectionStrategy(n_select=1)`を使い、判断1回につき候補1点という原法どおりのペースを保っています。例における交叉、突然変異、親選択、環境選択の組み合わせはsaealibの構成上の選択であり、CORSの定義には含まれません。論文の数値実験では、一次多項式項$p(x)$を持つ薄板スプラインカーネル（$\phi(r) = r^2 \log r$）を使います。上記で使った`kernel=GaussianKernel()`に対しては、`RBFSurrogate`の既定値`polynomial_degree="auto"`が定数項に解決されます（`GaussianKernel`の`auto_polynomial_degree`が`0`のため）。そのため、カーネルだけを差し替えても論文の一次項は再現できません。`kernel=ThinPlateSplineKernel()`と`polynomial_degree=1`を同時に渡すと再現できます。薄板スプラインのような条件付き正定値カーネルでは、線形多項式項が補間系を適切に定めるためです。

## パラメータと変種

**kernel（RBFカーネルの選択）**：`RBFSurrogate(kernel=...)`は`RBFKernel`インスタンスを要求する必須引数であり、既定値はありません。上記の例では`GaussianKernel()`を渡しています。

**delta**: 既定の`delta=None`では、`CORSDistance`は論文の反復ごとの$\Delta_i$を、候補プールにおけるmaximin距離`max_candidate min_evaluated distance`で近似します。有限の正の数値を`delta`に渡すと、代わりに固定の距離スケールを使います。

## betaの進み方

`CORSDistance.prepare(archive, *, ctx)`は`search_pattern[ctx.decision_count % len(search_pattern)]`を選びます。`decision_count`はランタイムが確定した評価計画の数なので、最初の判断では`search_pattern[0]`が使われ、以降は判断1回につき1要素ずつパターンを巡回します。同じ判断に対して再びprepareが呼ばれたとき（アーカイブの変更でキャッシュされた参照が無効になった場合など）は、その判断の要素を使い直し、先へ進めることはありません。`score()`を繰り返し呼んでも、用意済みのbetaを使うだけでパターンは進みません。

## 関連

- [文献リファレンス](../references.md)：出典の完全な書誌情報
- [Surrogate](../concepts/surrogate_modeling/surrogate.md)：`RBFSurrogate`/`GaussianKernel`を含むサロゲートモデル一覧
- [AcquisitionFunction](../concepts/surrogate_modeling/acquisition_functions.md)：CORSの距離制約付き`CORSDistance`を含む獲得関数一覧
- [SurrogateManager](../concepts/surrogate_modeling/surrogate_manager.md)：`GlobalSurrogateManager`の詳しい使い方
- [OptimizationStrategy](../concepts/execution_and_evaluation/strategies.md): `PreSelectionStrategy`を含む戦略の一覧
