---
primary_layer: layer2
related_layers: []
page_type: concept
---

# SMS-EMOA（S metric selection EMOA）

SMS-EMOAは、被支配超体積（$\mathcal{S}$メトリック）を選択基準に直接組み込んだ、定常状態の多目的進化アルゴリズムです。 非優越ソートで個体群をフロントに分割したうえで、最下位フロント内の超体積寄与度が最小の個体を1体ずつ淘汰し、世代を追うごとに個体群全体の被支配超体積を単調に増大させます。

## 概要

超体積指標は、パレートフロント近似の質を測る指標として広く使われています。 基準点$\mathbf{y}_{\mathrm{ref}}$を1つ固定すると、解集合$B$が支配する領域のルベーグ測度$\mathcal{S}(B, \mathbf{y}_{\mathrm{ref}})$が定義でき、有限のパレートフロント近似において$\mathcal{S}$を最大化することが真のパレート集合を求めることと等価になることが知られています。

SMS-EMOAは、この超体積指標を評価用途にとどめず、選択演算子そのものに採用します。 NSGA-IIと同じ**非優越ソート**で個体群をフロント$\mathcal{R}_1, \ldots, \mathcal{R}_v$に分割し、最下位フロント$\mathcal{R}_v$の中で、除いたときに$\mathcal{S}$メトリックの減少量が最小となる個体を1体だけ淘汰します。 この減少量を**排他的超体積寄与度** $\Delta_{\mathcal{S}}(s, \mathcal{R}_v) := \mathcal{S}(\mathcal{R}_v) - \mathcal{S}(\mathcal{R}_v \setminus \{s\})$と呼びます。

超体積の計算コストが高いため、SMS-EMOAは**定常状態**の世代交代を採ります。 1世代につき交叉と突然変異で新個体を1体だけ生成し、個体群サイズ$\mu$を保つために既存個体を1体だけ淘汰します。 $(\mu+\lambda)$世代交代のように$\binom{\mu+\lambda}{\mu}$通りの組み合わせを比較する必要がなく、最下位フロント内で高々$\mu+1$回の$\mathcal{S}$メトリック評価に抑えられます。

出典は{cite}`beume2007smsemoa`。具体的な手順は次の擬似コードに示します。

## 擬似コード

```{prf:algorithm} SMS-EMOA
:label: alg-sms-emoa

**Inputs** 目的関数群、個体数 $\mu$、初期個体群 $P_0$ **Output** 最終世代の個体群 $P_{t+1}$

1. $t=0$ とし、$\mu$ 個体からなる初期個体群 $P_0$ を生成する
2. 交叉と突然変異により、$P_t$ から新個体 $q_{t+1}$ を1体だけ生成する
3. $Q = P_t \cup \{q_{t+1}\}$（サイズ $\mu+1$）を非優越ソートし、フロント列 $\mathcal{R}_1, \ldots, \mathcal{R}_v$ を得る
4. 最下位フロント $\mathcal{R}_v$ を特定する（$|\mathcal{R}_v|=1$ ならその1個体がそのまま淘汰対象になる）
5. $|\mathcal{R}_v| > 1$ のとき、$\mathcal{R}_v$ 内の各個体 $s$ について排他的超体積寄与度 $\Delta_{\mathcal{S}}(s, \mathcal{R}_v) = \mathcal{S}(\mathcal{R}_v) - \mathcal{S}(\mathcal{R}_v \setminus \{s\})$ を計算し、最小の個体 $r$ を選ぶ
6. $P_{t+1} = Q \setminus \{r\}$ とする（$\mathcal{S}(P_t) \leq \mathcal{S}(P_{t+1})$ が常に成り立つ）
7. $t=t+1$ として2へ戻り、終了条件に達するまで繰り返す
```

## フローチャート

```{mermaid}
flowchart TD
    INIT["Initializer<br/>Generate initial population P0 of μ individuals<br/>(L1)"] --> GEN
    subgraph GEN["One generation (SteadyStateStrategy.step)"]
        direction TB
        ASK["GA.ask(n_offspring=1)<br/>Randomly select parent →<br/>SBX crossover →<br/>Polynomial mutation to generate one new individual q_t+1<br/>(手順 L2)"] --> EVAL["True evaluation<br/>(no surrogate involved)"]
        EVAL --> COMB["GA.tell()<br/>Combine Q = Pt ∪ {q_t+1}<br/>(L3)"]
        COMB --> SORT["HypervolumeComparator.sort_population()<br/>Non-dominated sorting →<br/>HV contribution within front<br/>(L3〜5)"]
        SORT --> TRUNC["TruncationSelection<br/>Cull the last individual<br/>(smallest contribution in the lowest front)<br/>(L4〜6)"]
    end
    GEN --> TERM{"Termination condition<br/>reached?"}
    TERM -- "Not yet (L7)" --> GEN
    TERM -- "Reached" --> RESULT(["Population of the final generation"])
```

## saealibでの構成

| 役割 | saealibでの実装 | 対応ステップ |
|---|---|---|
| 探索アルゴリズム本体 | `GA`（`ask(n_offspring=1)`で新個体を1体だけ生成、`tell()`で $Q=P_t\cup\{q_{t+1}\}$ の結合と生存選択を実行） | L2〜3、L6 |
| 親選択 | `TournamentSelection(tournament_size=1)`（比較処理を伴わない一様ランダム選択。論文は親選択方式を明記していない） | 手順 L2 |
| 交叉 | `CrossoverSBX(prob=0.9, eta=20.0)` | 手順 L2 |
| 突然変異 | `MutationPolynomial(eta=20.0)` | 手順 L2 |
| 非優越ソート＋フロント内HV寄与度 | `HypervolumeComparator`（`sort_population`が内部で非優越ソートと`hypervolume_contributions`を呼ぶ） | L3〜5 |
| 生存選択 | `TruncationSelection()`（`comparator.sort_population`の末尾1個体、すなわち最下位フロントで寄与度最小の個体を淘汰） | L4〜6 |
| 評価戦略 | `SteadyStateStrategy`と`SerialEvaluator`（逐次評価、`max_workers=1`、`max_pending=1`相当） | L2, L6-7（世代内の評価） |

`SteadyStateStrategy`は1ステップに1候補を要求します。以下の忠実な構成例は逐次評価を行い、非同期評価は別の拡張として説明します。

```python
from saealib import (
    GA,
    HypervolumeComparator,
    Optimizer,
    SerialEvaluator,
    SteadyStateStrategy,
)
from saealib.benchmarks import zdt1
from saealib.operators.crossover import CrossoverSBX
from saealib.operators.mutation import MutationPolynomial
from saealib.operators.selection import TournamentSelection, TruncationSelection
from saealib.termination import Termination, max_fe


problem = zdt1(n_var=10)

algorithm = GA(
    CrossoverSBX(prob=0.9, eta=20.0),
    MutationPolynomial(eta=20.0),
    TournamentSelection(tournament_size=1),
    TruncationSelection(),
)
evaluator = SerialEvaluator()

opt = (
    Optimizer(problem)
    .set_comparator(HypervolumeComparator())
    .set_algorithm(algorithm)
    .set_strategy(SteadyStateStrategy())
    .set_evaluator(evaluator)
    .set_termination(Termination(max_fe(2000)))
)
ctx = opt.run()
pareto_f = ctx.pareto_archive.get_array("f")
```

`set_comparator(HypervolumeComparator())`の呼び出しは省略できません。 NSGA-IIでは`NSGA2Comparator`が`n_obj > 1`のときの既定値なので同じ呼び出しを省略できましたが、SPEA2やNSGA-IIIと同様、SMS-EMOAでも明示的なcomparatorの設定が必要になります。

**非同期拡張**：評価時間が異なる場合は、`SteadyStateStrategy`を`AsyncEvaluator`と`AsyncEvaluationScheduler`（例：`max_workers=2`、`max_pending=2`）と組み合わせられます。ただし、これは上の逐次構成とは評価順序が異なる拡張です。

```python
from saealib import AsyncEvaluator, AsyncEvaluationScheduler

async_evaluator = AsyncEvaluator(SerialEvaluator(), max_workers=2)
async_schedule = AsyncEvaluationScheduler(async_evaluator, max_pending=2)
async_opt = (
    Optimizer(problem)
    .set_comparator(HypervolumeComparator())
    .set_algorithm(algorithm)
    .set_strategy(SteadyStateStrategy())
    .set_evaluator(async_evaluator)
    .set_async_evaluation_scheduler(async_schedule)
)
```

## 原法との差分

原法のSMS-EMOAは一度に1個体を評価する定常状態手続きを使います。この設計により、最下位フロントのハイパーボリューム計算は最大でも$\mu+1$個に制限されます。上の構成も`SteadyStateStrategy`で一度に1個体を生成します。代わりに`DirectStrategy`が世代ごとに複数個体を生成すると、`HypervolumeComparator`は全フロントにわたるハイパーボリューム寄与を順位付けするため、最下位フロントだけを対象とする原法の手続きには従わなくなります。原法は親選択を指定していないため、例では比較を行わない`TournamentSelection(tournament_size=1)`を使います。`HypervolumeComparator`は論文に記載されたSMS-EMOAのdp変種も実装しておらず、基本的なハイパーボリューム寄与だけを提供します。

## パラメータと変種

### 計算量

超体積計算自体は、点数について多項式だが目的数について指数的な計算量を持ちます。 saealibの`hypervolume`（再帰的スライシング）は$O(n^{m-1} n \log n)$（$n$は点数、$m$は目的数）です。

排他的寄与度は1点抜きでサイズ$k$のフロントに対し$k$回のHV計算を要するため、フロント1つ分の計算は$O(k^{m} \log k)$になります。 論文の削減手順は最下位フロント（サイズ高々$\mu+1$）だけにこれを適用するため、1世代あたり$O(\mu^{m} \log \mu)$に収まります{cite}`beume2007smsemoa`。

`HypervolumeComparator`は全フロントに対して寄与度を計算する一般化を行いますが、フロントサイズの総和は$\mu+1$を超えないため、漸近的な上界は$O(\mu^{m} \log \mu)$のまま変わりません。

**代替のreduce手続き（"SMS-EMOA dp"）**：論文は、超体積寄与度の代わりに支配点数$d(s, P(t))$を使う高速な変種も提案しています。

**参照点の扱い**：`HypervolumeComparator(reference_point=...)`では固定値を指定できます。`None`の場合は、論文の定義どおり各目的の最悪値に1.0を加えて、各世代の各フロントの参照点を計算します。2目的の場合は、参照点を計算せずに両端の境界解を残す論文の扱いに従います。

**支配述語の差し替え**：`HypervolumeComparator(reference_point=..., dominator=...)`で、既定の`ParetoDominator`以外の[Dominator](../concepts/problem_and_ranking/dominance.md)を注入できます。非劣ソートの結果が変わるため、フロント分割と寄与度計算の対象となる個体群もこの支配述語に依存します。

## 関連

- [文献リファレンス](../references.md)：出典の完全な書誌情報
- [Comparator](../concepts/problem_and_ranking/comparators.md)：`HypervolumeComparator`の詳しい仕様、個体群相対的なComparatorの扱い
- [Crossover](../concepts/search_algorithms/crossover.md): `CrossoverSBX`を含む交叉演算子の一覧
- [Mutation](../concepts/search_algorithms/mutation.md): `MutationPolynomial`を含む突然変異演算子の一覧
- [ParentSelection](../concepts/search_algorithms/parent_selection.md)：`TournamentSelection`の詳しい使い方
- [SurvivorSelection](../concepts/search_algorithms/survivor_selection.md)：`TruncationSelection`の詳しい使い方
- [OptimizationStrategy](../concepts/execution_and_evaluation/strategies.md)：独自Strategyの実装方法、`AskStage`の`n_offspring`
- [NonDominatedSorting](../concepts/problem_and_ranking/nondominated_sorting.md)：非劣ソートの実装詳細
- [Dominator](../concepts/problem_and_ranking/dominance.md)：`dominator`引数で差し替えられる支配述語の一覧
