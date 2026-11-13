---
primary_layer: layer2
related_layers: []
page_type: concept
---

# NSGA-II（Nondominated Sorting Genetic Algorithm II）

NSGA-IIは、多目的最適化の選択機構として広く使われている遺伝的アルゴリズムです。非優越ソートと混雑度距離を組み合わせ、パレートフロントへの収束と解集合の多様性維持を同時に達成します。

## 概要

多目的最適化には、パレートフロントへの収束と解集合内の多様性維持という、独立した2つの目標があります。

従来のNSGA（Nondominated Sorting Genetic Algorithm）は非優越ソートと共有関数によって両方を実現しましたが、共有関数には分散パラメータ$\sigma_{\mathrm{share}}$の手動調整が必要で、計算量は$O(N^2)$でした。

NSGA-IIはこの共有関数を**混雑度距離**に基づく**混雑比較演算子**（$\prec_n$）に置き換え、パラメータ不要の多様性維持を実現します。

各個体は非優越ランク$i_{\mathrm{rank}}$と混雑度距離$i_{\mathrm{distance}}$の2つの属性を持ちます。$\prec_n$はランクが低い（良い）個体を優先し、同ランク内では混雑度距離が大きい（周囲が疎な）個体を優先します。

さらにNSGA-IIは、親個体群$P_t$と子個体群$Q_t$を結合した$2N$個体からエリート選択を行い、優れた解が世代を跨いで失われないようにします（elitism）。

出典は{cite}`deb2002nsga2`です。具体的な手順を次の擬似コードに示します。

## 擬似コード

```{prf:algorithm} NSGA-II
:label: alg-nsga2

**Inputs** 目的関数群、個体数 $N$、初期個体群 $P_0$
**Output** 最終世代のパレートフロント

1. $t=0$とし、$P_0$をランダム生成した上で、二項トーナメント選択、交叉、突然変異により子個体群$Q_0$を生成する
2. 親子を結合した個体群 $R_t = P_t \cup Q_t$（サイズ$2N$）を作る
3. $R_t$を非優越ソートし、フロント列$\mathcal{F} = (\mathcal{F}_1, \mathcal{F}_2, \ldots)$を得る
4. $P_{t+1} = \emptyset$とし、フロントが丸ごと収まる限り順に$P_{t+1}$へ追加する
5. 丸ごと収まらない最後のフロント$\mathcal{F}_l$について、各個体の混雑度距離を計算する
6. $\mathcal{F}_l$を$\prec_n$で降順ソートし、$P_{t+1}$が$N$個体になるまで先頭から採用する
7. $P_{t+1}$に$\prec_n$を選択基準とする二項トーナメント選択、交叉、突然変異を適用し、$Q_{t+1}$を生成する
8. $t=t+1$として2へ戻り、終了条件に達するまで繰り返す
```

## フローチャート

```{mermaid}
flowchart TD
    INIT["Initializer<br/>Generate initial population P0<br/>(L1)"] --> GEN
    subgraph GEN["One generation (DirectStrategy.step)"]
        direction TB
        ASK["GA.ask()<br/>Binary tournament selection →<br/>SBX crossover →<br/>Polynomial mutation to generate Qt<br/>(L1, L7)"] --> EVAL["True evaluation<br/>(no surrogate involved)"]
        EVAL --> COMB["GA.tell()<br/>Combine Rt = Pt ∪ Qt<br/>(L2)"]
        COMB --> SORT["NSGA2Comparator.sort_population()<br/>Non-dominated sorting →<br/>crowding distance<br/>(L3-6)"]
        SORT --> TRUNC["TruncationSelection<br/>Take top N individuals as Pt+1<br/>(L4-6)"]
    end
    GEN --> TERM{"Termination condition<br/>reached?"}
    TERM -- "Not yet (L8)" --> GEN
    TERM -- "Reached" --> RESULT(["Pareto front"])
```

## saealibでの構成

| 役割 | saealibでの実装 | 対応ステップ |
|---|---|---|
| 探索アルゴリズム本体 | `GA`（`ask()`で交叉と突然変異、`tell()`で$R_t=P_t\cup Q_t$の結合と生存選択を実行） | L1-2, L7 |
| 親選択 | `TournamentSelection(tournament_size=2)`（`ctx.comparator.compare_population`で勝者を決定） | L1, L7 |
| 交叉 | `CrossoverSBX(prob=0.9, eta=20.0)` | L1, L7 |
| 突然変異 | `MutationPolynomial(eta=20.0)` | L1, L7 |
| 非優越ソート＋混雑度距離 | `NSGA2Comparator`（`sort_population`が内部で`non_dominated_sort`と`crowding_distance_all_fronts`を呼ぶ） | L3-6 |
| 生存選択 | `TruncationSelection()`（`comparator.sort_population`の順に上位$N$個体を残す） | L4-6 |
| 評価戦略 | `DirectStrategy`（サロゲートを介さず、`GA.ask()`が生成した候補を全て真の目的関数で評価する） | L2 |

```python
from saealib import GA, NSGA2Comparator, Optimizer
from saealib.benchmarks import zdt1
from saealib.operators.crossover import CrossoverSBX
from saealib.operators.mutation import MutationPolynomial
from saealib.operators.selection import TournamentSelection, TruncationSelection
from saealib.strategies import DirectStrategy
from saealib.termination import Termination, max_fe


problem = zdt1(n_var=10)

algorithm = GA(
    CrossoverSBX(prob=0.9, eta=20.0),
    MutationPolynomial(eta=20.0),
    TournamentSelection(tournament_size=2),
    TruncationSelection(),
)

opt = (
    Optimizer(problem)
    .set_algorithm(algorithm)
    .set_strategy(DirectStrategy())
    .set_termination(Termination(max_fe(2000)))
)
ctx = opt.run()
pareto_f = ctx.pareto_archive.get_array("f")
```

`NSGA2Comparator`は`n_obj > 1`のときの既定値なので、明示的に設定する必要はありません。サロゲートを一切使わないため、`Optimizer`に`set_surrogate_manager()`を呼ぶ必要はありません。

## 原法との差分

NSGA-IIの原法と比べると、例では`GA`、SBX、Polynomial mutation、二者トーナメント選択、`TruncationSelection`を組み合わせています。この具体的な組み合わせはsaealibの構成例であり、原法が定義する選択機構そのものではありません。同じ混雑距離の個体が打ち切り境界に達した場合、既定の`TruncationSelection(randomize_ties=False)`は`sort_population`による決定論的な順序を保持します。その境界で同順位の個体をシャッフルするのは`randomize_ties=True`の場合だけです。

## パラメータと変種

### 計算量

非優越ソートは$O(MN^2)$（$M$は目的数、$N$は個体数）、混雑度距離の計算は$O(MN\log N)$、$\prec_n$によるソートは$O(N\log N)$です。

1世代あたりの支配的なコストは非優越ソートであり、全体の計算量は$O(MN^2)$になります{cite}`deb2002nsga2`。

**$\eta_c$と$\eta_m$（分布指数）**：`CrossoverSBX(eta=...)`と`MutationPolynomial(eta=...)`で調整します。値が大きいほど親に近い子個体を生成します（探索が保守的になります）。論文の実数値実験では両方に$20$を使い、コード例でも同じ分布指数を明示的に設定しています{cite}`deb2002nsga2`。

**$p_m$（変数単位の突然変異確率）**：論文は$p_m = 1/n$（$n$は決定変数の数）を使います。これは個体レベルの`prob`ではなく、変数ごとの適用確率`prob_var`に対応します。`MutationPolynomial(prob_var=None)`では、既定値として$\min(0.5,\, 1/\mathrm{dim})$が自動設定されます。したがって、論文の$p_m=1/n$は論文側の設定であり、ライブラリのすべての構成がこの値を使うという意味ではありません。

## 関連

- [文献リファレンス](../references.md)：出典の完全な書誌情報
- [Comparator](../concepts/problem_and_ranking/comparators.md)：`NSGA2Comparator`の詳しい仕様
- [Crossover](../concepts/search_algorithms/crossover.md)：`CrossoverSBX`を含む交叉演算子の一覧
- [Mutation](../concepts/search_algorithms/mutation.md)：`MutationPolynomial`を含む突然変異演算子の一覧
- [ParentSelection](../concepts/search_algorithms/parent_selection.md)：`TournamentSelection`の詳しい使い方
- [SurvivorSelection](../concepts/search_algorithms/survivor_selection.md)：`TruncationSelection`の詳しい使い方
- [OptimizationStrategy](../concepts/execution_and_evaluation/strategies.md)：`DirectStrategy`を含む戦略の一覧
- [NonDominatedSorting](../concepts/problem_and_ranking/nondominated_sorting.md)：非優越ソートの実装詳細
