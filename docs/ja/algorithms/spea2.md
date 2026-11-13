---
primary_layer: layer2
related_layers: []
page_type: concept
---

# SPEA2 (Strength Pareto Evolutionary Algorithm 2)

SPEA2は、初代SPEA（Strength Pareto Evolutionary Algorithm）の適応度割り当てとアーカイブ管理を改良した多目的進化アルゴリズムです。
支配関係に基づく適応度と$k$最近傍密度を組み合わせて個体を順位付けし、固定サイズの外部アーカイブを境界解が失われない打ち切り手続きで維持します。

## 概要

SPEAは初期の多目的進化アルゴリズムとして高い性能を示しましたが、2つの弱点を抱えていました。

1つは、同じアーカイブ個体に支配される個体同士が同一の適応度を持ってしまい、優劣を区別できない点です。
もう1つは、アーカイブが上限を超えたときに使うクラスタリング手法が、非優越解集合の外周にある境界解を失うことがある点です。

SPEA2は、この2つの弱点をそれぞれ独立した仕組みで解消します。
各個体$i$には、自分が支配する個体数を表す**強度** $S(i)$が割り当てられ、$i$を支配する個体群の強度の総和が**生の適応度** $R(i)$になります。
$R(i)=0$は$i$が非優越であることを意味し、値が大きいほど多くの（かつ強い）個体に支配されていることを示します。
同一の$R(i)$を持つ個体同士を区別するため、目的空間上で$k$番目に近い個体との距離$\sigma_i^k$の逆数を**密度** $D(i)=1/(\sigma_i^k+2)$として加えます。
最終的な**適応度** $F(i)=R(i)+D(i)$は値が小さいほど優れており、非優越個体は常に$F(i)<1$になります。

SPEA2は個体群とは別に、サイズを固定した外部アーカイブを維持します。
各世代で、個体群とアーカイブを合わせた集合から非優越個体（$F(i)<1$）を新しいアーカイブへコピーします。
コピー後のアーカイブがちょうど規定サイズに収まればそのまま採用し、不足する場合は$F(i)$が小さい劣解から順に補充します。
規定サイズを超える場合は、**打ち切り演算子**を適用し、最近傍距離が最小の個体を1体ずつ、距離を再計算しながら取り除きます。
この手続きは、同じ距離を持つ個体が並ぶタイを2番目と3番目に近い個体との距離で順に解消するため、境界解が誤って除去されにくいです。

出典は{cite}`zitzler2001spea2`。具体的な手順は次の擬似コードに示します。

## 擬似コード

```{prf:algorithm} SPEA2
:label: alg-spea2

**Inputs** population size $N$, archive size $\bar N$, maximum generations $T$
**Output** the non-dominated solution set $A$

1. Initialization: generate an initial population $P_0$, prepare an empty archive $\bar P_0 = \emptyset$, and set $t=0$
2. Fitness assignment: compute the fitness $F(i) = R(i) + D(i)$ of every individual in $P_t$ and $\bar P_t$
3. Environmental selection: copy the non-dominated individuals of $P_t \cup \bar P_t$ into $\bar P_{t+1}$. If $|\bar P_{t+1}| > \bar N$, reduce it with the truncation operator; if $|\bar P_{t+1}| < \bar N$, fill it with inferior solutions in order of increasing $F(i)$
4. Termination check: if $t \geq T$ or another termination condition is met, output the non-dominated individuals in $\bar P_{t+1}$ as $A$ and stop
5. Mating selection: perform binary tournament selection (with replacement) on $\bar P_{t+1}$ to fill the mating pool
6. Variation: apply crossover and mutation to the mating pool to generate $P_{t+1}$, set $t=t+1$, and return to step 2
```

## フローチャート

```{mermaid}
flowchart TD
    INIT["LHSInitializer<br/>Sample n_init_archive individuals (≈ literature P0) →<br/>rank_population() ranks them<br/>(fitness assignment + environmental selection) →<br/>take top n_init_population as<br/>initial ctx.population (≈ literature P̄1)<br/>(L1, L2, L3)"] --> GEN
    subgraph GEN["One generation (DirectStrategy.step)"]
        direction TB
        ASK["GA.ask()<br/>Binary tournament selection →<br/>SBX crossover →<br/>Polynomial mutation to generate<br/>N offspring<br/>(L5, L6)"] --> EVAL["True evaluation<br/>(no surrogate involved;<br/>no independent pseudocode step)"]
        EVAL --> COMB["GA.tell()<br/>Combine population (P̄t) and offspring<br/>into a single pool"]
        COMB --> RANK["SPEA2Comparator.rank_population()<br/>prepare_population(): S(i)→R(i)→D(i)→F(i)<br/>sort_population(): F(i)&lt;1 spea2_truncation_order,<br/>F(i)≥1 ascending F(i),<br/>infeasible: ascending cv<br/>(L2, L3)"]
        RANK --> TRUNC["TruncationSelection<br/>Take top N̄ individuals as P̄t+1"]
    end
    GEN --> TERM{"Termination condition<br/>reached?"}
    TERM -- "Not yet (L4)" --> GEN
    TERM -- "Reached" --> RESULT(["Non-dominated solution set A"])
```

## saealibでの構成

| 役割 | saealibでの実装 | 対応ステップ |
|---|---|---|
| 初期化 | `LHSInitializer`（`n_init_archive`個体をサンプルし真の評価を行い、`rank_population()`を呼ぶ——原論文の初期fitness assignmentと最初のenvironmental selectionを初期化に畳み込んでいる——その後上位`n_init_population`個体を初期populationへ渡す） | 「L1, L2, L3」 |
| 探索アルゴリズム本体 | `GA`（`ask()`が交配選択と変異を行い、`tell()`が個体群と子個体を1つのプールに統合し環境選択を行う） | 「L3, L5, L6」 |
| 親選択 | `TournamentSelection(tournament_size=2)`（二項トーナメント。勝者は`compare_population`によって決まります） | 手順 L5 |
| Crossover | `CrossoverSBX(prob=0.9, eta=20.0)` | 手順 L6 |
| Mutation | `MutationPolynomial(eta=20.0)` | L6 |
| 適応度計算と環境選択 | `SPEA2Comparator.rank_population()`（`prepare_population()`が`spea2_fitness`を介して$S(i)$/$R(i)$/$D(i)$/$F(i)$を計算し個体群へ永続化、`sort_population()`が$F(i)<1$の非劣解ブロックを`spea2_truncation_order`で、$F(i)\geq1$の劣解ブロックを$F(i)$昇順で、実行不可能解を`cv`昇順で並べる）+ `TruncationSelection()`（上位$\bar N$個体を採用） | 「L2, L3」 |
| 評価戦略 | `DirectStrategy`（サロゲートを介さず、`GA.ask()`が生成した候補を全て真の目的関数で評価する） | 独立したステップ番号なし（L6の後、次のL2の前に発生） |

```python
from saealib import GA, LHSInitializer, SPEA2Comparator, Optimizer
from saealib.benchmarks import zdt1
from saealib.operators.crossover import CrossoverSBX
from saealib.operators.mutation import MutationPolynomial
from saealib.operators.selection import TournamentSelection, TruncationSelection
from saealib.strategies import DirectStrategy
from saealib.termination import Termination, max_fe


N = 30  # offspring per generation, and the initial candidate pool size
N_BAR = 20  # SPEA2 external archive size

problem = zdt1(n_var=10)

algorithm = GA(
    CrossoverSBX(prob=0.9, eta=20.0),
    MutationPolynomial(eta=20.0),
    TournamentSelection(tournament_size=2),
    TruncationSelection(),
)

opt = (
    Optimizer(problem)
    .set_comparator(SPEA2Comparator())
    .set_initializer(LHSInitializer(n_init_archive=N, n_init_population=N_BAR))
    .set_algorithm(algorithm)
    .set_strategy(DirectStrategy(n_offspring=N))
    .set_termination(Termination(max_fe(2000)))
)
ctx = opt.run()
pareto_f = ctx.pareto_archive.get_array("f")
```

`set_comparator(SPEA2Comparator())`の呼び出しは省略できません。
NSGA-IIでは`NSGA2Comparator`が`n_obj > 1`のときの既定値なので同じ行を省略できましたが、SPEA2ではそうではありません。

`ctx.pareto_archive`はsaealib独自の累積非劣解アーカイブであり、特定のアルゴリズムとは独立に実行全体を通じて追跡されます。
SPEA2の外部アーカイブ（`ctx.population`に対応）と同じ状態コンテナではありません。原論文の出力$A$（最終的な外部アーカイブ$\bar P_{t+1}$内の非劣解群）そのものが必要な場合は`ctx.population`を直接参照してください。

## 文献との差分

既定の`ParetoDominator`を制約なし問題で使う場合、saealibはSPEA2のselection semantics——strength/raw fitness/densityによる適応度割り当てと打ち切りベースの環境選択——を原論文の記述通りに再現します。
実行構造については、文献と異なる点が2箇所あります。

**初期化**：原論文は初期個体群$P_0$（サイズ$N$）を生成し、空のアーカイブ$\bar P_0 = \emptyset$から始めます。
最初の適応度割り当てと環境選択（L2, L3）は、メインループが$t=0$で始まって初めて実行されます。
saealibの`LHSInitializer`は、この最初の適応度割り当てと環境選択を構築処理そのものに畳み込みます。`n_init_archive`個体をサンプルし（原論文の$P_0$の役割を担う）、真の評価を行い、`rank_population()`を呼んでランク付けした上で、上位`n_init_population`個体を初期`ctx.population`へ渡します。したがって$t=0$時点の`ctx.population`は、空の$\bar P_0$ではなく、原論文の選択後のアーカイブ$\bar P_1$に既に対応しています。

**制約処理**：制約付き問題に対して、`SPEA2Comparator`はさらにDeb (2000)のfeasibility-firstルール{cite}`deb2000feasibility`を適用します——feasible個体は常にinfeasible個体より優先され、infeasible個体同士は制約違反量の昇順で並びます。
発表されたSPEA2アルゴリズムは制約処理を扱っておらず、これはsaealib側の拡張です。

## パラメータと変種

### 計算量

適応度計算($S(i)$/$R(i)$/$D(i)$)は$O(M^2)$（$M=N+\bar N$）で、密度推定のための距離ソートを含めると$O(M^2\log M)$になります。
打ち切り演算子は最悪$O(M^3)$、平均では$O(M^2\log M)$です{cite}`zitzler2001spea2`。

**$N$と$\bar N$の独立設定**：`ctx.population`はSPEA2のアーカイブ$\bar P$の役割を担います——`GA.tell()`がこれを子個体群と統合し、`TruncationSelection`（`set_comparator()`で設定した`SPEA2Comparator`を利用）がその結果に対して環境選択を行います——そのため`LHSInitializer(n_init_population=...)`が$\bar N$を、`DirectStrategy(n_offspring=...)`が各世代で生成（かつ真の評価が行われる）子個体数$N$を設定します。
`LHSInitializer`のもう一方のサイズパラメータ`n_init_archive`は、saealib自身の累積`Archive`（`ctx.archive`）——SPEA2の外部アーカイブとは異なり、以降の各世代の評価によって増え続ける、より長命なコンテナ——に最初に評価されるエントリ数を設定します。
$t=0$時点に限っては、これは最初の適応度割り当てと環境選択の対象となる候補プールのサイズも兼ねるため（前述の*Differences from the source*参照）、文献に忠実な初期化では`n_init_archive=N`とします。
組み込みのInitializerは`n_init_population <= n_init_archive`を前提とするため、$\bar N > N$という初期状態を直接構成することはできません。

**支配述語の差し替え**：`SPEA2Comparator(dominator=...)`では、既定の`ParetoDominator`以外の[Dominator](../concepts/problem_and_ranking/dominance.md)を注入できます。
$S(i)$/$R(i)$の計算はこの支配述語に依存するため、差し替えるとSPEA2の適応度自体が変わります。

## 関連

- [文献リファレンス](../references.md)：出典の完全な書誌情報
- [Comparator](../concepts/problem_and_ranking/comparators.md)：`SPEA2Comparator`の詳細仕様
- [Crossover](../concepts/search_algorithms/crossover.md): `CrossoverSBX`を含む交叉演算子の一覧
- [Mutation](../concepts/search_algorithms/mutation.md): `MutationPolynomial`を含む突然変異演算子の一覧
- [ParentSelection](../concepts/search_algorithms/parent_selection.md)：`TournamentSelection`の詳しい使い方
- [SurvivorSelection](../concepts/search_algorithms/survivor_selection.md)：`TruncationSelection`の詳しい使い方
- [OptimizationStrategy](../concepts/execution_and_evaluation/strategies.md): `DirectStrategy`を含む戦略の一覧
- [Dominator](../concepts/problem_and_ranking/dominance.md)：`dominator`引数で差し替えられる支配述語の一覧
- [Initializer](../concepts/execution_and_evaluation/initialization.md)：`LHSInitializer`の`n_init_archive`/`n_init_population`引数が初期アーカイブと個体群のサイズをどう決めるか
