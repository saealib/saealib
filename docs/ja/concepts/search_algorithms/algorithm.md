---
primary_layer: layer3
page_type: concept
---

# 探索アルゴリズム

`saealib` は候補生成とフィードバックの消費を分離し、その契約を `Algorithm` にまとめます。 `Algorithm` は `Optimizer.set_algorithm(algorithm)` で差し替えられます。

現行の `Algorithm` は、候補を生成するProposerと、評価結果を消費するFeedbackConsumerを兼ねる契約です。 通常の利用では、利用者がこの契約を意識する必要はありません。 独自の探索アルゴリズムを実装するときだけ、次の状態境界を実装します。

## Algorithmの役割

`Algorithm` は、読み取り専用の `StateView` を受け取り、状態を直接変更せずに結果を返します。

**`ask(request, state) -> ProposalBatch`**：`ProposalRequest` と `StateView` から候補のバッチを生成します。 `request.n_offspring` が `None` の場合の候補数は、アルゴリズムが現在のPopulationから決定します。

**`tell(feedback, state) -> StatePatch`**：`FeedbackBatch` を消費し、適用すべき状態変更を `StatePatch` として返します。 `tell()` は `OptimizationState` や `Population` を直接変更しません。

この分離によって、どの候補を高コストな真の評価へ送るかを `Algorithm` の外側に置けます。
その候補は評価プランナーが評価計画に基づいて選びます。`FeedbackRequirement`は`ProposalBatch.requirements`に含まれ、proposerが提案に対して動作する前に必要とするフィードバックを指定します。

## 組み込みAlgorithm

| クラス | 探索手法 |
|---|---|
| `GA` | 交叉と突然変異による遺伝的アルゴリズム |
| `PSO` | 速度と位置の更新による粒子群最適化 |
| `GenomeGA` | `SearchSpace`のGenomeを直接扱う遺伝的アルゴリズム |

`GenomeGA` は、`GenomeBatch` と `SearchSpace` の契約を使うGenome-native経路のAlgorithmです。 固定幅のdense vectorを前提とする `GA` と同じ `x` 列の経路へ暗黙に変換せず、Genomeの表現と空間サービスを維持します。

### GA：演算子を組み合わせるコンテナ

`GA`自体には探索ロジックがなく、`crossover`/`mutation`/`parent_selection`/`survivor_selection`という4つの演算子を注入して組み立てるコンテナです。

```python
GA(crossover, mutation, parent_selection, survivor_selection, *,
   duplicate_elimination=None,
   variation_execution="batch",
   integer_crossover=None, integer_mutation=None,
   categorical_crossover=None, categorical_mutation=None)
```

各演算子の動作とパラメータは、それぞれの専用ページで説明します。

- [Crossover](crossover.md)：親個体から子個体を生成します。
- [Mutation](mutation.md)：子個体に摂動を加えます。
- [ParentSelection](parent_selection.md)：交叉に使う親個体を選びます。
- [SurvivorSelection](survivor_selection.md)：次世代に残す個体を選びます。

`variation_execution`は交叉と突然変異のバッチ実行または逐次実行を選択し、辞書でそれぞれを個別に制御できます。既定値はバッチです。操作・フックの完全な順序と再現性の仕様については、下記の`GA` APIリファレンスを参照してください。

`GA.tell()`は現在の個体群と`ask()`が生成した候補を1つのプールにまとめ、`survivor_selection`で次の個体群を選びます。これは$(\mu+\lambda)$型の更新に相当します。プールに何を入れるか（親を含めるかどうか）は`GA`自身の責務であり、`SurvivorSelection`のインターフェース自体には現れません。

### GA：混合変数問題

設計変数に整数変数とカテゴリ変数が混在する問題では、`GA`は連続変数用の`crossover`/`mutation`に加えて、型ごとの演算子を使います。`integer_crossover`/`integer_mutation`/`categorical_crossover`/`categorical_mutation`を省略すると、それぞれ`CrossoverIntegerSBX`/`MutationIntegerUniform`/`CrossoverCategorical`/`MutationCategorical`が自動的に設定されます。自動設定された演算子の確率パラメータ（`prob`/`prob_var`）は、連続変数用の`crossover`/`mutation`から引き継がれます。

型ごとに設定する演算子でも、`n_children`/`n_parents`は連続変数用の`crossover`と一致しなければなりません。一致しない場合は`ConfigurationError`を送出します。これは、`GA`が親個体を変数型ごとの列に分割し、対応する列にだけ各演算子を適用してから、結果を1つの個体に再構成するためです。この実装では、型によって子個体数と親個体数を変えられません。既定のバッチモードでは、各型専用演算子のバッチメソッドをその列に対して呼び出します。`variation_execution="sequential"`では従来のペア単位・個体単位の振り分けを使い、乱数列を維持します。

Variable types は defined via [Problem](../problem_and_ranking/problem.md)'s `variables` argument.。

### GA：補助ユーティリティ

**`duplicate_elimination`**引数に`DuplicateElimination(atol=1e-16, rtol=0.0, max_retries=100)`を渡すと、現在の個体群と重複する子個体を再生成して置き換えます。重複検出の許容誤差は`atol`/`rtol`で、再生成の最大試行回数は`max_retries`で指定します。省略した場合（既定値は`None`）、重複除去は行われません。

`saealib.repair_clipping(candidates, bounds)`は、候補集合を`(lb, ub)`に単純に`np.clip`する独立したユーティリティ関数です。`GA`自身は[ConstraintHandler](../problem_and_ranking/constraints.md)の`repair()`（これも既定では`np.clip`）と`Problem.repair()`（`Variable`ごとの射影）を通じて候補を修復します。`repair_clipping`はGAの内部処理に自動接続されません。`ConstraintHandler`を通らないカスタム評価パイプラインを書く場合など、同じクリッピング動作を単独で使いたいときのために公開されています。

### PSO法

`PSO(w=0.7, c1=1.5, c2=1.5, v_max=None)`は、慣性項`w`、個体最良値への誘引`c1`、群最良値への誘引`c2`の重みを使って速度を更新し、その速度で位置を進めます。`v_max`を指定すると、各次元の速度の大きさをその値に制限します。

`GA`と異なり、演算子を注入して組み立てるものではありません。`ask()`/`tell()`が速度と位置の更新を直接行い、各粒子のpbest（個体最良値）を追跡します。群最良値（リーダー）は`ctx.comparator`を使って全粒子のpbestから選ぶため、任意の単目的`Comparator`を自動的に利用できます。多目的PSO（MOPSO）には、非劣解集合を管理する専用サブクラスが必要です。組み込みの`PSO`は単目的問題を対象とします。

### 外部ライブラリアダプタ

`PymooAlgorithm(pymoo_algorithm, *, allow_partial_tell=False)`は、構築済みの[pymoo](https://pymoo.org/)アルゴリズム（例：`NSGA2()`、`DE()`）をラップし、既存のpymooアルゴリズムでsaealibのask-tellループとサロゲート支援戦略を変更せずに動かせるようにします。

`GA`/`PSO`は`ctx.population`を正本とみなしますが、`PymooAlgorithm`は「エンジンモード」で動作します。ラップしたpymooアルゴリズムが独自の個体群と内部の生存状態を保持し、各 `tell()` の終了時に`ctx.population`がそこから更新されます。これは正本ではなくミラーです。pymooアルゴリズム自身の検証済みの生存選択ロジックを変更せずに再利用する唯一の方法ですが、利用前に知っておくべき実際の制約があります。

- チェックポイントからの保存・再開には対応しません — `OptimizationState.save()` はsaealib独自の配列だけをシリアライズし、ラップしたアルゴリズムの内部状態は対象にしません。
- `ask()`に渡した`n_offspring`は無視されます。子個体数はラップしたアルゴリズム自身の設定（例：`pop_size`）で固定されます。
- `PreSelectionStrategy`の上位k件への打ち切りでは、`ask()`が生成したものの一部だけを`tell()`に渡します。既定では`ConfigurationError`を送出します。インデックスに依存するアルゴリズム（例：差分進化）では、部分的な`tell()`により親と子個体の対応が気付かないままずれるためです。それでも許可する場合は`allow_partial_tell=True`を渡します。

`pymoo` extraのインストールについては[インストール](../../getting_started/installation.md)を参照してください。

## 独自Algorithmの実装方法

独自の探索アルゴリズムを実装するときは、`Algorithm` を継承して `ask()` と `tell()` を実装します。 それだけが必須です。 基底は `get_required_attrs() -> []`、`population_class` -> `Population`、`archive_class` -> `Archive` の既定を提供するため、ask/tellだけを実装する最小のAlgorithmが、初期化子が構築する標準の `x`/`f`/`g`/`cv`/`id` スキーマと標準の `Population`/`Archive` でそのまま動作します。

メタデータは、アルゴリズムがより多くを必要とするときだけオーバーライドします:

- `get_required_attrs()` は、PSOの `velocity`/`pbest_*` 列のようなアルゴリズム固有の状態を個体群に格納するときに追加属性を宣言します。
- `population_class`/`archive_class` は入れ物を `Population` サブクラスへ差し替えます。
- `create_pareto_archive()`（すでに非抽象）は、独自のParetoコレクションのために引き続きオーバーライドできます。

algorithmスロットはダックタイピングです。 `set_algorithm()`、検証、実行準備、初期化子のどれも `Algorithm` の継承や `ask`/`tell` を要求しません。 初期化子のメタデータフック（`get_required_attrs`、`population_class`、`archive_class`、`create_pareto_archive`）だけを持つオブジェクトは、独自StrategyとStageによる非ask/tellの制御フローで実行できます。

新しい実装では、`ask()` は `ProposalBatch` を返し、`tell()` は `StatePatch` を返します。 `state` は宣言された状態キーだけを参照できる `StateView` です。 候補数、候補ID、評価結果の対応付けは `ProposalBatch` と `FeedbackBatch` の契約に従います。

次の骨格は、独自Algorithmが実装する境界だけを示しています。

```python
from saealib import Algorithm
from saealib.algorithms import ProposalRequest
from saealib.core.contracts import FeedbackBatch, ProposalBatch
from saealib.core import StatePatch, StateView


class CustomAlgorithm(Algorithm):
    """Skeleton for an Algorithm that generates candidates and consumes Feedback."""

    def ask(self, request: ProposalRequest, state: StateView) -> ProposalBatch:
        # Read the current Population from state.context and declared StateKeys.
        # Return a ProposalBatch with candidate IDs and FeedbackRequirement.
        ...

    def tell(self, feedback: FeedbackBatch, state: StateView) -> StatePatch:
        # Read the FeedbackBatch and collect changes to Population and other state in a StatePatch.
        ...
```

次の最小Algorithmは `ask`/`tell` だけを実装し、標準のPopulationとArchiveでそのまま使えます:

```python
import numpy as np
from saealib import Algorithm, Optimizer, Problem, DirectStrategy, LHSInitializer, Termination, max_gen
from saealib.algorithms import ProposalRequest
from saealib.algorithms.base import algorithm_context
from saealib.core.contracts import (
    FeedbackBatch,
    FeedbackRequirement,
    ProposalBatch,
    ProposalRelations,
)
from saealib.core import StatePatch, StateView


class KeepPopulation(Algorithm):
    """A minimal ask/tell-only algorithm that keeps the population as-is."""

    def ask(self, request: ProposalRequest, state: StateView) -> ProposalBatch:
        ctx = algorithm_context(state)
        candidates = ctx.population.empty_like(capacity=0)
        return ProposalBatch.from_allocator(
            ctx.proposal_id_allocator,
            candidates=candidates,
            relations=ProposalRelations(row_count=0),
            requirements=FeedbackRequirement(quantities=()),
        )

    def tell(self, feedback: FeedbackBatch, state: StateView) -> StatePatch:
        return StatePatch(writes={})


problem = Problem(
    lambda x: np.sum(x**2),
    dim=2,
    n_obj=1,
    direction=np.array([-1.0]),
    lb=[-2.0] * 2,
    ub=[2.0] * 2,
)
optimizer = (
    Optimizer(problem, seed=0)
    .set_initializer(LHSInitializer(4, 4, seed=0))
    .set_algorithm(KeepPopulation())
    .set_strategy(DirectStrategy())
    .set_termination(Termination(max_gen(2)))
)
result = optimizer.run()
```

`tell()` が受け取るFeedbackは、どの評価経路から届いたかと、候補IDとの対応を保持します。
各提案で必要なフィードバックは、Algorithmの契約と`FeedbackRequirement`（`ProposalBatch.requirements`に保持され、Feedback契約との適合確認と必要なフィードバックの完了判定に使われます）で示します。
PSOのpbestのような永続的な補助情報は、Population属性または宣言済みの`StateKey`として扱います。

## 関連コンポーネント

- [Crossover](crossover.md) / [Mutation](mutation.md) / [ParentSelection](parent_selection.md) / [SurvivorSelection](survivor_selection.md)：`GA`が組み合わせる4つの演算子。
- [Problem](../problem_and_ranking/problem.md)：`variables`による混合変数の定義
- [ConstraintHandler](../problem_and_ranking/constraints.md): `repair()` that `GA` uses へrepair 候補s。
- [OptimizationStrategy](../execution_and_evaluation/strategies.md): Decides whether へperform 真の評価 between `ask()`/`tell()`。
- [Population](../observation_and_state/population.md): Algorithmの互換性経路で利用するPopulationと`GenomeBatch`

## 参照

- {py:class}`saealib.Algorithm`
- {py:class}`saealib.GA`
- {py:class}`saealib.PSO`
- {py:class}`saealib.PymooAlgorithm`
- {py:class}`saealib.DuplicateElimination`
- {py:func}`saealib.repair_clipping`
