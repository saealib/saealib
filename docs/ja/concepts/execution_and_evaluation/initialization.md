---
primary_layer: layer2
related_layers: [layer3]
page_type: concept
---

# Initializer

実行の開始時に、`Optimizer`は初期`Population`、`Archive`、`ParetoArchive`、`OptimizationState`の構築を`Initializer`へ委譲します。
サンプリング方法を変える場合は、Optimizer全体ではなくInitializerを差し替えます。

現行の標準経路では、`GenomeInitializer` が `Problem.space` からGenomeをサンプリングします。
`LHSInitializer`、`RandomInitializer`、`SobolInitializer` は、ベクトルの `x` 配列を使う互換性用Initializerとして残っています。

## Initializerの役割

`Initializer` の実装境界は `initialize(provider, problem) -> OptimizationState` です。
`provider` は、AlgorithmやEvaluatorなど、すでに構築されたComponentへアクセスするためのProviderです。
`problem` は対象となる [Problem](../problem_and_ranking/problem.md) です。

## 組み込みInitializer

| クラス | サンプリング方法 |
|---|---|
| `GenomeInitializer` | `Problem.space.sample()` によるGenome生成 |
| `LHSInitializer` | `scipy.stats.qmc.LatinHypercube` |
| `RandomInitializer` | `rng.uniform` |
| `SobolInitializer` | `scipy.stats.qmc.Sobol(scramble=True)` |

`GenomeInitializer`は、SearchSpaceが提供するGenomeを直接Archiveへ登録します。
他の3クラスはコンストラクタ`(n_init_archive, n_init_population, seed=None)`を共有し、ベクトル形式で初期点を生成します。
4クラスとも、初期Archiveを評価し、Comparatorでランク付けしたうえで、上位の候補を初期Populationへ渡します。

```
Sample (n_init_archive points)
  -> Evaluate via provider.evaluator.evaluate_batch
  -> add to archive / pareto_archive
  -> Rank via problem.comparator.rank_population
  -> Feed the top n_init_population into population
```

これは、前回の選択の状態を再利用するのではなく新しく組み立てたArchiveをランク付けするため、`sort_population`を直接ではなく`rank_population`を呼び出します——両者の違いは[Comparator](../problem_and_ranking/comparators.md)を参照してください。
`SPEA2Comparator`のようなComparatorにとっては、これが初期Populationにその永続化されたランキング状態（`spea2_fitness`）を最初に書き込む場所でもあります。

3クラスの実装はサンプリング方法の1行以外ほぼ重複していますが、これは意図的な単純さの選択であり、共通処理を過度に抽象化しない設計になっています。

## 基底クラスのヘルパーメソッド

`Initializer`基底には、独自実装で再利用できる2つのヘルパーメソッドが用意されています。

**`_create_attrs(problem, provider)`**：`Population`と`Archive`用の`PopulationAttribute`値を構築します。
レガシーなベクトル経路では`x`/`f`/`g`/`cv`/`id`を使い、続けてAlgorithm（`Algorithm.get_required_attrs`）とComparator（`Comparator.get_required_attrs`）の両方が要求する補助属性を統合します——例えば`SPEA2Comparator`の`spea2_fitness`属性がスキーマに加わるのはこの仕組みによるものです。
`GenomeInitializer`はGenomeを専用カラムで管理するため、標準属性に`x`は含まれません。
`objective_source`列は各行の目的値の由来を記録し、コードの意味は`OptimizationState`のdocstringに記載されています。

**`_create_context(problem, archive, pareto_archive, population, rng)`**：`OptimizationState` を構築します。
Comparatorが `NSGA3Comparator` で内部乱数生成器をまだ持たない場合は、ここで `rng.spawn(1)[0]` を設定します。

## 独自Initializerの実装方法

独自のベクトルサンプリングを追加する場合は `Initializer` を継承して `initialize()` を実装します。
新しいGenome表現を使う場合は、`Problem.space.sample()` と `Problem.space.validate()` を使うInitializerを実装します。

Genome-nativeな初期化では、次の順序を保ちます。

1. Algorithmの `population_class`、`archive_class`、`create_pareto_archive()` で入れ物を作る。最初の2つは標準の `Population`/`Archive` が既定です。
2. `OptimizationState` を構築する。
3. `problem.space.sample(n, rng)` でGenomeを生成し、`problem.space.validate()` で検証する。
4. `provider.evaluator.evaluate_batch(genomes, problem)` で評価する。
5. 候補ID、Genome、評価結果をArchiveとParetoArchiveへ登録する。
6. 評価回数を更新し、Comparatorで並べた候補をPopulationへ渡す。

次の骨格は、Genome-nativeなInitializerが使う入力と出力の境界を示しています。

```python
from saealib import GenomeInitializer


class CustomGenomeInitializer(GenomeInitializer):
    def initialize(self, provider, problem):
        # Replace problem.space.sample() with custom sampling.
        # Validate each generated Genome with space.validate() before passing it to the Evaluator.
        ...
```

初期評価の開始と終了は `CallbackManager` から観測できます。
評価Requestの候補IDとGenomeを維持する必要があるため、Genomeを `x` 配列へ暗黙に変換して管理しません。

`Optimizer.set_initializer(initializer)` でInitializerを差し替えます。

## 関連コンポーネント

- [OptimizationState](../observation_and_state/optimization_state.md)：`initialize()`が最終的に返す状態オブジェクト
- [Population](../observation_and_state/population.md)：構築対象の`Population`/`Archive`/`ParetoArchive`
- [Evaluator](evaluation.md)：初期サンプルの評価に使う
- [CallbackManager](../observation_and_state/callbacks.md)：`InitialEvaluationStartEvent`/`InitialEvaluationEndEvent`の観察
- [Comparator](../problem_and_ranking/comparators.md)：初期Archiveのランク付けに使う`rank_population`

## 参照

- {py:class}`saealib.Initializer`
- {py:class}`saealib.LHSInitializer`
- {py:class}`saealib.RandomInitializer`
- {py:class}`saealib.SobolInitializer`
