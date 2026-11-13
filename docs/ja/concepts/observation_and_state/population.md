---
primary_layer: layer2
page_type: concept
---

# Population

`Population`は、個体と評価結果を扱うドメインデータ構造です。
旧来のベクトル経路では、`x`、`f`、`g`、`cv`列を持つ構造化配列です。
現在のグラフネイティブ経路では、`GenomeBatch`と`SearchSpace`が個体の表現を定義し、`Population.x`は可能な場合にだけ利用できる互換性ビューです。

`Initializer` はProblemのSearchSpaceからGenomeを生成し、候補IDとともに状態やArchiveへ登録します。
したがって、すべてのProblemを浮動小数点の `x` 配列へ変換できるとは限りません。

## Populationの役割

`Population`は、Genome、目的値、制約値、制約違反、Algorithm固有の補助属性を保持します。
Genomeは、`DenseVectorBatch`、`ObjectBatch`、`PermutationBatch`などの`GenomeBatch`実装で表現できます。
値を安全に数値ベクトルとして取得できるGenomeでは、`x`から数値ビューの値を参照できます。
`Population`のコンストラクタでは、数値ビューサービスを`dense_numeric_view`引数に指定します。
`dense_view`は`dense_numeric_view`の非推奨エイリアスであり、指定すると`DeprecationWarning`が出力されます。

`Population.genomes`は、実行中のPopulationのGenome格納領域から借用する読み取り専用ビューであり、不変スナップショットではありません。
`update_array`など後続のPopulation操作はその格納領域をインプレースで更新できるため、既存の`genomes`ビューから観測される内容も変化する場合があります。

`Population` は通常インスタンスとして使い、独自の個体管理が必要な場合だけサブクラス化します。

属性スキーマは `PopulationAttribute(name, dtype, shape, default)` のリストで定義します。
旧来のベクトル経路では `x`、`f`、`g`、`cv` と、`Algorithm.get_required_attrs(problem)` が返す補助属性をこのスキーマへ追加します。
`get_required_attrs` の既定は空です。アルゴリズムが固有の状態を個体群に格納するときだけ属性を宣言します。
Genome-native経路では、GenomeBatchとSearchSpaceの契約が表現の構造を担い、列スキーマだけで個体を表しません。

## Populationが保持するもの

| メソッド | 役割 |
|---|---|
| `get_array(key)` | 属性の生配列を取得する。書き込み禁止のviewを返す |
| `update_array(key, value)` | 属性配列を一括更新する |
| `get(key, default=None)` | 存在しない属性なら`default`を返す、安全な取得 |
| `append(element=None, **kwargs)` | 個体を1件追加する |
| `extend(other)` | 別の`Population`またはdictから個体群をまとめて追加する |
| `extract(indices)` | インデックス配列/スライスで部分集合を新しい`Population`として取り出す |
| `truncate(new_size)` / `delete(index)` / `clear()` | サイズ変更と削除 |
| `reorder(order)` / `argsort(name, reverse=False)` | 並べ替え |
| `empty_like(capacity=None)` | 同じスキーマの空`Population`を作る |
| `set_cache(key, value)` / `get_cache(key)` | 計算結果をPopulation変更まで有効なキャッシュとして保持する |
| `pop[i]` / `pop[a:b]` | 単一intなら`Individual`を、sliceなら部分集合の`Population`を返す |
| `len(pop)` | 個体数 |

`set_cache`/`get_cache`によるキャッシュは、`append`/`delete`/`update_array`など個体群を変更する操作を呼ぶたびに自動的に無効化されます。
[NSGA2Comparator](../problem_and_ranking/comparators.md)がフロント分割と混雑度距離の計算結果を世代内で使い回す際に、この仕組みを利用しています。

```python
import numpy as np
from saealib import Population, PopulationAttribute

attrs = [
    PopulationAttribute("x", np.float64, (2,)),
    PopulationAttribute("f", np.float64, (1,)),
    PopulationAttribute("cv", np.float64, ()),
]
pop = Population(attrs, init_capacity=4)
pop.append(x=np.array([0.1, 0.2]), f=np.array([1.0]), cv=0.0)
pop.append(x=np.array([0.3, 0.4]), f=np.array([2.0]), cv=0.0)

pop.x  # design-variable array with shape (2, 2)
pop[0]  # an Individual view of the first individual
pop[0:1]  # a Population containing only the first individual
```

### Individual

`Individual`は`pop[i]`で得られる、単一個体への軽量なビューです。
実データを複製せず、参照元の`Population`と自分のインデックスだけを保持します。

`get_readonly_value(key)`/`update_value(key, value)`で値の読み書きができるほか、`ind.x`/`ind.f = ...`のような属性アクセスでも同じ読み書きができます。
参照元の`Population`の構造（個体数や並び順）が変わった後に古い`Individual`を使うと、無効な参照として例外になります。

## 不変条件

個体属性の配列は同じ個体数とスキーマを共有し、変更操作はキャッシュを無効化します。行数やスキーマが一致しないまま更新すると、個体と評価結果の対応が壊れます。

## Archive

`Archive`は、`ArchiveMixin`を`Population`にミックスインした具象クラスで、評価済み解を重複なく蓄積する目的で使います。

`add(element, **kwargs)`は`append`とほぼ同じ引数を取りますが、重複解を無視する点が異なります。
重複判定に使う属性は`key_attr`引数（既定`"x"`）で指定し、`atol`/`rtol`で許容誤差を調整します。
`get_knn(x, k)`はkd-tree（初回呼び出し時に遅延構築される）による近傍検索を提供し、[LocalSurrogateManager](../surrogate_modeling/surrogate_manager.md)の既定`training_set`が候補ごとの局所学習データを集める際に使います。

```python
from saealib import Archive

arc = Archive(attrs, init_capacity=4, key_attr="x")
arc.add(x=np.array([0.1, 0.2]), f=np.array([1.0]), cv=0.0)
arc.add(
    x=np.array([0.1, 0.2]), f=np.array([1.0]), cv=0.0
)  # the duplicate solution is ignored
idx, dist = arc.get_knn(np.array([0.1, 0.2]), k=1)
```

## ParetoArchive

`ParetoArchive`は、`ParetoMixin`を`Population`にミックスインした具象クラスで、非優越解集合を常時維持します。

新規解を追加するたびに、その解に支配される既存解を削除し、新規解自体が既存解に支配されている場合はその新規解を破棄します。
支配関係の判定は実行可能性優先方式で行われます。
実行可能解は常に実行不可能解を支配し、両方が実行可能な場合にのみ[Dominator](../problem_and_ranking/dominance.md)の`dominates`が使われます。
アーカイブがProblemの制約を`constraints`引数で受け取り、行が`g`列を持つ場合、実行可能性は制約定義から決まります。制約ごとの違反量が許容幅の内側ですべて0の行が実行可能であり、`Result`や履歴と同じ判定です。
`Optimizer`はアルゴリズム経由でProblemの制約を渡してアーカイブを作り、checkpointからの復元でも同じ制約を渡します。

`dominator`引数で支配関係の定義を差し替えられます。
制約または`g`列が無い場合、実行可能性は`cv <= eps_cv`にフォールバックします。`eps_cv`の既定値は`0.0`で、実行可能な解だけが0になる集約に対しては制約による判定と一致します。
探索中のε緩和はcomparatorが持ち、アーカイブには書き込まれません。

## Populationの拡張

`ArchiveMixin`/`ParetoMixin`は、`Population`のサブクラスに多重継承でミックスインするという前提で設計されています。
独自の個体群管理ロジックが必要な場合、これらのMixinを組み合わせた新しいクラス（`class MyArchive(ArchiveMixin, Population): ...`）を定義できます。
また[Algorithm](../search_algorithms/algorithm.md)の`population_class`/`archive_class`をオーバーライドすれば、`Initializer`が生成するPopulation/Archiveを独自サブクラスに差し替えられます。 どちらも標準の`Population`/`Archive`が既定のため、オーバーライドは任意です。

## 関連コンポーネント

- [Initializer](../execution_and_evaluation/initialization.md)：`Population`/`Archive`/`ParetoArchive`を実行開始時に構築する
- [OptimizationState](optimization_state.md)：構築後の`Population`/`Archive`/`ParetoArchive`を保持する状態オブジェクト
- [Algorithm](../search_algorithms/algorithm.md)：`population_class`/`archive_class`で具象クラスを差し替える
- [Comparator](../problem_and_ranking/comparators.md)：`set_cache`/`get_cache`によるソート結果の使い回し
- [Dominance](../problem_and_ranking/dominance.md)：`ParetoArchive`が非優越解判定に使う`Dominator`
- [SurrogateManager](../surrogate_modeling/surrogate_manager.md)：`Archive.get_knn`を使う局所学習データの収集

## 参照

- {py:class}`saealib.Population`
- {py:class}`saealib.Individual`
- {py:class}`saealib.PopulationAttribute`
- {py:class}`saealib.Archive`
- {py:class}`saealib.ArchiveMixin`
- {py:class}`saealib.ParetoArchive`
- {py:class}`saealib.ParetoMixin`
