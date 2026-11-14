---
primary_layer: layer4
related_layers: [layer3]
page_type: concept
---

# OptimizationState

`OptimizationState`は、`Optimizer`の実行コンテキストとチェックポイントを保持する値です。Stage互換経路では、`Stage`がこの状態を直接受け取り、`replace()`で更新したコピーを次の`Stage`へ渡します。

構造化ランタイムでは、グラフネイティブコンポーネントが`OptimizationState`全体を受け取ることはありません。コンポーネントは宣言した`StateKey`に限定された`StateView`を受け取り、変更を`StatePatch`として返します。ランタイムはそのパッチを`StateStore`へ適用し、次の実行境界を生成します。

## OptimizationStateの役割

`OptimizationState`は、Stage互換経路が実行に使う値、利用者が確認する結果、再開に必要な値の3つをまとめます。コンポーネント契約、グラフ構造、コンパイラの規則を保持する場所ではありません。状態の所有者は、Stage互換経路では`Optimizer`と状態自体、graph-native経路では`StateStore`です。

`OptimizationState`は`replace()`で新しい値を生成し、`StateStore`は`StatePatch`を適用して新しい世代を生成します。コンポーネントが`StateStore`や`OptimizationState`を直接変更できると、コンパイラもランタイムも、契約外で行われた書き込みを追跡したり、もはや一致しない再開地点を検出したりできません。

## 二つの状態境界

`OptimizationState` と `StateView` は、異なる実行境界を表します。

| 境界 | 受け取る値 | 返す値 | 主な利用者 |
|---|---|---|---|
| Stage互換境界 | `OptimizationState` | `OptimizationState` | 既存Stage、sequential compatibility ランタイム |
| graph-native境界 | 宣言済みの `StateView` | `StatePatch` または `NodeResult` | Component、Compiler、構造化ランタイム |

この二つの境界を混同すると、グラフネイティブコンポーネントが任意の状態を読み書きできるように見えます。構造化パイプラインでは、`ComponentContract`が読み書きする`StateKey`を宣言し、ランタイムはその範囲だけをコンポーネントへ渡します。

## OptimizationStateが保持する値

`OptimizationState` は、利用者が結果を確認したり、実行を再開したりするための値を保持します。

| 値 | 内容 |
|---|---|
| `problem` | 最適化対象の`Problem` |
| `population` | 現在の`Population`への互換性用ショートカット |
| `archive` | 評価済み解の`Archive`への互換性用ショートカット |
| `pareto_archive` | 非劣解集合への互換性用ショートカット |
| `rng` | 乱数生成器 |
| `fe` | 真の評価回数 |
| `gen` | 世代数 |
| `step` | 完了したフロー実行数（単位：回、初期値：0） |
| `decision_count` | 確定した評価計画数（インフィル判断数、初期値：0） |
| `region_counts` | 修飾識別子ごとの名前付きRepeat領域とLoop領域の累積反復数（初期値：`{}`） |
| `comparator` | Optimizer向けに解決された`Comparator` |
| `data` | Stage互換拡張用の追加データ |

`population`、`archive`、`pareto_archive`は、内部で名前付きのコレクションを参照します。新しいグラフネイティブコンポーネントが任意の値を`data`へ入れる設計にはせず、代わりに`StateContract`で`StateKey`を宣言します。

## StateStore、StateView、StatePatch

`StateStore`は、型付きの`StateKey`を値に対応付ける状態ストアです。コンポーネントに渡される`StateView`は、`ComponentContract`が宣言した読み取りキーだけを公開します。

Componentが状態を変更するときは、直接Storeを変更せずに `StatePatch` を返します。 RuntimeはPatchを現在のStoreへ適用し、次の状態を生成します。 この方式によって、Componentが宣言していないStateを偶然に読み書きすることを防ぎます。

コンパイラは、`StateContract`のreads、writes、exportsが`StateBinding`と対応していることを確認します。ランタイムは`StateView`を宣言済みの読み取りキーに限定し、`StatePatch`の書き込みをストアの型付きキーと世代へ適用します。同じキーで競合するパッチや、存在しないキーを削除するパッチは、ランタイムの診断規則に従って扱います。

```python
from saealib.core import StatePatch, StateView
from saealib.core import RUNTIME_GENERATION


def execute(view: StateView) -> StatePatch:
    current = view.get(RUNTIME_GENERATION)
    return StatePatch(writes={RUNTIME_GENERATION: current + 1})
```

実際の`StateKey`は文字列ではなく、`saealib.core.state`が提供する型付きキーです。上のコードは、ビューから読み取り、パッチを返す境界だけを示すよう簡略化しています。

## Stage互換経路での更新

カスタム`Stage`は`OptimizationState`を受け取り、`state.replace(...)`で更新後の状態を返します。

```python
from saealib import Stage


class LogGenerationStage(Stage):
    name = "log_generation"

    def execute(self, state):
        print(state.gen)
        return state
```

Stage互換経路で追加の値を持たせる場合は、`state.data`をコピーしてから`replace()`に渡します。グラフネイティブコンポーネントで同じ値を扱う場合は、`StateKey`と`StatePatch`を使います。

## 更新とチェックポイント

`replace(**kwargs) -> OptimizationState` は、Stage互換経路で値を更新するためのメソッドです。 `archive` は評価結果を蓄積するために変更され、`rng` は乱数を生成するたびに内部状態が進みます。 これらは、他の値を `replace()` で渡す設計と異なる、明示された例外です。

`save(path)`と`load(path, problem)`は`OptimizationState`のnpzチェックポイントを扱います。自動チェックポイントについては[Checkpointing](../../tutorials/checkpoint.md)を参照してください。pickle形式では`Optimizer`側の状態も必要なため、どの保存経路を使うかは`CheckpointCallback`と`Optimizer`の組み合わせで決まります。

## どの状態境界を使うか

- 既存のStageを実装する：`OptimizationState` と `replace()` を使う。
- 新しいグラフネイティブComponentを実装する：`ComponentContract`、`StateView`、`StatePatch` を使う。
- Optimizerの実行結果を確認する：`Optimizer.run()` または `Optimizer.iterate()` が返す `OptimizationState` を使う。
- 実行を再開する：`saealib.context.OptimizationState.load()` と `Optimizer.run_from()` を使う。

## 代表的な失敗

Stage互換経路で誤った対象に`replace()`を呼ぶと、次のStageが期待する値を受け取れません。graph-native経路で`StateContract`にないキーを読み取り、書き込み、または公開すると、コンパイラの状態効果検証で診断になります。別の`Problem`や互換性のない計画にチェックポイントを適用すると、保存された状態と実行境界がずれます。

## 関連コンポーネント

- [Stage](stage.md)：`OptimizationState` を直接受け取る互換性用の実行単位
- [Framework](../../framework/index.md)：Component、Contract、Graph、Compilerの関係
- [Runtime](../../framework/runtime.md)：計画、ランタイム、`StatePatch`が適用される境界
- [Population](population.md)：`Population`、`Archive`、`ParetoArchive`のデータ構造
- [Checkpointing](../../tutorials/checkpoint.md)：チェックポイントの保存と再開

## 参照

- {py:class}`saealib.context.OptimizationState`
- {py:class}`saealib.core.StateStore`
- {py:class}`saealib.core.StateView`
- {py:class}`saealib.core.StatePatch`
- [Framework ComponentContract](../../framework/contract.md)：状態宣言を含む契約
- [Framework Compiler](../../framework/compiler.md)：状態効果の検証と診断
