---
primary_layer: cross
related_layers: [layer3, layer4]
page_type: guide
---

# 拡張ガイドライン

このページは、既存コンポーネントの軽量な差し替えと互換性用Pipelineの構成を扱います。 拡張は、既存コンポーネントの差し替えとフレームワークの拡張に分かれます。

- 既存の契約にコンポーネントを追加する場合は [独自Componentを実装する](../tutorials/custom_components.md) を参照してください。
- 契約、探索空間、Graph、Runtimeを追加する場合は [フレームワーク拡張](../framework/extensions.md) を参照してください。

import経路の規則は [Canonical Imports](../api/imports.md) にまとめています。

## 拡張点を選ぶ

変更したい責務に対応する最も浅い境界を選びます。 新しい実行契約が必要な場合だけFrameworkへ進み、既存の契約で表現できる差し替えはチュートリアルの手順で実装します。

| 変更したいこと | 主な境界 | 概念 | APIリファレンス |
|---|---|---|---|
| 候補生成や個体更新を変更する | `Algorithm` | [Algorithm](search_algorithms/algorithm.md) | [アルゴリズム](../api/algorithms.md) |
| 交叉、突然変異、選択を変更する | 演算子 | [Crossover](search_algorithms/crossover.md)、[Mutation](search_algorithms/mutation.md) | [演算子](../api/operators.md) |
| 予測モデルや学習データを変更する | `Surrogate`、`SurrogateManager` | [Surrogate](surrogate_modeling/surrogate.md) | [代理モデル](../api/surrogate.md) |
| 候補の選択や評価順序を変更する | `OptimizationStrategy`、`Stage` | [Strategy](execution_and_evaluation/strategies.md) | [戦略](../api/strategies.md) |
| 外部ライブラリを接続する | 明示的なAdapter | [Evaluation](execution_and_evaluation/evaluation.md) | [評価](../api/evaluation.md) |
| 候補表現や変数制約を追加する | `SearchSpace` | [SearchSpace](problem_and_ranking/search_space.md) | [Space](../api/space.md) |
| 入出力、状態、ライフサイクルを宣言する | `ComponentContract` | [Contract](../framework/contract.md) | [Coreリファレンス](../api/core.md) |
| 接続や互換性検査を変更する | `ComponentGraph`、Compiler rule | [Graph](../framework/graph.md)、[Compiler](../framework/compiler.md) | [Coreリファレンス](../api/core.md) |
| 実行、再開、非同期待機の意味を変更する | `ExecutionRuntime` | [Runtime](../framework/runtime.md) | [Executionリファレンス](../api/execution.md) |

フレームワーク契約は `saealib.core` から取得し、実装モジュールを安定した拡張APIとして扱いません。

`Algorithm`、`OptimizationStrategy`、`Surrogate`、`AcquisitionFunction`、`SurrogateManager`には、それぞれ差し替え可能な契約があります。 独自の探索や予測を実装するときは、対象のドメインコンポーネントを実装して `Optimizer.set_*()` から登録します。 既存コンポーネントの一部だけを変更する場合は、次の軽量な拡張手段を使えます。

## 文脈依存フック

文脈依存の公開フックは、末尾のキーワード専用・必須引数 `ctx`（型は `OptimizationState | None`）という共通の引数規則に従います。 `ParentSelection.select`、`SurvivorSelection.select`、`EvaluationPlanner.plan`、`FeedbackBuilder.build`、`TrainingSet.build`、`AcquisitionFunction.prepare`/`evaluate`、`SurrogateManager.fit`/`predict`が対象です。 `ctx`を位置引数で渡したり省略したりする呼び出しは `TypeError` になるため、キーワード引数で渡してください。 独自オーバーライドも同じ形で `ctx` を宣言します。

```python
from saealib import OptimizationState, ParentSelection


selection.select(
    population,
    n_pair=4,
    n_parents=2,
    rng=rng,
    ctx=state,
)


class MySelection(ParentSelection):
    def select(
        self,
        population,
        n_pair,
        n_parents,
        rng=np.random.default_rng(),
        *,
        ctx: OptimizationState | None,
    ):
        ...
```

Selectionは比較と乱数を必要とするため、組み込み実装は `ctx=None` に対して `ValidationError` を送出します。 実行の外から単独で呼ぶときは `ctx=None` を明示的に渡し、他のファミリーはそれぞれが必要とする文脈の拒否動作を保ちます。 純粋な計算にとどまるフック（`Surrogate.fit`/`predict`、交叉・突然変異の本体、`Comparator`、`Decomposition`、`Variable`）は `ctx` を受け取りません。

```{note}
コンポーネントに `OptimizationState` 全体を渡すのはフルコンテキストの退避路です。 独自コンポーネントは `ctx` を無視しても最小限の宣言で動作します。
```

## with_post / with_post_fitによる後処理

[Crossover](search_algorithms/crossover.md)/[Mutation](search_algorithms/mutation.md) の `with_post(fn)` と [Surrogate](surrogate_modeling/surrogate.md) の `with_post_fit(fn)` は、サブクラス化せずに既存インスタンスへ後処理を追加します。元のインスタンスは変更せず、フックを追加したコピーを返します。

用例は、`Crossover`/`Mutation`への修復関数の追加や、`Surrogate`のフィット後処理です。 各コンポーネントページの「拡張フック」節に、コンポーネントごとの具体例があります。

## PipelineとStage

`Stage` と `Pipeline` には、互換性用の実行経路と構造化DSLの二つの境界があります。 Stageは `OptimizationState` を受け取り、更新した状態を返す互換性用の実行単位です。 一方、構造化Pipelineはgraph-native component、入れ子のPipeline、`Repeat`、`Loop`、`Branch`を保持し、状態を直接実行しません。

構造化PipelineへStageを入れるときは、`stage_component(stage)` で互換性境界を明示します。 Optimizerは内部でこのDSLをGraphへlowerし、構造化Runtimeで実行できる計画へ変換します。 通常の利用者がCompilerを直接呼び出す必要はありません。

`pipeline.replace("name", entry)` はトップレベルの構造エントリのみを置き換えます。
入れ子のエントリを置き換えるには、`find(name, recursive=True)` で対象の入れ子パイプラインを取得してその上で `replace` を呼び出します。
準備済みの実行において差し替えが反映されるのは次の確定点（commit point）です。
`to_spec` はコンストラクタ引数をシリアライズするため、組み込み戦略での差し替え内容はspecに保持されません。
差し替えを永続化するには、宣言やコンストラクタ引数を利用します。
`pipeline.find("name", recursive=False)` は名前でエントリを検索します。

```python
from saealib import Pipeline, Stage
from saealib.stages import stage_component
from saealib.stages import CountGenerationStage


class DoubleCountStage(Stage):
    name = "count_generation"

    def execute(self, state):
        return state.replace(gen=state.gen + 2)


pipeline = Pipeline(
    steps=[
        stage_component(CountGenerationStage()),
        # Add a graph-native component or another Stage adapter next.
    ]
)
pipeline.replace("count_generation", stage_component(DoubleCountStage()))
```

[Stage](observation_and_state/stage.md) はStageの契約、組み込みStage、互換性用のカスタムStageを説明します。 このページでは、既存の構成を置き換えるときにどの境界を選ぶかを説明します。

## CallbackManager

[CallbackManager](observation_and_state/callbacks.md) はEvent発火時にハンドラを呼び出す観測機構です。 `candidates` などのEventフィールドは観測用であり、再代入してもPipelineの入力は変わりません。 データフローの後処理には `with_post()` / `with_post_fit()`、実行単位の置換には `Stage`、実行構成の変更には `Runtime` を選びます。

`iterate()` / `run()` のステップ境界で `optimizer.set_*()` やコンポーネント属性を変更すると、実行環境が変化を検出してplanを再コンパイルし、次世代から反映します。 この手順はStage互換経路とgraph-native経路のどちらでも利用できます。 Component側から再コンパイルを要求する経路については [OptimizationStrategy](execution_and_evaluation/strategies.md) の「Behavior of runtime swapping」を参照してください。 Callbackは観測専用で、ハンドラから `RuntimeCommand` を返すことはできません。 Callbackのクロージャで `optimizer.set_*()` を呼ぶ変更は上記の経路として反映されますが、構成変更は `iterate()` 側の手順として案内します。

## レジストリ

`saealib.registry`は、名前（文字列）またはspec（`{"type": "Name", "params": {...}}`）から実インスタンスを構築する仕組みです。 `with_post`/`Pipeline`-`Stage`/`CallbackManager`が「実行時の挙動を変える」ための機構であるのに対し、Registryは「コンポーネントを文字列や設定ファイルから組み立てる」ための機構であり、目的が異なります。 プリセットYAML経由の設定駆動構築（`Optimizer.set_preset()`）やチェックポイント再開のように、クラスを直接importしない場面で使います。

**`register(name=None)`**（デコレータ）：クラスや関数をレジストリへ登録します。 独自の`Algorithm`/`Surrogate`等のサブクラスに`@register()`を付けるだけで、組み込みコンポーネントと同じように短い名前で参照できるようになります。

```python
from saealib import register
from saealib.surrogate import Surrogate


@register()
class MyCustomSurrogate(Surrogate):
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    ...
```

`get`/`build`/`to_spec`はsaealibのトップレベルからは公開されておらず、`saealib.registry`から直接importします。

**`get(name)`**：登録名、または未登録なら`"module.submodule.ClassName"`形式のドットパスとして解決します。

**`build(spec)`**：specを再帰的に実インスタンスへ構築します。 spec内の値がさらに入れ子のspecであれば再帰的に構築されます。 `{"callable": "dotted.path"}`という形式は、関数や組み込み関数そのものを（呼び出さずに）解決します。

**`to_spec(obj)`**：`build()`の逆操作。 コンストラクタシグネチャを反映し、同名属性を読んでspecへ再帰的にシリアライズします。 `_registry_spec`属性を持つクラス（`TerminationCondition`など）はこの汎用リフレクションを使わず、その属性を直接返します。 `Optimizer.save_preset()`が使う経路です。

```python
from saealib.registry import build, get, to_spec

obj = build({"type": "MyCustomSurrogate", "params": {"alpha": 2.0}})
get("MyCustomSurrogate")  # -> the MyCustomSurrogate class
to_spec(obj)  # -> {"type": "MyCustomSurrogate", "params": {"alpha": 2.0}}
```

各コンポーネントページで「◯◯クラスは`@register()`未登録」という注記が複数箇所にありますが、いずれもRegistry経由でクラス名から解決する使い方をする場合にのみ影響します。

## 使い分けの指針

| やりたいこと | 使う機構 |
|---|---|
| 既存の演算子やサロゲートに後処理を足すだけ | `with_post` / `with_post_fit` |
| ステージの並び自体を変えたい | `Pipeline` / `Stage` |
| 外部観測、ログ、条件判断 | `CallbackManager` |
| 設定ファイルやプリセットから組み立てたい | `Registry` |

## 関連コンポーネント

- [Crossover](search_algorithms/crossover.md) / [Mutation](search_algorithms/mutation.md) / [Surrogate](surrogate_modeling/surrogate.md)：`with_post`系フックを持つコンポーネント
- [Stage](observation_and_state/stage.md)：`Pipeline`が組み合わせる各ステージの契約
- [CallbackManager](observation_and_state/callbacks.md)：イベント一覧と観測の仕組み
- [strategies](execution_and_evaluation/strategies.md)：パイプラインの再構築タイミング

## 参照

- {py:func}`saealib.register`
