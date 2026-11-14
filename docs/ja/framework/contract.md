---
primary_layer: layer4
related_layers: [layer3]
page_type: concept
---

# ComponentContract

Componentが要求する能力と提供する境界は、`ComponentContract`という一つの宣言値にまとめます。Componentとの関係は継承ではなく、Componentが返す契約の包含です。

## ComponentContractの役割

`ComponentContract`は、Componentが要求する能力と提供する境界を一つの宣言値として所有します。Componentの継承階層を表すものではなく、GraphとCompilerが接続・実行条件を検証するための境界です。

## 構造要素

| フィールド | 宣言する内容 |
|---|---|
| `ports` | roleごとの`PortContract` |
| `required_services` | Component全体が必要とする`ServiceRequirement` |
| `parts` | 保持するコンポーネントの`PartSpec` |
| `lifecycle` | `events`とFeedbackの境界 |
| `state` | `reads`、`writes`、`exports`、`reads_enumerable` |
| `execution` | `required_runtime_capabilities`と`offered_runtime_capabilities` |
| `assumptions` | コンパイラが扱う前提条件 |

`ports`のroleは、同じComponentをGraph上の役割ごとに接続するための名前付き集合です。`parts`は子Componentの実装を複製する値ではなく、子の契約を親の契約へ含める宣言です。

## 各契約が保持する情報

| 契約 | 保持する情報 | 主な検証 |
|---|---|---|
| `PortContract` | 入力と出力の`PortSpec` | ポート名、方向、DataSpec、cardinality |
| `StateContract` | 読み取り、書き込み、公開する`StateKey` | 宣言外の状態アクセス、状態効果 |
| `LifecycleContract` | 消費するイベントとFeedbackContract | イベントとFeedbackの互換性 |
| `ExecutionContract` | 必要または提供するRuntimeの機能 | 実行環境の能力不足 |
| `AssumptionSet` | コンポーネントが置く前提条件 | 前提条件の登録と既定値 |

## 生成時点と利用時点

Componentは`contract()`で契約を生成し、ComponentNodeはコンパイル単位のスナップショットを保持します。Compilerは契約をポート互換性、サービス解決、状態効果、ライフサイクル、Runtimeの機能の検証に使います。Runtimeは検証済みのExecutablePlanに含まれる契約境界に従ってStateViewを作り、結果のStatePatchを適用します。

## 不変条件と診断

契約は凍結されたデータクラスとして扱われ、Compilerの検証中に変更されません。`ports`、`required_services`、`parts`は対応する宣言値だけを含み、role名とコンポーネント名は識別子として検証されます。状態の`reads`、`writes`、`exports`は型付きStateKeyを使い、Runtimeは宣言されていないキーをComponentに公開しません。必要な実行能力がCompileContextの提供する能力にない場合、CompilerはExecutablePlanを実行可能としてマークできません。呼び出せない契約や誤った戻り値の型は`contract_unavailable`となり、保持されたコンポーネントの契約が宣言と異なる場合は`part_contract_mismatch`となります。 子コンポーネントを保持するが`parts`に宣言されていないコンポーネント属性は、親と属性を示す非ブロッキングの`undeclared_part` WARNINGになります。 これは構造化診断ストリームに現れ、実行を妨げません。対象ポート、必須サービス、状態効果、ライフサイクル、Runtime能力の不整合もCompilerの診断として報告されます。無効な契約型を作成すると`ValidationError`が発生します。

## 宣言のないコンポーネントの状態効果

`contract()`の実装がfamily既定で状態を持たないコンポーネントは、自身の状態効果を宣言していません。
そのようなコンポーネントについて、Compilerはコンポーネントのソースから状態の読み取りと書き込みを推論し、診断だけに使います。コンテキストに公開操作がないため、状態のexportsは推論せず、宣言から取得します。
判定には解決された`contract`実装の同一性を使います。手書きの宣言、組み込みから継承した宣言、`super()`を呼ぶだけのoverride、`Algorithm`family既定が宣言する実際の状態は、いずれも宣言として扱い、置き換えません。

推論した効果は、一つの検証規則だけが読む内部の別記録に保持します。
実行契約、analysis contract、plan signature、binding version、実行時のwrite gateの判定、checkpointは変わりません。
推論した効果から見つかった衝突は、非ブロッキングの`inferred_state_effect_conflict` WARNINGになります。
推論はfindingを追加するだけで、Stageの`opaque_state_effects` WARNINGを含む既存のfindingを消しません。

効果を一つでも推論できない場合、family既定のコンポーネントは検証時にnon-enumerableとして扱います。Stage以外のfamily既定では、効果を推論できなかったことを示す`opaque_state_effects` WARNINGを出します。
解析器は解析対象関数のソースを構文解析し、クラスとmoduleの辞書を通して名前を解決します。ソース式を評価せず、解析対象のメソッドを呼ばず、例外も投げません。失敗した場合はコンポーネントをnon-enumerableにします。
解析する関数は、モデル化した構文の小さなallowlistに含まれるノードだけで構成される場合に限ります。ネストした定義、lambda、クラス、内包表記、`global`/`nonlocal`、`match`、代入式、`except*`、拡張代入、`del`、f-string、subscript、starred式、コンテキストの再束縛・shadowingはunknownです。
コンテキスト値は、モデル化したコンテキストの属性・メソッドのreceiver、解析対象の呼び出し先または`typing.cast`呼び出しへの直接の位置引数・キーワード引数、別名代入の単一の名前target、戻り値の位置でだけ追跡します。条件式や真偽値式、コンテナリテラル、反復対象など、それ以外の位置にコンテキストが現れると、その関数はunknownになります。
関数自身が束縛する名前は、パラメータ、代入のtarget、`for`・`with`・`except`のtarget、importの別名であっても、moduleやクラスの値として解決しません。そのような名前を状態キーや呼び出し先に使うと、その関数はunknownになります。
未知の呼び出し先、コンテキストに対する`getattr`、`exec`で定義されたコード、`functools.partial`の属性、独自メタクラス、ソースを読めない場合も、コンポーネントをnon-enumerableにします。
コンテキストを受け取るメソッドは、フローが呼ぶかどうかにかかわらず数えます。実行されない分岐も数えます。
`__module__`が`saealib.`で始まる利用者クラスは、frameworkのクラスと区別しません。
コンテキストではなく明示引数を通る効果（渡された候補への書き込みや、渡されたRNGの読み取りなど）は表しません。その効果は`contract()`で宣言してください。
推論結果はクラス単位でキャッシュされ、同じクラスはプロセス内で一度だけ解析されます。
graphのnodeになれる組み込みのfamily既定クラスのうち推論がunknownだったものは、実測で確認した効果を`contract()`で宣言します。nodeにならないクラスは継承した宣言のままにします。

## 最小例

```python
from saealib.core import ComponentContract, StateContract


def contract() -> ComponentContract:
    return ComponentContract(state=StateContract())
```

## 拡張点

コンパイラ規則の公開経路はリリースごとに異なる可能性があるため、具体的なインポートは[APIリファレンス](../api/index.md)で確認してください。

## 関連ページ

ポートの宣言には[Specs](specs.md)を、Graphへの配置には[Graph](graph.md)を参照してください。

## 参照

- {py:class}`saealib.core.ComponentContract`
- {py:class}`saealib.core.PortContract`
- {py:class}`saealib.core.StateContract`
