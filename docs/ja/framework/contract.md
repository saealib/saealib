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
