---
primary_layer: layer4
related_layers: [layer3]
page_type: concept
---

# Specs

契約や探索空間の境界では、処理そのものではなく、接続条件や表現を宣言値として渡します。
このページでは、これらの宣言値をまとめて宣言要素と呼びます。
宣言要素はComponentの派生型ではなく、Compilerが接続と要求を判定するための不変値です。

## Specsの役割

Specsは処理自体を行うのではなく、接続条件、表現、要求サービスをCompilerへ渡す境界を所有します。
Componentの実行状態は保持せず、`ComponentContract`と`SearchSpace`が宣言値を所有します。

## PortSpecとPortContract

`PortSpec`はポート名、方向、`DataSpec`、カーディナリティ、要求サービス、任意性を含む1つのポート宣言です。
`PortContract`は1つの役割に属する入力ポートと出力ポートの集合です。
ポート名は各方向内で一意でなければなりません。

`ONE`、`MANY`、`OPTIONAL`のcardinalityは、提供側の値が消費側の要求を満たすかをCompilerが判定します。
方向、登録済みのdata kind、schema binding、cardinalityが一致しない接続は互換ではありません。

## DataSpecとServiceRequirement

`DataSpec`は登録済みのnominal data kindとschema bindingを表します。
固定値、変数、包含条件、productによるbindingを使って、ポート間で同じスキーマ変数を統合できます。
DataSpecのkind互換性、schemaのunification、サービス解決は別々の検査です。

`ServiceRequirement`はポートまたはComponentが必要とする名前付きサービスを宣言します。
SearchSpaceの`SamplingService`や`ValidationService`などは空間の能力として照合されます。
`EvaluationAdapter`は評価境界のアダプターです。
`FeatureEncoder`は`saealib.space.services`で定義される`Protocol`で、`saealib.space.FeatureEncoder`として公開されます。
`FeatureEncoder`は`SearchSpace.services`へサービスとして登録・解決されます。
`FeatureEncoder`は`GenomeBatch`を`FeatureBatch`へ変換し、サロゲートモデルが学習できる特徴を決めます。
`FeatureEncoder`はAdapterインスタンスではありません。
`FeatureEncoder`を使う戦略は、実行準備時に空間の明示設定を優先してエンコーダを選びます。明示設定がない場合、`VectorSpace`は恒等エンコーディングを選び、選択元と理由をINFOで報告します。`PermutationSpace`と`SequenceSpace`は組み込みの既定エンコーダを使い、エンコーディングがサロゲートの学習対象を決めるため、明示選択を勧めるWARNINGを報告します。このWARNINGは実行を妨げません。
`SearchSpace.services`に明示されたエンコーダは常に既定より優先され、既定を使ったという診断は出ません。`ObjectSpace`と独自の空間にはエンコーダを推測して割り当てないため、利用する戦略が`FeatureEncoder`を必要とする場合は空間が明示的に提供します。
診断ストリームは`optimizer.contract_diagnostics()`で確認できます。`optimizer.validate()`を呼んだ後は`optimizer.last_contract_diagnostics`にも診断が残ります。`Optimizer.validate()`が返すのはERRORだけなので、INFOとWARNINGの選択理由は診断ストリームで確認します。

## RepresentationSpecの境界

`RepresentationSpec`はSearchSpace側の候補表現仕様です。
候補の型、形状、表現を記述しますが、ComponentContractのポート、状態、実行の一部ではありません。
`FeatureEncoder`はSearchSpaceサービスとして登録・解決され、`RepresentationSpec`は空間が保持する候補表現仕様です。

| 宣言値 | 所有者 | 利用する境界 |
|---|---|---|
| `PortSpec`、`PortContract` | `ComponentContract` | Graphの接続とポート互換性 |
| `DataSpec` | ポートまたは契約 | data kindとschema bindingの統合 |
| `ServiceRequirement` | Componentまたはポート | SearchSpace、Problemが提供するサービスの解決 |
| `RepresentationSpec` | `SearchSpace` | 候補表現と表現サービスの整合性 |

## 構築時点と利用時点

`PortSpec`、`DataSpec`、`ServiceRequirement`はComponentContractの構築時に作られ、Compilerがグラフ接続時に使います。
これらの値は契約またはSearchSpaceが所有し、Runtimeは実行中に書き換えません。

## 不変条件と診断

未登録のkind、未定義のcardinality、schema bindingの不一致、要求サービスの不足はDiagnosticsになります。
実装上は`unknown_data_spec`、`unknown_cardinality`、`unknown_schema_variable`、`unresolved_service`などの診断コードで報告されます。

## 拡張点

新しいデータkindやサービスを追加する場合は、登録規則と互換性規則を明示し、暗黙の型変換で接続しない設計にします。

## 関連ページ

具体的なポート宣言型と公開インポート経路は[APIリファレンス](../api/index.md)を参照してください。
候補表現の拡張は[SearchSpace](../concepts/problem_and_ranking/search_space.md)、Compilerルールの拡張は[Framework extensions](extensions.md)を参照してください。

## 参照

- {py:class}`saealib.core.PortSpec`・クラス
- {py:class}`saealib.core.PortContract`
- {py:class}`saealib.core.DataSpec`・クラス
