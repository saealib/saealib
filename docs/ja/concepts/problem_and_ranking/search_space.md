---
primary_layer: layer4
related_layers: [layer3]
page_type: concept
---

# SearchSpace

`SearchSpace` は候補の表現と、その表現を扱うためのサービスを定義します。 アルゴリズムや代理モデルは空間の内部表現を直接変更せず、空間が提供する `GenomeBatch` とサービスを利用します。

## SearchSpaceの役割

`SearchSpace` は `representation`、`services`、`sample()`、`validate()` を提供します。 `GenomeBatch` は候補を行方向にまとめた値であり、候補IDや目的値を直接所有するデータ構造ではありません。 `validate()` は各行の妥当性とバッチ全体のエラーを `ValidationResult` で返します。

`ServiceRegistry` は空間が提供するサービスを名前で登録します。 サンプリング、検証、複製、同値性、距離、ゲノムのシリアライズなどはSearchSpace servicesとして、必要な空間が公開します。 `EvaluationAdapter` は`GenomeBatch`を評価Payloadへ変換する評価境界のAdapterです。 `FeatureEncoder` は `GenomeBatch → FeatureEncoder → FeatureBatch → Surrogate` の意味変換を担うAdapter subtypeです。`SamplingService`などの空間の能力とは性格が異なり、代理モデルが学習できる特徴を決めます。 公開importは `from saealib.space import FeatureEncoder` です。

## SearchSpaceが保持するもの

組み込み空間は表現の制約に応じて次のように分かれます。

- **`VectorSpace`**：固定幅の密な数値ベクトルを扱います。
- **`ObjectSpace`**：任意のオブジェクト表現を保持する最小の空間です。
- **`SequenceSpace`**：長さが変化する系列を扱います。
- **`PermutationSpace`**：順列制約を満たす表現を扱います。

各空間は共通の `SearchSpace` 契約を満たしますが、サービスの集合と `RepresentationSpec` の内容は異なります。 たとえば `ObjectSpace` は既定のサービスを登録しないため、アルゴリズムが要求するサービスを別途提供できる構成にする必要があります。

## 不変条件

`RepresentationSpec` はパラメータの種類、形状、表現種別を記述します。 表現の記述は、候補をどのように保存するかを示すものであり、サンプリングや特徴量化の実装そのものではありません。

`FeatureEncoder`を使う戦略では、実行準備時に明示設定の有無を確認してエンコーダを選びます。`VectorSpace`は恒等エンコーディングを既定にし、選択元と理由をINFOで報告します。`PermutationSpace`と`SequenceSpace`は組み込みの既定エンコーダを使い、エンコーディングがサロゲートの学習対象を決めることを示して、明示選択を勧めるWARNINGを報告します。このWARNINGは実行を妨げません。
`SearchSpace.services`に明示されたエンコーダは既定より優先され、既定選択の診断は出ません。`ObjectSpace`と独自の空間にはエンコーダを推測して割り当てないため、選択した戦略が必要とする場合は空間が提供します。
`encode_features()`を直接呼ぶ場合も、組み込み空間では同じ既定値を使い、その選択をPythonの`UserWarning`で知らせます。対応する既定値がない独自空間では、引き続き明示エンコーダが必要です。
構造化診断は`optimizer.contract_diagnostics()`で確認できます。`optimizer.validate()`を呼んだ後は`optimizer.last_contract_diagnostics`からも確認できます。`Optimizer.validate()`はERRORだけを返すため、INFOとWARNINGは診断ストリームで確認します。

## SearchSpaceの拡張

`SearchSpace`は候補の表現、候補生成サービス、表現の妥当性を管理します。
候補の目的関数値、観測の由来、コンポーネント状態、Graphの実行順序は管理しません。
空間は`GenomeBatch`値の表現規則とサービスを管理し、`Population`は実行中のゲノム格納領域を管理します。
`Population.genomes`はその格納領域を読み取り専用ビューとして公開しますが、不変スナップショットではありません。後続のPopulation更新で内容が変わることがあります。
候補IDやFeedback結果を暗黙に追加することもありません。

新しい候補表現を追加するときは、その`RepresentationSpec`、サンプリング、検証、必要なサービスを1つの`SearchSpace`境界内で定義します。`RepresentationSpec`を`ComponentContract`へ移さず、ポートの`DataSpec`との接続と必要なサービスをCompilerで宣言します。

## よくある失敗

`RepresentationSpec`とサービス集合は空間の構築時に固定され、Compilerはコンポーネントの`ServiceRequirement`と照合します。`sample()`が返す`GenomeBatch`は空間の`RepresentationSpec`に適合し、`validate()`は不一致を黙って修復せず`ValidationResult`で報告します。`SearchSpace`はサービス実装の状態を所有し、コンポーネントは登録済みサービスを要求して使用します。種別や形状の不一致、必要なサービスの欠落、無効な`GenomeBatch`値は、Compilerまたは`ValidationResult`のDiagnosticsになります。

候補表現の拡張は[フレームワーク拡張](../../framework/extensions.md)を、ポートと`DataSpec`の境界は[宣言要素（Specs）](../../framework/specs.md)を、候補に対する観測の対応は[Feedback](../observation_and_state/feedback.md)を参照してください。 公開型のimport経路は[APIリファレンス](../../api/index.md)で確認してください。

## 関連コンポーネント

- [Problem](problem.md)：目的関数、制約、最適化方向とSearchSpaceを組み合わせます。
- [Population](../observation_and_state/population.md)：`GenomeBatch`と個体の評価結果を扱います。
- [SurrogateManager](../surrogate_modeling/surrogate_manager.md)：`FeatureEncoder`サービスを要求します。

## 参照

- {py:class}`saealib.space.SearchSpace`
- {py:class}`saealib.space.VectorSpace`
- {py:class}`saealib.space.ObjectSpace`
- {py:class}`saealib.space.SequenceSpace`
- {py:class}`saealib.space.PermutationSpace`
- {py:class}`saealib.space.FeatureEncoder`
