---
primary_layer: layer4
related_layers: [layer3]
page_type: concept
---

# Compiler

`ComponentGraph`を解決して検証すると、Compilerは実行から分離された`ExecutablePlan`を返します。
CompilerはRuntimeを実行せず、Runtimeが必要とする能力と診断情報を計画に記録します。

## Compilerの役割

CompilerはComponentGraphを解決・検証し、実行から分離されたExecutablePlanを返す境界を担います。
Runtimeは実行せず、Runtimeが計画で必要とする能力と診断情報を記録します。

## コンパイル、検証、解決

コンパイルは、契約のスナップショット取得、Resolution、Verificationの順に進みます。
`ResolutionRule`はサービス、アダプター、スキーマバインディングを解決する候補を、担当するGraph位置への提案として返します。
`VerificationRule`は解決済みGraphを観測し、Graphを変更せずに診断情報を返します。
共通の`CompilationRule`境界にはnamespace、name、phase、`apply(RuleContext)`が含まれます。

Resolutionは提案が収束するまで反復されます。
同じGraph位置への競合や未申告の変更はエラー診断になります。VerificationはGraphの整形式性、ポート互換性、サービス、データフロー、状態効果、ライフサイクル、Runtime能力を検査します。

## 実行可能な計画

`ExecutablePlan`は、解決済みGraph、Diagnostics、必要なRuntime能力、有効なルール、挿入されたアダプター、契約スナップショットを保持する不変値です。
計画は実行位置も現在の状態も所有しないため、同じ計画をRuntimeSessionに渡して実行と再開を分離できます。
Diagnosticsにエラーが含まれる場合に計画を実行可能な入力として受け入れるかは、呼び出し側のチェックとRuntime契約に従います。

### 検証から実行まで

```{mermaid}
flowchart LR
    C[Component] --> N[ComponentNode]
    N --> G[ComponentGraph]
    G --> K[Compiler]
    K --> P[ExecutablePlan]
    P --> R[ExecutionRuntime]
```

ComponentからGraphまでは構成、Compilerは解決と検証、ExecutablePlan以降はRuntimeの責務です。
Runtimeは計画ノードの結果からStatePatch、イベント、コマンドを適用し、検証済みの状態境界を次のステップへ渡します。

## 構築時点と利用時点

Graphの構築後、Compilerは契約スナップショットを読み取り、ResolutionとVerificationを経てExecutablePlanを生成します。
Runtimeは生成されたExecutablePlanを使い、実行中にCompilerの解決や検証を繰り返しません。

## 不変条件と診断

`Diagnostic`はseverity、code、message、contract path、resolution hintを持つ検証結果です。
Compilerは失敗を任意の例外だけで表さず、計画の判断のために複数の問題をDiagnosticBagへ収集します。
典型的な失敗には、未知のノード端点、未接続の必須ポート、不一致のDataSpec、未提供の必須サービス、未申告の状態アクセス、Runtime能力の不足、子コンポーネントを保持するが未宣言のコンポーネント（`undeclared_part` WARNING）があります。
実装は`invalid_graph_edge`、`incompatible_port`、`unresolved_service`、`unknown_runtime_capability`などのコードで報告します。
Compilerルールは未申告のGraph位置を変更せず、解決中にStructuredRegionの実行ツリーを任意に変更しません。
違反は`unclaimed_rewrite`、`conflicting_rewrite`、`structured_execution_mutation`などのDiagnosticsになります。
アダプターは変換の意味と無損失性を示す必要があり、ポート互換性検査を迂回してはなりません。
解決不能または曖昧なアダプターは`ambiguous_adapter`などのDiagnosticsになります。

## 拡張点

Compilerのルール型とimport経路はリリースごとの[APIリファレンス](../api/index.md)で確認してください。
[フレームワーク拡張](extensions.md)では拡張方法を説明します。

## 関連ページ

構成要素の宣言は[Contract](contract.md)と[Specs](specs.md)を、実行計画の利用は[Runtime](runtime.md)を参照してください。

## 参照

- {py:class}`saealib.core.CompilationRule`・クラス
- {py:class}`saealib.core.ExecutablePlan`・クラス
- {py:class}`saealib.core.ComponentGraph`
