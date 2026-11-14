---
primary_layer: layer4
related_layers: [layer3]
page_type: reference
---

# 探索空間API

SearchSpace APIは、候補表現と空間固有サービスの公開ファサードです。空間の実装モジュールではなく`saealib.space`の名前を使います。

## 空間、サービス、アダプター

### 探索空間

`GenomeBatch`表現を提供する探索空間です。

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.space.SearchSpace
   saealib.space.ServiceRegistry
   saealib.space.ValidationResult
   saealib.core.contracts.RepresentationSpec
   saealib.space.VectorSpace
   saealib.space.ObjectSpace
   saealib.space.SequenceSpace
   saealib.space.PermutationSpace
```

### SearchSpaceのサービス

サンプリング・検証などの探索空間のサービスは、空間が提供する能力です。

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.space.GenomeCodec
   saealib.space.SamplingService
   saealib.space.ValidationService
```

### アダプター

`EvaluationAdapter`は評価境界に置かれ、`GenomeBatch`を`Problem.evaluate_batch()`が受け取る評価payloadに変換するアダプターです。
`FeatureEncoder`は`saealib.space.services`で定義される`Protocol`で、`saealib.space.FeatureEncoder`として公開されます。
`FeatureEncoder`は`SearchSpace.services`のサービスとして登録・解決され、`GenomeBatch`を`FeatureBatch`に変換してサロゲートが学習できる特徴量を決めます。
`FeatureEncoder`はAdapterインスタンスではありません。
`FeatureEncoder`を使う戦略は、実行準備時に`SearchSpace.services`の明示設定を優先してエンコーダを選びます。明示設定がない場合、`VectorSpace`は恒等エンコーディングを選び、選択元と理由をINFOで報告します。`PermutationSpace`と`SequenceSpace`は組み込みの既定エンコーダを使い、明示選択を勧めるWARNINGを報告します。エンコーディングはサロゲートの学習対象を決めますが、このWARNINGは実行を妨げません。
`ObjectSpace`と独自の空間にはエンコーダを推測して割り当てません。選択した戦略が必要とする場合は、空間が`FeatureEncoder`を提供します。診断は`optimizer.contract_diagnostics()`で確認できます。`optimizer.validate()`の後は`optimizer.last_contract_diagnostics`にも残ります。`Optimizer.validate()`が返すのはERRORだけなので、INFOとWARNINGは診断ストリームを確認してください。
`encode_features(space, genomes)`を直接呼ぶ場合も、エンコーダが未登録なら同じ組み込み既定値を使い、Pythonの`UserWarning`を出します。明示エンコーダがあれば警告なしで使い、対応する既定値がない独自空間では`ValidationError`になります。このヘルパーは既定エンコーダを空間へ登録しません。

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.space.FeatureEncoder
```

