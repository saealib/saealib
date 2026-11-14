---
primary_layer: layer2
related_layers: [layer3, layer4]
---

# 評価

## Evaluatorとグラフネイティブ契約

graph-native経路では、評価プランナーの出力ポートに、候補選択用のデータ種別として `EvaluationRequestBatch` という `DataSpec` のkindが宣言されています。Evaluatorは選択された `GenomeBatch` を受け取ります。`Problem.evaluation_adapter`が設定されている場合、`EvaluationAdapter.transform()`の戻り値を`Problem.evaluate_batch()`に渡します。未設定時に`DenseVectorBatch`を受けた`SerialEvaluator`は、`DenseVectorBatch.array`をpayloadとして使います。Evaluatorは`ObservationBatch`を出力します。

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.Evaluator
   saealib.SerialEvaluator
   saealib.JoblibEvaluator
   saealib.ThreadPoolEvaluator
   saealib.AsyncEvaluator
   saealib.EvaluationAdapter
   saealib.EvaluationRequest
   saealib.EvaluationHandle
   saealib.EvaluationQuery
   saealib.EvaluationUpdate
   saealib.PendingEvaluation
```

## 評価ポリシーと計画

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.EvaluationPlanner
   saealib.EvaluationPlan
   saealib.EvaluateAll
   saealib.TopKEvaluation
   saealib.RatioEvaluation
   saealib.RepeatedEvaluation
   saealib.FidelityEvaluation
   saealib.FidelityPromotion
   saealib.ReplicateSummary
   saealib.aggregate_replicates
```

## Stage互換型

Stage互換経路では `Evaluator.evaluate_batch(...) -> EvaluationResult` を使います。`EvaluationResult` の `f`、`g`、`cv` のshape・行数整合性、および `candidate_ids` を指定した場合の結果行数との一致と一意性が、この互換境界の制約です。`Evaluator.submit` で `candidate_ids` を返す場合は、requestのcandidate IDsと完全一致する必要があります。

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.EvaluationResult
```
