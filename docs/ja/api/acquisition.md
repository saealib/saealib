---
primary_layer: layer2
related_layers: [layer3]
---

# 獲得関数

## 基底

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.AcquisitionFunction
```

## 実装

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.ExpectedImprovement
   saealib.BatchExpectedImprovement
   saealib.LowerConfidenceBound
   saealib.MaxUncertainty
   saealib.MeanPrediction
   saealib.CORSDistance
   saealib.acquisition.CORSReference
   saealib.ProbabilityOfFeasibility
   saealib.ProductOfFeasibility
   saealib.EHVIAcquisition
   saealib.ParEGOAcquisition
   saealib.SMSEGOAcquisition
```

```{eval-rst}
.. autofunction:: saealib.gp_ucb_beta_schedule
```

## CORSDistance

`CORSDistance`は、予測平均のスコアにCORSの距離制約を課します。`prepare(archive, *, ctx)`は`ctx.decision_count`からbetaを選び、評価済み点と合わせて保持する`CORSReference`を返します。そのため最初の判断では探索パターンの先頭要素が使われ、以降の判断はパターンを巡回します。原法に忠実な標準構成では、判断1回につき1点だけを真の評価に送ります。たとえば`PreSelectionStrategy(..., n_select=1)`や`TopKEvaluation(1)`を使います。

`delta=None`のとき、距離スケールは現在の候補プールから、評価済みアーカイブへの最小距離の最大値として近似されます。固定のスケールを使いたい場合は、有限の正の`delta`を渡します。
