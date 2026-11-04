---
primary_layer: layer2
related_layers: [layer3]
page_type: reference
---

# 最適化戦略

組み込みStrategyは、候補生成、評価計画、Feedback、Population更新を構成します。
新しいStrategyの正規の拡張点は`build_graph(provider) -> ComponentGraph`です。
`build_pipeline()`は既存のPipeline DSLに対する互換性用の記述形式です。

## 基底

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.OptimizationStrategy
   saealib.strategies.TemplateStrategy
```

## 実装

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.GenerationBasedStrategy
   saealib.IndividualBasedStrategy
   saealib.PreSelectionStrategy
   saealib.DirectStrategy
   saealib.SteadyStateStrategy
```

## Island実行

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.IslandModel
```
