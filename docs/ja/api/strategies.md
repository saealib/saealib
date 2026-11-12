---
primary_layer: layer2
related_layers: [layer3]
page_type: reference
---

# 最適化戦略

組み込みStrategyは、候補生成、評価計画、Feedback、Population更新を構成します。
新しいStrategyの正規の拡張点は、`Pipeline` 宣言を渡した `TemplateStrategy` です。 `build_graph(provider)` の上書きは非推奨で、`DeprecationWarning` が出力されます。

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
