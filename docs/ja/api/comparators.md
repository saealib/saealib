---
primary_layer: layer2
related_layers: [layer3]
---

# 比較器

## 基底

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.Comparator
```

## 実装

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.SingleObjectiveComparator
   saealib.WeightedSumComparator
   saealib.ParetoComparator
   saealib.NSGA2Comparator
   saealib.NSGA3Comparator
   saealib.RNSGA2Comparator
   saealib.SPEA2Comparator
   saealib.HypervolumeComparator
   saealib.EpsilonDominanceComparator
```

## 支配関係

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.Dominator
   saealib.ParetoDominator
   saealib.EpsilonDominator
```

## 非劣ソート

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.NonDominatedSorter
```

```{eval-rst}
.. autofunction:: saealib.non_dominated_sort
```

```{eval-rst}
.. autofunction:: saealib.dda_non_dominated_sort
```

```{eval-rst}
.. autofunction:: saealib.crowding_distance
```

```{eval-rst}
.. autofunction:: saealib.crowding_distance_all_fronts
```

```{eval-rst}
.. autofunction:: saealib.spea2_fitness
```

```{eval-rst}
.. autofunction:: saealib.comparators.default_comparator
```
