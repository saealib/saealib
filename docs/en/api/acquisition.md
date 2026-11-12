---
primary_layer: layer2
related_layers: [layer3]
---

# Acquisition Functions

## Base

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.AcquisitionFunction
```

## Implementations

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

`CORSDistance` applies the CORS distance constraint to predicted-mean scores.
`prepare(archive, *, ctx)` selects the beta value from `ctx.decision_count` and
returns a `CORSReference` carrying it alongside the evaluated points, so the
first decision uses the first search-pattern entry and later decisions repeat
the pattern cyclically. The canonical source-faithful configuration sends one
candidate to true evaluation per decision, for example with
`PreSelectionStrategy(..., n_select=1)` or `TopKEvaluation(1)`.

With `delta=None`, the distance scale is approximated from the current
candidate pool as the maximum minimum distance to the evaluated archive. A
finite positive `delta` can be supplied for a fixed scale.
