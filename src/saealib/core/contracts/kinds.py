from saealib.core.contracts.data import register_data_spec

for _name, _description in (
    ("GenomeBatch", "genomes in one representation"),
    ("Population", "candidate ids + genomes + columns"),
    ("ProposalBatch", "a proposal with its relations and requirements"),
    (
        "EvaluationRequestBatch",
        "evaluation payloads with the ids and fidelity they were issued at",
    ),
    (
        "CandidateSelection",
        "candidate IDs paired with the rows selected for evaluation",
    ),
    ("ObservationBatch", "schema + observation records"),
    ("FeedbackBatch", "observations delivered against a proposal"),
    ("FeatureBatch", "surrogate model inputs"),
    ("SurrogatePrediction", "model outputs, per channel"),
    (
        "Ordering",
        "a reference to some of the producer's rows, with the criterion "
        "that produced it",
    ),
    (
        "RowPredicate",
        "a per-row answer over all of the producer's rows, with the criterion "
        "that produced it",
    ),
    (
        "ArchiveUpdate",
        "rows offered to an archive, before its duplicate policy applies",
    ),
    ("OptimizationEvent", "one dispatched event"),
    ("StatePatch", "writes and deletes"),
):
    register_data_spec(
        _name,
        description=_description,
        variables=(),
        supertypes=(),
    )
