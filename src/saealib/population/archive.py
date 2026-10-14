"""Archive classes built on top of Population."""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any, cast

import numpy as np
from scipy.spatial import cKDTree  # type: ignore  # cKDTree has no bundled type stubs

from saealib.comparators import Dominator
from saealib.exceptions import ValidationError
from saealib.population.genome import DenseVectorBatch, GenomeBatch, ObjectBatch
from saealib.population.population import Individual, Population, PopulationAttribute
from saealib.space.services import (
    DenseNumericView,
    DistanceService,
    EquivalenceService,
    FingerprintService,
)

_UNSET = object()


def _extract_id_value(
    schema: dict[str, PopulationAttribute], element: Any, kwargs: dict[str, Any]
) -> int | None:
    """Return the effective ``id`` implied by an ``add()``/``append()`` call.

    Returns ``None`` when the schema has no ``id`` column.
    """
    if "id" not in schema:
        return None
    id_val = kwargs.get("id")
    if id_val is None:
        if isinstance(element, dict):
            id_val = element.get("id")
        elif element is not None and hasattr(element, "id"):
            id_val = getattr(element, "id")
    if id_val is None:
        id_val = schema["id"].default
    return int(id_val)


def _collect_data(
    schema: dict[str, PopulationAttribute], element: Any, kwargs: dict[str, Any]
) -> dict[str, Any]:
    data: dict[str, Any] = {}
    if isinstance(element, dict):
        data.update({key: value for key, value in element.items() if key in schema})
    elif element is not None:
        for key in schema:
            if hasattr(element, key):
                data[key] = getattr(element, key)
    data.update({key: value for key, value in kwargs.items() if key in schema})
    return data


def _validate_data(
    schema: dict[str, PopulationAttribute], data: dict[str, Any]
) -> None:
    for key, value in data.items():
        attr = schema[key]
        array = np.asarray(value)
        if array.dtype == np.dtype(object) or array.dtype != np.dtype(attr.dtype):
            raise ValidationError(f"invalid dtype for archive column {key!r}")
        if array.shape != attr.shape:
            raise ValidationError(f"invalid shape for archive column {key!r}")


def _validate_observation_schema(
    attrs: list[PopulationAttribute], duplicate_policy: str
) -> None:
    if duplicate_policy != "append":
        return
    schema = {attr.name: attr for attr in attrs}
    if not {"id", "request_id"}.issubset(schema):
        raise ValidationError("append archives require id and request_id columns")
    for name in ("id", "request_id"):
        attr = schema[name]
        if np.dtype(attr.dtype) != np.dtype(np.int64) or attr.shape != ():
            raise ValidationError(f"append archive {name} must be an int64 scalar")


def _service_from_registry(registry: Any, name: str) -> object | None:
    """Return a named service from either a registry or a search space.

    ``Archive`` receives resolved services in the compiled/runtime design, but
    the public constructor still needs a small compatibility seam while that
    wiring is being introduced.  Accepting either ``space`` or its registry
    keeps the seam independent of the concrete registry implementation.
    """
    if registry is None:
        return None
    provider = getattr(registry, "services", registry)
    get = getattr(provider, "get", None)
    if callable(get):
        return get(name)
    return None


def _service_tolerance(service: object | None, name: str) -> float:
    """Read a service-owned tolerance without copying it into the archive."""
    if service is None:
        return 0.0
    value = getattr(service, name, _UNSET)
    if value is _UNSET:
        value = getattr(service, f"_{name}", 0.0)
    return float(cast(Any, value))


def _as_single_genome_batch(value: Any) -> GenomeBatch | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        elif array.ndim != 2 or array.shape[0] != 1:
            raise ValidationError("Archive.add() expects one genome")
        return DenseVectorBatch(array)
    if isinstance(value, (DenseVectorBatch, ObjectBatch)):
        if len(value) != 1:
            raise ValidationError("Archive.add() expects one genome")
        return value
    if isinstance(value, GenomeBatch):
        if len(value) != 1:
            raise ValidationError("Archive.add() expects one genome")
        return value
    # A raw Python object is a valid one-row object genome.  A custom batch
    # still takes the branch above through the runtime-checkable protocol.
    return ObjectBatch([value])


def _extract_genome(
    element: Any, kwargs: dict[str, Any], population: Population
) -> GenomeBatch | None:
    genome_value = kwargs.get("genome", kwargs.get("genomes"))
    if genome_value is None and isinstance(element, dict):
        genome_value = element.get("genome", element.get("genomes"))
    if genome_value is None and isinstance(element, Individual):
        # Individual predates the opaque-genome API and exposes columns only;
        # use its live population view for service-backed archive calls.
        genome_value = element.pop.genomes.take([element._index])
    if genome_value is None and element is not None and hasattr(element, "genome"):
        genome_value = getattr(element, "genome")
    if genome_value is None and "x" in population.schema:
        # ``x`` is accepted only for dense compatibility. The service-backed
        # path does not use it as an identity key after this row is formed.
        return None
    return _as_single_genome_batch(genome_value)


class ArchiveMixin:
    """Provide archive operations for a :class:`Population` subclass.

    Duplicate identity is normally supplied by search-space services. The
    ``key_attr``, ``atol``, and ``rtol`` arguments remain supported for direct
    schema-backed construction without those services.

    Attributes
    ----------
    duplicate_policy : str
        Duplicate handling policy selected at construction.
    key_attr : str
        Schema column used by compatibility identity handling.
    atol : float
        Effective absolute tolerance, including any service-owned value.
    rtol : float
        Effective relative tolerance, including any service-owned value.
    """

    def __init__(
        self,
        attrs: list[PopulationAttribute],
        init_capacity: int = 100,
        key_attr: str = "x",
        atol: float | object = _UNSET,
        rtol: float | object = _UNSET,
        duplicate_policy: str = "keep_first",
        genomes: GenomeBatch | None = None,
        dense_numeric_view: DenseNumericView | None = None,
        dense_view: DenseNumericView | None = None,
        services: Any | None = None,
        space: Any | None = None,
        fingerprint_service: FingerprintService | None = None,
        equivalence_service: EquivalenceService | None = None,
        distance_service: DistanceService | None = None,
        **kwargs,
    ):
        """Initialize archive storage and duplicate-resolution services.

        Parameters
        ----------
        attrs : list of PopulationAttribute
            Column definitions for archive rows. With
            ``duplicate_policy="append"``, the schema must contain scalar
            ``int64`` columns named ``id`` and ``request_id``.
        init_capacity : int, optional
            Initial row-storage capacity passed to ``Population``.
        key_attr : str, optional
            Schema column used by compatibility duplicate detection. In a
            service-backed archive, ``"x"`` may also provide the dense genome
            used to resolve an incoming candidate.
        atol : float, optional
            Explicit absolute-tolerance override. When omitted, the archive
            uses the equivalence service's tolerance when applicable, or zero
            for exact matching.
        rtol : float, optional
            Explicit relative-tolerance override, with the same resolution
            rules as ``atol``.
        duplicate_policy : {"keep_first", "replace", "append"}, optional
            Action for a duplicate candidate. ``"keep_first"`` leaves the
            existing row unchanged, ``"replace"`` updates the existing row,
            and ``"append"`` retains every observation while requiring the
            observation schema described for ``attrs``.
        genomes : GenomeBatch or None, optional
            Initial opaque genome batch passed to ``Population``.
        dense_numeric_view : DenseNumericView or None, optional
            Service used when archive operations need dense coordinates for a
            non-dense genome representation.
        dense_view : DenseNumericView or None, optional
            Compatibility alias for ``dense_numeric_view``.
        services : object or None, optional
            Service provider or registry used when ``space`` is not supplied.
            Explicit service arguments take precedence over services resolved
            from this provider.
        space : object or None, optional
            Search-space or service provider. When supplied, it takes
            precedence over ``services`` and its ``services`` registry is
            queried for omitted service arguments.
        fingerprint_service : FingerprintService or None, optional
            Explicit service for exact identity lookup of opaque genomes.
        equivalence_service : EquivalenceService or None, optional
            Explicit service for tolerance-based identity lookup of opaque
            genomes.
        distance_service : DistanceService or None, optional
            Explicit service for nearest-neighbor queries on opaque genomes.
        **kwargs : Any
            Additional constructor keywords accepted for population
            compatibility.

        Raises
        ------
        ValueError
            If ``duplicate_policy`` is unknown or a compatibility
            ``key_attr`` is absent from the schema.
        ValidationError
            If an append archive has an invalid observation schema, or if
            service-backed duplicate detection lacks its required identity
            service.
        """
        if duplicate_policy not in {"keep_first", "replace", "append"}:
            raise ValidationError(
                "duplicate_policy must be 'keep_first', 'replace', or 'append'"
            )
        _validate_observation_schema(attrs, duplicate_policy)
        cast(Any, super().__init__)(
            attrs=attrs,
            init_capacity=init_capacity,
            genomes=genomes,
            dense_numeric_view=dense_numeric_view,
            dense_view=dense_view,
            space=space,
        )

        self.duplicate_policy = duplicate_policy
        self.key_attr = key_attr
        self._atol_override = None if atol is _UNSET else float(cast(float, atol))
        self._rtol_override = None if rtol is _UNSET else float(cast(float, rtol))

        service_provider = space if space is not None else services
        self._service_provider = getattr(service_provider, "services", service_provider)
        self._fingerprint_service = (
            fingerprint_service
            if fingerprint_service is not None
            else cast(
                FingerprintService | None,
                _service_from_registry(self._service_provider, "FingerprintService"),
            )
        )
        self._equivalence_service = (
            equivalence_service
            if equivalence_service is not None
            else cast(
                EquivalenceService | None,
                _service_from_registry(self._service_provider, "EquivalenceService"),
            )
        )
        self._distance_service = (
            distance_service
            if distance_service is not None
            else cast(
                DistanceService | None,
                _service_from_registry(self._service_provider, "DistanceService"),
            )
        )
        self._identity_mode = "unresolved"
        self._service_configuration_supplied = service_provider is not None or any(
            service is not None
            for service in (
                fingerprint_service,
                equivalence_service,
                distance_service,
            )
        )

        # Direct schema-backed archives remain usable without resolved
        # services. Opaque genomes and explicit service configuration use the
        # service path and fail early when identity services are missing.
        legacy_dense = isinstance(
            getattr(self, "_genome_batch", None), DenseVectorBatch
        )
        self._legacy_identity = (
            not self._service_configuration_supplied
            and "x" in self.schema  # ty: ignore[unresolved-attribute]
            and (genomes is None or legacy_dense)
        )
        if duplicate_policy == "append":
            self._identity_mode = "none"
        else:
            effective_atol = self.atol
            effective_rtol = self.rtol
            if effective_atol == 0.0 and effective_rtol == 0.0:
                if self._fingerprint_service is None and not self._legacy_identity:
                    raise ValidationError(
                        "Archive duplicate detection requires FingerprintService"
                    )
                self._identity_mode = (
                    "legacy" if self._legacy_identity else "fingerprint"
                )
            else:
                if self._equivalence_service is None and not self._legacy_identity:
                    raise ValidationError(
                        "Archive duplicate detection requires EquivalenceService"
                    )
                self._identity_mode = (
                    "legacy" if self._legacy_identity else "equivalence"
                )

        if self._legacy_identity and key_attr not in self.schema:  # ty: ignore[unresolved-attribute]
            raise ValidationError(f"key_attr '{key_attr}' is not defined in attrs")

        self._kdtree: cKDTree | None = None
        self._fingerprint_index: object | None = None
        self._distance_index: object | None = None

    @staticmethod
    def _has_index_api(service: object | None, method: str) -> bool:
        return service is not None and callable(getattr(service, method, None))

    def _invalidate_service_indexes(self) -> None:
        self._fingerprint_index = None
        self._distance_index = None

    @property
    def atol(self) -> float:
        """Return an explicit or service-owned absolute tolerance."""
        if self._atol_override is not None:
            return self._atol_override
        if getattr(self, "_identity_mode", "legacy") == "equivalence" or (
            getattr(self, "_identity_mode", "legacy") == "unresolved"
            and getattr(self, "_service_configuration_supplied", False)
        ):
            return _service_tolerance(self._equivalence_service, "atol")
        return 0.0

    @atol.setter
    def atol(self, value: float) -> None:
        self._atol_override = float(value)

    @property
    def rtol(self) -> float:
        """Return an explicit or service-owned relative tolerance."""
        if self._rtol_override is not None:
            return self._rtol_override
        if getattr(self, "_identity_mode", "legacy") == "equivalence" or (
            getattr(self, "_identity_mode", "legacy") == "unresolved"
            and getattr(self, "_service_configuration_supplied", False)
        ):
            return _service_tolerance(self._equivalence_service, "rtol")
        return 0.0

    @rtol.setter
    def rtol(self, value: float) -> None:
        self._rtol_override = float(value)

    def add(
        self: Any, element: Individual | dict[str, Any] | None = None, **kwargs
    ) -> int:
        """
        Add a solution according to the configured duplicate policy.

        Parameters
        ----------
        element : Individual | dict | None
            Data for the additional individual
        **kwargs :
            Set attribute values individually and add them.
            Alternatively, overwrite based on the element's value and add it.

        Returns
        -------
        idx : int
            Index of the retained or newly added row.

        Notes
        -----
        ``"keep_first"`` returns the existing row for a duplicate,
        ``"replace"`` updates or replaces that row, and ``"append"`` retains
        every observation subject to its ID and request-ID constraints.

        Examples
        --------
        >>> arcv.add(ind)
        >>> arcv.add({"x": x_val})
        >>> arcv.add(x=x_val, f=0.1)
        >>> arcv.add(ind, f=0.1)
        """
        data = _collect_data(self._schema, element, kwargs)  # type: ignore[unresolved-attribute]
        key_attr_val = data.get(self.key_attr)
        if self._legacy_identity and key_attr_val is None:
            raise ValidationError(f"Solution must have {self.key_attr} attribute")

        _validate_data(self._schema, data)  # type: ignore[unresolved-attribute]
        incoming_genome = _extract_genome(element, kwargs, self)
        if (
            incoming_genome is None
            and key_attr_val is not None
            and self._identity_mode in {"fingerprint", "equivalence"}
            and self.key_attr == "x"
        ):
            incoming_genome = _as_single_genome_batch(np.asarray(key_attr_val))
        if (
            self._identity_mode in {"fingerprint", "equivalence"}
            and incoming_genome is None
        ):
            raise ValidationError(
                "Archive.add() requires a genome for duplicate detection"
            )

        incoming_id = _extract_id_value(self._schema, element, kwargs)  # type: ignore[unresolved-attribute]
        if incoming_id == -1:
            raise ValidationError(
                "Archive.add() requires a real candidate id when the "
                "schema declares an 'id' column (got the -1 sentinel)"
            )
        if self.duplicate_policy == "append":
            request_id = data.get("request_id")
            if request_id is None:
                raise ValidationError("append observations require request_id")
            if (
                self._size
                and "id" in self.schema
                and np.any(
                    (self.get_array("request_id") == request_id)
                    & (self.get_array("id") == incoming_id)
                )
            ):
                raise ValidationError("Duplicate (request_id, candidate_id) pair")

        idx = self._find_idx(key_attr_val, incoming_genome)

        if idx is not None and self.duplicate_policy != "append":
            if self.duplicate_policy == "keep_first":
                return idx

            # A retry of the same candidate is a value-only update.  A spatial
            # duplicate from another candidate must replace the whole row so
            # that the ID remains provenance-correct.
            existing_id = (
                int(self.get_array("id")[idx]) if "id" in self.schema else incoming_id
            )
            if incoming_id == existing_id and "id" in self.schema:
                values = {
                    key: np.asarray(value, dtype=self._schema[key].dtype).reshape(
                        (1, *self._schema[key].shape)
                    )
                    for key, value in data.items()
                    if key != "id"
                    and not (
                        key == "x"
                        and self._identity_mode in {"fingerprint", "equivalence"}
                    )
                }
                self.update_rows(
                    np.array([idx]),
                    values,
                    genome=(
                        incoming_genome
                        if self._identity_mode in {"fingerprint", "equivalence"}
                        else None
                    ),
                )
                self._kdtree = None
                self._invalidate_service_indexes()
                return idx

            ids = self.get_array("id") if "id" in self.schema else np.empty(0)
            if (
                "id" in self.schema
                and incoming_id != -1
                and np.any(ids[np.arange(len(ids)) != idx] == incoming_id)
            ):
                raise ValidationError(f"Duplicate candidate id {incoming_id}")
            self.delete(idx)
            # Delete compacts rows; append the incoming observation below.
            idx = None
        else:
            pass

        new_idx = self._size
        append_kwargs = dict(kwargs)
        append_kwargs.pop("genome", None)
        append_kwargs.pop("genomes", None)
        if incoming_genome is not None and (
            "x" not in self.schema or key_attr_val is None
        ):
            append_kwargs["genome"] = incoming_genome
        # Population._append_internal() calls mod_structure(), which reaches
        # ArchiveMixin.mod_value() and invalidates the exact-match index. Keep
        # that handle across the expected append mutation so the fingerprint
        # service can append just the new row instead of rebuilding it.
        fingerprint_index = (
            self._fingerprint_index
            if self._has_index_api(self._fingerprint_service, "add_to_index")
            else None
        )
        super()._append_internal(
            element,
            preserve_ids=True,
            allow_duplicate_ids=self.duplicate_policy == "append",
            **append_kwargs,
        )
        if incoming_genome is not None and fingerprint_index is not None:
            self._fingerprint_service.add_to_index(fingerprint_index, incoming_genome)
            self._fingerprint_index = fingerprint_index
        self._kdtree = None
        # The fingerprint index is incrementally updated above; compatibility
        # remain invalidated as before.
        if not self._has_index_api(self._fingerprint_service, "find_matches"):
            self._fingerprint_index = None
        return new_idx

    def _find_idx(
        self,
        element: np.ndarray | np.floating | None,
        genome: GenomeBatch | None = None,
    ) -> int | None:
        """
        Search for duplicate indexes and return them if found.

        Parameters
        ----------
        element : np.ndarray | np.floating
            Search target

        Returns
        -------
        int | None
            Duplicate index. Return None if it does not exist.
        """
        if self._identity_mode == "none":
            return None
        if self._identity_mode == "fingerprint":
            if genome is None:
                raise ValidationError("Archive duplicate detection requires a genome")
            return self._find_fingerprint_idx(genome)
        if self._identity_mode == "equivalence":
            if genome is None:
                raise ValidationError("Archive duplicate detection requires a genome")
            return self._find_equivalence_idx(genome)

        if self._size == 0:  # ty: ignore[unresolved-attribute]
            return None
        key_attr_arr = self.get_array(self.key_attr)  # ty: ignore[unresolved-attribute]
        element = np.array(element, dtype=self._schema[self.key_attr].dtype)  # ty: ignore[unresolved-attribute]
        if element.ndim == 0:
            element = element.reshape(1)
        if element.shape != key_attr_arr.shape[1:]:
            element = element.reshape(key_attr_arr.shape[1:])
        matching = np.all(
            np.isclose(key_attr_arr, element, atol=self.atol, rtol=self.rtol), axis=1
        )
        indices = np.where(matching)[0]
        if indices.size > 0:
            return int(indices[0])
        return None

    def _ensure_fingerprint_index(self: Any) -> None:
        if self._fingerprint_index is not None:
            return
        if self._fingerprint_service is None:
            raise ValidationError(
                "Archive duplicate detection requires FingerprintService"
            )
        if self._has_index_api(self._fingerprint_service, "create_index"):
            self._fingerprint_index = self._fingerprint_service.create_index()
            self._fingerprint_service.add_to_index(
                self._fingerprint_index, self.genomes
            )
            return
        fingerprints = self._fingerprint_service.fingerprint(self.genomes)
        if len(fingerprints) != self._size:
            raise ValidationError(
                "FingerprintService returned an invalid number of fingerprints"
            )
        index: dict[Hashable, int] = {}
        for row, fingerprint in enumerate(fingerprints):
            index.setdefault(fingerprint, row)
        self._fingerprint_index = index

    def _find_fingerprint_idx(self: Any, genome: GenomeBatch) -> int | None:
        if self._fingerprint_service is None:
            raise ValidationError(
                "Archive duplicate detection requires FingerprintService"
            )
        if self._has_index_api(
            self._fingerprint_service, "create_index"
        ) and self._has_index_api(self._fingerprint_service, "find_matches"):
            if self._fingerprint_index is None:
                self._fingerprint_index = self._fingerprint_service.create_index()
                self._fingerprint_service.add_to_index(
                    self._fingerprint_index, self.genomes
                )
            matches = np.asarray(
                self._fingerprint_service.find_matches(self._fingerprint_index, genome),
                dtype=np.intp,
            )
            if matches.shape != (1,):
                raise ValidationError(
                    "FingerprintService returned an invalid match array"
                )
            return None if matches[0] < 0 else int(matches[0])
        fingerprints = self._fingerprint_service.fingerprint(genome)
        if len(fingerprints) != 1:
            raise ValidationError(
                "FingerprintService must return one fingerprint per genome"
            )
        self._ensure_fingerprint_index()
        index = cast(dict[Hashable, int], self._fingerprint_index)
        return index.get(fingerprints[0])

    def _find_equivalence_idx(self: Any, genome: GenomeBatch) -> int | None:
        if self._equivalence_service is None:
            raise ValidationError(
                "Archive duplicate detection requires EquivalenceService"
            )
        find_matches = getattr(self._equivalence_service, "find_matches", None)
        if callable(find_matches):
            matches = np.asarray(
                find_matches(self.genomes, genome),
                dtype=np.intp,
            )
            if matches.shape != (1,):
                raise ValidationError(
                    "EquivalenceService returned an invalid match array"
                )
            return None if matches[0] < 0 else int(matches[0])
        existing = self.genomes
        for index in range(self._size):
            pair = type(existing).concat((existing.take([index]), genome))
            duplicate_mask = np.asarray(
                self._equivalence_service.find_duplicates(pair), dtype=bool
            )
            if duplicate_mask.shape != (2,):
                raise ValidationError(
                    "EquivalenceService returned an invalid duplicate mask"
                )
            if duplicate_mask[1]:
                return index
        return None

    def delete(self, index):
        """Delete candidates selected by ``index`` in place.

        Parameters
        ----------
        index : int, slice, list of int, or ndarray
            Row index or indices to remove from the active archive.

        Notes
        -----
        Surviving rows retain their relative order. The operation invalidates
        the archive's identity and nearest-neighbor caches.
        """
        super().delete(index)  # ty: ignore[unresolved-attribute]
        self._kdtree = None
        self._invalidate_service_indexes()

    def extend(self, other: Any) -> None:
        """Extend the archive with another candidate collection.

        Parameters
        ----------
        other : Population or dict
            Source population or column-data mapping accepted by
            :meth:`Population.extend`.

        Notes
        -----
        The operation invalidates the archive's identity and nearest-neighbor
        caches after copying the source rows.
        """
        super().extend(other)  # ty: ignore[unresolved-attribute]
        self._kdtree = None
        self._invalidate_service_indexes()

    def reorder(self, order: np.ndarray) -> None:
        """Reorder rows according to a complete row permutation.

        Parameters
        ----------
        order : ndarray of int
            Permutation containing one row index for every active candidate.

        Raises
        ------
        ValueError
            If ``order`` does not contain one entry per active row.

        Notes
        -----
        Reordering is in place and invalidates caches whose values are
        row-indexed.
        """
        super().reorder(order)  # ty: ignore[unresolved-attribute]
        self._kdtree = None
        self._invalidate_service_indexes()

    def truncate(self, new_size: int) -> None:
        """Keep the first ``new_size`` rows in place.

        Parameters
        ----------
        new_size : int
            Number of rows to retain.

        Raises
        ------
        ValueError
            If ``new_size`` is negative.

        Notes
        -----
        Truncation invalidates the archive's identity and nearest-neighbor
        caches.
        """
        super().truncate(new_size)  # ty: ignore[unresolved-attribute]
        self._kdtree = None
        self._invalidate_service_indexes()

    def clear(self) -> None:
        """Clear rows and invalidate identity/neighbor caches."""
        super().clear()  # ty: ignore[unresolved-attribute]
        self._kdtree = None
        self._invalidate_service_indexes()

    def mod_value(self) -> None:
        """Invalidate the kNN cache on every value-only mutation, too."""
        self._kdtree = None
        self._invalidate_service_indexes()
        super().mod_value()  # ty: ignore[unresolved-attribute]

    def _dense_archive_array(self: Any) -> np.ndarray:
        """Return dense genome coordinates for the intentionally dense kNN API."""
        if self._identity_mode == "legacy":
            return self.get_array(self.key_attr)
        genomes = self.genomes
        if isinstance(genomes, DenseVectorBatch):
            return genomes.array
        dense_view = getattr(self, "_dense_numeric_view", None)
        if dense_view is None:
            raise ValidationError(
                "Archive.get_knn requires the DenseNumericView service"
            )
        return np.asarray(dense_view.get_view(genomes))

    def _ensure_kdtree(self) -> None:
        if self._kdtree is None:
            self._kdtree = cKDTree(self._dense_archive_array())

    def get_knn(self, x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Get k-nearest neighbors of the given solution from the archive.

        Parameters
        ----------
        x : np.ndarray
            The solution to find neighbors for.
        k : int
            The number of neighbors to retrieve.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Indices and distances of the k-nearest neighbors.
        """
        if self._size == 0:  # ty: ignore[unresolved-attribute]
            return np.array([]), np.array([])
        create_index = getattr(self._distance_service, "create_index", None)
        query_knn = getattr(self._distance_service, "query_knn", None)
        if callable(create_index) and callable(query_knn):
            distance_service = cast(DistanceService, self._distance_service)
            if self._distance_index is None:
                self._distance_index = distance_service.create_index(
                    cast(Any, self).genomes
                )
            return distance_service.query_knn(self._distance_index, x, k)
        self._ensure_kdtree()
        k = min(k, self._size)  # ty: ignore[unresolved-attribute]
        dist, idx = self._kdtree.query(x, k=k)  # ty: ignore[unresolved-attribute]
        return np.atleast_1d(idx), np.atleast_1d(dist)

    def empty_like(self: Any, capacity: int | None = None):
        """Create an empty archive while retaining resolved identity services.

        Parameters
        ----------
        capacity : int or None, optional
            Storage capacity for the new archive. If omitted, the current
            capacity is reused.

        Returns
        -------
        Archive
            An empty archive of the same concrete type and schema with
            independent row storage and the same identity services.
        """
        if capacity is None:
            capacity = self._capacity
        if isinstance(getattr(self, "_genome_batch", None), DenseVectorBatch):
            genome_template = None
        else:
            genome_template = self.genomes.take([])

        kwargs: dict[str, Any] = {
            "key_attr": self.key_attr,
            "duplicate_policy": self.duplicate_policy,
            "dense_numeric_view": getattr(self, "_dense_numeric_view", None),
            "services": self._service_provider
            if self._service_configuration_supplied
            else None,
            "fingerprint_service": self._fingerprint_service,
            "equivalence_service": self._equivalence_service,
            "distance_service": self._distance_service,
            "genomes": genome_template,
            "space": getattr(self, "space", None),
        }
        if self._atol_override is not None:
            kwargs["atol"] = self._atol_override
        if self._rtol_override is not None:
            kwargs["rtol"] = self._rtol_override
        return self.__class__(self.attrs, capacity, **kwargs)


class Archive(ArchiveMixin, Population):
    """Concrete archive combining ``ArchiveMixin`` with ``Population``.

    Its constructor is inherited from ``ArchiveMixin``.
    """

    pass


class ParetoMixin:
    """
    A mixin that maintains a Pareto-non-dominated archive.

    Must be used via multiple inheritance together with ``Population``
    (or a subclass thereof).  Only non-dominated solutions are retained:
    when a new solution is added any existing solutions it dominates are
    removed, and if the new solution is itself dominated it is discarded.

    Feasibility-first dominance is applied:

    - A feasible solution (cv ≤ eps_cv) dominates every infeasible one.
    - Among two infeasible solutions the one with lower cv dominates.
    - Among two feasible solutions ``dominator.dominates`` is used.

    Parameters
    ----------
    attrs : list[PopulationAttribute]
        Forwarded to ``Population.__init__``.
    init_capacity : int, optional
        Forwarded to ``Population.__init__``.
    direction : np.ndarray or None, optional
        Per-objective direction (+1 maximize, -1 minimize).
        ``None`` defaults to all-minimize.
    dominator : Dominator or None, optional
        Dominance predicate.  ``None`` defaults to ``ParetoDominator()``.
    eps_cv : float, optional
        Feasibility threshold for constraint violation, by default 0.0.
        Under ``Optimizer`` execution this value is overwritten every
        generation from ``problem.handler.feasibility_threshold``; the
        default of 0.0 (strictly feasible only) is only meaningful for
        standalone (non-``Optimizer``) use of ``ParetoArchive``.
    """

    def __init__(
        self,
        attrs: list[PopulationAttribute],
        init_capacity: int = 100,
        direction: np.ndarray | None = None,
        dominator: Dominator | None = None,
        eps_cv: float = 0.0,
        key_attr: str = "x",
        atol: float = 0.0,
        rtol: float = 0.0,
        duplicate_policy: str = "keep_first",
        genomes: GenomeBatch | None = None,
        dense_numeric_view: DenseNumericView | None = None,
        dense_view: DenseNumericView | None = None,
        **kwargs,
    ):
        if duplicate_policy not in {"keep_first", "replace", "append"}:
            raise ValidationError("invalid duplicate_policy")
        if duplicate_policy == "append":
            raise ValidationError("ParetoArchive does not support append policy")
        cast(Any, super().__init__)(
            attrs=attrs,
            init_capacity=init_capacity,
            genomes=genomes,
            dense_numeric_view=dense_numeric_view,
            dense_view=dense_view,
            space=kwargs.pop("space", None),
        )
        if key_attr not in getattr(self, "_schema"):
            raise ValidationError(f"key_attr '{key_attr}' is not defined in attrs")

        # Import here to avoid circular imports at module load time.
        from saealib.comparators import ParetoDominator

        self.direction = direction
        self.dominator: Dominator = (
            dominator if dominator is not None else ParetoDominator()
        )
        self.eps_cv = eps_cv
        self.key_attr = key_attr
        self.atol = atol
        self.rtol = rtol
        self.duplicate_policy = duplicate_policy

    def _extract_fv(
        self, element: Individual | dict[str, Any] | None, kwargs: dict[str, Any]
    ) -> tuple[np.ndarray | None, float]:
        """
        Extract (f, cv) from the supplied element / kwargs.

        Returns
        -------
        f : np.ndarray or None
            Objective vector, or None when the key is absent or all-NaN.
        cv : float
            Constraint violation (0.0 when absent).
        """
        f_val = kwargs.get("f")
        if f_val is None:
            if isinstance(element, dict):
                f_val = element.get("f")
            elif element is not None and hasattr(element, "f"):
                f_val = getattr(element, "f")

        if f_val is None:
            f = None
        else:
            f = np.asarray(f_val, dtype=float).ravel()
            if np.all(np.isnan(f)):
                f = None

        cv_val = kwargs.get("cv")
        if cv_val is None:
            if isinstance(element, dict):
                cv_val = element.get("cv")
            elif element is not None and hasattr(element, "cv"):
                cv_val = getattr(element, "cv")

        cv: float = float(cv_val) if cv_val is not None else 0.0

        return f, cv

    def _new_dominates_existing(
        self,
        f_new: np.ndarray | None,
        cv_new: float,
        f_ex: np.ndarray | None,
        cv_ex: float,
    ) -> bool:
        new_feasible = cv_new <= self.eps_cv
        ex_feasible = cv_ex <= self.eps_cv

        if new_feasible and not ex_feasible:
            return True
        if not new_feasible and ex_feasible:
            return False
        if new_feasible and ex_feasible:
            # Both feasible — use objective-space dominance.
            if f_new is None:
                return False
            if f_ex is None:
                # Existing has no objective value → new dominates it.
                return True
            return bool(self.dominator.dominates(f_new, f_ex, self.direction))
        # Both infeasible — lower cv wins.
        return cv_new < cv_ex

    def _existing_dominates_new(
        self,
        f_new: np.ndarray | None,
        cv_new: float,
        f_ex: np.ndarray | None,
        cv_ex: float,
    ) -> bool:
        return self._new_dominates_existing(f_ex, cv_ex, f_new, cv_new)

    def add(self, element: Individual | dict[str, Any] | None = None, **kwargs) -> int:
        """
        Add a solution to the Pareto archive.

        The solution is accepted only if it is not dominated by any existing
        member.  After insertion all existing members that are dominated by
        the new solution are removed.

        Parameters
        ----------
        element : Individual | dict | None
            Data for the new solution.
        **kwargs :
            Attribute values that override or supplement ``element``.

        Returns
        -------
        idx : int
            Index assigned to the new solution, or -1 when it was rejected.

        Examples
        --------
        >>> archive.add(ind)
        >>> archive.add({"x": x_val, "f": f_val})
        >>> archive.add(x=x_val, f=f_val)
        """
        f_new, cv_new = self._extract_fv(element, kwargs)

        id_val = _extract_id_value(self.schema, element, kwargs)  # ty: ignore[unresolved-attribute]
        if id_val == -1:
            raise ValidationError(
                "ParetoArchive.add() requires a real candidate id when the "
                "schema declares an 'id' column (got the -1 sentinel)"
            )
        if (
            id_val is not None
            and id_val != -1
            and self._size > 0  # ty: ignore[unresolved-attribute]
            and np.any(self.get_array("id") == id_val)  # ty: ignore[unresolved-attribute]
        ):
            raise ValidationError(f"Duplicate candidate id {id_val}")

        # Check whether any existing solution dominates the new one.
        if self._size > 0:  # ty: ignore[unresolved-attribute]
            f_arr = self.get_array("f") if "f" in self._schema else None  # ty: ignore[unresolved-attribute]
            cv_arr = self.get_array("cv") if "cv" in self._schema else None  # ty: ignore[unresolved-attribute]
            n = self._size  # ty: ignore[unresolved-attribute]

            has_nan = (f_new is not None and np.any(np.isnan(f_new))) or (
                f_arr is not None and np.any(np.isnan(f_arr))
            )

            # dominates_many requires NaN-free objective values.
            use_fast_path = not has_nan and f_new is not None and f_arr is not None

            existing_dominates_new = np.zeros(n, dtype=bool)
            new_dominates_existing = np.zeros(n, dtype=bool)

            if use_fast_path:
                cv_ex_arr = cv_arr.astype(float) if cv_arr is not None else np.zeros(n)
                # np.bool_ (not a plain Python bool) so that `~` below is a
                # correct logical negation rather than a bitwise int inversion.
                new_feasible = np.bool_(cv_new <= self.eps_cv)
                ex_feasible = cv_ex_arr <= self.eps_cv

                existing_dominates_new |= (~new_feasible) & ex_feasible
                new_dominates_existing |= new_feasible & (~ex_feasible)

                both_infeasible = (~new_feasible) & (~ex_feasible)
                existing_dominates_new |= both_infeasible & (cv_ex_arr < cv_new)
                new_dominates_existing |= both_infeasible & (cv_new < cv_ex_arr)

                both_feasible = new_feasible & ex_feasible
                if np.any(both_feasible):
                    # Only pass the both_feasible-masked subset to
                    # dominates_many: existing rows outside that mask may be
                    # infeasible with objective values that aren't guaranteed
                    # meaningful (e.g. non-positive under multiplicative
                    # epsilon-dominance's f > 0 requirement), so including
                    # them in the call -- even though their result would
                    # later be discarded by the mask -- can crash.
                    feasible_idx = np.where(both_feasible)[0]
                    new_dom, ex_dom = self.dominator.dominates_many(
                        f_new, f_arr[feasible_idx], self.direction
                    )
                    new_dominates_existing[feasible_idx] |= new_dom
                    existing_dominates_new[feasible_idx] |= ex_dom

            if use_fast_path:
                if existing_dominates_new.any():
                    return -1
                dominated_mask = new_dominates_existing
            else:
                for i in range(n):
                    f_ex = f_arr[i] if f_arr is not None else None
                    cv_ex = float(cv_arr[i]) if cv_arr is not None else 0.0
                    if self._existing_dominates_new(f_new, cv_new, f_ex, cv_ex):
                        return -1

                # Collect indices of existing solutions dominated by the new one.
                dominated_mask = np.zeros(n, dtype=bool)
                for i in range(n):
                    f_ex = f_arr[i] if f_arr is not None else None
                    cv_ex = float(cv_arr[i]) if cv_arr is not None else 0.0
                    if self._new_dominates_existing(f_new, cv_new, f_ex, cv_ex):
                        dominated_mask[i] = True

            # Remove dominated solutions in one pass using delete().
            if np.any(dominated_mask):
                dominated_indices = np.where(dominated_mask)[0]
                self.delete(dominated_indices)  # ty: ignore[unresolved-attribute]

        # Append the new solution and return its index.
        new_idx: int = self._size  # ty: ignore[unresolved-attribute]
        super()._append_internal(element, preserve_ids=True, **kwargs)  # type: ignore[misc]  # ty: ignore[unresolved-attribute]
        return new_idx


class ParetoArchive(ParetoMixin, Population):
    """Concrete Pareto archive: ``ParetoMixin`` mixed into ``Population``."""

    pass
