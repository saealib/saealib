"""Population and individual container classes."""

from __future__ import annotations

import warnings
import weakref
from collections.abc import Hashable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Generic, TypeVar, overload

import numpy as np
from typing_extensions import Self

from saealib.core.contracts.identity import CandidateIds
from saealib.exceptions import ValidationError
from saealib.identity import PopulationAttribute
from saealib.population.genome import DenseVectorBatch, GenomeBatch, ObjectBatch
from saealib.space.services import DenseNumericView

if TYPE_CHECKING:
    pass

T_Population = TypeVar("T_Population", bound="Population")
T_Individual = TypeVar("T_Individual", bound="Individual")
_T_Default = TypeVar("_T_Default")

ColumnStore = Mapping[str, np.ndarray]


class _ConflictBypassProperty(property):
    pass


class _LegacyDenseNumericView:
    """Dense view for schema-backed ``x`` compatibility populations."""

    _canonical_identity_backing = True

    def get_view(self, genomes: GenomeBatch) -> np.ndarray:
        if not isinstance(genomes, DenseVectorBatch):
            raise ValidationError("DenseNumericView requires DenseVectorBatch")
        return genomes.array


def bind_property(key: str, doc: str = "") -> Any:  # noqa: D103
    def fget(self):
        return self.get_readonly_value(key)

    def fset(self, value):
        self.update_value(key, value)

    return _ConflictBypassProperty(fget, fset, doc=doc)


def bind_property_array(key: str, doc: str = "") -> Any:  # noqa: D103
    def fget(self):
        return self.get_array(key)

    def fset(self, value):
        self.update_array(key, value)

    return _ConflictBypassProperty(fget, fset, doc=doc)


class Population(Generic[T_Individual]):
    """Container for schema-defined population attributes and genome data."""

    individual_class = None

    x: np.ndarray = bind_property_array("x", doc="Design variables")
    f: np.ndarray = bind_property_array("f", doc="Objective function values")
    g: np.ndarray = bind_property_array("g", doc="Constraint values")
    cv: np.ndarray = bind_property_array("cv", doc="Constraint violation")

    def __init__(
        self,
        attrs: list[PopulationAttribute],
        init_capacity: int = 100,
        *,
        genomes: GenomeBatch | None = None,
        space: Any | None = None,
        dense_numeric_view: DenseNumericView | None = None,
        dense_view: DenseNumericView | None = None,
    ) -> None:
        """
        Initialize a Population.

        Parameters
        ----------
        attrs
            Column definitions for per-candidate data.
        init_capacity
            Initial storage capacity.
        genomes
            Optional initial genome batch.
        space
            Search space associated with the genomes.
        dense_numeric_view
            Service used to obtain a dense numeric view of dense genomes.
        dense_view
            Compatibility alias for ``dense_numeric_view``.
        """
        if genomes is not None:
            init_capacity = max(init_capacity, len(genomes))
        self._capacity = init_capacity
        self.space = space
        self._size = 0
        self._structure_version = 0
        self._value_version = 0
        self._data: dict[str, np.ndarray] = {}
        self._cache: dict[Hashable, Any] = {}
        for attr in attrs:
            self._init_column(attr, self._capacity)
        self._schema = {attr.name: attr for attr in attrs}
        self._dense_genomes_view_cache: tuple[int, int, DenseVectorBatch] | None = None
        self._dense_numeric_view = (
            dense_numeric_view if dense_numeric_view is not None else dense_view
        )
        self._genome_items: list[object] | None = None
        self._legacy_scalar_x = False
        if genomes is not None:
            self._initialize_genomes(genomes)
        elif "x" in self._schema:
            # Compatibility path for schema-backed ``x`` populations without
            # an explicitly supplied GenomeBatch.
            if self._schema["x"].shape == ():
                self._legacy_scalar_x = True
                self._genome_items = []
                self._genome_batch = ObjectBatch()
            else:
                if self._dense_numeric_view is None:
                    self._dense_numeric_view = _LegacyDenseNumericView()
                self._genome_batch = DenseVectorBatch._from_borrowed_view(
                    self._data["x"]
                )
        else:
            self._genome_items = []
            self._genome_batch = ObjectBatch()
        self._check_name_conflicts()

    def _initialize_genomes(self, genomes: GenomeBatch) -> None:
        if len(genomes) > self._capacity:
            self._capacity = len(genomes)
        if isinstance(genomes, DenseVectorBatch):
            if self._dense_numeric_view is None:
                raise ValidationError(
                    "DenseVectorBatch genomes require a resolved DenseNumericView"
                )
            view = np.asarray(self._dense_numeric_view.get_view(genomes))
            if view.ndim != 2 or view.dtype != np.float64:
                raise ValidationError(
                    "DenseNumericView must return a 2-D float64 array"
                )
            storage = np.array(view, dtype=np.float64, order="C", copy=True)
            if "x" in self._schema:
                attr = self._schema["x"]
                if tuple(attr.shape) != (storage.shape[1],):
                    raise ValidationError("genome dimension does not match x column")
                self._data["x"][: len(storage)] = storage
                self._genome_batch = DenseVectorBatch._from_borrowed_view(
                    self._data["x"]
                )
            else:
                self._data["x"] = np.full(
                    (self._capacity, storage.shape[1]), np.nan, dtype=np.float64
                )
                self._data["x"][: len(storage)] = storage
                self._genome_batch = DenseVectorBatch._from_borrowed_view(
                    self._data["x"]
                )
                self._schema.setdefault(
                    "x",
                    PopulationAttribute(
                        name="x", dtype=np.float64, shape=(storage.shape[1],)
                    ),
                )
            self._size = len(storage)
        elif isinstance(genomes, ObjectBatch):
            self._genome_items = list(genomes.items)
            self._size = len(self._genome_items)
            self._genome_batch = ObjectBatch(self._genome_items)
        else:
            # Dense ``x`` compatibility is defined only for DenseVectorBatch.
            self._genome_batch = genomes
            self._size = len(genomes)

    def _check_name_conflicts(self):
        cls = type(self)
        for name in self.schema:
            if hasattr(cls, name):
                attr = getattr(cls, name)
                if isinstance(attr, _ConflictBypassProperty):
                    continue
                warnings.warn(
                    f"Attribute name '{name}' conflicts with a "
                    f"Population method/property. "
                    f"Access via pop.{name} will return the method. "
                    f"Use pop.get('{name}') or pop.get_array('{name}') "
                    f"to access the data.",
                    UserWarning,
                    stacklevel=3,
                )

    def _init_column(self, attr: PopulationAttribute, capacity: int) -> None:
        shape = (capacity, *attr.shape)
        if attr.default is not None:
            arr = np.full(
                shape=shape, fill_value=attr.default, dtype=attr.dtype, order="C"
            )
        elif np.issubdtype(attr.dtype, np.floating) and np.isnan(attr.default):
            arr = np.full(shape=shape, fill_value=np.nan, dtype=attr.dtype, order="C")
        else:
            arr = np.zeros(shape=shape, dtype=attr.dtype, order="C")
        self._data[attr.name] = arr

    def _resize(self, new_capacity: int) -> None:
        for k, v in self._data.items():
            attr = self._schema.get(k)
            if attr is None and k == "x":
                attr = PopulationAttribute(
                    name="x", dtype=v.dtype, shape=v.shape[1:], default=np.nan
                )
            if attr is None:
                raise RuntimeError(f"Missing schema for population storage '{k}'")
            shape = (new_capacity, *attr.shape)
            new_arr = np.full(
                shape=shape, fill_value=attr.default, dtype=attr.dtype, order="C"
            )
            if attr.default is not None:
                new_arr[:] = attr.default
            new_arr[: self._size] = v[: self._size]
            self._data[k] = new_arr
        self._capacity = new_capacity
        if isinstance(self._genome_batch, DenseVectorBatch):
            self._genome_batch = DenseVectorBatch._from_borrowed_view(self._data["x"])

    def mod_value(self) -> None:
        """Mark population values as changed and invalidate cached results."""
        self._value_version += 1
        if "_cache" in self.__dict__:
            self._cache.clear()

    def mod_structure(self) -> None:
        """Mark population structure as changed and invalidate cached results."""
        self._structure_version += 1
        self.mod_value()

    def set_cache(self, key: Hashable, value: Any) -> None:
        """Store a value in the population cache without changing its version.

        Parameters
        ----------
        key : Hashable
            Cache key. An existing value for the key is replaced.
        value : Any
            Value to retain until the population is modified.

        Raises
        ------
        TypeError
            If ``key`` is not hashable.

        Notes
        -----
        The cache is cleared by ``mod_value()`` and ``mod_structure()``.
        Storing a value replaces only the entry for ``key``; it does not clear
        other cached entries or bump the population's value or structure
        version.
        """
        self._cache[key] = value

    def get_cache(self, key: Hashable) -> Any | None:
        """Return a cached value without changing population state.

        Parameters
        ----------
        key : Hashable
            Cache key to look up.

        Returns
        -------
        Any or None
            The cached value, or ``None`` when no value is stored for
            ``key``. A cached ``None`` is therefore indistinguishable from a
            missing key through this method.

        Raises
        ------
        TypeError
            If ``key`` is not hashable.

        Notes
        -----
        Cached values are removed when the population is modified through
        ``mod_value()`` or ``mod_structure()``.
        """
        return self._cache.get(key)

    def append(
        self, element: T_Individual | dict[str, Any] | None = None, **kwargs
    ) -> None:
        """Append one candidate.

        Parameters
        ----------
        element : Individual-like object, dict, or None
            Candidate values read from an individual or mapping. Values passed
            through ``kwargs`` take precedence over values from ``element``.
        **kwargs : Any
            Column values to assign. ``genome`` or ``genomes`` supplies the
            single genome required by a non-dense genome representation.

        Raises
        ------
        ValidationError
            If an explicit candidate ID other than the unassigned ``-1``
            sentinel is supplied, or if a custom genome representation does
            not receive exactly one genome.

        Notes
        -----
        The population grows its storage as needed. Appending invalidates
        cached values and structure-dependent views.
        """
        self._append_internal(element, preserve_ids=False, **kwargs)

    def _append_internal(
        self,
        element: T_Individual | dict[str, Any] | None = None,
        *,
        preserve_ids: bool,
        allow_duplicate_ids: bool = False,
        **kwargs,
    ) -> None:
        """Append a candidate while optionally preserving validated candidate IDs."""
        data: dict[str, Any] = {}
        if element is not None:
            if isinstance(element, dict):
                data.update(element)
            else:
                for key in self._schema:
                    if hasattr(element, key):
                        data[key] = getattr(element, key)
        data.update(kwargs)

        genome_value = data.pop("genome", data.pop("genomes", None))
        if (
            genome_value is None
            and self._genome_items is None
            and not isinstance(self._genome_batch, DenseVectorBatch)
        ):
            raise ValidationError("custom GenomeBatch populations require a genome")

        if "id" in self._schema:
            id_val = int(data.get("id", self._schema["id"].default))
            if not preserve_ids and id_val != -1:
                raise ValidationError(
                    "append() does not accept an explicit 'id' other than the -1 "
                    "sentinel; internal lifecycle code uses "
                    "_append_internal(preserve_ids=True)"
                )
            if (
                id_val != -1
                and not allow_duplicate_ids
                and self._size > 0
                and np.any(self._get_mutable_array("id") == id_val)
            ):
                raise ValidationError(f"Duplicate candidate id {id_val}")

        if self._size >= self._capacity:
            self._resize(self._capacity * 2)

        idx = self._size
        for key, attr in self._schema.items():
            data_self = self._data[key]
            if key in data:
                data_self[idx] = data[key]
            else:
                if attr.default is not None:
                    data_self[idx] = attr.default
                elif np.issubdtype(attr.dtype, np.floating) and np.isnan(attr.default):
                    data_self[idx] = np.nan
                else:
                    data_self[idx] = 0

        if "x" not in self._schema and self._genome_items is not None:
            if genome_value is None:
                self._genome_items.append(None)
            else:
                if not isinstance(genome_value, ObjectBatch) or len(genome_value) != 1:
                    raise ValidationError("append() expects one genome")
                self._genome_items.append(genome_value.items[0])
        elif self._legacy_scalar_x:
            if self._genome_items is None:
                raise RuntimeError("Legacy scalar genome storage is not initialized")
            self._genome_items.append(data.get("x", self._schema["x"].default))
        elif genome_value is not None:
            if len(genome_value) != 1:
                raise ValidationError("append() expects one genome")
            if isinstance(self._genome_batch, DenseVectorBatch):
                self._replace_genome_rows(np.array([idx], dtype=np.intp), genome_value)
            elif self._genome_items is None:
                self._append_genomes(genome_value)

        self._size += 1
        if self._genome_items is not None:
            self._genome_batch = ObjectBatch(self._genome_items)
        self.mod_structure()

    def extend(self, other: Self | dict) -> None:
        """Extend the population with another candidate collection.

        Parameters
        ----------
        other : Population or dict
            Source population or column-data mapping. A mapping may provide
            ``genome`` or ``genomes`` for an opaque genome representation.

        Raises
        ------
        ValidationError
            If the source supplies assigned candidate IDs through this public
            method, or omits genomes required by a custom representation.

        Notes
        -----
        An empty source is a no-op. Otherwise rows are copied, storage grows
        as needed, and cached values and structure-dependent views are
        invalidated.
        """
        self._extend_internal(other, preserve_ids=False)

    def _validate_incoming_ids(
        self,
        other_data: Mapping[str, Any],
        other_size: int,
        *,
        preserve_ids: bool,
        allow_duplicate_ids: bool,
        check_existing: bool = True,
    ) -> None:
        if "id" not in self._schema:
            return

        if "id" in other_data:
            incoming_ids = np.asarray(other_data["id"]).astype(np.int64, copy=False)
            if not preserve_ids and np.any(incoming_ids != -1):
                raise ValidationError(
                    "extend() does not accept explicit 'id' values other than the "
                    "-1 sentinel; internal lifecycle code uses "
                    "_extend_internal(preserve_ids=True)"
                )
        else:
            incoming_ids = np.full(
                other_size, self._schema["id"].default, dtype=np.int64
            )
        real_incoming = incoming_ids[incoming_ids != -1]
        if not allow_duplicate_ids and len(real_incoming) != len(
            np.unique(real_incoming)
        ):
            raise ValidationError("Duplicate candidate id within the extended batch")
        if (
            check_existing
            and self._size > 0
            and len(real_incoming) > 0
            and not allow_duplicate_ids
            and np.any(np.isin(self._get_mutable_array("id"), real_incoming))
        ):
            raise ValidationError(
                "Duplicate candidate id already present in population"
            )

    def _extend_internal(
        self,
        other: Any,
        *,
        preserve_ids: bool,
        allow_duplicate_ids: bool = False,
    ) -> None:
        """Extend while optionally preserving validated candidate IDs."""
        exact_dense_population = False
        canonical_identity_backing = False
        if isinstance(other, Population):
            other_size = len(other)
            exact_dense_population = (
                type(other) is type(self)
                and other._schema == self._schema
                and type(other).get_array is Population.get_array
                and type(self).get_array is Population.get_array
                and isinstance(other._genome_batch, DenseVectorBatch)
                and isinstance(self._genome_batch, DenseVectorBatch)
            )
            if exact_dense_population:
                canonical_identity_backing = (
                    getattr(
                        other._dense_numeric_view,
                        "_canonical_identity_backing",
                        False,
                    )
                    is True
                    and getattr(
                        self._dense_numeric_view,
                        "_canonical_identity_backing",
                        False,
                    )
                    is True
                )
                # The canonical dense path copies the x backing column below;
                # defer constructing the compatibility genome view until it is
                # actually needed by a non-canonical fallback.
                other_genomes = None
                other_data = {
                    k: other._data[k][:other_size] for k in other.schema if k != "x"
                }
                if not canonical_identity_backing:
                    other_genomes = other.genomes
            else:
                other_genomes = other.genomes
                other_data = {k: other.get_array(k) for k in other.schema}
        elif isinstance(other, dict):
            other_size = np.asarray(next(iter(other.values()))).shape[0]
            other_data = other
            other_genomes = other.get("genomes", other.get("genome"))

        if (
            other_genomes is None
            and self._genome_items is None
            and not isinstance(self._genome_batch, DenseVectorBatch)
        ):
            raise ValidationError("custom GenomeBatch populations require genomes")

        if other_size == 0:
            return

        self._validate_incoming_ids(
            other_data,
            other_size,
            preserve_ids=preserve_ids,
            allow_duplicate_ids=allow_duplicate_ids,
        )

        if self._size + other_size > self._capacity:
            self._resize(max(self._capacity * 2, self._size + other_size))

        start = self._size
        for key, attr in self._schema.items():
            if key == "x" and exact_dense_population:
                continue
            val_self = self._data[key]
            if key in other_data:
                val_self[start : start + other_size] = other_data[key]
            else:
                if attr.default is not None:
                    val_self[start : start + other_size] = attr.default
                elif np.issubdtype(attr.dtype, np.floating) and np.isnan(attr.default):
                    val_self[start : start + other_size] = np.nan
                else:
                    val_self[start : start + other_size] = 0

        self._size += other_size
        if canonical_identity_backing:
            self._data["x"][start : start + other_size] = other._data["x"][:other_size]
        elif other_genomes is not None:
            self._append_genomes(other_genomes)
        elif self._genome_items is not None:
            self._genome_items.extend([None] * other_size)
            self._genome_batch = ObjectBatch(self._genome_items)
        elif not isinstance(self._genome_batch, DenseVectorBatch):
            raise ValidationError("custom GenomeBatch populations require genomes")
        self.mod_structure()

    def _replace_from_population(
        self,
        source: Population,
        indices: np.ndarray | list[int] | slice,
        *,
        preserve_ids: bool,
        allow_duplicate_ids: bool = False,
    ) -> bool:
        """Replace rows from a dense population without an intermediate copy.

        Return ``False`` when the populations are not an exact dense match so
        callers can use the general extract/extend path.
        """
        if (
            source is self
            or type(source) is not type(self)
            or not isinstance(source, Population)
            or not isinstance(source._genome_batch, DenseVectorBatch)
            or not isinstance(self._genome_batch, DenseVectorBatch)
            or source._schema != self._schema
            or type(source).get_array is not Population.get_array
            or type(self).get_array is not Population.get_array
        ):
            return False

        if isinstance(indices, slice):
            start, stop, step = indices.indices(source._size)
            indices_arr: Any = slice(start, stop, step)
            n_selected = len(range(start, stop, step))
        else:
            indices_arr = np.asarray(indices)
            if indices_arr.ndim != 1 or indices_arr.dtype.kind not in "iu":
                return False
            n_selected = len(indices_arr)

        # Match GA.tell's existing clear-then-extend state transition, including
        # the empty-on-validation-error behavior of the old path.
        self.clear()
        # Validate IDs before writing any destination columns.  Only the ID
        # column needs selection materialized for validation; materializing
        # every selected column here used to create a full intermediate copy
        # of the survivor payload.
        selected_ids = None
        if "id" in self._schema:
            selected_ids = {"id": source._data["id"][: source._size][indices_arr]}
        self._validate_incoming_ids(
            selected_ids or {},
            n_selected,
            preserve_ids=preserve_ids,
            allow_duplicate_ids=allow_duplicate_ids,
            check_existing=False,
        )
        if n_selected == 0:
            return True
        if n_selected > self._capacity:
            self._resize(max(self._capacity * 2, n_selected))
        source_rows = source._size
        for key in self._schema:
            if key == "x":
                continue
            source_column = source._data[key][:source_rows]
            destination_column = self._data[key][:n_selected]
            if isinstance(indices_arr, slice):
                destination_column[...] = source_column[indices_arr]
            else:
                # np.take(out=...) writes directly into the destination and
                # avoids retaining a separate selected_data dictionary.
                np.take(source_column, indices_arr, axis=0, out=destination_column)
        source_view = source._dense_numeric_view
        target_view = self._dense_numeric_view
        canonical_identity_backing = (
            getattr(source_view, "_canonical_identity_backing", False) is True
            and getattr(target_view, "_canonical_identity_backing", False) is True
        )
        if canonical_identity_backing:
            # Built-in identity views expose the population's canonical x
            # backing directly, so no batch selection or service conversion is
            # needed for this dense survivor tail.
            source_x = source._data["x"][:source_rows]
            destination_x = self._data["x"][:n_selected]
            if isinstance(indices_arr, slice):
                destination_x[...] = source_x[indices_arr]
            else:
                np.take(source_x, indices_arr, axis=0, out=destination_x)
        else:
            selected_indices = np.arange(source._size, dtype=np.intp)[indices_arr]
            self._replace_genome_rows(
                np.arange(n_selected, dtype=np.intp),
                source.genomes.take(selected_indices),
            )
        self._size = n_selected
        self.mod_structure()
        return True

    def extract(self, indices: np.ndarray | list[int] | slice) -> Self:
        """Return a new population containing rows selected by ``indices``.

        Parameters
        ----------
        indices : ndarray, list of int, or slice
            Indices into the active rows. Rows appear in the requested order;
            a slice follows normal Python slicing semantics.

        Returns
        -------
        Population
            A new population of the same concrete type, schema, and genome
            representation. Row and genome data have independent storage, and
            the source population is unchanged.
        """
        if isinstance(indices, slice):
            start, stop, step = indices.indices(self._size)
            n_extract = len(range(start, stop, step))
            indices_arr = slice(start, stop, step)
        else:
            indices_arr = np.array(indices)
            n_extract = len(indices_arr)

        new_pop = self.empty_like(capacity=n_extract)

        for key, val in self._data.items():
            new_pop._data[key][:n_extract] = val[: self._size][indices_arr]

        # Dense genomes are backed by the x column copied above.  The helper
        # is needed only for representations whose genome storage is not in
        # _data (or for schema-backed scalar ``x``, whose object mirror must
        # be synced).
        if not isinstance(self._genome_batch, DenseVectorBatch):
            new_pop._copy_genomes_from(self, indices_arr)

        new_pop._size = n_extract
        new_pop.mod_structure()
        return new_pop

    def truncate(self, new_size: int) -> None:
        """Keep the first ``new_size`` candidates in place.

        Parameters
        ----------
        new_size : int
            Number of leading active rows to retain. Values greater than or
            equal to the current size leave the population unchanged.

        Raises
        ------
        ValueError
            If ``new_size`` is negative.

        Notes
        -----
        When the population is reduced, trailing rows become inactive and
        non-dense genome storage is reduced accordingly. Structure-dependent
        state and cached values are invalidated. No version or cache change
        occurs for a no-op truncation.
        """
        if new_size < 0:
            raise ValidationError("new_size must be non-negative")
        if new_size < self._size:
            self._size = new_size
            if self._genome_items is not None:
                del self._genome_items[new_size:]
                self._genome_batch = ObjectBatch(self._genome_items)
            elif not isinstance(self._genome_batch, DenseVectorBatch):
                self._genome_batch = self.genomes.take(
                    np.arange(new_size, dtype=np.intp)
                )
            self.mod_structure()

    def delete(self, index: int | slice | list[int] | np.ndarray) -> None:
        """Delete candidates selected by ``index`` in place.

        Parameters
        ----------
        index : int, slice, list of int, or ndarray
            Row index or indices to remove from the active population.

        Notes
        -----
        Surviving rows retain their relative order. The operation mutates the
        population and invalidates cached values and structure-dependent
        views.
        """
        bool_mask = np.ones(self._size, dtype=bool)
        bool_mask[index] = False
        new_size = np.sum(bool_mask)
        for k, v in self._data.items():
            valid_data = v[: self._size]
            v[:new_size] = valid_data[bool_mask]
        self._reorder_genomes(np.flatnonzero(bool_mask))
        self._size = new_size
        self.mod_structure()

    def reorder(self, order: np.ndarray) -> None:
        """Reorder candidates according to a complete row permutation.

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
        Reordering is in place and invalidates cached values and
        structure-dependent views.
        """
        if len(order) != self._size:
            raise ValidationError(
                f"Order length {len(order)} must match population size {self._size}"
            )
        for k, v in self._data.items():
            valid_data = v[: self._size]
            v[: self._size] = valid_data[order]
        self._reorder_genomes(order)
        self.mod_structure()

    def argsort(self, name: str, reverse: bool = False) -> np.ndarray:
        """Return row indices that order a named column.

        Parameters
        ----------
        name : str
            Name of a column in the population schema.
        reverse : bool, optional
            If true, return the indices in descending rather than ascending
            column order.

        Returns
        -------
        ndarray of int
            Indices for the active rows in sorted order. The population is not
            mutated.

        Raises
        ------
        KeyError
            If ``name`` is not present in the population schema.
        """
        if name not in self._data:
            raise KeyError(f"Key '{name}' not found in population schema")
        sort_arg = np.argsort(self._data[name][: self._size])
        if reverse:
            sort_arg = sort_arg[::-1]
        return sort_arg

    def clear(self) -> None:
        """Clear the population."""
        self._size = 0
        if self._genome_items is not None:
            self._genome_items.clear()
            self._genome_batch = ObjectBatch()
        elif not isinstance(self._genome_batch, DenseVectorBatch):
            self._genome_batch = self.genomes.take([])
        self.mod_structure()

    def empty_like(self, capacity: int | None = None):
        """Create an empty population with the same schema and representation.

        Parameters
        ----------
        capacity : int or None, optional
            Storage capacity for the new population. If omitted, the current
            capacity is reused.

        Returns
        -------
        Population
            An empty population of the same concrete type, schema, and genome
            representation. It reuses the associated space and dense-view
            service but has independent row storage.
        """
        if capacity is None:
            capacity = self._capacity
        # Dense populations can reconstruct their empty x-backed genome view
        # from the schema; creating an empty batch here is redundant.  Other
        # representations still need an empty batch to preserve their type.
        genome_template = (
            None
            if self._legacy_scalar_x or isinstance(self._genome_batch, DenseVectorBatch)
            else self.genomes.take([])
        )
        return self.__class__(
            self.attrs,
            capacity,
            genomes=genome_template,
            space=self.space,
            dense_numeric_view=self._dense_numeric_view,
        )

    @overload
    def get(self, key: str) -> np.ndarray | None: ...

    @overload
    def get(self, key: str, default: _T_Default) -> np.ndarray | _T_Default: ...

    def get(
        self, key: str, default: _T_Default | None = None
    ) -> np.ndarray | _T_Default | None:
        """
        Get the array of a specific attribute.

        Parameters
        ----------
        key : str
            The attribute name to get the array for.
        default : any, optional
            Returned when the key is absent. Defaults to ``None``.

        Returns
        -------
        np.ndarray or default
            The attribute array if ``key`` exists, otherwise ``default``.
        """
        if key in self._data:
            return self.get_array(key)
        return default

    def _get_mutable_array(self, key: str) -> np.ndarray:
        """
        Get a mutable view of a specific attribute's backing storage.

        Bypasses read-only enforcement and bumps no version — callers are
        responsible for calling ``mod_value()`` (or ``mod_structure()``)
        themselves after mutating through this view.

        Parameters
        ----------
        key : str
            The attribute name to get the array for.
        """
        return self._data[key][: self._size]

    def _validate_genomes(
        self, genomes: GenomeBatch | np.ndarray | None, count: int
    ) -> GenomeBatch | None:
        if genomes is None:
            return None
        candidate: GenomeBatch
        if isinstance(genomes, np.ndarray):
            candidate = DenseVectorBatch(genomes)
        else:
            candidate = genomes
        if len(candidate) != count:
            raise ValidationError(
                f"genome batch length must be {count}, got {len(candidate)}"
            )
        if isinstance(self._genome_batch, DenseVectorBatch):
            if self._dense_numeric_view is None:
                raise ValidationError(
                    "Dense genome updates require the DenseNumericView service"
                )
            view = np.asarray(self._dense_numeric_view.get_view(candidate))
            target = self._data["x"][: self._size]
            if view.shape != (count, target.shape[1]) or view.dtype != np.float64:
                raise ValidationError(
                    "genome batch shape/dtype does not match DenseNumericView"
                )
        elif self._genome_items is not None and not isinstance(candidate, ObjectBatch):
            raise ValidationError("Object populations require an ObjectBatch genome")
        return candidate

    def _require_dense_view(self) -> DenseNumericView:
        if self._dense_numeric_view is None:
            raise AttributeError(
                "Population genome access requires the DenseNumericView service"
            )
        return self._dense_numeric_view

    def _replace_genome_rows(self, indices: np.ndarray, genomes: GenomeBatch) -> None:
        if isinstance(self._genome_batch, DenseVectorBatch):
            values = np.asarray(self._require_dense_view().get_view(genomes))
            self._data["x"][indices] = values
        elif self._genome_items is not None:
            if not isinstance(genomes, ObjectBatch):
                raise ValidationError(
                    "Object populations require an ObjectBatch genome"
                )
            for index, item in zip(indices, genomes.items):
                self._genome_items[int(index)] = item
            self._genome_batch = ObjectBatch(self._genome_items)
        else:
            current = self._genome_batch
            replacements = {int(index): pos for pos, index in enumerate(indices)}
            batches = []
            for row in range(len(current)):
                if row in replacements:
                    batches.append(genomes.take([replacements[row]]))
                else:
                    batches.append(current.take([row]))
            self._genome_batch = (
                type(current).concat(batches) if batches else current.take([])
            )

    def _append_genomes(self, genomes: GenomeBatch) -> None:
        if isinstance(self._genome_batch, DenseVectorBatch):
            values = np.asarray(self._require_dense_view().get_view(genomes))
            self._data["x"][self._size - len(genomes) : self._size] = values
        elif isinstance(genomes, ObjectBatch) and self._genome_items is not None:
            self._genome_items.extend(genomes.items)
            self._genome_batch = ObjectBatch(self._genome_items)
        elif self._genome_items is None:
            self._genome_batch = type(self._genome_batch).concat(
                [self._genome_batch, genomes]
            )

    def _copy_genomes_from(self, source: Population, indices: Any) -> None:
        if isinstance(indices, slice):
            indices = np.arange(len(source), dtype=np.intp)[indices]
        selected = source.genomes.take(indices)
        if isinstance(self._genome_batch, DenseVectorBatch):
            values = np.asarray(self._require_dense_view().get_view(selected))
            self._data["x"][: len(selected)] = values
        else:
            if self._genome_items is not None:
                if not isinstance(selected, ObjectBatch):
                    raise ValidationError("incompatible genome batch representation")
                self._genome_items = list(selected.items)
                self._genome_batch = ObjectBatch(self._genome_items)
            else:
                self._genome_batch = selected

    def _reorder_genomes(self, order: Any) -> None:
        if self._genome_items is not None:
            values = self._genome_items
            self._genome_items = list(np.asarray(values, dtype=object)[order])
            self._genome_batch = ObjectBatch(self._genome_items)
        elif not isinstance(self._genome_batch, DenseVectorBatch):
            self._genome_batch = self.genomes.take(np.asarray(order, dtype=np.intp))

    @property
    def genomes(self) -> GenomeBatch:
        """Return the current genomes as a read-only batch value."""
        if isinstance(self._genome_batch, DenseVectorBatch):
            batch_id = id(self._genome_batch)
            cached = self._dense_genomes_view_cache
            if (
                cached is not None
                and cached[0] == self._structure_version
                and cached[1] == batch_id
            ):
                return cached[2]
            view = DenseVectorBatch._from_borrowed_view(self._data["x"][: self._size])
            self._dense_genomes_view_cache = (
                self._structure_version,
                batch_id,
                view,
            )
            return view
        if self._genome_items is not None:
            return ObjectBatch(self._genome_items)
        return self._genome_batch.take(np.arange(self._size, dtype=np.intp))

    @property
    def candidate_ids(self) -> CandidateIds:
        """Return candidate IDs as a read-only row-aligned array."""
        if "id" in self._data:
            return self.get_array("id")
        ids = np.full(len(self), -1, dtype=np.int64)
        ids.setflags(write=False)
        return ids

    @property
    def columns(self) -> ColumnStore:
        """Return read-only row-aligned column arrays, excluding genome and IDs."""
        return MappingProxyType(
            {key: self.get_array(key) for key in self._schema if key not in {"x", "id"}}
        )

    def get_array(self, key: str) -> np.ndarray:
        """Return a read-only view of the active rows in a named column.

        Parameters
        ----------
        key : str
            Column name. For ``"x"``, the value is obtained through the
            configured ``DenseNumericView`` unless this is a legacy scalar-x
            population.

        Returns
        -------
        np.ndarray
            A non-writeable view covering the active rows of the column. The
            returned array for stored columns shares backing storage rather
            than being an owned copy; ``"x"`` uses the service-provided view.

        Raises
        ------
        KeyError
            If ``key`` is not a stored non-genome column.
        AttributeError
            If ``key`` is ``"x"`` and the required ``DenseNumericView`` is
            unavailable or reports that it cannot provide a view.

        Notes
        -----
        The population is not modified by this method. Mutating the returned
        view raises ``ValueError``; use ``update_array()`` or ``update_rows()``
        so value versions and caches remain synchronized.

        """
        if key == "x":
            if self._legacy_scalar_x:
                view = self._get_mutable_array(key).view()
                view.flags.writeable = False
                return view
            if self._dense_numeric_view is None:
                raise AttributeError(
                    "Population.x requires the DenseNumericView service"
                )
            try:
                view = np.asarray(
                    self._dense_numeric_view.get_view(self.genomes)
                ).view()
            except KeyError as exc:
                raise AttributeError(
                    "Population.x requires the DenseNumericView service"
                ) from exc
            view.flags.writeable = False
            return view
        view = self._get_mutable_array(key).view()
        view.flags.writeable = False
        return view

    def update_array(self, key: str, value: Any) -> None:
        """Update array in place and bump the value version."""
        if key == "x":
            if self._legacy_scalar_x:
                self._get_mutable_array(key)[:] = value
                self.mod_value()
                return
            if self._dense_numeric_view is None:
                raise AttributeError(
                    "Population.update_array('x') requires the DenseNumericView service"
                )
            arr = self._data["x"][: self._size]
            normalized = self._copy_validated_array(
                value, arr.shape, np.dtype(arr.dtype)
            )
            arr[:] = normalized
            self.mod_value()
            return
        if key == "id":
            raise ValidationError(
                "Cannot update the reserved 'id' column via update_array()"
            )
        self._get_mutable_array(key)[:] = value
        self.mod_value()

    def _copy_validated_array(
        self, value: Any, expected_shape: tuple[int, ...], expected_dtype: np.dtype
    ) -> np.ndarray:
        """
        Validate ``value`` against an exact shape/dtype and return an owned copy.

        Parameters
        ----------
        value : Any
            The array-like value to validate.
        expected_shape : tuple[int, ...]
            The exact shape the value must have.
        expected_dtype : np.dtype
            The exact dtype the value must have (no silent casting).

        Returns
        -------
        np.ndarray
            A native-endian, C-contiguous owned copy of ``value``.
        """
        arr = np.asarray(value)
        if expected_dtype == np.dtype(object) or arr.dtype == np.dtype(object):
            raise ValidationError(
                "update_rows() rejects object-dtype columns and values"
            )
        if arr.shape != expected_shape or arr.dtype != expected_dtype:
            raise ValidationError(
                f"Expected array of shape {expected_shape} and dtype "
                f"{expected_dtype}, got shape {arr.shape} and dtype {arr.dtype}"
            )
        return np.array(arr, dtype=expected_dtype, order="C", copy=True)

    def update_rows(
        self,
        indices: np.ndarray | list[int],
        values: dict[str, np.ndarray],
        genome: GenomeBatch | np.ndarray | None = None,
    ) -> None:
        """Atomically update multiple columns for the given rows.

        Validates every index and value before mutating any backing storage,
        so a validation failure never leaves a partial update applied. Bumps
        the value version exactly once for a non-empty update; an update with
        empty ``indices`` or an empty ``values`` mapping is a no-op and does
        NOT bump the version.

        Parameters
        ----------
        indices : np.ndarray | list[int]
            Row indices to update. Must be 1-D, integer dtype, within
            ``[0, len(self))``, and free of duplicates.
        values : dict[str, np.ndarray]
            Mapping of column name to the new values for those rows. Each
            value array must match shape ``(len(indices), *schema[key].shape)``
            and dtype ``schema[key].dtype`` exactly.

        Raises
        ------
        ValidationError
            If ``indices`` is invalid (wrong dtype/ndim, out of range, or
            contains duplicates), if a key in ``values`` is unknown (or is
            ``"id"``), or if a value array's shape/dtype does not match the
            schema exactly.
        """
        genome_value = genome
        # Preserve schema-backed ``x`` input when no explicit genome is supplied;
        # the explicit genome argument is canonical.
        if "x" in values and genome_value is not None:
            raise ValidationError(
                "'x' is not a column; pass the genome through the explicit "
                "genome argument"
            )
        if genome_value is None and "x" in values and not self._legacy_scalar_x:
            genome_value = values["x"]
            values = {key: value for key, value in values.items() if key != "x"}
        if len(values) == 0 and genome_value is None:
            return

        indices_arr = np.asarray(indices)

        if indices_arr.ndim != 1:
            raise ValidationError(
                "indices must be a 1-D integer array, not a scalar/0-D or "
                "higher-dimensional array"
            )
        if indices_arr.dtype == np.bool_:
            raise ValidationError(
                "update_rows() expects an integer row-index array, not a boolean mask"
            )
        if indices_arr.dtype.kind not in "iu":
            raise ValidationError("indices must be a 1-D integer array")

        if len(indices_arr) == 0:
            return

        normalized_indices = indices_arr.astype(np.intp, copy=False)

        if np.any((normalized_indices < 0) | (normalized_indices >= self._size)):
            raise ValidationError(f"indices must be within [0, {self._size})")
        if len(np.unique(normalized_indices)) != len(normalized_indices):
            raise ValidationError("indices must not contain duplicates")

        normalized_values: dict[str, np.ndarray] = {}
        for key, value in values.items():
            if key == "id":
                raise ValidationError("Cannot update the reserved 'id' column")
            if key not in self._schema:
                raise ValidationError(f"Cannot update unknown column '{key}'")
            attr = self._schema[key]
            expected_shape = (len(normalized_indices), *attr.shape)
            normalized_values[key] = self._copy_validated_array(
                value, expected_shape, np.dtype(attr.dtype)
            )

        normalized_genome = self._validate_genomes(
            genome_value, len(normalized_indices)
        )

        for key, arr in normalized_values.items():
            self._data[key][normalized_indices] = arr
        if normalized_genome is not None:
            self._replace_genome_rows(normalized_indices, normalized_genome)

        self.mod_value()

    def _assign_ids(self, indices: np.ndarray, ids: np.ndarray) -> None:
        """Assign real candidate IDs to rows currently holding the -1 sentinel.

        Rejects reassigning a row that already has a real (non -1) id, and
        rejects an assignment that would create a duplicate id in the live
        population. Validates fully before mutating (atomic).
        """
        idx = np.asarray(indices, dtype=np.intp)
        id_col = self._get_mutable_array("id")
        if np.any(id_col[idx] != -1):
            raise ValidationError("Cannot reassign an already-assigned candidate id")
        prospective = id_col.copy()
        prospective[idx] = ids
        live = prospective[prospective != -1]
        if len(live) != len(np.unique(live)):
            raise ValidationError("Duplicate candidate id after assignment")
        id_col[idx] = ids
        self.mod_value()

    @property
    def schema(self) -> MappingProxyType[str, PopulationAttribute]:
        """Return the schema of the population."""
        return MappingProxyType(self._schema)

    @property
    def attrs(self) -> list[PopulationAttribute]:
        """Return the list of attributes in the population."""
        return list(self._schema.values())

    @property
    def value_version(self) -> int:
        """Return the current value-version counter, bumped by ``mod_value()``."""
        return self._value_version

    @property
    def structure_version(self) -> int:
        """Return the current structure-version counter, bumped by ``mod_structure``."""
        return self._structure_version

    def __len__(self) -> int:
        return self._size

    def __getattr__(self, name: str) -> np.ndarray:
        if name == "x" and "_dense_numeric_view" in self.__dict__:
            raise AttributeError("Population.x requires the DenseNumericView service")
        if "_data" in self.__dict__ and name in self.__dict__["_data"]:
            return self.get_array(name)

        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def __setattr__(self, name: str, value: Any) -> None:
        if "_data" in self.__dict__ and name in self.__dict__["_data"]:
            self.update_array(name, value)
        else:
            super().__setattr__(name, value)

    @overload
    def __getitem__(self, index: int) -> T_Individual: ...

    @overload
    def __getitem__(self, index: slice) -> Self: ...

    def __getitem__(self, index: int | slice) -> T_Individual | Self:
        """
        Support bracket access.

        ``pop[0]`` returns an Individual; ``pop[:10]`` returns a new Population.
        """
        if isinstance(index, int):
            if index < 0 or index >= self._size:
                raise IndexError("Index out of range")
            return self.individual_class(self, index)  # type: ignore  # individual_class is generic; ty can't verify constructor signature
        elif isinstance(index, slice):
            return self.extract(index)
        else:
            raise TypeError("Invalid argument type.")


class Individual(Generic[T_Population]):
    """Live view of one population row; structural changes invalidate it."""

    __slots__ = ("_index", "_popref", "_structure_version")

    def __init__(self, population: T_Population, index: int):
        self._popref = weakref.ref(population)
        self._index = index
        self._structure_version = population._structure_version

    def _get_pop(self) -> T_Population:
        """
        Get the referenced population, checking for validity.

        Returns
        -------
        Population
            The referenced population.
        """
        pop = self._popref()
        if pop is None or pop._structure_version != self._structure_version:
            raise RuntimeError("Invalid Individual reference")
        return pop

    def get_readonly_value(self, key: str) -> Any:
        """
        Retrieve the value of the specified key.

        If the value is a NumPy array, return a read-only view.
        """
        value = self._get_pop().get_array(key)[self._index]
        if isinstance(value, np.ndarray):
            view = value.view()
            view.flags.writeable = False
            return view
        return value

    def update_value(self, key: str, value: Any) -> None:
        """Update value in place and bump the value version."""
        if key == "id":
            raise ValidationError(
                "Cannot update the reserved 'id' column via update_value()"
            )
        self._get_pop()._get_mutable_array(key)[self._index] = value
        self._get_pop().mod_value()

    def __getattr__(self, name: str) -> Any:
        pop = self._get_pop()
        if name in pop._data:
            return self.get_readonly_value(name)
        else:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.__slots__:
            super().__setattr__(name, value)
            return
        else:
            pop = self._get_pop()
            if name in pop._data:
                self.update_value(name, value)
                return
            else:
                raise AttributeError(
                    f"'{type(self).__name__}' object has no attribute '{name}'"
                )

    @property
    def pop(self) -> T_Population:
        """Return the parent population."""
        pop = self._get_pop()
        return pop

    @property
    def genome(self) -> GenomeBatch:
        """Return this individual's genome as a one-row batch."""
        return self._get_pop().genomes.take([self._index])


Population.individual_class = Individual
