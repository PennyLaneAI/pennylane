# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Lightweight containers for Hamiltonians expressed as a bundle of per-fragment tensors.

These classes wrap pre-computed numeric data in a named type so that it can be passed
around, validated, and used as operator input data consistently, whether the data is
concrete or only known abstractly at compile time.

Adding a new representation requires only a shape family:

.. code-block:: python

    class THCHamiltonian(NumericHamiltonian):
        core_shape = ("R", "R")
        leaf_shape = ("R", "N")
        symbol_metadata = {"R": ("tensor_rank", 0), "N": ("num_orbitals", 0)}

Symbols repeated within or across the two templates must take the same size; the base
class derives the named dimensions from the shapes and reports them as attributes.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Number
from typing import Any, ClassVar

import numpy as np

from pennylane import math
from pennylane.pytrees import register_pytree
from pennylane.typing import AbstractArray

__all__ = ["NumericHamiltonian", "CDFHamiltonian", "CGFHamiltonian"]

_TENSOR_NAMES = ("core_tensors", "leaf_tensors", "nuc_constant")
"""The tensor fields every fragmented Hamiltonian carries, in pytree leaf order."""


def _shape_of(tensor):
    """Return the shape of a concrete tensor or of an :class:`~.AbstractArray` spec."""
    if isinstance(tensor, AbstractArray):
        return tensor.shape
    return tuple(math.shape(tensor))


def _dtype_of(tensor):
    """Return a ``numpy`` dtype for a concrete tensor or an ``AbstractArray`` spec."""
    if isinstance(tensor, AbstractArray):
        return tensor.dtype
    if isinstance(tensor, Number):
        return np.dtype(type(tensor))
    return np.dtype(math.get_dtype_name(tensor))


class NumericHamiltonian:
    r"""Base class for Hamiltonians expressed as a bundle of per-fragment tensors.

    Subclasses describe their tensor shapes symbolically via ``core_shape`` and
    ``leaf_shape``. Each entry is a symbol, and a symbol repeated within or across the
    two templates must take the same size — this is what lets a single validator enforce
    shape consistency for every representation, for concrete and abstract data alike.

    ``symbol_metadata`` maps each symbol to the attribute that reports it, together with
    the offset between the two. The leading axis holds ``L + 1`` entries for ``L``
    two-body fragments (index ``0`` being the one-body fragment), so its offset is ``1``.

    Subclasses are registered as pytrees whose leaves are the three tensors, so the data
    flows through program capture and lowering with no per-representation special-casing.
    """

    core_shape: ClassVar[tuple[str, ...]]
    """Symbolic shape template for ``core_tensors``."""

    leaf_shape: ClassVar[tuple[str, ...]]
    """Symbolic shape template for ``leaf_tensors``."""

    symbol_metadata: ClassVar[dict[str, tuple[str, int]]]
    """Maps each shape symbol to ``(attribute name, offset)``."""

    core_tensors: Any
    leaf_tensors: Any
    nuc_constant: Any

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # IT's required for Catalyst to treat the Hamiltonian as a pytree, so that it can be
        # passed as an argument to a decomposition rule and reconstructed from abstract avals
        register_pytree(cls, cls._flatten, cls._unflatten)

    def _unify_shapes(self):
        """Check both tensors against their templates and unify the shared symbols."""
        sizes: dict[str, int] = {}

        for name, template in (
            ("core_tensors", self.core_shape),
            ("leaf_tensors", self.leaf_shape),
        ):
            shape = _shape_of(getattr(self, name))

            if shape is Ellipsis or len(shape) != len(template):
                raise ValueError(
                    f"'{name}' must have {len(template)} dimensions {template}, "
                    f"got shape {shape}."
                )

            for size, symbol in zip(shape, template, strict=True):
                # ``-1`` marks an axis whose size is unknown. It is permissive and does
                # not pin the symbol.
                if size == -1:
                    sizes.setdefault(symbol, -1)
                    continue

                if symbol in sizes and sizes[symbol] not in (size, -1):
                    meta_name, _ = self.symbol_metadata[symbol]
                    raise ValueError(
                        f"inconsistent '{meta_name}' ({symbol}): {sizes[symbol]} vs {size}."
                    )

                sizes[symbol] = size

        for symbol, size in sizes.items():
            meta_name, offset = self.symbol_metadata[symbol]
            if size != -1 and size - offset < 1:
                raise ValueError(f"'{meta_name}' must be at least 1, got {size - offset}.")

        return sizes

    def __post_init__(self):
        if self.nuc_constant is None:
            zero = AbstractArray((), float) if self.is_abstract else np.asarray(0.0)
            object.__setattr__(self, "nuc_constant", zero)
        elif isinstance(self.nuc_constant, Number):
            # Stored as an array so the pytree leaf has a stable shape and dtype,
            # Note it's important when a Hamiltonian is used as a control-flow carry.
            object.__setattr__(self, "nuc_constant", np.asarray(self.nuc_constant))

        nuc_shape = _shape_of(self.nuc_constant)
        if nuc_shape != ():
            raise ValueError(f"'nuc_constant' must be a scalar, got shape {nuc_shape}.")

        sizes = self._unify_shapes()
        for symbol, (name, offset) in self.symbol_metadata.items():
            size = sizes[symbol]
            object.__setattr__(self, name, size - offset if size >= 0 else None)

    @property
    def is_abstract(self) -> bool:
        """bool: Whether the tensors are abstract specifications rather than data.

        Note that traced values are *not* abstract in this sense: a tracer has a
        concrete shape and dtype, so a Hamiltonian built from ``qjit`` arguments is
        concrete.
        """
        return isinstance(self.core_tensors, AbstractArray)

    @property
    def tensors(self) -> tuple:
        """tuple: The ``(core_tensors, leaf_tensors, nuc_constant)`` triple."""
        return (self.core_tensors, self.leaf_tensors, self.nuc_constant)

    @property
    def dimensions(self) -> dict[str, int | None]:
        """dict: The named dimensions derived from the tensor shapes."""
        return {name: getattr(self, name) for name, _ in self.symbol_metadata.values()}

    def _flatten(self):
        """Split into tensor leaves and hashable metadata.

        The derived dimensions travel in the metadata so that ``_unflatten`` does not
        have to re-derive them.
        """
        return self.tensors, tuple(self.dimensions.values())

    @classmethod
    def _unflatten(cls, data, metadata):
        """Rebuild from leaves and metadata, bypassing validation."""
        obj = cls.__new__(cls)
        for name, value in zip(_TENSOR_NAMES, data, strict=True):
            object.__setattr__(obj, name, value)
        for value, (name, _) in zip(metadata, cls.symbol_metadata.values(), strict=True):
            object.__setattr__(obj, name, value)
        return obj

    def _hash_key(self):
        return tuple((_shape_of(t), _dtype_of(t)) for t in self.tensors)

    def __hash__(self):
        # Deliberately keyed on shapes rather than values: this is the information that
        # matters for compilation and resource analysis, and it makes an abstract
        # Hamiltonian hash equal to the concrete one it was derived from.
        return hash((type(self).__name__, self._hash_key()))

    def __eq__(self, other):
        if type(other) is not type(self):
            return NotImplemented
        if self._hash_key() != other._hash_key():
            return False
        if self.is_abstract or other.is_abstract:
            # Shapes and dtypes already match and there are no values to compare.
            return True
        return all(math.allclose(a, b) for a, b in zip(self.tensors, other.tensors, strict=True))

    def __repr__(self):
        def render(tensor):
            # Abstract specs and scalars are small enough to show outright; a full
            # tensor is not, so it is summarized by its shape.
            if isinstance(tensor, AbstractArray) or _shape_of(tensor) == ():
                return repr(tensor)
            return f"tensor(shape={_shape_of(tensor)})"

        body = ", ".join(f"{n}={render(getattr(self, n))}" for n in _TENSOR_NAMES)
        return f"{type(self).__name__}({body})"


@dataclass(frozen=True, eq=False, repr=False)
class CDFHamiltonian(NumericHamiltonian):
    r"""A compressed double-factorized (CDF) electronic Hamiltonian.

    The form of this Hamiltonian is described in
    `arXiv:2506.15784, Sec. III A <https://arxiv.org/abs/2506.15784>`__.
    Briefly, this Hamiltonian form makes approximations to the
    two-body term in an electronic Hamiltonian in the molecular
    orbital basis via a sum of :math:`L` fragments, each parameterized
    by orthogonal rotation matrices (`leaf_tensors`) and a diagonal
    interaction core (`core_tensors`).

    Args:
        core_tensors (TensorLike | AbstractArray): the core tensors, of shape
            ``(L+1, N, N)``. The leading dimension's ``0`` index is the one-body core tensor, while the rest represent the two-body cores.
        leaf_tensors (TensorLike | AbstractArray): the leaf tensors, of shape
            ``(L+1, N, N)``. The leading dimension's ``0`` index represents the one-body leaf tensor, while the rest represents the two-body leaves.
        nuc_constant (float | AbstractArray | None): the nuclear constant energy offset.
            Defaults to ``0.0``.

    Here ``N`` is the number of spatial orbitals and ``L`` the number of two-body
    fragments; both are derived from the shapes and reported as
    :attr:`num_orbitals` and :attr:`num_fragments`.

    Raises:
        ValueError: if the tensor ranks or shared dimensions are inconsistent.

    **Example**

    >>> import numpy as np
    >>> L, N = 2, 3
    >>> ham = qp.CDFHamiltonian(
    ...     core_tensors=np.random.rand(L + 1, N, N),
    ...     leaf_tensors=np.random.rand(L + 1, N, N),
    ...     nuc_constant=0.5,
    ... )
    >>> ham.num_orbitals, ham.num_fragments
    (3, 2)

    The numeric data is directly accessible as follows:

    >>> ham.core_tensors.shape
    (3, 3, 3)
    >>> ham.nuc_constant
    array(0.5)

    The same Hamiltonian can be described with abstract data for the purposes of fast, low-fidelity
    resource-estimation workflows where only shape information is available:

    >>> from pennylane.typing import Float
    >>> qp.CDFHamiltonian(Float[L + 1, N, N], Float[L + 1, N, N]).core_tensors
    AbstractArray((3, 3, 3), float64, weak_type=True)
    """

    core_shape: ClassVar[tuple[str, ...]] = ("L1", "N", "N")
    leaf_shape: ClassVar[tuple[str, ...]] = ("L1", "N", "N")
    symbol_metadata: ClassVar[dict[str, tuple[str, int]]] = {
        "L1": ("num_fragments", 1),
        "N": ("num_orbitals", 0),
    }

    core_tensors: Any
    leaf_tensors: Any
    nuc_constant: Any = None


@dataclass(frozen=True, eq=False, repr=False)
class CGFHamiltonian(NumericHamiltonian):
    r"""A Christiansen greedy-fragmentation (CGF) vibrational Hamiltonian.

    The form of this Hamiltonian is described in
    `arXiv:2508.11865, Sec. III C <https://arxiv.org/abs/2508.11865>`__.
    Briefly, this Hamiltonian form makes approximations to the two-mode
    terms of a vibrational Hamiltonian in the vibrational self-consistent
    field (VSCF)-rotated modal basis via a sum of :math:`L` fragments,
    each parameterized by orthogonal rotation matrices (`leaf_tensors`)
    and a diagonal interaction core (`core_tensors`).

    Args:
        core_tensors (TensorLike | AbstractArray): The core tensors of shape
            ``(L+1, M, M, N, N)``. Index ``0`` along the leading dimension represents
            the one-body core tensor, while indices ``1`` through ``L`` represent the
            two-mode interaction cores.
        leaf_tensors (TensorLike | AbstractArray): The leaf tensors of shape
            ``(L+1, M, N, N)``. Index ``0`` along the leading dimension represents
            the one-body leaf tensor, while indices ``1`` through ``L`` represent the
            two-mode rotation leaves.
        nuc_constant (float | AbstractArray | None): the nuclear constant energy offset.
            Defaults to ``0.0``.

    Here ``M`` is the number of modes, ``N`` the number of modals per mode, and ``L`` the
    number of two-body fragments; all three are derived from the shapes and reported as
    :attr:`num_modes`, :attr:`num_modals`, and :attr:`num_fragments`.

    Raises:
        ValueError: if the tensor ranks or shared dimensions are inconsistent.

    **Example**

    >>> import numpy as np
    >>> L, M, N = 2, 2, 3
    >>> ham = qp.CGFHamiltonian(
    ...     core_tensors=np.random.rand(L + 1, M, M, N, N),
    ...     leaf_tensors=np.random.rand(L + 1, M, N, N),
    ...     nuc_constant=0.5,
    ... )
    >>> ham.num_modes, ham.num_modals, ham.num_fragments
    (2, 3, 2)

    The numeric data is directly accessible as follows:

    >>> ham.core_tensors.shape
    (3, 2, 2, 3, 3)
    >>> ham.leaf_tensors.shape
    (3, 2, 3, 3)

    Note that ``core_shape`` is the *symbolic* shape family shared by every instance of the
    class, not the shape of this instance's data:

    >>> ham.core_shape
    ('L1', 'M', 'M', 'N', 'N')

    The same Hamiltonian can be described with abstract data for the purposes of fast, low-fidelity
    resource-estimation workflows where only shape information is available:

    >>> from pennylane.typing import Float
    >>> qp.CGFHamiltonian(
    ...     core_tensors=qp.typing.Float[L + 1, M, M, N, N],
    ...     leaf_tensors=qp.typing.Float[L + 1, M, N, N],
    ... ).core_tensors
    AbstractArray((3, 2, 2, 3, 3), float64, weak_type=True)
    """

    core_shape: ClassVar[tuple[str, ...]] = ("L1", "M", "M", "N", "N")
    leaf_shape: ClassVar[tuple[str, ...]] = ("L1", "M", "N", "N")
    symbol_metadata: ClassVar[dict[str, tuple[str, int]]] = {
        "L1": ("num_fragments", 1),
        "M": ("num_modes", 0),
        "N": ("num_modals", 0),
    }

    core_tensors: Any
    leaf_tensors: Any
    nuc_constant: Any = None
