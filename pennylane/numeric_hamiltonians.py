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
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Number
from typing import Any, ClassVar

import numpy as np

from pennylane import math
from pennylane.typing import AbstractArray

__all__ = ["NumericHamiltonian", "CDFHamiltonian", "CGFHamiltonian"]


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

    .. seealso:: :class:`~.CDFHamiltonian`, :class:`~.CGFHamiltonian`
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
        """The named dimensions derived from the tensor shapes."""
        return {name: getattr(self, name) for name, _ in self.symbol_metadata.values()}

    def _flatten(self):
        """Split into tensor leaves and hashable metadata."""
        return self.tensors, tuple(self.dimensions.values())

    @classmethod
    def _unflatten(cls, data, metadata):
        """Rebuild from leaves and metadata, bypassing validation."""
        obj = cls.__new__(cls)
        tensor_names = ("core_tensors", "leaf_tensors", "nuc_constant")
        for name, value in zip(tensor_names, data, strict=True):
            object.__setattr__(obj, name, value)
        for value, (name, _) in zip(metadata, cls.symbol_metadata.values(), strict=True):
            object.__setattr__(obj, name, value)
        return obj

    def __repr__(self):
        def render(tensor):
            # Abstract specs and scalars are small enough to show outright; a full
            # tensor is not, so it is summarized by its shape.
            if isinstance(tensor, AbstractArray) or _shape_of(tensor) == ():
                return repr(tensor)
            return f"tensor(shape={_shape_of(tensor)})"

        tensor_names = ("core_tensors", "leaf_tensors", "nuc_constant")
        dims = ", ".join(f"{k}={v}" for k, v in self.dimensions.items())
        body = ", ".join(f"{n}={render(getattr(self, n))}" for n in tensor_names)
        return f"{type(self).__name__}({dims}, {body})"


@dataclass(frozen=True, eq=False, repr=False)
class CDFHamiltonian(NumericHamiltonian):
    r"""A compressed double-factorized (CDF) electronic Hamiltonian.

    The form of this Hamiltonian is described in
    `arXiv:2506.15784, Sec. III A <https://arxiv.org/abs/2506.15784>`__.

    Args:
        core_tensors (TensorLike | AbstractArray): the core tensors, of shape
            ``(L+1, N, N)``. Index ``0`` along the leading axis is the one-body fragment.
        leaf_tensors (TensorLike | AbstractArray): the leaf tensors, of shape
            ``(L+1, N, N)``
        nuc_constant (float | AbstractArray | None): the nuclear constant energy offset.
            Defaults to ``0.0``.

    Here ``N`` is the number of spatial orbitals and ``L`` the number of two-body
    fragments; both are derived from the shapes and reported as
    :attr:`num_orbitals` and :attr:`num_fragments`.

    Raises:
        ValueError: if the tensor ranks or shared dimensions are inconsistent

    .. seealso:: :class:`~.CGFHamiltonian`, :class:`~.NumericHamiltonian`

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

    The numeric data is directly accessible:

    >>> ham.core_tensors.shape
    (3, 3, 3)

    The same Hamiltonian can be described abstractly, which is what the compile-time
    typing and resource-analysis paths consume:

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

    Args:
        core_tensors (TensorLike | AbstractArray): the core tensors, of shape
            ``(L+1, M, M, N, N)``. Index ``0`` along the leading axis is the one-body
            fragment.
        leaf_tensors (TensorLike | AbstractArray): the leaf tensors, of shape
            ``(L+1, M, N, N)``
        nuc_constant (float | AbstractArray | None): the nuclear constant energy offset.
            Defaults to ``0.0``.

    Here ``M`` is the number of modes, ``N`` the number of modals per mode, and ``L`` the
    number of two-body fragments; all three are derived from the shapes and reported as
    :attr:`num_modes`, :attr:`num_modals`, and :attr:`num_fragments`.

    Raises:
        ValueError: if the tensor ranks or shared dimensions are inconsistent

    .. seealso:: :class:`~.CDFHamiltonian`, :class:`~.NumericHamiltonian`

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

    Inconsistent shapes are reported against the named dimension:

    >>> qp.CGFHamiltonian(np.zeros((3, 2, 2, 3, 3)), np.zeros((3, 2, 4, 4)))
    Traceback (most recent call last):
        ...
    ValueError: CGFHamiltonian has an inconsistent 'num_modals' (N): 'core_tensors' axis 4 has size 3 but 'leaf_tensors' axis 2 has size 4. Expected core_tensors (L+1, M, M, N, N) and leaf_tensors (L+1, M, N, N).
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
