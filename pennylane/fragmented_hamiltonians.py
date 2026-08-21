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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from pennylane.pytrees import register_pytree

__all__ = ["CDFHamiltonian", ]

class FragmentedHamiltonian:
    r"""Base class for Hamiltonians expressed as a bundle of per-fragment tensors.
    """

    core_shape: ClassVar[tuple[str, ...]]
    """Symbolic shape template for ``core_tensors``."""

    leaf_shape: ClassVar[tuple[str, ...]]
    """Symbolic shape template for ``leaf_tensors``."""

    symbol_metadata: ClassVar[dict[str, tuple[str, int]]]
    """Maps each shape symbol to ``(attribute name, offset)``."""

    _symbol_display: ClassVar[dict[str, str]] = {"L1": "L+1"}
    """Human-readable rendering of symbols in error messages."""

    core_tensors: Any
    leaf_tensors: Any
    nuc_constant: Any

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        register_pytree(cls, cls._flatten, cls._unflatten)

@dataclass(frozen=True, eq=False, repr=False)
class CDFHamiltonian(FragmentedHamiltonian):
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

    .. seealso:: :class:`~.CGFHamiltonian`, :class:`~.FragmentedHamiltonian`

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


