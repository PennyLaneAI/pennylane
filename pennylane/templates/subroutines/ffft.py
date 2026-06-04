# Copyright 2018-2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This module contains the fast fermionic Fourier transform. Implemented based on the arXiv paper
by Andrew J. Ferris: https://arxiv.org/pdf/1310.7605."""

from collections import defaultdict

import numpy as np

from pennylane import capture, math
from pennylane.control_flow import for_loop, while_loop
from pennylane.core.operator import Operator
from pennylane.decomposition import add_decomps, pow_resource_rep, register_resources, resource_rep
from pennylane.ops import FermionicSWAP, PauliZ, pow
from pennylane.wires import WiresLike

INV_SQRT2 = 1 / math.sqrt(2)


class TwoWireFFT(Operator):
    r"""
    The two-qubit unitary operator that corresponds to a Fourier transform on fermions, encoded
    using a Jordan-Wigner transformation.

    .. math::

        \hat{F}_2 = \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & 2^{-\frac{1}{2}} & 2^{-\frac{1}{2}} & 0 \\
            0 & 2^{-\frac{1}{2}} & -2^{-\frac{1}{2}} & 0 \\
            0 & 0 & 0 & -1 \\
        \end{bmatrix}

    Args:
        wires (WiresLike): The two wires to apply the operator to. Ideally, they are adjacent
            wires, since the cost of simulating interactions between Fermionic modes depends
            on their distance in the encoding.
    """

    num_wires = 2
    num_params = 0

    def __init__(self, wires: WiresLike):
        super().__init__(wires=wires)

    @staticmethod
    def compute_matrix(*_, **__):
        """
        Computes the matrix of the two-site Fourier operator.

        Returns:
            The matrix representation of the operator.
        """
        return math.array(
            [
                [1, 0, 0, 0],
                [0, INV_SQRT2, INV_SQRT2, 0],
                [0, INV_SQRT2, -INV_SQRT2, 0],
                [0, 0, 0, -1],
            ]
        )


class FFFT(Operator):
    r"""Performs a Fast Fermionic Fourier Transform (FFFT) operation based on
    `arXiv:1310.7605 <https://arxiv.org/pdf/1310.7605>`_. This assumes that fermions are encoded
    using the Jordan-Wigner transformation, where the supplied ordering of the ``wires`` passed to
    ``FFFT`` is used therein.

    The FFFT over a number of wires :math:`n = 2^k` is decomposed recursively into two parallel
    FFFTs over :math:`\tfrac{n}{2}` sites in each iteration of the recursion. These parallel Fourier
    transforms are followed by a series of two-site linear gates.

    .. math::

        \sum_{x=0}^{n-1} e^{\frac{2 \pi i k x}{n}} c_x^\dagger = \sum_{x'=0}^{n/2-1} e^{\frac{2 \pi i k x'}{n/2}} c_{2x'}^\dagger + e^{\frac{2 \pi i k}{n}} \sum_{x'=0}^{n/2-1} e^{\frac{2 \pi i k x'}{n/2}} c_{2x'+1}^\dagger

    Here, :math:`k` is the momentum mode with wave number :math:`2 \pi k / n`, :math:`x` is a site
    targeted by an operator such as the Fermionic creation operator :math:`c_{x}^\dagger`.

    Iterating this decomposition :math:`k` times realizes the full Fourier transform over
    :math:`n = 2^{k}` sites.

    .. seealso:: :class:`~.TwoWireFFT`

    Args:

        wires (WiresLike):
            The wires to apply the FFFT to. The number of wires must be a power of 2 greater than or
            equal to 2.

    Raises:

        ValueError: If ``len(wires)`` is not at least 2.
        NotImplementedError: If ``len(wires)`` is not a power of 2.

    **Example**

    Consider the ``FFFT`` operation performed on 4 wires:

    .. code-block:: python

        import pennylane as qp

        dev = qp.device("default.qubit")

        @qp.qnode(dev)
        def circuit():
            qp.FFFT(wires=(0, 1, 2, 3))
            return qp.state()


    >>> print(qp.draw(circuit, level="device")())
    0: ─╭TwoWireFFT────────────────────╭TwoWireFFT──────────────┤ ╭State
    1: ─╰TwoWireFFT───────╭FSWAP(3.14)─╰TwoWireFFT─╭FSWAP(3.14)─┤ ├State
    2: ─╭TwoWireFFT──Z⁰⋅⁰─╰FSWAP(3.14)─╭TwoWireFFT─╰FSWAP(3.14)─┤ ├State
    3: ─╰TwoWireFFT──Z⁰⋅⁵──────────────╰TwoWireFFT──────────────┤ ╰State


    The ``FFFT`` operation is decomposed recursively into :class:`~.TwoWireFFT` operations
    (two-site Fermionic Fourier transforms) according to the equation above.
    """

    resource_keys = {"num_wires"}

    def __init__(self, wires: WiresLike):
        if len(wires) <= 1:
            raise ValueError("The number of wires must be at least 2 for the FFFT algorithm.")
        if not math.log2(len(wires)).is_integer():
            raise NotImplementedError(
                "FFFT is currently only implemented for numbers of wires that are powers of two."
            )

        super().__init__(wires=wires)

    @classmethod
    def _primitive_bind_call(cls, *args, **kwargs):
        return cls._primitive.bind(*args, **kwargs)

    @property
    def resource_params(self) -> dict:
        return {
            "num_wires": len(self.wires),
        }


def _fast_fermionic_fourier_transform_resources(num_wires):
    resources = defaultdict(int)

    two_qubit_gates = num_wires * math.log2(num_wires) // 2
    resources[resource_rep(TwoWireFFT)] = two_qubit_gates

    def _count_one_recursive(wires, resources):
        if wires > 2:
            for mode in range(wires // 2):
                resources[pow_resource_rep(PauliZ, {}, z=2 * mode / wires)] += 1
            resources = _count_one_recursive(wires // 2, resources)
            resources = _count_one_recursive(wires // 2, resources)
        return resources

    resources = _count_one_recursive(num_wires, resources)

    if num_wires > 2:
        resources[resource_rep(FermionicSWAP)] = (
            num_wires * (num_wires - math.log2(num_wires) - 1) / 2
        )

    return resources


@register_resources(_fast_fermionic_fourier_transform_resources)
def _fast_fermionic_fourier_transform_decomposition(*_, wires: WiresLike, **__):
    if capture.enabled():
        wires = math.array(wires, like="jax")

    # Rather than performing a bit-reversal permutation, we expect the user to label their wires
    # correctly at the beginning.

    _recursive_decompose(wires)


def _recursive_decompose(wires: WiresLike):
    # base case is that we have two wires
    if len(wires) == 2:
        TwoWireFFT(wires)
    else:
        _recursive_decompose(wires[: len(wires) // 2])
        _recursive_decompose(wires[len(wires) // 2 :])

        @for_loop(len(wires) // 2)
        def twiddle(mode):
            pow(PauliZ(wires[len(wires) // 2 + mode]), z=2 * mode / len(wires))

        twiddle()  # pylint: disable=no-value-for-parameter

        _permute_and_apply_parallel(wires, TwoWireFFT)


def _permute_and_apply_parallel(wires, operator):
    """
    A permutation algorithm specific to permuting all the 2-site fourier inputs into range
    at the same time, parallelizing as many FSWAPs as possible and cutting the total number
    of them in half.

    Args:
        wires (WiresLike): The wires to permute.
        operator (Type[Operator]): The operator to apply once the Fermions are adjacent in the encoding.
    """

    @while_loop(lambda i, count, num_parallel_swaps, wires: count < num_parallel_swaps)
    def apply_swaps(i, count, num_parallel_swaps, wires):
        # apply the FSWAP
        FermionicSWAP(np.pi, [wires[i], wires[i + 1]])

        # increase index of next FSWAP and count of FSWAPs
        return i + 2, count + 1, num_parallel_swaps, wires

    @while_loop(lambda num_parallel_swaps, curr_start, wires: num_parallel_swaps < len(wires) // 2)
    def permutation_in_layers(num_parallel_swaps, curr_start, wires):
        # applies a layer of parallel FSWAPs
        apply_swaps((len(wires) - 2) // 2 - curr_start, 0, num_parallel_swaps, wires)

        return num_parallel_swaps + 1, curr_start + 1, wires

    # applies several layers of FSWAPs that achieve the parallel permutation of all needed indices
    permutation_in_layers(1, 0, wires)

    # applies the operator on the indices that are now in range
    @for_loop(len(wires) // 2)
    def apply_op(i):
        # apply the op
        operator([wires[2 * i], wires[2 * i + 1]])

    apply_op()  # pylint: disable=no-value-for-parameter

    @while_loop(lambda num_parallel_swaps, curr_start, wires: num_parallel_swaps > 0)
    def permutation_out_layers(num_parallel_swaps, curr_start, wires):
        # applies a layer of parallel FSWAPs
        apply_swaps((len(wires) - 2) // 2 - curr_start, 0, num_parallel_swaps, wires)

        # number of parallel swaps and their starting index is decreasing this time
        return num_parallel_swaps - 1, curr_start - 1, wires

    # applies the inverse permutation
    permutation_out_layers(len(wires) // 2 - 1, len(wires) // 2 - 2, wires)


add_decomps(FFFT, _fast_fermionic_fourier_transform_decomposition)
