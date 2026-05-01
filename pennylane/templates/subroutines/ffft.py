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

import copy
from collections import defaultdict

import numpy as np

from pennylane import math
from pennylane.control_flow import for_loop, while_loop
from pennylane.decomposition import add_decomps, pow_resource_rep, register_resources, resource_rep
from pennylane.operation import Operator
from pennylane.ops import FermionicSWAP, PauliZ, pow
from pennylane.wires import Wires, WiresLike


class TwoQubitFFT(Operator):
    r"""
    The two-qubit unitary operator that corresponds to a Fourier transform on
    Fermions, encoded using a Jordan-Wigner Transformation (JWT).

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
        super().__init__(wires=wires, id=None)

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
                [0, 2 ** -(1 / 2), 2 ** -(1 / 2), 0],
                [0, 2 ** -(1 / 2), -(2 ** -(1 / 2)), 0],
                [0, 0, 0, -1],
            ]
        )


class FFFT(Operator):
    r"""Performs a Fast Fermionic Fourier Transform (FFFT) operation based on `arXiv:1310.7605 <https://arxiv.org/pdf/1310.7605>`_. This assumes that
    the fermions are encoded using the Jordan-Wigner transformation. Assumes the Fermions are encoded using the ordering
    of the wires as passed to the FFFT.

    The FFFT over a number of wires :math:`n` (a power of two)
    is decomposed recursively into two parallel FFFTs over :math:`\tfrac{n}{2}`
    sites in each iteration of the recursion. These parallel Fourier transforms are followed by a series of
    2-site linear gates.

    Args:

        wires (WiresLike): The wires to apply the FFFT to. The number of wires must be a power of 2 greater than or equal to 2.

    Raises:

        ValueError: If ``len(wires)`` is not at least 2.
        NotImplementedError: If ``len(wires)`` is not a power of 2.

    .. math::

        \sum_{x=0}^{n-1} e^{\frac{2 \pi i k x}{n}} c_x^\dagger = \sum_{x'=0}^{n/2-1} e^{\frac{2 \pi i k x'}{n/2}} c_{2x'}^\dagger + e^{\frac{2 \pi i k}{n}} \sum_{x'=0}^{n/2-1} e^{\frac{2 \pi i k x'}{n/2}} c_{2x'+1}^\dagger


    This is a transform between real and momentum space. The momentum mode is
    :math:`k`, wave number :math:`2 \pi k / n`. :math:`x` is a site targeted
    by an operator such as the Fermionic creation operator :math:`c_{x}^\dagger`.

    A phase-delay implemented using Pauli Z gates raised to various powers is
    necessary to take into account the twiddle-factor :math:`e^{\frac{2 \pi i k}{n}}`.

    Iterating the decomposition :math:`k` times realizes the full Fourier transform over
    :math:`2^{k}` sites.

    **Example**

    Consider the FFFT operation performed on 4 wires:

    .. code-block:: python

        import pennylane as qp

        dev = qp.device("default.qubit")

        @qp.qnode(dev)
        def circuit():
            qp.FFFT(wires=(0, 1, 2, 3))
            return qp.state()


    >>> print(qp.draw(circuit, level="device")())
    0: ─╭TwoQubitFFT────────────────────╭TwoQubitFFT─────────────────────────────────────── ···
    1: ─╰TwoQubitFFT───────╭fSWAP(3.14)─╰TwoQubitFFT─╭fSWAP(3.14)──────────────╭TwoQubitFFT ···
    2: ─╭TwoQubitFFT──Z⁰⋅⁰─╰fSWAP(3.14)──────────────╰fSWAP(3.14)─╭fSWAP(3.14)─╰TwoQubitFFT ···
    3: ─╰TwoQubitFFT──Z⁰⋅⁵────────────────────────────────────────╰fSWAP(3.14)───────────── ···
    <BLANKLINE>
    0: ··· ──────────────┤  State
    1: ··· ──────────────┤  State
    2: ··· ─╭fSWAP(3.14)─┤  State
    3: ··· ─╰fSWAP(3.14)─┤  State

    The FFFT operation is decomposed recursively into :class:`~TwoQubitFFT` operations (2-site Fermionic Fourier transforms) according to the equation above.
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
        return {"num_wires": len(self.wires)}


def _fast_fermionic_fourier_transform_resources(num_wires):
    resources = defaultdict(int)

    def _count_two_recursive(
        wires,
    ):
        two_qubit_gates = wires // 2
        if wires > 2:
            two_qubit_gates += _count_two_recursive(wires // 2) * 2
        return two_qubit_gates

    two_qubit_gates = _count_two_recursive(num_wires)
    resources[resource_rep(TwoQubitFFT)] = two_qubit_gates

    def _count_one_recursive(wires, resources):
        if wires > 2:
            for mode in range(wires // 2):
                resources[pow_resource_rep(PauliZ, {}, z=2 * mode / wires)] += 1
            resources = _count_one_recursive(wires // 2, resources)
            resources = _count_one_recursive(wires // 2, resources)
        return resources

    resources = _count_one_recursive(num_wires, resources)

    def _count_swaps(wires):
        swaps = 2 * (wires // 2) * (wires // 2 - 1)
        if wires > 2:
            swaps += _count_swaps(wires // 2) * 2
        return swaps

    if num_wires > 2:
        resources[resource_rep(FermionicSWAP)] = _count_swaps(num_wires)

    bit_reversal_swaps = 0
    for i in range(num_wires // 2):
        left, right, l_end, r_end = i, num_wires - i - 1, num_wires - i - 1, i
        finished = False
        while not finished:
            finished = (left == l_end) and (right == r_end)
            if left < l_end:
                bit_reversal_swaps += 1
                if left + 1 == right:
                    right -= 1
                left += 1
            if right > r_end:
                bit_reversal_swaps += 1
                right -= 1

    resources[resource_rep(FermionicSWAP)] += bit_reversal_swaps

    return resources


@register_resources(_fast_fermionic_fourier_transform_resources)
def _fast_fermionic_fourier_transform_decomposition(*_, wires: WiresLike, **__):
    wires = math.array(wires)
    num_wires = len(wires)

    # bit-reversal permutation
    @for_loop(num_wires // 2)
    def swaps(i):
        @while_loop(lambda finished, _, __, ___, ____: not finished)
        def fswaps(f, left, right, l_end, r_end):
            f = (left == l_end) and (right == r_end)
            if left < l_end:
                FermionicSWAP(np.pi, Wires([wires[left], wires[left + 1]]))
                if left + 1 == right:
                    right -= 1
                left += 1
            if right > r_end:
                FermionicSWAP(np.pi, Wires([wires[right], wires[right - 1]]))
                right -= 1
            return f, left, right, l_end, r_end

        fswaps(False, i, num_wires - i - 1, num_wires - i - 1, i)

    swaps()  # pylint: disable=no-value-for-parameter

    _recursive_decompose(wires)


def _recursive_decompose(wires: WiresLike):
    # base case is that we have two wires
    if len(wires) == 2:
        TwoQubitFFT(wires)
    else:
        _recursive_decompose(wires[: len(wires) // 2])
        _recursive_decompose(wires[len(wires) // 2 :])

        @for_loop(len(wires) // 2)
        def twiddle(mode):
            pow(PauliZ(wires[len(wires) // 2 + mode]), z=2 * mode / len(wires))

        twiddle()  # pylint: disable=no-value-for-parameter

        @for_loop(len(wires) // 2)
        def fouriers(i):
            _permute_and_apply(wires, Wires([wires[i], wires[len(wires) // 2 + i]]), TwoQubitFFT)

        fouriers()  # pylint: disable=no-value-for-parameter


def _permute_and_apply(order, wires, operator):
    """
    Makes the sites in question adjacent in the ordering, applies the given operator,
    and permutes them back.

    Args:
        wires (WiresLike): The wires to permute.
        operator (Type[Operator]): The operator to apply once the Fermions are adjacent in the encoding.
    """
    first = list(order).index(wires.labels[0])
    second = list(order).index(wires.labels[1])
    second_copy = copy.copy(second)

    # permute into adjacency
    @while_loop(lambda s: s > first + 1)
    def permute_in(s):
        FermionicSWAP(np.pi, Wires([order[s], order[s - 1]]))
        s -= 1
        return s

    permute_in(second)  # pylint: disable=no-value-for-parameter

    # apply the operator
    operator(Wires([order[first], order[first + 1]]))

    # permute back
    @while_loop(lambda s: s < second_copy)
    def permute_out(s):
        FermionicSWAP(np.pi, Wires([order[s], order[s + 1]]))
        s += 1
        return s

    permute_out(first + 1)  # pylint: disable=no-value-for-parameter


add_decomps(FFFT, _fast_fermionic_fourier_transform_decomposition)
