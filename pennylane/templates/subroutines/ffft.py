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

from pennylane import math
from pennylane.control_flow import for_loop
from pennylane.decomposition import add_decomps, pow_resource_rep, register_resources, resource_rep
from pennylane.operation import Operator
from pennylane.ops import PauliZ, pow
from pennylane.wires import Wires, WiresLike


class TwoQubitFFT(Operator):
    """
    The two-qubit unitary operator that corresponds to a Fourier transform on
    Fermions, encoded using a Jordan-Wigner Transformation (JWT).

    Args:
        wires (WiresLike): The two wires to apply the operator to. Ideally, they are adjacent
            wires, since the cost of simulating interactions between Fermionic modes depends
            on their distance in the encoding.
    """

    num_wires = 2
    num_params = 0

    def __init__(self, wires: WiresLike):
        super().__init__(wires=wires, id=None)

    def compute_matrix(self, *_):
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
    """Performs a Fast Fermionic Fourier Transform (FFFT) operation based on `arXiv:1310.7605 <https://arxiv.org/pdf/1310.7605>`_. This assumes that
    Fermions are encoded using a Jordan Wigner Transformation (JWT).

    The Fermionic Fourier transform over a number of wires n (a power of two)
    is decomposed recursively into two parallel Fourier transforms over n/2
    sites in each stack frame. These parallel FTs are followed by a series of
    2-site linear gates.

    Args:

        wires (WiresLike): The wires to apply the FFFT to. Must be a power of 2 greater than or equal to 2.

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

    i.e. for 4 sites:

    .. code-block:: python

        from pennylane.templates.subroutines.ffft import FFFT
        from pennylane import device, state

        dev = device("default.qubit")

        wires = (0, 1, 2, 3)

        @qnode(dev)
        def circuit(wires):
            FFFT(wires)
            return state()


    >>> print(qml.draw(circuit, level="device")(wires))
    0: ─╭TwoQubitFFT───────╭TwoQubitFFT──────────────┤  State
    1: ─╰TwoQubitFFT───────│────────────╭TwoQubitFFT─┤  State
    2: ─╭TwoQubitFFT──Z⁰⋅⁰─╰TwoQubitFFT─│────────────┤  State
    3: ─╰TwoQubitFFT──Z⁰⋅⁵──────────────╰TwoQubitFFT─┤  State"""

    resource_keys = {"num_wires"}

    def __init__(self, wires: WiresLike):
        if len(wires) <= 1:
            raise ValueError("The number of wires must be at least 2 for the FFFT algorithm.")
        if not math.log2(len(wires)).is_integer():
            raise NotImplementedError(
                "FFFT is currently only implemented for numbers of wires that are powers of two."
            )

        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        return {"num_wires": len(self.wires)}


def _fast_fermionic_fourier_transform_resources(num_wires):
    resources = defaultdict(int)

    def _count_two_recursive(wires, two_qubit_gates):
        two_qubit_gates += wires // 2
        if wires > 2:
            two_qubit_gates += _count_two_recursive(wires // 2, 0) * 2
        return two_qubit_gates

    two_qubit_gates = _count_two_recursive(num_wires, 0)
    resources[resource_rep(TwoQubitFFT)] = two_qubit_gates

    def _count_one_recursive(wires, resources):
        if wires > 2:
            for mode in range(wires // 2):
                resources[pow_resource_rep(PauliZ, {}, z=2 * mode / wires)] += 1
            resources = _count_one_recursive(wires // 2, resources)
            resources = _count_one_recursive(wires // 2, resources)
        return resources

    resources = _count_one_recursive(num_wires, resources)

    return resources


@register_resources(_fast_fermionic_fourier_transform_resources)
def _fast_fermionic_fourier_transform_decomposition(*_, wires: WiresLike, **__):
    wires = math.array(wires)
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
            TwoQubitFFT(Wires([wires[i], wires[len(wires) // 2 + i]]))

        fouriers()  # pylint: disable=no-value-for-parameter


add_decomps(FFFT, _fast_fermionic_fourier_transform_decomposition)
