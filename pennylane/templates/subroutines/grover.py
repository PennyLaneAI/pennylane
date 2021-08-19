# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
Contains the Grover Operation template.
"""
import itertools
import functools
import numpy as np
import pennylane as qml
from pennylane.operation import AnyWires, Operation
from pennylane.ops import Hadamard, PauliZ, MultiControlledX


class GroverOperator(Operation):
    r"""Performs the Grover Diffusion Operator.

    .. math::

        G = 2 |s \rangle \langle s | - I
        = H^{\bigotimes n} \left( 2 |0\rangle \langle 0| - I \right) H^{\bigotimes n}

    where :math:`n` is the number of wires, and :math:`|s\rangle` is the uniform superposition:

    .. math::

        |s\rangle = H^{\bigotimes n} |0\rangle =  \frac{1}{\sqrt{2^n}} \sum_{i=0}^{2^n-1} | i \rangle.

    For this template, the operator is implemented with a layer of Hadamards, a layer of :math:`X`,
    followed by a multi-controlled :math:`Z` gate, then another layer of :math:`X` and Hadamards.
    This is expressed in a compact form by the circuit below:

    .. figure:: ../../_static/templates/subroutines/grover.svg
        :align: center
        :width: 60%
        :target: javascript:void(0);

    The open circles on the controlled gate indicate control on 0 instead of 1.
    The ``Z`` gates on the last wire result from leveraging the circuit identity :math:`HXH = Z`,
    where the last ``H`` gate converts the multi-controlled :math:`Z` gate into a
    multi-controlled :math:`X` gate.

    Args:
        wires (Union[Wires, Sequence[int], or int]): the wires to apply to
        work_wires (Union[Wires, Sequence[int], or int]): optional auxiliary wires to assist
            in the decomposition of :class:`~.MultiControlledX`.

    **Example**

    The Grover Diffusion Operator amplifies the magnitude of the basis state with
    a negative phase.  For example, if the solution to the search problem is the :math:`|111\rangle`
    state, we require an oracle that flips its phase; this could be implemented using a `CCZ` gate:

    .. code-block:: python

        n_wires = 3
        wires = list(range(n_wires))

        def oracle():
            qml.Hadamard(wires[-1])
            qml.Toffoli(wires=wires)
            qml.Hadamard(wires[-1])

    We can then implement the entire Grover Search Algorithm for ``num_iterations`` iterations by alternating calls to the oracle and the diffusion operator:

    .. code-block:: python

        dev = qml.device('default.qubit', wires=wires)

        @qml.qnode(dev)
        def GroverSearch(num_iterations=1):
            for wire in wires:
                qml.Hadamard(wire)

            for _ in range(num_iterations):
                oracle()
                qml.templates.GroverOperator(wires=wires)
            return qml.probs(wires)

    >>> GroverSearch(num_iterations=1)
    tensor([0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125,
            0.78125], requires_grad=True)
    >>> GroverSearch(num_iterations=2)
    tensor([0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125,
        0.0078125, 0.9453125], requires_grad=True)

    We can see that the marked :math:`|111\rangle` state has the greatest probability amplitude.

    Optimally, the oracle-operator pairing should be repeated :math:`\lceil \frac{\pi}{4}\sqrt{2^{n}} \rceil` times.

    """
    num_params = 0
    num_wires = AnyWires
    par_domain = None

    def __init__(self, wires=None, work_wires=None, do_queue=True, id=None):
        if (not hasattr(wires, "__len__")) or (len(wires) < 2):
            raise ValueError("GroverOperator must have at least two wires provided.")

        self.work_wires = work_wires
        super().__init__(wires=wires, do_queue=do_queue, id=id)

    def expand(self):
        ctrl_str = "0" * (len(self.wires) - 1)

        with qml.tape.QuantumTape() as tape:
            for wire in self.wires[:-1]:
                Hadamard(wire)

            PauliZ(self.wires[-1])
            MultiControlledX(
                control_values=ctrl_str,
                control_wires=self.wires[:-1],
                wires=self.wires[-1],
                work_wires=self.work_wires,
            )

            PauliZ(self.wires[-1])

            for wire in self.wires[:-1]:
                Hadamard(wire)

        return tape

    @property
    def matrix(self):
        # Redefine the property here to allow for a custom _matrix signature
        mat = self._matrix(len(self.wires))
        return mat

    @classmethod
    def _matrix(cls, *params):
        num_wires = params[0]

        # s1 = H|0>, Hadamard on a single qubit in the ground state
        s1 = np.array([1, 1]) / np.sqrt(2)

        # uniform superposition state |s>
        s = functools.reduce(np.kron, list(itertools.repeat(s1, num_wires)))

        # Grover diffusion operator
        G = 2 * np.outer(s, s) - np.identity(2 ** num_wires)
        return G
