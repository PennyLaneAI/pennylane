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
This submodule contains the discrete-variable quantum operations that perform
arithmetic operations on their input states.
"""
# pylint:disable=abstract-method,arguments-differ,protected-access
import numpy as np

import pennylane as qml
from pennylane.operation import Operation


class QubitCarry(Operation):
    r"""QubitCarry(wires)
    Apply the ``QubitCarry`` operation to four input wires.

    This operation performs the transformation:

    .. math::
        |a\rangle |b\rangle |c\rangle |d\rangle \rightarrow |a\rangle |b\rangle |b\oplus c\rangle |bc \oplus d\oplus (b\oplus c)a\rangle

    .. figure:: ../../_static/ops/QubitCarry.svg
        :align: center
        :width: 60%
        :target: javascript:void(0);

    See `here <https://arxiv.org/abs/quant-ph/0008033v1>`__ for more information.

    .. note::
        The first wire should be used to input a carry bit from previous operations. The final wire
        holds the carry bit of this operation and the input state on this wire should be
        :math:`|0\rangle`.

    **Details:**

    * Number of wires: 4
    * Number of parameters: 0

    Args:
        wires (Sequence[int]): the wires the operation acts on

    **Example**

    The ``QubitCarry`` operation maps the state :math:`|0110\rangle` to :math:`|0101\rangle`, where
    the last qubit denotes the carry value:

    .. code-block::

        input_bitstring = (0, 1, 1, 0)

        @qml.qnode(dev)
        def circuit(basis_state):
            qml.BasisState(basis_state, wires=[0, 1, 2, 3])
            qml.QubitCarry(wires=[0, 1, 2, 3])
            return qml.probs(wires=[0, 1, 2, 3])

        probs =  circuit(input_bitstring)
        probs_indx = np.argwhere(probs == 1).flatten()[0]
        bitstrings = list(itertools.product(range(2), repeat=4))
        output_bitstring = bitstrings[probs_indx]

    The output bitstring is

    >>> output_bitstring
    (0, 1, 0, 1)

    The action of ``QubitCarry`` is to add wires ``1`` and ``2``. The modulo-two result is output
    in wire ``2`` with a carry value output in wire ``3``. In this case, :math:`1 \oplus 1 = 0` with
    a carry, so we have:

    >>> bc_sum = output_bitstring[2]
    >>> bc_sum
    0
    >>> carry = output_bitstring[3]
    >>> carry
    1
    """
    num_params = 0
    num_wires = 4
    par_domain = None
    _mat = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        ]
    )

    @classmethod
    def _matrix(cls, *params):
        return QubitCarry._mat

    @staticmethod
    def decomposition(wires):
        decomp_ops = [
            qml.Toffoli(wires=wires[1:]),
            qml.CNOT(wires=[wires[1], wires[2]]),
            qml.Toffoli(wires=[wires[0], wires[2], wires[3]]),
        ]
        return decomp_ops


class QubitSum(Operation):
    r"""QubitSum(wires)
    Apply a ``QubitSum`` operation on three input wires.

    This operation performs the transformation:

    .. math::
        |a\rangle |b\rangle |c\rangle \rightarrow |a\rangle |b\rangle |a\oplus b\oplus c\rangle


    .. figure:: ../../_static/ops/QubitSum.svg
        :align: center
        :width: 40%
        :target: javascript:void(0);

    See `here <https://arxiv.org/abs/quant-ph/0008033v1>`__ for more information.

    **Details:**

    * Number of wires: 3
    * Number of parameters: 0

    Args:
        wires (Sequence[int]): the wires the operation acts on

    **Example**

    The ``QubitSum`` operation maps the state :math:`|010\rangle` to :math:`|011\rangle`, with the
    final wire holding the modulo-two sum of the first two wires:

    .. code-block::

        input_bitstring = (0, 1, 0)

        @qml.qnode(dev)
        def circuit(basis_state):
            qml.BasisState(basis_state, wires = [0, 1, 2])
            qml.QubitSum(wires=[0, 1, 2])
            return qml.probs(wires=[0, 1, 2])

        probs = circuit(input_bitstring)
        probs_indx = np.argwhere(probs == 1).flatten()[0]
        bitstrings = list(itertools.product(range(2), repeat=3))
        output_bitstring = bitstrings[probs_indx]

    The output bitstring is

    >>> output_bitstring
    (0, 1, 1)

    The action of ``QubitSum`` is to add wires ``0``, ``1``, and ``2``. The modulo-two result is
    output in wire ``2``. In this case, :math:`0 \oplus 1 \oplus 0 = 1`, so we have:

    >>> abc_sum = output_bitstring[2]
    >>> abc_sum
    1
    """
    num_params = 0
    num_wires = 3
    par_domain = None
    _mat = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ]
    )

    @classmethod
    def _matrix(cls, *params):
        return QubitSum._mat

    @staticmethod
    def decomposition(wires):
        decomp_ops = [
            qml.CNOT(wires=[wires[1], wires[2]]),
            qml.CNOT(wires=[wires[0], wires[2]]),
        ]
        return decomp_ops

    def adjoint(self):
        return QubitSum(wires=self.wires)
