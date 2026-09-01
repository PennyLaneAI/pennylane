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
"""Contains the HammingFour template, a specialized subroutine for parallelized rotations."""

import pennylane as qp
from pennylane.core.operator import Operation
from pennylane.wires import Wires


class HammingFour(Operation):
    r"""Computes the Hamming weight of four qubits in a specialized verbose output format
    of five qubits.

    Args:
        input_wires (WiresLike): Input wires for which to compute the Hamming weight. Must have
            length four.
        output_wires (WiresLike): Output wires in which to store the computed Hamming weight.
            Must have length five, with the wires storing the binary representation
            :math:`(w_0w_1w_2)_2` of the Hamming weight in little endian (first three qubits),
            the product of the ones and twos bits, :math:`w_0w_1` (fourth qubit),
            and a temporary cache qubit storing :math:`k=(i_0^i_1^i_2)&i_3` (fifth bit).

    The additional output bit storing :math:`k` saves non-Clifford cost, and can be reused and
    uncomputed in the uncomputation of :class:`~.HammingFour`.

    **Example**

    Consider the specific scenario where we want to load a distinct bitstring based on the
    Hamming weight of four input qubits, but no bitstring for a Hamming weight of zero.
    This is useful when we want to perform parallel rotations with the same angle, where the
    bitstring to load for Hamming weight :math:`n` is the binary representation of :math:`n\theta`
    for some angle :math:`\theta`.
    Then we can use the following circuit:

    .. code-block:: python

        def load(bitstrings, input_wires, target_wires, work_wires):
            '''Load the n'th bitstring for Hamming weight n in the input_wires.'''
            # Compute Hamming weight and extra qubit with product w0w1
            HammingFour(input_wires, work_wires)

            # Load the bitstrings for Hamming weight 1, 2, and 4, controlled on the respective
            # output wire of HammingFour. This wrongly loads bitstrings[0]^bitstrings[1] if
            # the Hamming weight is 3.
            qp.ctrl(qp.MultiX(bitstrings[0], target_wires), work_wires[0])
            qp.ctrl(qp.MultiX(bitstrings[1], target_wires), work_wires[1])
            qp.ctrl(qp.MultiX(bitstrings[3], target_wires), work_wires[2])

            # Make use of the product w0w1 stored on the fourth output wire to correct the loaded
            # bitstring for Hamming weight being 3.
            relative_string = np.sum(bitstrings[:2], axis=0) % 2
            qp.ctrl(qp.MultiX(relative_string, target_wires), work_wires[3])

            # Uncompute the Hamming weight
            qp.adjoint(HammingFour)(input_wires, work_wires)

    Note that this loading function can replace a four-control :class:`~.QROM` in this highly
    specialized scenario, at a cost of just four elbow gates.

    """

    resource_keys = {}

    def __init__(self, input_wires, output_wires):
        input_wires = Wires(input_wires)
        output_wires = Wires(output_wires)
        assert len(input_wires) == 4
        assert len(output_wires) == 5
        super().__init__(input_wires + output_wires)
        self.hyperparameters["input_wires"] = input_wires
        self.hyperparameters["output_wires"] = output_wires


@qp.register_resources({qp.CNOT: 10, qp.TemporaryAND: 4})
def _hamming_four(input_wires, output_wires):
    w0, w1, w2, t, k = output_wires
    _ = [qp.CNOT(inp, w0) for inp in input_wires]
    i0, i1, i2, i3 = input_wires
    qp.CNOT([i2, i0])
    qp.CNOT([i2, i1])
    qp.TemporaryAND([i0, i1, w1])
    qp.CNOT([i2, i1])
    qp.CNOT([i2, i0])
    qp.CNOT([i2, w1])
    qp.TemporaryAND([i3, w0, k], control_values=(1, 0))
    qp.TemporaryAND([w1, k, w2])
    qp.CNOT([k, w1])
    qp.TemporaryAND([w0, w1, t])


qp.add_decomps(HammingFour, _hamming_four)
