# Copyright 2024 Xanadu Quantum Technologies Inc.

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
Contains the OutAdder template.
"""

import pennylane as qml
from pennylane.operation import Operation


class OutAdder(Operation):
    r"""Performs the Outplace Addition operation.

    This operator adds the integer :math:`k` modulo :math:`mod` in the computational basis:

    .. math::

        \text{OutAdder}(mod) |m \rangle | k \rangle | 0 \rangle = |m \rangle | k \rangle | m+k \, \text{mod} \, mod \rangle ,

    The decomposition of this operator is based on the QFT-based method presented in `Atchade-Adelomou and Gonzalez (2023) <https://arxiv.org/abs/2311.08555>`_.

    Args:
        x_wires (Sequence[int]): the wires that stores the integer :math:`x`.
        y_wires (Sequence[int]): the wires that stores the integer :math:`y`.
        output_wires (Sequence[int]): the wires that stores the sum modulo mod :math:`x+y \, \text{mod} \, mod`.
        mod (int): modulo with respect to which the sum is performed, default value will be ``2^len(wires)``.
        work_wires (Sequence[int]): the auxiliary wires to use for the sum modulo :math:`mod` when :math:`mod \neq 2^{\text{len(wires)}}`

    **Example**

    Sum of two integers :math:`x=8` and :math:`y=5` modulo :math:`mod=15`. Note that to perform this sum using qml.OutAdder we need that :math:`x,y < mod`.

    .. code-block::

        x=5
        y=6
        mod=7
        x_wires=[0,1,2]
        y_wires=[3,4,5]
        output_wires=[7,8,9]
        work_wires=[6,10]
        dev = qml.device("default.qubit", shots=1)
        @qml.qnode(dev)
        def circuit_OutAdder():
            qml.BasisEmbedding(x, wires=x_wires)
            qml.BasisEmbedding(y, wires=y_wires)
            qml.OutAdder(x_wires, y_wires, output_wires, mod, work_wires)
            return qml.sample(wires=output_wires)

    .. code-block:: pycon

        >>> print(f"The ket representation of {x} + {y} mod {mod} is {circuit_OutAdder()}")
        The ket representation of 5 + 6 mod 7 is [1 0 0]

    We can see that the result [1 0 0] corresponds to 4, which comes from :math:`5+6=11 \longrightarrow 11 \, \text{mod} \, 7 = 4`.
    """

    grad_method = None

    def __init__(self, x_wires, y_wires, output_wires, mod=None, work_wires=None, id=None):

        if mod is None:
            mod = 2 ** (len(output_wires))
        if (not hasattr(output_wires, "__len__")) or (mod > 2 ** len(output_wires)):
            raise ValueError("OutAdder must have at least enough wires to represent mod.")
        if work_wires is not None:
            if any(wire in work_wires for wire in x_wires):
                raise ValueError("None of the wires in work_wires should be included in x_wires.")
            if any(wire in work_wires for wire in y_wires):
                raise ValueError("None of the wires in work_wires should be included in y_wires.")
        else:
            max_wire = max(max(x_wires), max(y_wires), max(output_wires))
            work_wires = [max_wire + 1, max_wire + 2]
        for key in ["x_wires", "y_wires", "output_wires", "work_wires"]:
            self.hyperparameters[key] = qml.wires.Wires(locals()[key])
        all_wires = sum(
            self.hyperparameters[key]
            for key in ["x_wires", "y_wires", "output_wires", "work_wires"]
        )
        self.hyperparameters["mod"] = mod
        super().__init__(wires=all_wires, id=id)

    @property
    def num_params(self):
        return 0

    @staticmethod
    def compute_decomposition(x_wires, y_wires, output_wires, mod, work_wires, **kwargs):
        r"""Representation of the operator as a product of other operators.
        Args:
            x_wires (Sequence[int]): the wires that stores the integer :math:`x`.
            y_wires (Sequence[int]): the wires that stores the integer :math:`y`.
            output_wires (Sequence[int]): the wires that stores the sum modulo mod :math:`x+y mod mod`.
            mod (int): modulo with respect to which the sum is performed, default value will be ``2^len(output_wires)``.
            work_wires (Sequence[int]): the auxiliary wires to use for the sum modulo :math:`mod` when :math:`mod \neq 2^{\textrm{len(output_wires)}}`.
        Returns:
            list[.Operator]: Decomposition of the operator

        **Example**

        >>> qml.OutAdder.compute_decomposition(x_wires=[0,1], y_wires=[2,3], output_wires=[5,6], mod=4, work_wires=[4,7])
        [CNOT(wires=[2, 5]),
        CNOT(wires=[3, 6]),
        QFT(wires=[5, 6]),
        ControlledSequence(PhaseAdder(wires=[5, 6]), control=[0, 1]),
        Adjoint(QFT(wires=[5, 6]))]
        """
        op_list = []
        if mod != 2 ** (len(output_wires)):
            qft_new_output_wires = work_wires[:1] + output_wires
        else:
            qft_new_output_wires = output_wires
        for i in range(len(y_wires)):
            op_list.append(qml.CNOT(wires=[y_wires[i], output_wires[i]]))
        op_list.append(qml.QFT(wires=qft_new_output_wires))
        op_list.append(
            qml.ControlledSequence(
                qml.PhaseAdder(1, output_wires, mod, work_wires), control=x_wires
            )
        )
        op_list.append(qml.adjoint(qml.QFT)(wires=qft_new_output_wires))

        return op_list
