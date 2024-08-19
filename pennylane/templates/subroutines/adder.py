# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
Contains the Adder template.
"""

import pennylane as qml
from pennylane.operation import Operation


class Adder(Operation):
    r"""Performs the Inplace Addition operation.

        This operator adds the integer :math:`k` modulo :math:`mod` in the computational basis:

        .. math::
            Adder(k,mod) |m \rangle = | m+k \, \textrm{mod} \, mod \rangle,

    The decomposition of this operator is based on the QFT-based method presented in `Atchade-Adelomou and Gonzalez (2023) <https://arxiv.org/abs/2311.08555>`_.

        Args:
            k (int): number that wants to be added.
            wires (Sequence[int]): the wires the operation acts on. There are needed at least enough wires to represent k and mod.
            mod (int): modulo with respect to which the sum is performed, default value will be ``2^len(wires)``.
            work_wires (Sequence[int]): the auxiliary wires to use for the sum modulo :math:`mod` when :math:`mod \neq 2^{\text{len(wires)}}`.

        **Example**

        Sum of two integers :math:`m=8` and :math:`k=5` modulo :math:`mod=15`. Note that to perform this sum using qml.Adder we need that :math:`m,k < mod`.

        .. code-block::

            m = 8
            k = 5
            mod = 15
            wires_m =[0,1,2,3]
            work_wires=[4,5]
            dev = qml.device("default.qubit", shots=1)
            @qml.qnode(dev)
            def adder_modulo(m, k, mod, wires_m, work_wires):
                # Function that performs m + k modulo mod in the computational basis
                qml.BasisEmbedding(m, wires=wires_m)
                qml.Adder(k, wires_m, mod, work_wires)
                return qml.sample(wires=wires_m)

        .. code-block:: pycon

            >>> print(f"The ket representation of {m} + {k} mod {mod} is {adder_modulo(m, k, mod,wires_m, work_wires)}")
            The ket representation of 8 + 5 mod 15 is [1 1 0 1]

        We can see that the result [1 1 0 1] corresponds to 13, which comes from :math:`8+5=13 \longrightarrow 13 \, \text{mod} \, 15 = 13`.
    """

    grad_method = None

    def __init__(
        self, k, wires, mod=None, work_wires=None, id=None
    ):  # pylint: disable=too-many-arguments
        if mod is None:
            mod = 2 ** (len(wires))
        if work_wires is not None:
            if any(wire in work_wires for wire in wires):
                raise ValueError("Any wire in work_wires should not be included in wires.")
        else:
            work_wires = [wires[-1] + 1, wires[-1] + 2]
        self.hyperparameters["k"] = k
        self.hyperparameters["mod"] = mod
        self.hyperparameters["work_wires"] = qml.wires.Wires(work_wires)
        super().__init__(wires=wires, id=id)

    @property
    def num_params(self):
        return 0

    @classmethod
    def _primitive_bind_call(cls, *args, **kwargs):
        return cls._primitive.bind(*args, **kwargs)

    @staticmethod
    def compute_decomposition(k, mod, work_wires, wires):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a product of other operators.
        Args:
            k (int): number that wants to be added
            mod (int): modulo of the sum
            work_wires (Sequence[int]): the auxiliary wires to use for the sum modulo :math:`mod` when :math:`mod \neq 2^{\textrm{len(wires)}}`
            wires (Sequence[int]): the wires the operation acts on
        Returns:
            list[.Operator]: Decomposition of the operator

        **Example**

        >>> qml.Adder.compute_decomposition(k=2,mod=8,wires=[0,1,2],work_wires=[3,4])
        [QFT(wires=[0, 1, 2]),
        PhaseAdder(wires=[0, 1, 2]),
        Adjoint(QFT(wires=[0, 1, 2]))]
        """
        op_list = []
        if mod == 2 ** len(wires):
            qft_wires = wires
        else:
            qft_wires = work_wires[:1] + wires
        op_list.append(qml.QFT(qft_wires))
        op_list.append(qml.PhaseAdder(k, wires, mod, work_wires))
        op_list.append(qml.adjoint(qml.QFT)(qft_wires))

        return op_list
