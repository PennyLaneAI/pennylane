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
Contains the Multiplier template.
"""

import numpy as np
import pennylane as qml
from pennylane.operation import Operation


def _mul_out_k_mod(k, wires_m, mod, work_wires_aux, wires_aux):
    """Performs m*k in the registers wires_aux"""
    op_list = []
    if mod == (2 ** len(wires_m)):
        qft_wires = wires_aux
    else:
        qft_wires = work_wires_aux[:1] + wires_aux
    op_list.append(qml.QFT(wires=qft_wires))
    op_list.append(
        qml.ControlledSequence(qml.PhaseAdder(k, wires_aux, mod, work_wires_aux), control=wires_m)
    )
    op_list.append(qml.adjoint(qml.QFT(wires=qft_wires)))
    return op_list


class Multiplier(Operation):
    r"""Performs the Inplace Multiplication operation.

    This operator multiplies the integer :math:`k` modulo :math:`mod` in the computational basis:

    .. math::
        Multiplier(k,mod) |m \rangle = | m*k mod mod \rangle,

    The quantum circuit that represents the Multiplier operator is:


    Args:
        k (int): number that wants to be added
        wires (Sequence[int]): the wires the operation acts on. There are needed at least enough wires to represent :math:`k` and :math:`mod`.
        mod (int): modulo with respect to which the multiplication is performed, default value will be ``2^len(wires)``
        work_wires (Sequence[int]): the auxiliary wires to use for the multiplication modulo :math:`mod`

    **Example**

    Multiplication of two integers :math:`m=3` and :math:`k=4` modulo :math:`mod=7`. Note that to perform this multiplication using qml.Multiplier we need that :math:`m,k < mod`
    and that :math:`k` has inverse, :math:`k^-1`, modulo :math:`mod`. That means :math:`k*k^-1 modulo mod = 1`, which will only be possible if :math:`k` and :math:`mod` are coprime.

    .. code-block::
        m = 3
        k = 4
        mod = 7
        wires_m =[0,1,2]
        work_wires=[3,4,5,6,7]
        dev = qml.device("default.qubit", shots=1)
        @qml.qnode(dev)
        def multiplier_modulo(m, k, mod, wires_m, work_wires):
            # Function that performs m * k modulo mod in the computational basis
            qml.BasisEmbedding(m, wires=wires_m)
            qml.Multiplier(k, wires_m, mod, work_wires)
            return qml.sample(wires=wires_m)

    .. code-block:: pycon

        >>> print(f"The ket representation of {m} * {k} mod {mod} is {multiplier_modulo(m, k, mod, wires_m, work_wires)}")
        The ket representation of 3 * 4 mod 7 is [1 0 1]

    We can see that the result [1 0 1] corresponds to 5, which comes from :math:`3+4=12 \longrightarrow 12 mod 7 = 5`.
    """

    grad_method = None

    def __init__(self, k, wires, mod=None, work_wires=None, id=None):
        if mod == None:
            mod = 2 ** (len(wires))
        if k >= mod:
            raise ValueError("The module mod must be larger than k.")
        if (not hasattr(wires, "__len__")) or (mod > 2 ** (len(wires))):
            raise ValueError("Multiplier must have at least enough wires to represent mod.")
        if work_wires != None:
            if any(wire in work_wires for wire in wires):
                raise ValueError("Any wire in work_wires should not be included in wires.")
            if len(work_wires) < (len(wires) + 2):
                raise ValueError("Multiplier needs as many work_wires as wires plus two.")
        else:
            work_wires = list(range(len(wires), 2 * len(wires) + 2))

        if np.gcd(k, mod) != 1:
            raise ValueError("Since k has no inverse modulo mod, the work_wires cannot be cleaned.")

        self.hyperparameters["k"] = k
        self.hyperparameters["mod"] = mod
        self.hyperparameters["work_wires"] = qml.wires.Wires(work_wires)
        super().__init__(wires=wires, id=id)

    @property
    def num_params(self):
        return 0

    @staticmethod
    def compute_decomposition(k, mod, work_wires, wires):
        r"""Representation of the operator as a product of other operators.
        Args:
            k (int): number that wants to be added
            mod (int): modulo of the sum
            work_wires (Sequence[int]): the auxiliary wires to use for the sum modulo :math:`mod` when :math:`mod \neq 2^{\textrm{len(wires)}}`
            wires (Sequence[int]): the wires the operation acts on
        Returns:
            list[.Operator]: Decomposition of the operator

        **Example**

        >>> qml.Multiplier.compute_decomposition(k=3,mod=8,wires=[0,1,2],work_wires=[3,4,5,6,7])
        [QFT(wires=[5, 6, 7]),
        ControlledSequence(PhaseAdder(wires=[5, 6, 7]), control=[0, 1, 2]),
        Adjoint(QFT(wires=[5, 6, 7])),
        SWAP(wires=[0, 5]),
        SWAP(wires=[1, 6]),
        SWAP(wires=[2, 7]),
        Adjoint(Adjoint(QFT(wires=[5, 6, 7]))),
        Adjoint(ControlledSequence(PhaseAdder(wires=[5, 6, 7]), control=[0, 1, 2])),
        Adjoint(QFT(wires=[5, 6, 7]))]
        """

        op_list = []
        work_wires_aux = work_wires[0:2]
        wires_aux = work_wires[2:]
        op_list.extend(_mul_out_k_mod(k, wires, mod, work_wires_aux, wires_aux))
        for i in range(len(wires)):
            op_list.append(qml.SWAP(wires=[wires[i], wires_aux[i]]))
        inv_k = pow(k, -1, mod)
        op_list.extend(qml.adjoint(_mul_out_k_mod)(inv_k, wires, mod, work_wires_aux, wires_aux))
        return op_list
