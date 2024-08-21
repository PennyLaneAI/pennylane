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
Contains the Adder template.
"""

import pennylane as qml
from pennylane.operation import Operation


class Adder(Operation):
    r"""Performs the Inplace Addition operation.

    This operator adds the integer :math:`k` modulo :math:`mod` in the computational basis:

    .. math::

        \text{Adder}(k,mod) |x \rangle = | x+k \, \text{mod} \, mod \rangle,

    The decomposition of this operator is based on the QFT-based method presented in `arXiv:2311.08555 <https://arxiv.org/abs/2311.08555>`_.

    Args:
        k (int): the number that needs to be added
        x_wires (Sequence[int]): the wires the operation acts on
        mod (int): modulo with respect to which the sum is performed, default value will be ``2^len(wires)``.
        work_wires (Sequence[int]): the auxiliary wires to use for the sum modulo :math:`mod` when :math:`mod \neq 2^{\text{len(x_wires)}}`

    **Example**

    This example computes the sum of two integers :math:`x=8` and :math:`k=5` modulo :math:`mod=15`. Note that to perform this sum using qml.Adder, when :math:`mod \neq \text{len(x_wires)}` we need that :math:`x < \text{len(x_wires)}/2`.

    .. code-block::

        x = 8
        k = 5
        mod = 15
        x_wires =[0,1,2,3]
        work_wires=[4,5]
        dev = qml.device("default.qubit", shots=1)
        @qml.qnode(dev)
        def adder_modulo(x, k, mod, x_wires, work_wires):
            qml.BasisEmbedding(x, wires=x_wires)
            qml.Adder(k, x_wires, mod, work_wire)
            return qml.sample(wires=x_wires)

    .. code-block:: pycon

        >>> adder_modulo(x, k, mod,x_wires, work_wire)
        [1 1 0 1]

    The result [1 1 0 1] is the ket representation of :math:`8 + 5  \, \text{mod} \, 15 = 13`.
    """

    grad_method = None

    def __init__(
        self, k, x_wires, mod=None, work_wires=None, id=None
    ):  # pylint: disable=too-many-arguments

        if work_wires is not None:
            if any(wire in work_wires for wire in x_wires):
                raise ValueError("None wire in work_wires should be included in x_wires.")

        self.hyperparameters["k"] = k
        self.hyperparameters["mod"] = mod
        self.hyperparameters["work_wires"] = qml.wires.Wires(work_wires)
        self.hyperparameters["x_wires"] = qml.wires.Wires(x_wires)
        all_wires = qml.wires.Wires(x_wires) + qml.wires.Wires(work_wires)
        super().__init__(wires=all_wires, id=id)

    @property
    def num_params(self):
        return 0

    def _flatten(self):
        metadata = tuple((key, value) for key, value in self.hyperparameters.items())
        return tuple(), metadata

    @classmethod
    def _unflatten(cls, data, metadata):
        hyperparams_dict = dict(metadata)
        return cls(**hyperparams_dict)

    def map_wires(self, wire_map: dict):
        new_dict = {
            key: [wire_map.get(w, w) for w in self.hyperparameters[key]]
            for key in ["x_wires", "work_wires"]
        }

        return Adder(
            self.hyperparameters["k"],
            new_dict["x_wires"],
            self.hyperparameters["mod"],
            new_dict["work_wires"],
        )

    @property
    def x_wires(self):
        """The wires where x is loaded."""
        return self.hyperparameters["x_wires"]

    @property
    def work_wire(self):
        """The work_wire."""
        return self.hyperparameters["work_wire"]

    @property
    def wires(self):
        """All wires involved in the operation."""
        return self.hyperparameters["x_wires"] + self.hyperparameters["work_wires"]

    def decomposition(self):  # pylint: disable=arguments-differ

        return self.compute_decomposition(
            self.hyperparameters["k"],
            self.hyperparameters["x_wires"],
            self.hyperparameters["mod"],
            self.hyperparameters["work_wires"],
        )

    @classmethod
    def _primitive_bind_call(cls, *args, **kwargs):
        return cls._primitive.bind(*args, **kwargs)

    @staticmethod
    def compute_decomposition(k, x_wires, mod, work_wires):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a product of other operators.
        Args:
            k (int): number that wants to be added
            x_wires (Sequence[int]): the wires the operation acts on. There are needed at least enough wires to represent mod.
            mod (int): modulo of the sum
            work_wires (Sequence[int]): the auxiliary wires to use for the sum modulo :math:`mod` when :math:`mod \neq 2^{\textrm{len(x_wires)}}`
        Returns:
            list[.Operator]: Decomposition of the operator

        **Example**

        >>> qml.Adder.compute_decomposition(k=2,x_wires=[0,1,2], mod = 8, work_wires=None)
        [QFT(wires=[0, 1, 2]),
        PhaseAdder(wires=[0, 1, 2]),
        Adjoint(QFT(wires=[0, 1, 2]))]
        """
        op_list = []
        if mod == 2 ** (len(x_wires)):
            qft_wires = x_wires
        else:
            qft_wires = work_wires[:1] + x_wires
        op_list.append(qml.QFT(qft_wires))
        op_list.append(qml.PhaseAdder(k, x_wires, mod, work_wires[1:]))
        op_list.append(qml.adjoint(qml.QFT)(qft_wires))

        return op_list
