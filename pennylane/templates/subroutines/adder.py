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

    The decomposition of this operator is based on the QFT-based method presented in `Atchade-Adelomou and Gonzalez (2023) <https://arxiv.org/abs/2311.08555>`_.

        Args:
            k (int): number that wants to be added.
            x_wires (Sequence[int]): the wires the operation acts on. There are needed at least enough wires to represent mod.
            mod (int): modulo with respect to which the sum is performed, default value will be ``2^len(wires)``.
            work_wire (Sequence[int]): the auxiliary wires to use for the sum modulo :math:`mod` when :math:`mod \neq 2^{\text{len(x_wires)}}`.

        **Example**

        Sum of two integers :math:`x=8` and :math:`k=5` modulo :math:`mod=15`. Note that to perform this sum using qml.Adder, when :math:`mod \neq \text{len(x_wires)}` we need that :math:`x < \text{len(x_wires)}/2`.

        .. code-block::

            x = 8
            k = 5
            mod = 15
            x_wires =[0,1,2,3]
            work_wires=[4]
            dev = qml.device("default.qubit", shots=1)
            @qml.qnode(dev)
            def adder_modulo(x, k, mod, x_wires, work_wires):
                # Function that performs x + k modulo mod in the computational basis
                qml.BasisEmbedding(x, wires=x_wires)
                qml.Adder(k, x_wires, mod, work_wire)
                return qml.sample(wires=x_wires)

        .. code-block:: pycon

            >>> print(f"The ket representation of {x} + {k} mod {mod} is {adder_modulo(x, k, mod,x_wires, work_wire)}")
            The ket representation of 8 + 5 mod 15 is [1 1 0 1]

        We can see that the result [1 1 0 1] corresponds to 13, which comes from :math:`8+5=13 \longrightarrow 13 \, \text{mod} \, 15 = 13`.
    """

    grad_method = None

    def __init__(
        self, k, x_wires, mod=None, work_wire=None, id=None
    ):  # pylint: disable=too-many-arguments
        
        if work_wire is not None:
            if any(wire in work_wire for wire in x_wires):
                raise ValueError("work_wire should not be included in x_wires.")

        self.hyperparameters["k"] = k
        self.hyperparameters["mod"] = mod
        self.hyperparameters["work_wire"] = qml.wires.Wires(work_wire)
        self.hyperparameters["x_wires"] = qml.wires.Wires(x_wires)
        all_wires=qml.wires.Wires(x_wires)+qml.wires.Wires(work_wire)
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
            for key in ["x_wires", "work_wire"]
        }

        return PhaseAdder(
            self.hyperparameters["k"],
            new_dict["x_wires"],
            self.hyperparameters["mod"],
            new_dict["work_wire"]
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
        return (
            self.hyperparameters["x_wires"]
            + self.hyperparameters["work_wire"]
        )
    def decomposition(self):  # pylint: disable=arguments-differ

        return self.compute_decomposition(
            self.hyperparameters["k"],
            self.hyperparameters["x_wires"],
            self.hyperparameters["mod"],
            self.hyperparameters["work_wire"],
        )
    @classmethod
    def _primitive_bind_call(cls, *args, **kwargs):
        return cls._primitive.bind(*args, **kwargs)
    @staticmethod
    def compute_decomposition(k, x_wires, mod, work_wire):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a product of other operators.
        Args:
            k (int): number that wants to be added
            x_wires (Sequence[int]): the wires the operation acts on. There are needed at least enough wires to represent mod.
            mod (int): modulo of the sum
            work_wire (Sequence[int]): the auxiliary wires to use for the sum modulo :math:`mod` when :math:`mod \neq 2^{\textrm{len(wires)}}`
        Returns:
            list[.Operator]: Decomposition of the operator

        **Example**

        >>> qml.Adder.compute_decomposition(k=2,x_wires=[0,1,2], mod = 8, work_wire=None)
        [QFT(wires=[0, 1, 2]),
        PhaseAdder(wires=[0, 1, 2]),
        Adjoint(QFT(wires=[0, 1, 2]))]
        """
        op_list = []
        op_list.append(qml.QFT(x_wires))
        op_list.append(qml.PhaseAdder(k, x_wires, mod, work_wire))
        op_list.append(qml.adjoint(qml.QFT)(x_wires))

        return op_list