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
Contains the PhaseAdder template.
"""

import numpy as np

import pennylane as qml
from pennylane.operation import Operation


def _add_k_fourier(k, wires):
    """Adds k in the Fourier basis"""
    op_list = []
    for j, wire in enumerate(wires):
        op_list.append(qml.PhaseShift(k * np.pi / (2**j), wires=wire))
    return op_list


class PhaseAdder(Operation):
    r"""Performs the Inplace Phase Addition operation.

    This operator adds the integer :math:`k` modulo :math:`mod` in the Fourier basis:

    .. math::

        \text{PhaseAdder}(k,mod) |\phi (x) \rangle = |\phi (x+k \, \text{mod} \, mod) \rangle,

    where :math:`|\phi (x) \rangle` represents the :math:`| x \rangle`: state in the Fourier basis such:

    .. math::

        QFT |x \rangle = |\phi (x) \rangle.

    The decomposition of this operator is based on the QFT-based method presented in `arXiv:2311.08555 <https://arxiv.org/abs/2311.08555>`_.

    Args:
        k (int): the number that needs to be added
        x_wires (Sequence[int]): the wires the operation acts on
        mod (int): modulo with respect to which the sum is performed, default value will be ``2^len(wires)``.
        work_wire (Sequence[int]): the auxiliary wires to use for the sum modulo :math:`mod` when :math:`mod \neq 2^{\text{len(x_wires)}}`

    **Example**

    This example computes the sum of two integers :math:`x=5` and :math:`k=4` modulo :math:`mod=7`. Note that to perform this sum using qml.PhaseAdder, when :math:`mod \neq \text{len(x_wires)}` we need that :math:`x < \text{len(wires)}/2`.

    .. code-block::

        x = 5
        k = 4
        mod = 7
        x_wires =[0,1,2,3]
        work_wire=[4]
        dev = qml.device("default.qubit", shots=1)
        @qml.qnode(dev)
        def adder_modulo(x, k, mod, wires_m, work_wire):
            qml.BasisEmbedding(x, wires=x_wires)
            qml.QFT(wires=x_wires)
            PhaseAdder(k, x_wires, mod, work_wire)
            qml.adjoint(qml.QFT)(wires=x_wires)
            return qml.sample(wires=x_wires)

    .. code-block:: pycon

        >>> adder_modulo(x, k, mod, x_wires, work_wire)
        [0 1 0]

    The result [0 1 0] is the ket representation of :math:`5 + 4  \text{mod} \, 7 = 2`.
    """

    grad_method = None

    def __init__(
        self, k, x_wires, mod=None, work_wire=None, id=None
    ):  # pylint: disable=too-many-arguments

        if mod is None:
            mod = 2 ** len(x_wires)
        elif work_wire is None:
            raise ValueError(f"If mod is not 2^{len(x_wires)} you should provide one work_wire")
        k = k % mod
        if not hasattr(x_wires, "__len__") or mod > 2 ** len(x_wires):
            raise ValueError("PhaseAdder must have at least enough x_wires to represent mod.")
        if work_wire is not None:
            if any(wire in work_wire for wire in x_wires):
                raise ValueError("work_wire should not be included in x_wires.")

        self.hyperparameters["k"] = k
        self.hyperparameters["mod"] = mod
        self.hyperparameters["work_wire"] = qml.wires.Wires(work_wire)
        self.hyperparameters["x_wires"] = qml.wires.Wires(x_wires)
        all_wires = qml.wires.Wires(x_wires) + qml.wires.Wires(work_wire)
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
            new_dict["work_wire"],
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
        return self.hyperparameters["x_wires"] + self.hyperparameters["work_wire"]

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

        >>> qml.PhaseAdder.compute_decomposition(k=2,x_wires=[0,1,2],mod=8,work_wire=None)
        [PhaseShift(6.283185307179586, wires=[1]),
        PhaseShift(3.141592653589793, wires=[2]),
        PhaseShift(1.5707963267948966, wires=[3])]
        """
        op_list = []

        if mod == 2 ** (len(x_wires)):
            op_list.extend(_add_k_fourier(k, x_wires))
        else:
            aux_k = x_wires[0]
            op_list.extend(_add_k_fourier(k, x_wires))
            op_list.extend(qml.adjoint(_add_k_fourier)(mod, x_wires))
            op_list.append(qml.adjoint(qml.QFT)(wires=x_wires))
            op_list.append(qml.ctrl(qml.PauliX(work_wire), control=aux_k, control_values=1))
            op_list.append(qml.QFT(wires=x_wires))
            op_list.extend(qml.ctrl(op, control=work_wire) for op in _add_k_fourier(mod, x_wires))
            op_list.extend(qml.adjoint(_add_k_fourier)(k, x_wires))
            op_list.append(qml.adjoint(qml.QFT)(wires=x_wires))
            op_list.append(qml.ctrl(qml.PauliX(work_wire), control=aux_k, control_values=0))
            op_list.append(qml.QFT(wires=x_wires))
            op_list.extend(_add_k_fourier(k, x_wires))

        return op_list
