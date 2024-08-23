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

        \text{OutAdder}(mod) |x \rangle | k \rangle | 0 \rangle = |x \rangle | k \rangle | x+k \, \text{mod} \, mod \rangle ,

    The implementation is based on the quantum Fourier transform method presented in
    `arXiv:2311.08555 <https://arxiv.org/abs/2311.08555>`_.

    .. note::

        Note that :math:`x` and :math:`y` must be smaller than :math:`mod` to get the correct result.

    Args:
        x_wires (Sequence[int]): the wires that store the integer :math:`x`.
        y_wires (Sequence[int]): the wires that store the integer :math:`y`.
        output_wires (Sequence[int]): the wires that store the addition modulo mod :math:`x + y \text{mod}`.
        mod (int): the modulus for performing the addition, default value is :math:`2^{\text{len(output\_wires)}`
        work_wires (Sequence[int]): the auxiliary wires to use for the addition modulo :math:`mod` when
        :math:`mod \neq 2^{\text{len(output\_wires)}}`.

    **Example**

    Sum of two integers :math:`x=8` and :math:`y=5` modulo :math:`mod=15`.

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
        def circuit():
            qml.BasisEmbedding(x, wires=x_wires)
            qml.BasisEmbedding(y, wires=y_wires)
            qml.OutAdder(x_wires, y_wires, output_wires, mod, work_wires)
            return qml.sample(wires=output_wires)

    .. code-block:: pycon

        >>> print(f"The ket representation of {x} + {y} mod {mod} is {circuit()}")
        The ket representation of 5 + 6 mod 7 is [1 0 0]

    We can see that the result [1 0 0] corresponds to 4, which comes from :math:`5+6=11 \longrightarrow 11 \, \text{mod} \, 7 = 4`.
    """

    grad_method = None

    def __init__(
        self, x_wires, y_wires, output_wires, mod=None, work_wires=None, id=None
    ):  # pylint: disable=too-many-arguments

        if mod is None:
            mod = 2 ** (len(output_wires))
        if (not hasattr(output_wires, "__len__")) or (mod > 2 ** len(output_wires)):
            raise ValueError("OutAdder must have enough wires to represent mod.")
        if work_wires is not None:
            if any(wire in work_wires for wire in x_wires):
                raise ValueError("None of the wires in work_wires should be included in x_wires.")
            if any(wire in work_wires for wire in y_wires):
                raise ValueError("None of the wires in work_wires should be included in y_wires.")
        if any(wire in y_wires for wire in x_wires):
            raise ValueError("None of the wires in y_wires should be included in x_wires.")
        if any(wire in x_wires for wire in output_wires):
            raise ValueError("None of the wires in x_wires should be included in output_wires.")
        if any(wire in y_wires for wire in output_wires):
            raise ValueError("None of the wires in y_wires should be included in output_wires.")
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
            for key in ["x_wires", "y_wires", "output_wires", "work_wires"]
        }

        return OutAdder(
            new_dict["x_wires"],
            new_dict["y_wires"],
            new_dict["output_wires"],
            self.hyperparameters["mod"],
            new_dict["work_wires"],
        )

    @property
    def wires(self):
        """All wires involved in the operation."""
        return self.hyperparameters["x_wires"] + self.hyperparameters["work_wires"]

    def decomposition(self):  # pylint: disable=arguments-differ

        return self.compute_decomposition(**self.hyperparameters)

    @classmethod
    def _primitive_bind_call(cls, *args, **kwargs):
        return cls._primitive.bind(*args, **kwargs)

    @staticmethod
    def compute_decomposition(
        x_wires, y_wires, output_wires, mod, work_wires
    ):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a product of other operators.
        Args:
            x_wires (Sequence[int]): the wires that store the integer :math:`x`.
            y_wires (Sequence[int]): the wires that store the integer :math:`y`.
            output_wires (Sequence[int]): the wires that store the addition modulo mod :math:`x + y \text{mod}`.
            mod (int): the modulus for performing the addition, default value is :math:`2^{\text{len(output\_wires)}`
            work_wires (Sequence[int]): the auxiliary wires to use for the addition modulo :math:`mod` when
            :math:`mod \neq 2^{\text{len(output\_wires)}}`.
        Returns:
            list[.Operator]: Decomposition of the operator

        **Example**

        >>> qml.OutAdder.compute_decomposition(x_wires = [0, 1], y_wires = [2, 3], output_wires = [5, 6], mod = 4, work_wires = [4, 7])
        [QFT(wires=[5, 6]),
        ControlledSequence(PhaseAdder(wires=[5, 6, None]), control=[0, 1])
        ControlledSequence(PhaseAdder(wires=[5, 6, None]), control=[2, 3]),
        Adjoint(QFT(wires=[5, 6]))]
        """
        op_list = []
        if mod != 2 ** len(output_wires) and mod is not None:
            qft_new_output_wires = work_wires[:1] + output_wires
            work_wire = work_wires[1:]
        else:
            qft_new_output_wires = output_wires
            work_wire = None
        op_list.append(qml.QFT(wires=qft_new_output_wires))
        op_list.append(
            qml.ControlledSequence(
                qml.PhaseAdder(1, qft_new_output_wires, mod, work_wire), control=x_wires
            )
        )
        op_list.append(
            qml.ControlledSequence(
                qml.PhaseAdder(1, qft_new_output_wires, mod, work_wire), control=y_wires
            )
        )
        op_list.append(qml.adjoint(qml.QFT)(wires=qft_new_output_wires))

        return op_list
