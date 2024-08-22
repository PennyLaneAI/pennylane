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
Contains the ModExp template.
"""
import pennylane as qml
from pennylane.operation import Operation


class ModExp(Operation):
    r"""Performs the Exponentiation operation outplace in the computational basis.

    This operator exponentiates the integer :math:`base` to the power :math:`x` modulo :math:`mod` in the computational basis:

    .. math::

        \text{ModExp}(base,mod) |x \rangle |k \rangle = |x \rangle |k*base^x \, \text{mod} \, mod \rangle,

    The decomposition of this operator is based on the QFT-based method presented in `Atchade-Adelomou and Gonzalez (2023) <https://arxiv.org/abs/2311.08555>`_.

    Args:
        x_wires (Sequence[int]): the wires that stores the integer :math:`x`.
        output_wires (Sequence[int]): the wires that stores the exponentiation modulo mod :math:`base^x \, \text{mod} \, mod`.
        base (int): integer we want to exponentiate.
        mod (int): modulo with respect to which the exponentiation is performed, default value will be ``2^len(output_wires)``.
        work_wires (Sequence[int]): the auxiliary wires to use for the exponentiation modulo :math:`mod` when :math:`mod \neq 2^{\text{len(output_wires)}}`

    **Example**

    Exponentiation of :math:`base=2` to the power :math:`x=3` modulo :math:`mod=7`. Note that to perform this multiplication using qml.ModExp we need that :math:`x < mod`
    and that :math:`base` has inverse, :math:`base^-1` modulo :math:`mod`. That means :math:`base*base^-1 modulo mod = 1`, which will only be possible if :math:`base` and :math:`mod` are coprime.

    .. code-block::

        x, k = 3, 1
        base = 2
        mod = 7
        x_wires = [0, 1]
        output_wires = [2, 3, 4]
        work_wires = [5, 6, 7, 8, 9]
        dev = qml.device("default.qubit", shots=1)
        @qml.qnode(dev)
        def circuit_ModExp():
            qml.BasisEmbedding(x, wires = x_wires)
            qml.BasisEmbedding(k, wires = output_wires)
            qml.ModExp(input_wires, output_wires, base, mod, work_wires)
            return qml.sample(wires = output_wires)

    .. code-block:: pycon

        >>> print(f"The ket representation of {base} ^ {x} mod {mod} is {circuit_ModExp()}")
        The ket representation of 2 ^ 3 mod 7 is [0 0 1]

    We can see that the result [0 0 1] corresponds to 1, which comes from :math:`2^3=8 \longrightarrow 8 \, \text{mod} \, 7 = 1`.
    """

    grad_method = None

    def __init__(
        self, x_wires, output_wires, base, mod=None, work_wires=None, id=None
    ):  # pylint: disable=too-many-arguments

        if mod is None:
            mod = 2 ** (len(output_wires))
        if (not hasattr(output_wires, "__len__")) or (mod > 2 ** (len(output_wires))):
            raise ValueError("ModExp must have at least enough wires to represent mod.")
        if work_wires is not None:
            if any(wire in work_wires for wire in x_wires):
                raise ValueError("None of the wires in work_wires should be included in x_wires.")
            if any(wire in work_wires for wire in output_wires):
                raise ValueError(
                    "None of the wires in work_wires should be included in output_wires."
                )
        if any(wire in x_wires for wire in output_wires):
            raise ValueError("None of the wires in x_wires should be included in output_wires.")
        wire_keys = ["x_wires", "output_wires", "work_wires"]
        for key in wire_keys:
            self.hyperparameters[key] = qml.wires.Wires(locals()[key])
        all_wires = sum(self.hyperparameters[key] for key in wire_keys)
        base = base % mod
        self.hyperparameters["base"] = base
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
            for key in ["x_wires", "output_wires", "work_wires"]
        }

        return ModExp(
            new_dict["x_wires"],
            new_dict["output_wires"],
            self.hyperparameters["base"],
            self.hyperparameters["mod"],
            new_dict["work_wires"],
        )

    @property
    def wires(self):
        """All wires involved in the operation."""
        return (
            self.hyperparameters["x_wires"]
            + self.hyperparameters["output_wires"]
            + self.hyperparameters["work_wires"]
        )

    def decomposition(self):  # pylint: disable=arguments-differ

        return self.compute_decomposition(
            self.hyperparameters["x_wires"],
            self.hyperparameters["output_wires"],
            self.hyperparameters["base"],
            self.hyperparameters["mod"],
            self.hyperparameters["work_wires"],
        )

    @classmethod
    def _primitive_bind_call(cls, *args, **kwargs):
        return cls._primitive.bind(*args, **kwargs)

    @staticmethod
    def compute_decomposition(
        x_wires, output_wires, base, mod, work_wires
    ):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a product of other operators.
        Args:
            x_wires (Sequence[int]): the wires that stores the integer :math:`x`.
            output_wires (Sequence[int]): the wires that stores the exponentiation modulo mod :math:`base^x \, \text{mod} \, mod`.
            base (int): integer we want to exponentiate.
            mod (int): modulo with respect to which the exponentiation is performed, default value will be ``2^len(output_wires)``
            work_wires (Sequence[int]): the auxiliary wires to use for the exponentiation modulo :math:`mod` when :math:`mod \neq 2^{\textrm{len(output_wires)}}`
        Returns:
            list[.Operator]: Decomposition of the operator

        **Example**

        >>> qml.ModExp.compute_decomposition(x_wires = [0, 1], output_wires = [2, 3, 4], base = 3, mod = 8, work_wires = [5, 6, 7, 8, 9])
        [ControlledSequence(Multiplier(wires=[2, 3, 4]), control=[0, 1])]
        """
        op_list = []
        op_list.append(
            qml.ControlledSequence(
                qml.Multiplier(base, output_wires, mod, work_wires), control=x_wires
            )
        )
        return op_list
