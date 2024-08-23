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
    r"""Performs the out-place modular exponentiation operation.

    This operator performs the modular exponentiation of the integer :math:`base` to the power
    :math:`x` modulo :math:`mod` in the computational basis:

    .. math::

        \text{ModExp}(base,mod) |x \rangle |k \rangle = |x \rangle |k*base^x \, \text{mod} \, mod \rangle,

    The implementation is based on the quantum Fourier transform method presented in
    `arXiv:2311.08555 <https://arxiv.org/abs/2311.08555>`_.

    .. note::

        Note that :math:`x` must be smaller than :math:`mod` to get the correct result.
        Also, it is required that :math:`base` has inverse, :math:`base^-1` modulo :math:`mod`.
        That means :math:`base*base^-1 modulo mod = 1`, which will only be possible if :math:`base`
        and :math:`mod` are coprime.

    Args:
        x_wires (Sequence[int]): the wires that store the integer :math:`x`
        output_wires (Sequence[int]): the wires that store the exponentiation result
        base (int): integer that needs to be exponentiated
        mod (int): the modulus for performing the exponentiation, default value is :math:`2^{len(output\_wires)}`
        work_wires (Sequence[int]): the auxiliary wires to be used for the exponentiation. There
            must be as many as ``output_wires`` and if :math:`mod \neq 2^{len(x\_wires)}`, two more
            wires must be added.

    **Example**

    This example performs the exponentiation of :math:`base=2` to the power :math:`x=3` modulo :math:`mod=7`.

    .. code-block::

        x, k = 3, 1
        base = 2
        mod = 7
        x_wires = [0, 1]
        output_wires = [2, 3, 4]
        work_wires = [5, 6, 7, 8, 9]
        dev = qml.device("default.qubit", shots=1)
        @qml.qnode(dev)
        def circuit():
            qml.BasisEmbedding(x, wires = x_wires)
            qml.BasisEmbedding(k, wires = output_wires)
            qml.ModExp(x_wires, output_wires, base, mod, work_wires)
            return qml.sample(wires = output_wires)

    .. code-block:: pycon

        >>> print(circuit())
        [0 0 1]

    The result :math:`[0 0 1]`, is the ket representation of
    :math:`2^3 \, \text{modulo} \, 7 = 1`.
    """

    grad_method = None

    def __init__(
        self, x_wires, output_wires, base, mod=None, work_wires=None, id=None
    ):  # pylint: disable=too-many-arguments

        output_wires = qml.wires.Wires(output_wires)

        if mod is None:
            mod = 2 ** (len(output_wires))
        if len(output_wires) == 0 or (mod > 2 ** (len(output_wires))):
            raise ValueError("ModExp must have enough wires to represent mod.")
        if mod != 2 ** len(x_wires):
            if len(work_wires) < (len(output_wires) + 2):
                raise ValueError("ModExp needs as many work_wires as output_wires plus two.")
        else:
            if len(work_wires) < len(output_wires):
                raise ValueError("ModExp needs as many work_wires as output_wires.")
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

        return self.compute_decomposition(**self.hyperparameters)

    @classmethod
    def _primitive_bind_call(cls, *args, **kwargs):
        return cls._primitive.bind(*args, **kwargs)

    @staticmethod
    def compute_decomposition(
        x_wires, output_wires, base, mod, work_wires
    ):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a product of other operators.
        Args:
            x_wires (Sequence[int]): the wires that store the integer :math:`x`
            output_wires (Sequence[int]): the wires that store the exponentiation result
            base (int): integer that needs to be exponentiated
            mod (int): the modulus for performing the exponentiation, default value is :math:`2^{len(output\_wires)}`
            work_wires (Sequence[int]): the auxiliary wires to be used for the exponentiation. There
                must be as many as ``output_wires`` and if :math:`mod \neq 2^{len(x\_wires)}`, two more
                wires must be added.

        Returns:
            list[.Operator]: Decomposition of the operator

        **Example**

        >>> qml.ModExp.compute_decomposition(x_wires=[0,1], output_wires=[2,3,4], base=3, mod=8, work_wires=[5,6,7,8,9])
        [ControlledSequence(Multiplier(wires=[2, 3, 4, 5, 6, 7, 8, 9]), control=[0, 1])]
        """

        # TODO: Cancel the QFTs of consecutive Multipliers
        op_list = []
        op_list.append(
            qml.ControlledSequence(
                qml.Multiplier(base, output_wires, mod, work_wires), control=x_wires
            )
        )
        return op_list
