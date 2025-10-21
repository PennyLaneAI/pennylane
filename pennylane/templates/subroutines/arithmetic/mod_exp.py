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
import numpy as np

from pennylane.decomposition import add_decomps, register_resources, resource_rep
from pennylane.operation import Operation
from pennylane.templates.subroutines.controlled_sequence import ControlledSequence
from pennylane.wires import Wires, WiresLike

from .multiplier import Multiplier


class ModExp(Operation):
    r"""Performs the out-place modular exponentiation operation.

    This operator performs the modular exponentiation of the integer :math:`base` to the power
    :math:`x` modulo :math:`mod` in the computational basis:

    .. math::

        \text{ModExp}(base,mod) |x \rangle |b \rangle = |x \rangle |b \cdot base^x \; \text{mod} \; mod \rangle,

    The implementation is based on the quantum Fourier transform method presented in
    `arXiv:2311.08555 <https://arxiv.org/abs/2311.08555>`_.

    .. note::

        To obtain the correct result, :math:`x` must be smaller than :math:`mod`.
        Also, it is required that :math:`base` has a modular inverse, :math:`base^{-1}`, with respect to :math:`mod`.
        That means :math:`base \cdot base^{-1}` modulo :math:`mod` is equal to 1, which will only be possible if :math:`base`
        and :math:`mod` are coprime.

    .. seealso:: :class:`~.Multiplier`.

    Args:
        x_wires (Sequence[int]): the wires that store the integer :math:`x`
        output_wires (Sequence[int]): the wires that store the operator result. These wires also encode :math:`b`.
        base (int): integer that needs to be exponentiated
        mod (int): the modulo for performing the exponentiation. If not provided, it will be set to its maximum value, :math:`2^{\text{len(output_wires)}}`
        work_wires (Sequence[int]): the auxiliary wires to use for the exponentiation. If
            :math:`mod=2^{\text{len(output_wires)}}`, the number of auxiliary wires must be ``len(output_wires)``. Otherwise
            ``len(output_wires) + 2`` auxiliary wires are needed. Defaults to empty tuple.

    **Example**

    This example performs the exponentiation of :math:`base=2` to the power :math:`x=3` modulo :math:`mod=7`.

    .. code-block:: python

        x, b = 3, 1
        base = 2
        mod = 7

        x_wires = [0, 1]
        output_wires = [2, 3, 4]
        work_wires = [5, 6, 7, 8, 9]

        dev = qml.device("default.qubit")

        @qml.qnode(dev, shots=1)
        def circuit():
            qml.BasisEmbedding(x, wires = x_wires)
            qml.BasisEmbedding(b, wires = output_wires)
            qml.ModExp(x_wires, output_wires, base, mod, work_wires)
            return qml.sample(wires = output_wires)

    >>> print(circuit())
    [[0 0 1]]

    The result :math:`[0 0 1]`, is the binary representation of
    :math:`2^3 \; \text{modulo} \; 7 = 1`.

    .. details::
        :title: Usage Details

        This template takes as input three different sets of wires.

        The first one is ``x_wires`` which is used
        to encode the integer :math:`x < mod` in the computational basis. Therefore, ``x_wires`` must contain at least
        :math:`\lceil \log_2(x)\rceil` wires to represent :math:`x`.

        The second one is ``output_wires`` which is used
        to encode the integer :math:`b \cdot base^x \; \text{mod} \; mod` in the computational basis. Therefore, at least
        :math:`\lceil \log_2(mod)\rceil` ``output_wires`` are required to represent :math:`b \cdot base^x \; \text{mod} \; mod`. Note that these wires can be initialized with any integer
        :math:`b`, but the most common choice is :math:`b=1` to obtain as a final result :math:`base^x \; \text{mod} \; mod`.

        The third set of wires is ``work_wires`` which consist of the auxiliary qubits used to perform the modular exponentiation operation.

        - If :math:`mod = 2^{\text{len(output_wires)}}`,  the length of ``work_wires`` must be equal to the length of ``output_wires``.

        - If :math:`mod \neq 2^{\text{len(output_wires)}}`, the length of ``work_wires`` must be ``len(output_wires) + 2``

        Note that the ``ModExp`` template allows us to perform modular exponentiation in the computational basis. However if one just wants to perform standard exponentiation (with no modulo),
        that would be equivalent to setting the modulo :math:`mod` to a large enough value to ensure that :math:`base^x < mod`.

        Also, to perform the out-place modular exponentiation operator it is required that :math:`base` has inverse, :math:`base^{-1} \; \text{mod} \; mod`. That means
        :math:`base \cdot base^{-1}` modulo :math:`mod` is equal to 1, which will only be possible if :math:`base` and
        :math:`mod` are coprime. In other words, :math:`base` and :math:`mod` should not have any common factors other than 1.
    """

    grad_method = None

    resource_keys = {"num_x_wires", "num_output_wires", "mod", "num_work_wires"}

    def __init__(
        self, x_wires: WiresLike, output_wires, base, mod=None, work_wires: WiresLike = (), id=None
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments

        output_wires = Wires(output_wires)
        work_wires = Wires(() if work_wires is None else work_wires)

        if len(work_wires) == 0:
            raise ValueError("Work wires must be specified for ModExp")

        if mod is None:
            mod = 2 ** (len(output_wires))
        if len(output_wires) == 0 or (mod > 2 ** (len(output_wires))):
            raise ValueError("ModExp must have enough wires to represent mod.")
        if mod != 2 ** len(output_wires):
            if len(work_wires) < (len(output_wires) + 2):
                raise ValueError("ModExp needs as many work_wires as output_wires plus two.")
        else:
            if len(work_wires) < len(output_wires):
                raise ValueError("ModExp needs as many work_wires as output_wires.")
        if len(work_wires) != 0:
            if any(wire in work_wires for wire in x_wires):
                raise ValueError("None of the wires in work_wires should be included in x_wires.")
            if any(wire in work_wires for wire in output_wires):
                raise ValueError(
                    "None of the wires in work_wires should be included in output_wires."
                )
        if any(wire in x_wires for wire in output_wires):
            raise ValueError("None of the wires in x_wires should be included in output_wires.")

        if np.gcd(base, mod) != 1:
            raise ValueError("The operator cannot be built because base has no inverse modulo mod.")

        wire_keys = ["x_wires", "output_wires", "work_wires"]
        for key in wire_keys:
            self.hyperparameters[key] = Wires(locals()[key])
        all_wires = sum(self.hyperparameters[key] for key in wire_keys)
        base = base % mod
        self.hyperparameters["base"] = base
        self.hyperparameters["mod"] = mod
        super().__init__(wires=all_wires, id=id)

    @property
    def resource_params(self) -> dict:
        return {
            "num_x_wires": len(self.hyperparameters["x_wires"]),
            "num_output_wires": len(self.hyperparameters["output_wires"]),
            "mod": self.hyperparameters["mod"],
            "num_work_wires": len(self.hyperparameters["work_wires"]),
        }

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

    def decomposition(self):

        return self.compute_decomposition(**self.hyperparameters)

    @classmethod
    def _primitive_bind_call(cls, *args, **kwargs):
        return cls._primitive.bind(*args, **kwargs)

    @staticmethod
    def compute_decomposition(
        x_wires, output_wires: WiresLike, base, mod, work_wires: WiresLike
    ):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a product of other operators.

        Args:
            x_wires (Sequence[int]): the wires that store the integer :math:`x`
            output_wires (Sequence[int]): the wires that store the operator result. These wires also encode :math:`b`.
            base (int): integer that needs to be exponentiated
            mod (int): the modulo for performing the exponentiation. If not provided, it will be set to its maximum value, :math:`2^{\text{len(output_wires)}}`
            work_wires (Sequence[int]): the auxiliary wires to use for the exponentiation. If
                :math:`mod=2^{\text{len(output_wires)}}`, the number of auxiliary wires must be ``len(output_wires)``. Otherwise
                ``len(output_wires) + 2`` auxiliary wires are needed.
        Returns:
            list[.Operator]: Decomposition of the operator

        **Example**

        >>> qml.ModExp.compute_decomposition(x_wires=[0,1], output_wires=[2,3,4], base=3, mod=8, work_wires=[5,6,7,8,9])
        [ControlledSequence(Multiplier(wires=[2, 3, 4, 5, 6, 7, 8, 9]), control=[0, 1])]
        """

        # TODO: Cancel the QFTs of consecutive Multipliers
        op_list = []
        op_list.append(
            ControlledSequence(Multiplier(base, output_wires, mod, work_wires), control=x_wires)
        )
        return op_list


def _mod_exp_decomposition_resources(num_x_wires, num_output_wires, mod, num_work_wires) -> dict:
    return {
        resource_rep(
            ControlledSequence,
            base_class=Multiplier,
            base_params={
                "num_x_wires": num_output_wires,
                "num_work_wires": num_work_wires,
                "mod": mod,
            },
            num_control_wires=num_x_wires,
        ): 1,
    }


@register_resources(_mod_exp_decomposition_resources)
def _mod_exp_decomposition(
    x_wires, output_wires: WiresLike, base, mod, work_wires: WiresLike, **__
):
    ControlledSequence(Multiplier(base, output_wires, mod, work_wires), control=x_wires)


add_decomps(ModExp, _mod_exp_decomposition)
