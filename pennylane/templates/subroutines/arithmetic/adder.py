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
from pennylane.decomposition import (
    add_decomps,
    change_op_basis_resource_rep,
    register_resources,
)
from pennylane.decomposition.resources import resource_rep
from pennylane.operation import Operation
from pennylane.ops.op_math import change_op_basis
from pennylane.templates.subroutines.qft import QFT
from pennylane.wires import Wires, WiresLike

from .phase_adder import PhaseAdder


class Adder(Operation):
    r"""Performs the in-place modular addition operation.

    This operator performs the modular addition by an integer :math:`k` modulo :math:`mod` in the
    computational basis:

    .. math::

        \text{Adder}(k, mod) |x \rangle = | x+k \; \text{mod} \; mod \rangle.

    The implementation is based on the quantum Fourier transform method presented in
    `arXiv:2311.08555 <https://arxiv.org/abs/2311.08555>`_.

    .. note::

        To obtain the correct result, :math:`x` must be smaller than :math:`mod`.

    .. seealso:: :class:`~.PhaseAdder` and :class:`~.OutAdder`.

    Args:
        k (int): the number that needs to be added
        x_wires (Sequence[int]): the wires the operation acts on. The number of wires must be enough
            for encoding `x` in the computational basis. The number of wires also limits the
            maximum value for `mod`.
        mod (int): the modulo for performing the addition. If not provided, it will be set to its maximum value, :math:`2^{\text{len(x_wires)}}`.
        work_wires (Sequence[int]): the auxiliary wires to use for the addition. The
            work wires are not needed if :math:`mod=2^{\text{len(x_wires)}}`, otherwise two work wires
            should be provided. Defaults to empty tuple.

    **Example**

    This example computes the sum of two integers :math:`x=8` and :math:`k=5` modulo :math:`mod=15`.

    .. code-block:: python

        x = 8
        k = 5
        mod = 15

        x_wires =[0,1,2,3]
        work_wires=[4,5]

        dev = qml.device("default.qubit")
        @qml.qnode(dev, shots=1)
        def circuit():
            qml.BasisEmbedding(x, wires=x_wires)
            qml.Adder(k, x_wires, mod, work_wires)
            return qml.sample(wires=x_wires)

    >>> print(circuit())
    [[1 1 0 1]]

    The result, :math:`[[1 1 0 1]]`, is the binary representation of
    :math:`8 + 5  \; \text{modulo} \; 15 = 13`.

    .. details::
        :title: Usage Details

        This template takes as input two different sets of wires.

        The first one is ``x_wires``, used to encode the integer :math:`x < \text{mod}` in the Fourier basis.
        To represent :math:`x`, ``x_wires`` must include at least :math:`\lceil \log_2(x) \rceil` wires.
        After the modular addition, the result can be as large as :math:`\text{mod} - 1`,
        requiring at least :math:`\lceil \log_2(\text{mod}) \rceil` wires. Since :math:`x < \text{mod}`,
        :math:`\lceil \log_2(\text{mod}) \rceil` is a sufficient length for ``x_wires`` to cover all possible inputs and outputs.

        The second set of wires is ``work_wires`` which consist of the auxiliary qubits used to perform the modular addition operation.

        - If :math:`mod = 2^{\text{len(x_wires)}}`, there will be no need for ``work_wires``, hence ``work_wires=()``. This is the case by default.

        - If :math:`mod \neq 2^{\text{len(x_wires)}}`, two ``work_wires`` have to be provided.

        Note that the ``Adder`` template allows us to perform modular addition in the computational basis. However if one just wants to perform standard addition (with no modulo), that would be equivalent to setting
        the modulo :math:`mod` to a large enough value to ensure that :math:`x+k < mod`.
    """

    grad_method = None

    resource_keys = {"num_x_wires", "mod"}

    def __init__(
        self, k, x_wires: WiresLike, mod=None, work_wires: WiresLike = (), id=None
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments

        x_wires = Wires(x_wires)
        work_wires = Wires(() if work_wires is None else work_wires)

        num_works_wires = len(work_wires)

        if mod is None:
            mod = 2 ** len(x_wires)
        elif mod != 2 ** len(x_wires) and num_works_wires != 2:
            raise ValueError(f"If mod is not 2^{len(x_wires)}, two work wires should be provided")
        if not isinstance(k, int) or not isinstance(mod, int):
            raise ValueError("Both k and mod must be integers")
        if num_works_wires != 0:
            if any(wire in work_wires for wire in x_wires):
                raise ValueError("None of the wires in work_wires should be included in x_wires.")
        if mod > 2 ** len(x_wires):
            raise ValueError(
                "Adder must have enough x_wires to represent mod. The maximum mod "
                f"with len(x_wires)={len(x_wires)} is {2 ** len(x_wires)}, but received {mod}."
            )

        all_wires = x_wires + work_wires

        self.hyperparameters["k"] = k
        self.hyperparameters["mod"] = mod
        self.hyperparameters["work_wires"] = work_wires
        self.hyperparameters["x_wires"] = x_wires

        super().__init__(wires=all_wires, id=id)

    @property
    def resource_params(self) -> dict:
        return {
            "num_x_wires": len(self.hyperparameters["x_wires"]),
            "mod": self.hyperparameters["mod"],
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
            for key in ["x_wires", "work_wires"]
        }

        return Adder(
            self.hyperparameters["k"],
            new_dict["x_wires"],
            self.hyperparameters["mod"],
            new_dict["work_wires"],
        )

    def decomposition(self):
        return self.compute_decomposition(**self.hyperparameters)

    @classmethod
    def _primitive_bind_call(cls, *args, **kwargs):
        return cls._primitive.bind(*args, **kwargs)

    @staticmethod
    def compute_decomposition(
        k, x_wires: WiresLike, mod, work_wires: WiresLike
    ):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a product of other operators.

        Args:
            k (int): the number that needs to be added
            x_wires (Sequence[int]): the wires the operation acts on. The number of wires must be enough
                for encoding `x` in the computational basis. The number of wires also limits the
                maximum value for `mod`.
            mod (int): the modulo for performing the addition. If not provided, it will be set to its maximum value, :math:`2^{\text{len(x_wires)}}`.
            work_wires (Sequence[int]): the auxiliary wires to use for the addition. The
                work wires are not needed if :math:`mod=2^{\text{len(x_wires)}}`, otherwise two work wires
                should be provided.
        Returns:
            list[.Operator]: Decomposition of the operator

        **Example**

        >>> qml.Adder.compute_decomposition(k=2, x_wires=[0,1,2], mod=8, work_wires=[3])
        [(Adjoint(QFT(wires=[0, 1, 2]))) @ PhaseAdder(wires=[0, 1, 2]) @ QFT(wires=[0, 1, 2])]
        """
        if mod == 2 ** len(x_wires):
            qft_wires = x_wires
            work_wire = ()
        else:
            qft_wires = work_wires[:1] + x_wires
            work_wire = work_wires[1:]

        op_list = [change_op_basis(QFT(qft_wires), PhaseAdder(k, qft_wires, mod, work_wire))]

        return op_list


def _adder_decomposition_resources(num_x_wires, mod) -> dict:
    qft_wires = num_x_wires if mod == 2**num_x_wires else 1 + num_x_wires
    return {
        change_op_basis_resource_rep(
            resource_rep(QFT, num_wires=qft_wires),
            resource_rep(PhaseAdder, num_x_wires=qft_wires, mod=mod),
        ): 1,
    }


@register_resources(_adder_decomposition_resources)
def _adder_decomposition(k, x_wires: WiresLike, mod, work_wires: WiresLike, **__):
    if mod == 2 ** len(x_wires):
        qft_wires = x_wires
        work_wire = ()
    else:
        qft_wires = work_wires[:1] + x_wires
        work_wire = work_wires[1:]

    change_op_basis(QFT(qft_wires), PhaseAdder(k, qft_wires, mod, work_wire))


add_decomps(Adder, _adder_decomposition)
