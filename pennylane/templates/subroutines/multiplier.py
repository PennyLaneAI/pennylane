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
from pennylane.wires import WiresLike


def _mul_out_k_mod(k, x_wires: WiresLike, mod, work_wire_aux: WiresLike, wires_aux: WiresLike):
    """Performs :math:`x \times k` in the registers wires wires_aux"""
    op_list = []

    op_list.append(qml.QFT(wires=wires_aux))
    op_list.append(
        qml.ControlledSequence(qml.PhaseAdder(k, wires_aux, mod, work_wire_aux), control=x_wires)
    )
    op_list.append(qml.adjoint(qml.QFT(wires=wires_aux)))
    return op_list


class Multiplier(Operation):
    r"""Performs the in-place modular multiplication operation.

    This operator performs the modular multiplication by an integer :math:`k` modulo :math:`mod` in
    the computational basis:

    .. math::

        \text{Multiplier}(k,mod) |x \rangle = | x \cdot k \; \text{mod} \; mod \rangle.

    The implementation is based on the quantum Fourier transform method presented in
    `arXiv:2311.08555 <https://arxiv.org/abs/2311.08555>`_.

    .. note::

        To obtain the correct result, :math:`x` must be smaller than :math:`mod`. Also, it
        is required that :math:`k` has modular inverse :math:`k^{-1}` with respect to :math:`mod`. That means
        :math:`k \cdot k^{-1}` modulo :math:`mod` is equal to 1, which will only be possible if :math:`k` and
        :math:`mod` are coprime.

    .. seealso:: :class:`~.PhaseAdder` and :class:`~.OutMultiplier`.

    Args:
        k (int): the number that needs to be multiplied
        x_wires (Sequence[int]): the wires the operation acts on. The number of wires must be enough for encoding `x` in the computational basis. The number of wires also limits the maximum value for `mod`.
        mod (int): the modulo for performing the multiplication. If not provided, it will be set to its maximum value, :math:`2^{\text{len(x_wires)}}`.
        work_wires (Sequence[int]): the auxiliary wires to use for the multiplication. If :math:`mod=2^{\text{len(x_wires)}}`, the number of auxiliary wires must be ``len(x_wires)``. Otherwise ``len(x_wires) + 2`` auxiliary wires are needed.

    **Example**

    This example performs the multiplication of two integers :math:`x=3` and :math:`k=4` modulo :math:`mod=7`.

    .. code-block::

        x = 3
        k = 4
        mod = 7

        x_wires = [0,1,2]
        work_wires = [3,4,5,6,7]

        dev = qml.device("default.qubit", shots=1)
        @qml.qnode(dev)
        def circuit():
            qml.BasisEmbedding(x, wires=x_wires)
            qml.Multiplier(k, x_wires, mod, work_wires)
            return qml.sample(wires=x_wires)

    .. code-block:: pycon

        >>> print(circuit())
        [1 0 1]

    The result :math:`[1 0 1]`, is the binary representation of
    :math:`3 \cdot 4 \; \text{modulo} \; 7 = 5`.

    .. details::
        :title: Usage Details

        This template takes as input two different sets of wires.

        The first one is ``x_wires``, used to encode the integer :math:`x < \text{mod}` in the Fourier basis.
        To represent :math:`x`, ``x_wires`` must include at least :math:`\lceil \log_2(x) \rceil` wires.
        After the modular addition, the result can be as large as :math:`\text{mod} - 1`,
        requiring at least :math:`\lceil \log_2(\text{mod}) \rceil` wires. Since :math:`x < \text{mod}`,
        :math:`\lceil \log_2(\text{mod}) \rceil` is a sufficient length for ``x_wires`` to cover all possible inputs and outputs.

        The second set of wires is ``work_wires`` which consist of the auxiliary qubits used to perform the modular multiplication operation.

        - If :math:`mod = 2^{\text{len(x_wires)}}`, the length of ``work_wires`` must be equal to the length of ``x_wires``.

        - If :math:`mod \neq 2^{\text{len(x_wires)}}`, the length of ``work_wires`` must be ``len(x_wires) + 2``.

        Note that the ``Multiplier`` template allows us to perform modular multiplication in the computational basis. However if one just want to perform standard multiplication (with no modulo),
        that would be equivalent to setting the modulo :math:`mod` to a large enough value to ensure that :math:`x \cdot k < mod`.

        Also, to perform the in-place multiplication operator it is required that :math:`k` has inverse, :math:`k^{-1} \; \text{mod} \; mod`. That means
        :math:`k \cdot k^{-1}` modulo :math:`mod` is equal to 1, which will only be possible if :math:`k` and
        :math:`mod` are coprime. In other words, :math:`k` and :math:`mod` should not have any common factors other than 1.
    """

    grad_method = None

    def __init__(
        self, k, x_wires: WiresLike, mod=None, work_wires: WiresLike = (), id=None
    ):  # pylint: disable=too-many-arguments

        x_wires = qml.wires.Wires(x_wires)
        work_wires = qml.wires.Wires(() if work_wires is None else work_wires)
        if len(work_wires) == 0:
            raise ValueError("Work wires must be specified for Multiplier")

        if any(wire in work_wires for wire in x_wires):
            raise ValueError("None of the wire in work_wires should be included in x_wires.")

        if mod is None:
            mod = 2 ** len(x_wires)
        if mod != 2 ** len(x_wires) and len(work_wires) != (len(x_wires) + 2):
            raise ValueError("Multiplier needs as many work_wires as x_wires plus two.")
        if len(work_wires) < len(x_wires):
            raise ValueError("Multiplier needs as many work_wires as x_wires.")
        if mod > 2 ** len(x_wires):
            raise ValueError(
                "Multiplier must have enough wires to represent mod. The maximum mod "
                f"with len(x_wires)={len(x_wires)} is {2 ** len(x_wires)}, but received {mod}."
            )

        k = k % mod
        if np.gcd(k, mod) != 1:
            raise ValueError("The operator cannot be built because k has no inverse modulo mod.")

        self.hyperparameters["k"] = k
        self.hyperparameters["mod"] = mod
        self.hyperparameters["work_wires"] = work_wires
        self.hyperparameters["x_wires"] = x_wires
        all_wires = x_wires + work_wires
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

        return Multiplier(
            self.hyperparameters["k"],
            new_dict["x_wires"],
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
        k, x_wires: WiresLike, mod, work_wires: WiresLike
    ):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a product of other operators.

        Args:
            k (int): the number that needs to be multiplied
            x_wires (Sequence[int]): the wires the operation acts on. The number of wires must be enough for encoding `x` in the computational basis. The number of wires also limits the maximum value for `mod`.
            mod (int): the modulo for performing the multiplication. If not provided, it will be set to its maximum value, :math:`2^{\text{len(x_wires)}}`.
            work_wires (Sequence[int]): the auxiliary wires to use for the multiplication. If :math:`mod=2^{\text{len(x_wires)}}`, the number of auxiliary wires must be ``len(x_wires)``. Otherwise ``len(x_wires) + 2`` auxiliary wires are needed.
        Returns:
            list[.Operator]: Decomposition of the operator

        **Example**

        >>> qml.Multiplier.compute_decomposition(k=3, mod=8, x_wires=[0,1,2], work_wires=[3,4,5])
        [QFT(wires=[3, 4, 5]),
        ControlledSequence(PhaseAdder(wires=[3, 4 , 5 , None]), control=[0, 1, 2]),
        Adjoint(QFT(wires=[3, 4, 5])),
        SWAP(wires=[0, 3]),
        SWAP(wires=[1, 4]),
        SWAP(wires=[2, 5]),
        Adjoint(Adjoint(QFT(wires=[3, 4, 5]))),
        Adjoint(ControlledSequence(PhaseAdder(wires=[3, 4, 5, None]), control=[0, 1, 2])),
        Adjoint(QFT(wires=[3, 4, 5]))]
        """

        op_list = []
        if mod != 2 ** len(x_wires):
            work_wire_aux = work_wires[:1]
            wires_aux = work_wires[1:]
            wires_aux_swap = wires_aux[1:]
        else:
            work_wire_aux = ()
            wires_aux = work_wires[: len(x_wires)]
            wires_aux_swap = wires_aux
        op_list.extend(_mul_out_k_mod(k, x_wires, mod, work_wire_aux, wires_aux))
        for x_wire, aux_wire in zip(x_wires, wires_aux_swap):
            op_list.append(qml.SWAP(wires=[x_wire, aux_wire]))
        inv_k = pow(k, -1, mod)

        for op in reversed(_mul_out_k_mod(inv_k, x_wires, mod, work_wire_aux, wires_aux)):
            op_list.append(qml.adjoint(op))

        return op_list
