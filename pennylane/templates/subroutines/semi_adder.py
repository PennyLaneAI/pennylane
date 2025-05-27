# Copyright 2025 Xanadu Quantum Technologies Inc.

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
Contains the SemiAdder template.
"""

import pennylane as qml
from pennylane.operation import Operation
from pennylane.wires import WiresLike


def left_operator(wires):
    op_list = []
    # notation from figure 2 in https://arxiv.org/pdf/1709.06648
    ck, ik, tk, aux = wires
    op_list.append(qml.CNOT([ck, ik]))
    op_list.append(qml.CNOT([ck, tk]))
    op_list.append(qml.Elbow([ik, tk, aux]))
    op_list.append(qml.CNOT([ck, aux]))
    return op_list

def right_operator(wires):
    op_list = []
    # notation from figure 2 in https://arxiv.org/pdf/1709.06648
    ck, ik, tk, aux = wires
    op_list.append(qml.CNOT([ck, aux]))
    op_list.append(qml.adjoint(qml.Elbow([ik, tk, aux])))
    op_list.append(qml.CNOT([ck, ik]))
    op_list.append(qml.CNOT([ik, tk]))
    return op_list


class SemiAdder(Operation):
    r"""Performs the semi-out-place modular addition operation.
    More specifically, the operation is an in-place quantum-quantum modular addition.

    This operator performs the modular addition of two integers :math:`x` and :math:`y` modulo
    :math:`mod` in the computational basis:

    .. math::

        \text{SemiAdder}(mod) |x \rangle | y \rangle = |x \rangle | x+y \; \text{mod} \; mod \rangle,

    The implementation is based on the quantum Fourier transform method presented in
    `arXiv:2311.08555 <https://arxiv.org/abs/2311.08555>`_.

    .. note::

        To obtain the correct result, :math:`x`, :math:`y` must be smaller than :math:`mod`.

    .. seealso:: :class:`~.PhaseAdder` and :class:`~.Adder`.

    Args:
        x_wires (Sequence[int]): the wires that store the integer :math:`x`
        y_wires (Sequence[int]): the wires that store the integer :math:`y`
        mod (int): the modulo for performing the addition. If not provided, it will be set to its maximum value, :math:`2^{\text{len(y_wires)}}`.
        work_wires (Sequence[int]): the auxiliary wires to use for the addition. The work wires are not needed if :math:`mod=2^{\text{len(y_wires)}}`,
            otherwise two work wires should be provided. Defaults to empty tuple.

    **Example**

    This example computes the sum of two integers :math:`x=5` and :math:`y=6` modulo :math:`mod=7`.

    .. code-block::

        x=5
        y=6
        mod=7

        x_wires=[0,1,2]
        y_wires=[3,4,5]
        work_wires=[6,7]

        dev = qml.device("default.qubit", shots=1)
        @qml.qnode(dev)
        def circuit():
            qml.BasisEmbedding(x, wires=x_wires)
            qml.BasisEmbedding(y, wires=y_wires)
            qml.SemiAdder(x_wires, y_wires, mod, work_wires)
            return qml.sample(wires=y_wires)

    .. code-block:: pycon

        >>> print(circuit())
        [1 0 0]

    The result :math:`[1 0 0]`, is the binary representation of
    :math:`5 + 6 \; \text{modulo} \; 7 = 4`.

    .. details::
        :title: Usage Details

        This template takes as input three different sets of wires.

        The first one is ``x_wires`` which is used
        to encode the integer :math:`x < mod` in the computational basis. Therefore, ``x_wires`` must contain
        at least :math:`\lceil \log_2(x)\rceil` to represent :math:`x`.

        The second one is ``y_wires`` which is used
        to encode the integer :math:`y < mod` in the computational basis. Therefore, ``y_wires`` must contain
        at least :math:`\lceil \log_2(y)\rceil` wires to represent :math:`y`.
        ``y_wires`` is also used
        to encode the integer :math:`x+y \; \text{mod} \; mod` in the computational basis. Therefore, it will require at least
        :math:`\lceil \log_2(mod)\rceil` wires to represent :math:`x+y \; \text{mod} \; mod`.

        The fourth set of wires is ``work_wires`` which consist of the auxiliary qubits used to perform the modular addition operation.

        - If :math:`mod = 2^{\text{len(y_wires)}}`, there will be no need for ``work_wires``, hence ``work_wires=None``. This is the case by default.

        - If :math:`mod \neq 2^{\text{len(y_wires)}}`, two ``work_wires`` have to be provided.

        Note that the ``SemiAdder`` template allows us to perform modular addition in the computational basis.
        However if one just wants to perform standard addition (with no modulo),
        that would be equivalent to setting the modulo :math:`mod` to a large enough value to ensure that :math:`x+k < mod`.
    """

    grad_method = None

    def __init__(
        self,
        x_wires: WiresLike,
        y_wires: WiresLike,
        work_wires,
        id=None,
    ):  # pylint: disable=too-many-arguments

        x_wires = qml.wires.Wires(x_wires)
        y_wires = qml.wires.Wires(y_wires)
        work_wires = qml.wires.Wires(work_wires)
        num_work_wires = len(work_wires)

        """
        if mod is None:
            mod = 2 ** (len(y_wires))

      
        if mod > 2 ** len(y_wires):
            raise ValueError(
                "SemiAdder must have enough wires to represent mod. The maximum mod "
                f"with len(y_wires)={len(y_wires)} is {2 ** len(y_wires)}, but received {mod}."
            )
        if mod != 2 ** len(y_wires) and num_work_wires != 2:
            raise ValueError(
                f"If mod is not 2^{len(y_wires)}, two work wires should be provided."
            )
        if len(work_wires) != 0:
            if any(wire in work_wires for wire in x_wires):
                raise ValueError("None of the wires in work_wires should be included in x_wires.")
            if any(wire in work_wires for wire in y_wires):
                raise ValueError("None of the wires in work_wires should be included in y_wires.")
        if any(wire in y_wires for wire in x_wires):
            raise ValueError("None of the wires in y_wires should be included in x_wires.")
        """
        for key in ["x_wires", "y_wires", "work_wires"]:
            self.hyperparameters[key] = qml.wires.Wires(locals()[key])

        # pylint: disable=consider-using-generator
        all_wires = sum(
            [self.hyperparameters[key] for key in ["x_wires", "y_wires"]], start=[]
        )
        if num_work_wires != 0:
            all_wires += self.hyperparameters["work_wires"]

        #self.hyperparameters["mod"] = mod
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
            for key in ["x_wires", "y_wires", "work_wires"]
        }

        return SemiAdder(
            new_dict["x_wires"],
            new_dict["y_wires"],
            #self.hyperparameters["mod"],
            new_dict["work_wires"],
        )

    def decomposition(self):  # pylint: disable=arguments-differ

        return self.compute_decomposition(**self.hyperparameters)

    @classmethod
    def _primitive_bind_call(cls, *args, **kwargs):
        return cls._primitive.bind(*args, **kwargs)

    @staticmethod
    def compute_decomposition(
        x_wires, y_wires, work_wires
    ):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a product of other operators.

        Args:
            x_wires (Sequence[int]): the wires that store the integer :math:`x`
            y_wires (Sequence[int]): the wires that store the integer :math:`y`
            mod (int): the modulo for performing the addition. If not provided, it will be set to its maximum value, :math:`2^{\text{len(y_wires)}}`.
            work_wires (Sequence[int]): the auxiliary wires to use for the addition. The work wires are not needed if :math:`mod=2^{\text{len(y_wires)}}`,
                otherwise two work wires should be provided. Defaults to ``None``.
        Returns:
            list[.Operator]: Decomposition of the operator

        **Example**

        >>> qml.SemiAdder.compute_decomposition(x_wires=[0,1], y_wires=[2,3], mod=4, work_wires=[4,5])
        [QFT(wires=[2, 3]),
        ControlledSequence(PhaseAdder(wires=[2, 3, None]), control=[0, 1]),
        Adjoint(QFT(wires=[2, 3]))]
        """
        op_list = []

        # revert wires to follow PennyLane convention

        x_wires_pl = x_wires[::-1][:len(y_wires)]
        y_wires_pl = y_wires[::-1]
        work_wires_pl = work_wires[::-1]
        op_list.append(qml.Elbow([x_wires_pl[0], y_wires_pl[0], work_wires_pl[0]]))

        for i in range(1, len(y_wires_pl)-1):
            if i < len(x_wires_pl):
                op_list += left_operator([work_wires_pl[i-1], x_wires_pl[i], y_wires_pl[i], work_wires_pl[i]])
            else:
                op_list.append(qml.CNOT([work_wires_pl[i-1], y_wires_pl[i]]))
                op_list.append(qml.Elbow([work_wires_pl[i-1], y_wires_pl[i], work_wires_pl[i]]))
                op_list.append(qml.CNOT([work_wires_pl[i-1], work_wires_pl[i]]))


        op_list.append(qml.CNOT([work_wires_pl[-1], y_wires_pl[-1]]))

        if len(x_wires_pl) >= len(y_wires_pl):
            op_list.append(qml.CNOT([x_wires_pl[-1], y_wires_pl[-1]]))

        for i in range(len(y_wires_pl)-2,0,-1):
            if i < len(x_wires_pl):
                op_list += right_operator([work_wires_pl[i-1], x_wires_pl[i], y_wires_pl[i], work_wires_pl[i]])
            else:
                op_list.append(qml.CNOT([work_wires_pl[i-1], work_wires_pl[i]]))
                op_list.append(qml.adjoint(qml.Elbow([work_wires_pl[i-1], y_wires_pl[i], work_wires_pl[i]])))

        op_list.append(qml.adjoint(qml.Elbow([x_wires_pl[0], y_wires_pl[0], work_wires_pl[0]])))

        op_list.append(qml.CNOT([x_wires_pl[0], y_wires_pl[0]]))

        return op_list
