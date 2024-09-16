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
    r"""Performs the out-place modular addition operation.

    This operator performs the modular addition of two integers :math:`x` and :math:`y` modulo
    :math:`mod` in the computational basis:

    .. math::

        \text{OutAdder}(mod) |x \rangle | y \rangle | b \rangle = |x \rangle | y \rangle | b+x+y \; \text{mod} \; mod \rangle,

    The implementation is based on the quantum Fourier transform method presented in
    `arXiv:2311.08555 <https://arxiv.org/abs/2311.08555>`_.

    .. note::

        To obtain the correct result, :math:`x`, :math:`y` and :math:`b` must be smaller than :math:`mod`.

    .. seealso:: :class:`~.PhaseAdder` and :class:`~.Adder`.

    Args:
        x_wires (Sequence[int]): the wires that store the integer :math:`x`
        y_wires (Sequence[int]): the wires that store the integer :math:`y`
        output_wires (Sequence[int]): the wires that store the addition result. If the register is in a non-zero state :math:`b`, the solution will be added to this value.
        mod (int): the modulo for performing the addition. If not provided, it will be set to its maximum value, :math:`2^{\text{len(output_wires)}}`.
        work_wires (Sequence[int]): the auxiliary wires to use for the addition. The work wires are not needed if :math:`mod=2^{\text{len(output_wires)}}`, otherwise two work wires should be provided. Defaults to ``None``.

    **Example**

    This example computes the sum of two integers :math:`x=5` and :math:`y=6` modulo :math:`mod=7`.
    We'll let :math:`b=0`. See Usage Details for :math:`b \neq 0`.

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

        >>> print(circuit())
        [1 0 0]

    The result :math:`[1 0 0]`, is the binary representation of
    :math:`5 + 6 \; \text{modulo} \; 7 = 4`.

    .. details::
        :title: Usage Details

        This template takes as input four different sets of wires.

        The first one is ``x_wires`` which is used
        to encode the integer :math:`x < mod` in the computational basis. Therefore, ``x_wires`` must contain
        at least :math:`\lceil \log_2(x)\rceil` to represent :math:`x`.

        The second one is ``y_wires`` which is used
        to encode the integer :math:`y < mod` in the computational basis. Therefore, ``y_wires`` must contain
        at least :math:`\lceil \log_2(y)\rceil` wires to represent :math:`y`.

        The third one is ``output_wires`` which is used
        to encode the integer :math:`b+x+y \; \text{mod} \; mod` in the computational basis. Therefore, it will require at least
        :math:`\lceil \log_2(mod)\rceil` ``output_wires`` to represent :math:`b+x+y \; \text{mod} \; mod`. Note that these wires can be initialized with any integer
        :math:`b < mod`, but the most common choice is :math:`b=0` to obtain as a final result :math:`x + y \; \text{mod} \; mod`.
        The following is an example for :math:`b = 1`.

        .. code-block::

            b=1
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
                qml.BasisEmbedding(b, wires=output_wires)
                qml.OutAdder(x_wires, y_wires, output_wires, mod, work_wires)
                return qml.sample(wires=output_wires)

        .. code-block:: pycon

            >>> print(circuit())
            [1 0 1]

        The result :math:`[1 0 1]`, is the binary representation of
        :math:`5 + 6 + 1\; \text{modulo} \; 7 = 5`.

        The fourth set of wires is ``work_wires`` which consist of the auxiliary qubits used to perform the modular addition operation.

        - If :math:`mod = 2^{\text{len(output_wires)}}`, there will be no need for ``work_wires``, hence ``work_wires=None``. This is the case by default.

        - If :math:`mod \neq 2^{\text{len(output_wires)}}`, two ``work_wires`` have to be provided.

        Note that the ``OutAdder`` template allows us to perform modular addition in the computational basis. However if one just wants to perform standard addition (with no modulo),
        that would be equivalent to setting the modulo :math:`mod` to a large enough value to ensure that :math:`x+k < mod`.
    """

    grad_method = None

    def __init__(
        self, x_wires, y_wires, output_wires, mod=None, work_wires=None, id=None
    ):  # pylint: disable=too-many-arguments

        x_wires = qml.wires.Wires(x_wires)
        y_wires = qml.wires.Wires(y_wires)
        output_wires = qml.wires.Wires(output_wires)

        num_work_wires = 0 if work_wires is None else len(work_wires)

        if mod is None:
            mod = 2 ** (len(output_wires))
        if mod > 2 ** len(output_wires):
            raise ValueError(
                "OutAdder must have enough wires to represent mod. The maximum mod "
                f"with len(output_wires)={len(output_wires)} is {2 ** len(output_wires)}, but received {mod}."
            )
        if mod != 2 ** len(output_wires) and num_work_wires != 2:
            raise ValueError(
                f"If mod is not 2^{len(output_wires)}, two work wires should be provided."
            )
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
            x_wires (Sequence[int]): the wires that store the integer :math:`x`
            y_wires (Sequence[int]): the wires that store the integer :math:`y`
            output_wires (Sequence[int]): the wires that store the addition result. If the register is in a non-zero state :math:`b`, the solution will be added to this value.
            mod (int): the modulo for performing the addition. If not provided, it will be set to its maximum value, :math:`2^{\text{len(output_wires)}}`.
            work_wires (Sequence[int]): the auxiliary wires to use for the addition. The work wires are not needed if :math:`mod=2^{\text{len(output_wires)}}`, otherwise two work wires should be provided. Defaults to ``None``.
        Returns:
            list[.Operator]: Decomposition of the operator

        **Example**

        >>> qml.OutAdder.compute_decomposition(x_wires=[0,1], y_wires=[2,3], output_wires=[5,6], mod=4, work_wires=[4,7])
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
