# Copyright 2026 Xanadu Quantum Technologies Inc.

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
Contains the OutSquare template.
"""
from itertools import combinations

from pennylane.decomposition import (
    add_decomps,
    controlled_resource_rep,
    register_condition,
    register_resources,
)
from pennylane.decomposition.resources import resource_rep
from pennylane.operation import Operation
from pennylane.ops import CNOT, Controlled, X
from pennylane.templates.subroutines.arithmetic import SemiAdder, TemporaryAND
from pennylane.wires import Wires, WiresLike


class OutSquare(Operation):
    r"""Performs out-of-place modular squaring.

    This operator performs the modular squaring of integers :math:`x` modulo
    :math:`2^n` in the computational basis, where ``n=len(output_wires)``:

    .. math::
        \text{OutSquare} |x \rangle |b \rangle = |x \rangle |(b + x^2) \; \text{mod} \; 2^n \rangle,

    .. seealso:: :class:`~.SemiAdder`, :class:`~.Multiplier` , :class:`~.OutMultiplier`
        and :class:`~.SignedOutMultiplier`.

    Args:
        x_wires (Sequence[int]): the wires that store the integer :math:`x`
        output_wires (Sequence[int]): the wires that store the squaring result. If the
            register is in a non-zero state :math:`b`, the solution will be added to this value.
        work_wires (Sequence[int]): the auxiliary wires to use for the multiplication.
            ``len(y_wires)`` work wires are required.

    **Example**

    This example performs the multiplication of two integers :math:`x=2` and :math:`y=7` modulo :math:`mod=12`.
    We'll let :math:`b=0`. See Usage Details for :math:`b \neq 0`.

    .. code-block:: python

        x = 2
        y = 7
        mod = 12

        x_wires = [0, 1]
        y_wires = [2, 3, 4]
        output_wires = [6, 7, 8, 9]
        work_wires = [5, 10]

        dev = qml.device("default.qubit")

        @qml.qnode(dev, shots=1)
        def circuit():
            qml.BasisEmbedding(x, wires=x_wires)
            qml.BasisEmbedding(y, wires=y_wires)
            qml.OutSquare(x_wires, y_wires, output_wires, mod, work_wires)
            return qml.sample(wires=output_wires)

    >>> print(circuit())
    [[0 0 1 0]]

    The result :math:`[[0 0 1 0]]`, is the binary representation of
    :math:`2 \cdot 7 \; \text{modulo} \; 12 = 2`.

    .. details::
        :title: Usage Details

        This template takes as input four different sets of wires.

        The first one is ``x_wires`` which is used
        to encode the integer :math:`x < mod` in the computational basis. Therefore, ``x_wires`` must contain
        at least :math:`\lceil \log_2(x)\rceil` wires to represent :math:`x`.

        The second one is ``y_wires`` which is used
        to encode the integer :math:`y < mod` in the computational basis. Therefore, ``y_wires`` must contain
        at least :math:`\lceil \log_2(y)\rceil` wires to represent :math:`y`.

        The third one is ``output_wires`` which is used
        to encode the integer :math:`b+ x \cdot y \; \text{mod} \; mod` in the computational basis. Therefore, it will require at least
        :math:`\lceil \log_2(mod)\rceil` ``output_wires`` to represent :math:`b + x \cdot y \; \text{mod} \; mod`.  Note that these wires can be initialized with any integer
        :math:`b < mod`, but the most common choice is :math:`b=0` to obtain as a final result :math:`x \cdot y \; \text{mod} \; mod`.
        The following is an example for :math:`b = 1`.

        .. code-block:: python

            b = 1
            x = 2
            y = 7
            mod = 12

            x_wires = [0, 1]
            y_wires = [2, 3, 4]
            output_wires = [6, 7, 8, 9]
            work_wires = [5, 10]

            dev = qml.device("default.qubit")

            @qml.qnode(dev, shots=1)
            def circuit():
                qml.BasisEmbedding(x, wires=x_wires)
                qml.BasisEmbedding(y, wires=y_wires)
                qml.BasisEmbedding(b, wires=output_wires)
                qml.OutSquare(x_wires, y_wires, output_wires, mod, work_wires)
                return qml.sample(wires=output_wires)

        >>> print(circuit())
        [[0 0 1 1]]

        The result :math:`[[0 0 1 1]]`, is the binary representation of
        :math:`2 \cdot 7 + 1\; \text{modulo} \; 12 = 3`.

        The fourth set of wires is ``work_wires`` which consist of the auxiliary qubits used to perform the modular multiplication operation.

        - If the cheaper decomposition based on :class:`~.SemiAdder` is used,
          ``len(y_wires)`` work wires are required, which are passed to the adders.

        - If :math:`mod = 2^{\text{len(output_wires)}}`, there will be no need for ``work_wires``, hence ``work_wires=()``. This is the case by default.

        - If :math:`mod \neq 2^{\text{len(output_wires)}}`, two ``work_wires`` have to be provided.

        Note that the ``OutSquare`` template allows us to perform modular multiplication in the computational basis. However if one just wants to perform
        standard multiplication (with no modulo), that would be equivalent to setting the modulo :math:`mod` to a large enough value to ensure that :math:`x \cdot y < mod`.
    """

    grad_method = None

    resource_keys = {"num_x_wires", "num_output_wires", "num_work_wires"}

    def __init__(
        self,
        x_wires: WiresLike,
        output_wires: WiresLike,
        work_wires: WiresLike,
        id=None,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments

        x_wires = Wires(x_wires)
        output_wires = Wires(output_wires)
        work_wires = Wires(work_wires)

        num_required_work_wires = 0  # TODO
        if len(work_wires) < num_required_work_wires:
            raise ValueError(
                f"OutSquare requires at least {num_required_work_wires} work wires for "
                f"{len(x_wires)} input wires."
            )

        registers = [
            (x_wires, "x_wires"),
            (output_wires, "output_wires"),
            (work_wires, "work_wires"),
        ]
        for (reg0, reg0_name), (reg1, reg1_name) in combinations(registers, r=2):
            if reg0.intersection(reg1):
                raise ValueError(
                    f"None of the wires in {reg0_name} should be included in {reg1_name}."
                )

        for wires, name in registers:
            self.hyperparameters[name] = wires

        all_wires = sum((self.hyperparameters[name] for _, name in registers), start=[])
        super().__init__(wires=all_wires, id=id)

    @property
    def resource_params(self) -> dict:
        return {
            "num_x_wires": len(self.hyperparameters["x_wires"]),
            "num_output_wires": len(self.hyperparameters["output_wires"]),
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

        return OutSquare(
            new_dict["x_wires"],
            new_dict["output_wires"],
            new_dict["work_wires"],
        )

    def decomposition(self):
        return self.compute_decomposition(**self.hyperparameters)

    @classmethod
    def _primitive_bind_call(cls, *args, **kwargs):
        return cls._primitive.bind(*args, **kwargs)

    @staticmethod
    def compute_decomposition(
        x_wires: WiresLike, output_wires: WiresLike, work_wires: WiresLike
    ):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a product of other operators.

        Args:
            x_wires (Sequence[int]): the wires that store the integer :math:`x`
            output_wires (Sequence[int]): the wires that store the squaring result. If the register is in a non-zero state :math:`b`, the solution will be added to this value
            work_wires (Sequence[int]): the auxiliary wires to use for the multiplication.

        Returns:
            list[.Operator]: Decomposition of the operator

        **Example**

        >>> qml.OutSquare.compute_decomposition(x_wires=[0,1], output_wires=[2,3], output_wires=[5,6], mod=4, work_wires=[4,7])
        """
        op_list = []
        op_list.append(CNOT([x_wires[0], output_wires[0]]))

        for w1, w2 in zip(x_wires[1:-1], output_wires[1:-1]):
            op_list.append(TemporaryAND([x_wires[0], w1, w2]))

        num_bits = len(x_wires)

        for ix in range(num_bits // 2):
            op_list.append(CNOT([x_wires[ix], work_wires[0]]))

        op_list.append(
            Controlled(
                SemiAdder(
                    x_wires=x_wires[:-1],
                    y_wires=output_wires[1 : num_bits + 1],
                    work_wires=work_wires[1:],
                ),
                control_wires=work_wires[0],
            )
        )

        for ix in range(num_bits // 2, num_bits):
            op_list.append(CNOT([x_wires[ix - 1], work_wires[0]]))

        op_list.append(
            Controlled(
                SemiAdder(
                    x_wires=x_wires[:-1],
                    y_wires=output_wires[2 : num_bits + 2],
                    work_wires=work_wires[1:],
                ),
                control_wires=work_wires[0],
            )
        )

        op_list.append(CNOT([x_wires[num_bits - 2], work_wires[0]]))
        op_list.append(CNOT([x_wires[num_bits - 1], output_wires[2 * num_bits - 2]]))

        op_list.append(CNOT([x_wires[num_bits - 1], work_wires[0]]))
        op_list.append(CNOT([work_wires[0], x_wires[num_bits - 1]]))

        for ix in range(num_bits, 2 * num_bits - 1):
            op_list.append(X(output_wires[ix]))

        op_list.append(
            Controlled(
                SemiAdder(
                    x_wires=x_wires,
                    y_wires=output_wires[num_bits : 2 * num_bits],
                    work_wires=work_wires[1:],
                ),
                control_wires=work_wires[0],
            )
        )

        for ix in range(num_bits, 2 * num_bits - 1):
            op_list.append(X(output_wires[ix]))

        op_list.append(CNOT([work_wires[0], x_wires[num_bits - 1]]))
        op_list.append(CNOT([x_wires[num_bits - 1], work_wires[0]]))

        return op_list


def _out_square_resources(num_x_wires, num_output_wires, num_work_wires) -> dict:
    # pylint: disable=unused-argument
    return {
        resource_rep(CNOT): num_x_wires + 7,
        resource_rep(TemporaryAND): num_x_wires - 2,
        resource_rep(X): 2 * num_x_wires - 2,
        controlled_resource_rep(
            base_class=SemiAdder,
            base_params={
                "num_y_wires": num_output_wires // 2 + 1,
            },
            num_control_wires=1,
        ): 2,
        controlled_resource_rep(
            base_class=SemiAdder,
            base_params={
                "num_y_wires": num_output_wires // 2,
            },
            num_control_wires=1,
        ): 1,
    }


def _out_square_condition(num_x_wires, num_output_wires, num_work_wires):
    return num_work_wires >= 0  # TODO


@register_condition(_out_square_condition)
@register_resources(_out_square_resources)
def _out_square(x_wires: WiresLike, output_wires: WiresLike, work_wires: WiresLike, **_):
    OutSquare.compute_decomposition(x_wires, output_wires, work_wires)


add_decomps(OutSquare, _out_square)
