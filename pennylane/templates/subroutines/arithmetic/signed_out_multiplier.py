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
Contains the SignedOutMultiplier template.
"""
from itertools import combinations

from pennylane.decomposition import (
    add_decomps,
    controlled_resource_rep,
    register_resources,
)
from pennylane.decomposition.resources import resource_rep
from pennylane.operation import Operation
from pennylane.ops import Toffoli, X, ctrl
from pennylane.templates.subroutines.arithmetic import OutMultiplier, SemiAdder
from pennylane.wires import Wires, WiresLike


class SignedOutMultiplier(Operation):
    r"""Performs the signed out-of-place modular multiplication operation.

    This operator performs the modular multiplication of signed integers :math:`x` and :math:`y`
    modulo :math:`2^{n}` in the computational basis, where :math:`n` is the size of the output
    register to which the product is added:

    .. math::
        \text{SignedOutMultiplier}(mod) |x \rangle |y \rangle |b \rangle = |x \rangle |y \rangle |b + x \cdot y \; \text{mod} \; 2^n \rangle,

    The signed multiplication is implemented with :class:`~.OutMultiplier`, controlled
    :class:`~.SemiAdder`\ s, and a :class:`~.Toffoli` gate.

    .. seealso:: :class:`~.SemiAdder`, :class:`~.OutMultiplier` and :class:`~.Multiplier`.

    Args:
        x_wires (Sequence[int]): the wires that store the integer :math:`x`
        y_wires (Sequence[int]): the wires that store the integer :math:`y`
        output_wires (Sequence[int]): the wires that store the multiplication result. If the register
            is in a non-zero state :math:`b`, the solution will be added to this value, modulo
            :math:`2^n`, where :math:`n` is ``len(output_wires)``.
        work_wires (Sequence[int]): the auxiliary wires to use for the multiplication.
            ``len(y_wires)-1`` work wires should be provided.

    TODO : Update from here on
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
            qml.SignedOutMultiplier(x_wires, y_wires, output_wires, mod, work_wires)
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
                qml.SignedOutMultiplier(x_wires, y_wires, output_wires, mod, work_wires)
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

        Note that the ``SignedOutMultiplier`` template allows us to perform modular multiplication in the computational basis. However if one just wants to perform
        standard multiplication (with no modulo), that would be equivalent to setting the modulo :math:`mod` to a large enough value to ensure that :math:`x \cdot y < mod`.
    """

    grad_method = None

    resource_keys = {"num_output_wires", "num_x_wires", "num_y_wires", "num_work_wires"}

    def __init__(
        self,
        x_wires: WiresLike,
        y_wires: WiresLike,
        output_wires: WiresLike,
        work_wires: WiresLike,
        id=None,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments

        x_wires = Wires(x_wires)
        y_wires = Wires(y_wires)
        output_wires = Wires(output_wires)
        work_wires = Wires(work_wires)
        num_required_work_wires = max(len(x_wires) - 1, len(y_wires) - 1)
        if len(work_wires) < num_required_work_wires:
            raise ValueError(
                f"SignedOutMultiplier requires at least {num_required_work_wires} work wires for "
                f"{len(x_wires)} and {len(y_wires)} input wires."
            )

        registers = [
            (x_wires, "x_wires"),
            (y_wires, "y_wires"),
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
            "num_output_wires": len(self.hyperparameters["output_wires"]),
            "num_x_wires": len(self.hyperparameters["x_wires"]),
            "num_y_wires": len(self.hyperparameters["y_wires"]),
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
            for key in ["x_wires", "y_wires", "output_wires", "work_wires"]
        }

        return SignedOutMultiplier(
            new_dict["x_wires"],
            new_dict["y_wires"],
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
        x_wires: WiresLike, y_wires: WiresLike, output_wires: WiresLike, work_wires: WiresLike
    ):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a product of other operators.

        Args:
            x_wires (Sequence[int]): the wires that store the integer :math:`x`
            y_wires (Sequence[int]): the wires that store the integer :math:`y`
            output_wires (Sequence[int]): the wires that store the multiplication result. If the
                register is in a non-zero state :math:`b`, the solution will be added to this value
            work_wires (Sequence[int]): the auxiliary wires to use for the multiplication. The work
                wires are not needed if :math:`mod=2^{\text{len(output_wires)}}`, otherwise two
                work wires should be provided.

        Returns:
            list[.Operator]: Decomposition of the operator

        **Example**

        >>> qml.SignedOutMultiplier.compute_decomposition(x_wires=[0,1], y_wires=[2,3], output_wires=[5,6], work_wires=[4,7])
        # TODO
        """
        nx = len(x_wires)
        ny = len(y_wires)
        m = len(output_wires)
        need_first_subtractor = max(0, m - nx) > max(0, m - (nx + ny))
        need_second_subtractor = max(0, m - ny) > max(0, m - (nx + ny))
        op_list = [OutMultiplier(x_wires[1:], y_wires[1:], output_wires, work_wires=work_wires)]
        if need_first_subtractor or need_second_subtractor:
            op_list.extend(X(w) for w in output_wires)
        if need_first_subtractor:
            op_list.append(
                ctrl(
                    SemiAdder(
                        y_wires[1:],
                        output_wires[max(0, m - (nx + ny) + 1) : max(0, m - nx + 1)],
                        work_wires=work_wires,
                    ),
                    control=x_wires[0],
                )
            )
        if need_second_subtractor:
            op_list.append(
                ctrl(
                    SemiAdder(
                        x_wires[1:],
                        output_wires[max(0, m - (nx + ny) + 1) : max(0, m - ny + 1)],
                        work_wires=work_wires,
                    ),
                    control=y_wires[0],
                )
            )
        if need_first_subtractor or need_second_subtractor:
            op_list.extend(X(w) for w in output_wires)
        op_list.append(Toffoli([x_wires[0], y_wires[0], output_wires[0]]))
        return op_list


def _signed_out_multiplier_resources(
    num_output_wires, num_x_wires, num_y_wires, num_work_wires
) -> dict:
    need_first_subtractor = max(0, num_output_wires - num_x_wires) > max(
        0, num_output_wires - (num_x_wires + num_y_wires)
    )
    need_second_subtractor = max(0, num_output_wires - num_y_wires) > max(
        0, num_output_wires - (num_x_wires + num_y_wires)
    )
    resources = {
        resource_rep(
            OutMultiplier,
            num_output_wires=num_output_wires,
            num_x_wires=num_x_wires - 1,
            num_y_wires=num_y_wires - 1,
            num_work_wires=num_work_wires,
            mod=2**num_output_wires,
        ): 1,
        controlled_resource_rep(
            base_class=SemiAdder,
            base_params={
                "num_y_wires": num_y_wires,
            },
            num_control_wires=1,
        ): need_first_subtractor,
        controlled_resource_rep(
            base_class=SemiAdder,
            base_params={
                "num_y_wires": num_x_wires,
            },
            num_control_wires=1,
        ): need_second_subtractor,
        resource_rep(X): 2 * num_output_wires * (need_first_subtractor or need_second_subtractor),
    }
    resources[resource_rep(Toffoli)] = 1
    return resources


@register_resources(_signed_out_multiplier_resources)
def _signed_out_multiplier(
    x_wires: WiresLike,
    y_wires: WiresLike,
    output_wires: WiresLike,
    work_wires: WiresLike,
    **__,
):
    SignedOutMultiplier.compute_decomposition(x_wires, y_wires, output_wires, work_wires)


add_decomps(SignedOutMultiplier, _signed_out_multiplier)
