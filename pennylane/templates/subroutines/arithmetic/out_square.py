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
from collections import defaultdict
from itertools import combinations

from pennylane.decomposition import (
    add_decomps,
    controlled_resource_rep,
    register_condition,
    register_resources,
)
from pennylane.decomposition.resources import resource_rep
from pennylane.operation import Operation
from pennylane.ops import CNOT, Controlled
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
        work_wires (Sequence[int]): the auxiliary wires to use for the squaring.
            ``len(output_wires)`` work wires are required if ``output_wires_zeroed=False``,
            otherwise ``min(len(output_wires), len(x_wires)+1)`` work wires are required.
        output_wires_zeroed (bool): Whether the output wires are guaranteed to be in the state
            :math:`|0\rangle` initially. Defaults to ``False``.

    THE REST OF THE DOCSTRING NEEDS UPDATING!
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

    resource_keys = {"num_x_wires", "num_output_wires", "num_work_wires", "output_wires_zeroed"}

    def __init__(
        self,
        x_wires: WiresLike,
        output_wires: WiresLike,
        work_wires: WiresLike,
        output_wires_zeroed: bool = False,
        id=None,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments

        x_wires = Wires(x_wires)
        output_wires = Wires(output_wires)
        work_wires = Wires(work_wires)

        if output_wires_zeroed:
            num_required_work_wires = min(len(x_wires) + 1, len(output_wires))
        else:
            num_required_work_wires = len(output_wires)
        if len(work_wires) < num_required_work_wires:
            raise ValueError(
                f"OutSquare requires at least {num_required_work_wires} work wires for "
                f"{len(x_wires)} input wires, {len(output_wires)} output wires "
                f"and {output_wires_zeroed=}."
            )

        registers = [
            (work_wires, "work_wires"),
            (output_wires, "output_wires"),
            (x_wires, "x_wires"),
        ]
        for (reg0, reg0_name), (reg1, reg1_name) in combinations(registers, r=2):
            if reg0.intersection(reg1):
                raise ValueError(
                    f"None of the wires in {reg0_name} should be included in {reg1_name}."
                )

        for wires, name in registers:
            self.hyperparameters[name] = wires

        self.hyperparameters["output_wires_zeroed"] = output_wires_zeroed
        all_wires = sum((self.hyperparameters[name] for _, name in registers), start=[])
        super().__init__(wires=all_wires, id=id)

    @property
    def resource_params(self) -> dict:
        return {
            "num_x_wires": len(self.hyperparameters["x_wires"]),
            "num_output_wires": len(self.hyperparameters["output_wires"]),
            "num_work_wires": len(self.hyperparameters["work_wires"]),
            "output_wires_zeroed": self.hyperparameters["output_wires_zeroed"],
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
            self.hyperparameters["output_wires_zeroed"],
        )

    def decomposition(self):
        return self.compute_decomposition(**self.hyperparameters)

    @classmethod
    def _primitive_bind_call(cls, *args, **kwargs):
        return cls._primitive.bind(*args, **kwargs)

    @staticmethod
    def compute_decomposition(
        x_wires: WiresLike,
        output_wires: WiresLike,
        work_wires: WiresLike,
        output_wires_zeroed: bool,
    ):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a product of other operators.

        Args:
            x_wires (Sequence[int]): the wires that store the integer :math:`x`
            output_wires (Sequence[int]): the wires that store the squaring result. If the register is in a non-zero state :math:`b`, the solution will be added to this value
            work_wires (Sequence[int]): the auxiliary wires to use for the multiplication.

        Returns:
            list[.Operator]: Decomposition of the operator

        **Example**

        >>> qml.OutSquare.compute_decomposition(x_wires=[0,1], output_wires=[2,3], work_wires=[4,7], output_wires_zeroed=True)
        """
        n = len(x_wires)
        m = len(output_wires)
        op_list = []

        if output_wires_zeroed:
            # Copy x, controlled on the least significant bit (LSB) of x, to the output register,
            # which is in |0>. This can be reduced to a CNOT for the LSB and temporary ANDs for
            # the other bits.
            op_list.append(
                CNOT([x_wires[-1], output_wires[-1]])
            )  # First control-copy reduces to CNOT
            op_list.extend(
                [
                    TemporaryAND([x_wires[-1], x_wire, out_wire])  # Subsequent control-copies
                    for x_wire, out_wire in zip(
                        x_wires[:-1][::-1], output_wires[:-1][::-1]
                    )  # todo unify slicing
                ]
            )
            # Mark that the copying has happened and does not have to happen via an adder below
            x_wires_to_multiply = x_wires[:-1]
            start = 1
        else:
            x_wires_to_multiply = x_wires
            start = 0

        for i, x_wire in enumerate(reversed(x_wires_to_multiply), start=start):
            # Add x to the output register, controlled on x_wire via the work_wires[0] and
            # shifted by i bit positions. For output_wires_zeroed=False, includes the initial copy
            # The output wires of the adder need to take all of the output register of square
            # into account due to carry values. For output_wires_zeroed=True, we can reduce to
            # a fixed size (`n`) instead.
            # In future, we could investigate whether controlled addition can be made cheaper if
            # the control is equal to one of the input bits.
            if output_wires_zeroed:
                add_y_wires = output_wires[max(0, m - n - i - 1) : max(0, m - i)]
            else:
                add_y_wires = output_wires[: max(0, m - i)]
            op_list.extend(
                [
                    CNOT([x_wire, work_wires[0]]),
                    Controlled(
                        SemiAdder(x_wires=x_wires, y_wires=add_y_wires, work_wires=work_wires[1:]),
                        control_wires=work_wires[:1],
                    ),
                    CNOT([x_wire, work_wires[0]]),
                ]
            )

        return op_list


def _out_square_resources(
    num_x_wires, num_output_wires, num_work_wires, output_wires_zeroed
) -> dict:
    # pylint: disable=unused-argument
    resources = defaultdict(int)
    resources[resource_rep(CNOT)] = 2 * (num_x_wires - 1)
    if output_wires_zeroed:
        # Copying of first bit is a CNOT, all other bits require a TemporaryAND
        resources[resource_rep(CNOT)] += 1
        resources[resource_rep(TemporaryAND)] = output_wires_zeroed * (num_x_wires - 1)
    else:
        # Copying is done via CNOT-wrapped controlled adder. Account for CNOTs here
        resources[resource_rep(CNOT)] += 2

    # Controlled adders, includes the one for copying if output_wires_zeroed=False
    for i in range(output_wires_zeroed, min(num_x_wires, num_output_wires)):
        if output_wires_zeroed:
            num_out = max(0, num_output_wires - i) - max(0, num_output_wires - num_x_wires - i - 1)
        else:
            num_out = max(0, num_output_wires - i)
        resources[
            controlled_resource_rep(
                base_class=SemiAdder,
                base_params={"num_y_wires": num_out},
                num_control_wires=1,
            )
        ] += 1
    return dict(resources)


def _out_square_condition(num_x_wires, num_work_wires, num_output_wires, **_):
    # This condition ensures that we can use an efficient C(SemiAdder) decomposition, which
    # requires num_x_wires work wires. One more work wire is required by the squaring itself.
    return num_work_wires >= min(num_x_wires, num_output_wires) + 1


@register_condition(_out_square_condition)
@register_resources(_out_square_resources)
def _out_square(
    x_wires: WiresLike,
    output_wires: WiresLike,
    work_wires: WiresLike,
    output_wires_zeroed: bool,
    **_,
):
    OutSquare.compute_decomposition(x_wires, output_wires, work_wires, output_wires_zeroed)


add_decomps(OutSquare, _out_square)
