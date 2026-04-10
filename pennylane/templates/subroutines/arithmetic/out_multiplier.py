# Copyright 2018-2026 Xanadu Quantum Technologies Inc.

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
Contains the OutMultiplier template.
"""

from collections import defaultdict

from pennylane.decomposition import (
    add_decomps,
    change_op_basis_resource_rep,
    controlled_resource_rep,
    register_condition,
    register_resources,
)
from pennylane.decomposition.resources import resource_rep
from pennylane.operation import Operation
from pennylane.ops import (
    BasisState,
    MidMeasure,
    MultiControlledX,
    X,
    change_op_basis,
    ctrl,
    measure,
)
from pennylane.templates.subroutines.arithmetic import SemiAdder
from pennylane.templates.subroutines.controlled_sequence import ControlledSequence
from pennylane.templates.subroutines.qft import QFT
from pennylane.wires import Wires, WiresLike

from .phase_adder import PhaseAdder


class OutMultiplier(Operation):
    r"""Performs the out-place modular multiplication operation.

    This operator performs the modular multiplication of integers :math:`x` and :math:`y` modulo
    :math:`mod` in the computational basis:

    .. math::
        \text{OutMultiplier}(mod) |x \rangle |y \rangle |b \rangle = |x \rangle |y \rangle |b + x \cdot y \; \text{mod} \; mod \rangle,

    There are three implementations available, which differ in the auxiliary wires
    and in the gate counts they require, and in whether or not they support arbitrary values for
    the modulus ``mod``. See the usage details for more information.

    .. note::

        To obtain the correct result, :math:`x`, :math:`y` and :math:`b` must be smaller than :math:`mod`.

    .. seealso:: :class:`~.PhaseAdder` and :class:`~.Multiplier`.

    Args:
        x_wires (Sequence[int]): the wires that store the integer :math:`x`
        y_wires (Sequence[int]): the wires that store the integer :math:`y`
        output_wires (Sequence[int]): the wires that store the multiplication result. If the register is in a non-zero state :math:`b`, the solution will be added to this value
        mod (int): the modulo for performing the multiplication. If not provided, it will be set to its maximum value, :math:`2^{\text{len(output_wires)}}`
        work_wires (Sequence[int]): the auxiliary wires to use for the multiplication. The
            work wires are not needed if :math:`mod=2^{\text{len(output_wires)}}`, otherwise at least two work wires
            should be provided. Defaults to empty tuple.
        zeroed_output_wires (bool): Whether the ``output_wires`` are guaranteed to be in state
            :math:`|0\rangle` initially.

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
            qml.OutMultiplier(x_wires, y_wires, output_wires, mod, work_wires)
            return qml.sample(wires=output_wires)

    >>> print(circuit())
    [[0 0 1 0]]

    The result :math:`[[0 0 1 0]]`, is the binary representation of
    :math:`2 \cdot 7 \; \text{modulo} \; 12 = 2`.

    .. details::
        :title: Usage Details

        This template takes as input four different registers of wires.

        The first register is ``x_wires`` which is used
        to encode the integer :math:`x < mod` in the computational basis.

        The second register is ``y_wires`` which is used
        to encode the integer :math:`y < mod` in the computational basis.

        The third register is ``output_wires`` which is used
        to encode the integer :math:`(b+ x \cdot y) \; \text{mod} \; mod` in the computational
        basis. Therefore, it will require at least :math:`\lceil \log_2(mod)\rceil` wires
        Note that these wires can be initialized with any integer :math:`b < mod`.

        The fourth register is ``work_wires`` containing the auxiliary qubits used to
        perform the modular multiplication operation. The number of auxiliary wires determines
        which decomposition is available (also see below).

        **Initial state of output wires**

        As indicated above, the initial state of ``output_wires`` can encode any value
        :math:`b<mod`. The following is an example for :math:`b = 1`.

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
                qml.OutMultiplier(x_wires, y_wires, output_wires, mod, work_wires)
                return qml.counts(wires=output_wires)

        >>> out = circuit()
        >>> print(out)
        {np._bin("0011"): 1}
        >>> print(int(list(out.keys())[0], 2))
        3

        The result :math:`(0011)_2`, is the binary representation of
        :math:`(1 + 2 \cdot 7)\; \text{modulo} \; 12 = 3`:

        If the initial state on the output wires is guaranteed to be :math:`|0\rangle`, this
        can be indicated to ``OutMultiplier`` by setting ``zeroed_output_wires=True``. This
        simplifies some of the available decompositions (also see below), saving quantum resources.

        **Different decompositions**

        There are three decompositions, which differ in the required number of work wires and
        gates, and in whether they support ``mod!=2**len(output_wires)``.

        - The first implementation is based on the quantum Fourier transform (QFT) method presented in
          `arXiv:2311.08555 <https://arxiv.org/abs/2311.08555>`_. It requires zero (two) auxiliary
          wires for ``mod=2**len(output_wires)`` (for other values of ``mod``), and it uses a doubly
          controlled sequence (a nested :class:`~.ControlledSequence`) and two QFTs. Any value
          for ``mod`` is supported, subject to the description above.

        - The second implementation uses controlled :class:`~.SemiAdder`\ s to realize the
          multiplication. For :math:`n` ``x_wires``, :math:`m` ``y_wires`` and :math:`k`
          ``output_wires``, we need :math:`L = \min(k, n)` adders with usually varying sizes
          :math:`\min(k - i, m + 1)` for :math:`0\leq i<L`. The concrete :class:`~.Toffoli` count
          resulting from this is a bit verbose in general.

        - The third implementation uses controlled addition/subtraction to replace the controlled
          ``SemiAdder``\ s from the previous implementation, based on
          `arXiv:2410.00899 <https://arxiv.org/abs/2410.00899>`__.
          For :math:`n` ``x_wires``, :math:`m` ``y_wires`` and :math:`k`
          ``output_wires``, we need :math:`L=\min(k, n)` controlled add/subtract operations
          of usually varying size :math:`\min(k + 1 - i, m + 1)` for :math:`0\leq i<L`,
          three ``SemiAdder``\ s of sizes :math:`\min(k + 1 - m, n + 1)`,
          :math:`\min(k + 1 - n, m + 1)` and :math:`k+1`, as well as an incrementer on
          :math:`\min(k + 1, n + m)` qubits and Pauli gates.

    """

    grad_method = None

    resource_keys = {
        "num_output_wires",
        "num_x_wires",
        "num_y_wires",
        "num_work_wires",
        "mod",
        "zeroed_output_wires",
    }

    def __init__(
        self,
        x_wires: WiresLike,
        y_wires: WiresLike,
        output_wires: WiresLike,
        mod=None,
        work_wires: WiresLike = (),
        zeroed_output_wires: bool = False,
        id=None,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments

        x_wires = Wires(x_wires)
        y_wires = Wires(y_wires)
        output_wires = Wires(output_wires)
        work_wires = Wires(() if work_wires is None else work_wires)

        num_work_wires = len(work_wires)

        if mod is None:
            mod = 2 ** len(output_wires)
        if mod != 2 ** len(output_wires):
            if num_work_wires < 2:
                raise ValueError(
                    f"If mod is not 2^{len(output_wires)}, at least two work wires should be provided."
                )
            work_wires = work_wires[:2]
        if mod > 2 ** (len(output_wires)):
            raise ValueError(
                "OutMultiplier must have enough wires to represent mod. The maximum mod "
                f"with len(output_wires)={len(output_wires)} is {2 ** len(output_wires)}, but received {mod}."
            )

        if len(work_wires) != 0:
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

        wires_list = [x_wires, y_wires, output_wires]
        wires_name = ["x_wires", "y_wires", "output_wires"]

        if len(work_wires) != 0:
            wires_list.append(work_wires)
            wires_name.append("work_wires")

        for name, wires in zip(wires_name, wires_list):
            self.hyperparameters[name] = Wires(wires)
        self.hyperparameters["mod"] = mod
        self.hyperparameters["zeroed_output_wires"] = zeroed_output_wires

        # pylint: disable=consider-using-generator
        all_wires = sum([self.hyperparameters[name] for name in wires_name], start=[])
        super().__init__(wires=all_wires, id=id)

    @property
    def resource_params(self) -> dict:
        return {
            "num_output_wires": len(self.hyperparameters["output_wires"]),
            "num_x_wires": len(self.hyperparameters["x_wires"]),
            "num_y_wires": len(self.hyperparameters["y_wires"]),
            "num_work_wires": len(self.hyperparameters["work_wires"]),
            "mod": self.hyperparameters["mod"],
            "zeroed_output_wires": self.hyperparameters["zeroed_output_wires"],
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

        return OutMultiplier(
            new_dict["x_wires"],
            new_dict["y_wires"],
            new_dict["output_wires"],
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
        x_wires: WiresLike,
        y_wires: WiresLike,
        output_wires: WiresLike,
        mod,
        work_wires: WiresLike,
        zeroed_output_wires: bool,
    ):  # pylint: disable=arguments-differ, too-many-arguments, unused-argument
        r"""Representation of the operator as a product of other operators.

        Args:
            x_wires (Sequence[int]): the wires that store the integer :math:`x`
            y_wires (Sequence[int]): the wires that store the integer :math:`y`
            output_wires (Sequence[int]): the wires that store the multiplication result. If the register is in a non-zero state :math:`b`, the solution will be added to this value
            mod (int): the modulo for performing the multiplication. If not provided, it will be set to its maximum value, :math:`2^{\text{len(output_wires)}}`
            work_wires (Sequence[int]): the auxiliary wires to use for the multiplication. The
                work wires are not needed if :math:`mod=2^{\text{len(output_wires)}}`, otherwise two work wires
                should be provided.

        Returns:
            list[.Operator]: Decomposition of the operator

        **Example**

        >>> qml.OutMultiplier.compute_decomposition(x_wires=[0,1], y_wires=[2,3], output_wires=[5,6], mod=4, work_wires=[4,7])
        [(Adjoint(QFT(wires=[5, 6]))) @ (ControlledSequence(ControlledSequence(PhaseAdder(wires=[5, 6]), control=[0, 1]), control=[2, 3])) @ QFT(wires=[5, 6])]
        """
        if mod != 2 ** len(output_wires):
            qft_output_wires = work_wires[:1] + output_wires
            work_wire = work_wires[1:2]
        else:
            qft_output_wires = output_wires
            work_wire = ()

        op_list = [
            change_op_basis(
                QFT(wires=qft_output_wires),
                ControlledSequence(
                    ControlledSequence(
                        PhaseAdder(1, qft_output_wires, mod, work_wire), control=x_wires
                    ),
                    control=y_wires,
                ),
            )
        ]
        return op_list


def _out_multiplier_with_qft_resources(
    num_output_wires, num_x_wires, num_y_wires, mod, **_
) -> dict:
    qft_wires = num_output_wires + 1 if mod != 2**num_output_wires else num_output_wires
    return {
        change_op_basis_resource_rep(
            resource_rep(QFT, num_wires=qft_wires),
            resource_rep(
                ControlledSequence,
                base_class=ControlledSequence,
                base_params={
                    "base_class": PhaseAdder,
                    "base_params": {"num_x_wires": qft_wires, "mod": mod},
                    "num_control_wires": num_x_wires,
                },
                num_control_wires=num_y_wires,
            ),
        ): 1
    }


def _out_multiplier_with_qft_condition(num_output_wires, mod, num_work_wires, **_):
    return mod in (None, 2**num_output_wires) or num_work_wires >= 2


@register_condition(_out_multiplier_with_qft_condition)
@register_resources(_out_multiplier_with_qft_resources)
def _out_multiplier_with_qft(
    x_wires: WiresLike,
    y_wires: WiresLike,
    output_wires: WiresLike,
    mod,
    work_wires: WiresLike,
    zeroed_output_wires: bool,
    **_,
):  # pylint: disable=too-many-arguments, unused-argument
    if mod != 2 ** len(output_wires):
        qft_output_wires = work_wires[:1] + output_wires
        work_wire = work_wires[1:2]
    else:
        qft_output_wires = output_wires
        work_wire = ()

    change_op_basis(
        QFT(wires=qft_output_wires),
        ControlledSequence(
            ControlledSequence(PhaseAdder(1, qft_output_wires, mod, work_wire), control=x_wires),
            control=y_wires,
        ),
    )


def _out_multiplier_with_adder_resources(num_output_wires, num_x_wires, num_y_wires, **_) -> dict:
    """Resources for OutMultiplier decomposition with controlled adders."""
    n = num_x_wires
    m = num_y_wires
    k = num_output_wires

    resources = defaultdict(int)
    for i in range(min(k, n)):
        size = min(k - i, m + 1)
        resources[
            controlled_resource_rep(
                base_class=SemiAdder,
                base_params={"num_y_wires": size},
                num_control_wires=1,
                num_zero_control_values=0,
            )
        ] += 1
    return dict(resources)


def _out_multiplier_with_adder_condition(
    num_output_wires, num_y_wires, mod, num_work_wires, zeroed_output_wires, **_
):
    k = num_output_wires
    m = num_y_wires
    # Controlled adder takes as many work wires as the output register size. The largest controlled
    # adder is the first one in the loop, with size min(k, m+1)
    min_num_work_wires = min(k, m + 1)
    return (
        mod in (None, 2**num_output_wires)
        and num_work_wires >= min_num_work_wires
        and zeroed_output_wires
    )


@register_condition(_out_multiplier_with_adder_condition)
@register_resources(_out_multiplier_with_adder_resources)
def _out_multiplier_with_adder(
    x_wires: WiresLike,
    y_wires: WiresLike,
    output_wires: WiresLike,
    mod,
    work_wires: WiresLike,
    zeroed_output_wires: bool,
    **__,
):  # pylint: disable=unused-argument, too-many-arguments
    """We add the y register to the output register, controlled by one bit in the x register,
    and shifted onto the output register by the same shift as the control qubit."""
    m = len(y_wires)
    k = len(output_wires)
    for i, x_wire in enumerate(x_wires[::-1][:k]):
        # Slice the output wires according to the shift in control, and bounded by its own size,
        # and the size of the y_wires
        out_wires = output_wires[max(0, k - (m + 1 + i)) : k - i]
        # Add y wires to shifted output, controlled by current x_wire
        ctrl(SemiAdder(y_wires, out_wires, work_wires=work_wires), control=x_wire)


def _out_multiplier_with_caddsub_resources(
    num_output_wires, num_x_wires, num_y_wires, num_work_wires, **_
) -> dict:
    n = num_x_wires
    m = num_y_wires
    k = num_output_wires + 1  # augmented output register

    resources = defaultdict(int)

    # Controlled add-subtract loop
    loop_size = min(k, n)
    if num_y_wires > 1:
        resources[
            controlled_resource_rep(
                BasisState,
                base_params={"num_wires": num_y_wires - 1},
                num_control_wires=1,
                num_zero_control_values=1,
            )
        ] += (
            2 * loop_size
        )
    resources[
        controlled_resource_rep(X, base_params={}, num_control_wires=1, num_zero_control_values=1)
    ] += (2 * loop_size)
    resources[
        controlled_resource_rep(X, base_params={}, num_control_wires=1, num_zero_control_values=1)
    ] += 2 * min(n, k - 1)
    for i in range(loop_size):
        size = min(k - i, m + 1)
        resources[resource_rep(SemiAdder, num_y_wires=size)] += 1

    # Add 2^m(x+1)
    resources[resource_rep(SemiAdder, num_y_wires=k - m)] += 1
    # bit flips corresponding to input carry activated. Accounts for the fact that
    # we don't need to flip a work wire if k=m+1, in which case there are no work wires.
    has_work_wires = int(k > m + 1)
    resources[resource_rep(X)] += 4 + has_work_wires
    # The work wire reset bit flip is done via measurement
    if has_work_wires:
        resources[resource_rep(MidMeasure)] += 1

    # Subtract y+2^(n+m)
    # First negation
    resources[resource_rep(X)] += k
    # Add y
    resources[resource_rep(SemiAdder, num_y_wires=k)] += 1
    # increment 2^(n+m) bit
    size = k - n - m
    if size > 0:
        for i in range(1, size):
            resources[
                resource_rep(
                    MultiControlledX,
                    num_control_wires=i,
                    num_zero_control_values=0,
                    num_work_wires=num_work_wires - 1,
                    work_wire_type="zeroed",
                )
            ] += 1
        resources[resource_rep(X)] += 1

    # Second negation
    resources[resource_rep(X)] += k

    # Add 2^n y
    if k > n:
        resources[resource_rep(SemiAdder, num_y_wires=k - n)] += 1

    return dict(resources)


def _out_multiplier_with_caddsub_condition(
    num_output_wires, mod, num_work_wires, zeroed_output_wires, **_
):
    # Adder sizes are (using n=num_x_wires, m=num_y_wires, k=num_output_wires+1):
    # - min(k, m+1) # Largest size occurring in CAddSub loop
    # - k-m, # Add 2^m(x+1)
    # - k, # Add y during subtracting 2^(n+m)+y     <-- Largest one
    # - k-n, # Add 2^n y
    largest_adder_size = num_output_wires + 1
    # One work wire for temporarily enlarged output register. Adder takes size-1 work wires.
    min_num_work_wires = 1 + (largest_adder_size - 1)
    return (
        mod in (None, 2**num_output_wires)
        and num_work_wires >= min_num_work_wires
        and zeroed_output_wires
    )


def _add_plus_one(x_wires, y_wires, work_wires):
    """This qfunc implements ``(x, y, 0) -> (x, (x+y+1) % 2**m, 0)`` for ``m`` the number of
    bits in ``y``. Note that it will produce the right behaviour in a circuit when decomposing
    the right elbows into measurement + CZ, but it will not yield the correct behaviour
    when using the decomposition into unitary operations. We need to resolve this somehow.
    """
    work_wires = work_wires[: len(y_wires) - 1]
    X(x_wires[-1])
    X(y_wires[-1])
    if work_wires:
        X(work_wires[-1])
    SemiAdder(x_wires, y_wires, work_wires)
    X(x_wires[-1])
    X(y_wires[-1])
    if work_wires:
        # In principle, we could just apply a bit flip here to reset the work wire.
        # However, the SemiAdder uses a decomposition with a right elbow, which has an assumption
        # about its output state attached to it, and we violate this assumption here.
        # In order to obtain a correct behaviour both for unitary decompositions of the right elbow
        # and for its implementation relying on the assumption, we replace the bit flip with
        # a measurement + reset.
        measure(work_wires[-1], reset=True)


def _increment(wires, work_wires):
    _ = [
        MultiControlledX(wires[::-1][:i], work_wires=work_wires, work_wire_type="zeroed")
        for i in range(len(wires), 1, -1)
    ]
    X(wires[-1])


def _c_add_sub(c_wire, x_wires, y_wires, work_wires):
    if len(x_wires) > 1:
        ctrl(BasisState([1] * (len(x_wires) - 1), x_wires[:-1]), control=c_wire, control_values=[0])

    work_wires = work_wires[: len(y_wires) - 1]
    ctrl(X(y_wires[-1]), control=c_wire, control_values=[0])
    if work_wires:
        ctrl(X(work_wires[-1]), control=c_wire, control_values=[0])
    SemiAdder(x_wires, y_wires, work_wires)
    ctrl(X(y_wires[-1]), control=c_wire, control_values=[0])
    if work_wires:
        ctrl(X(work_wires[-1]), control=c_wire, control_values=[0])

    if len(x_wires) > 1:
        ctrl(BasisState([1] * (len(x_wires) - 1), x_wires[:-1]), control=c_wire, control_values=[0])


@register_condition(_out_multiplier_with_caddsub_condition)
@register_resources(_out_multiplier_with_caddsub_resources)
def _out_multiplier_with_caddsub(
    x_wires: WiresLike,
    y_wires: WiresLike,
    output_wires: WiresLike,
    mod: None,
    work_wires: WiresLike,
    zeroed_output_wires: bool,
    **__,
):  # pylint: disable=unused-argument, too-many-arguments
    """We add the y register to the output register, controlled by one bit in the x register,
    and shifted onto the output register by the same shift as the control qubit."""
    # We extend our output by one wire because we need to
    # store 2x*y intermediately, instead of x*y.
    output_wires = output_wires + [work_wires[0]]
    # The other work wires can be used for arithmetic building blocks
    work_wires = work_wires[1:]
    n = len(x_wires)
    m = len(y_wires)
    k = len(output_wires)

    # Controlled add-subtract loop
    for i, x_wire in enumerate(x_wires[::-1][:k]):
        # Slice the output wires according to the shift in control, and bounded by its own size,
        # and the size of the y_wires.
        output = output_wires[max(0, k - (m + 1 + i)) : k - i]
        _c_add_sub(x_wire, y_wires, output, work_wires)

    # Add 2^m(x+1)
    _add_plus_one(x_wires, output_wires[: k - m], work_wires)

    # Implement |y> |z> -> |y> |z-2^(n+m)-y>, i.e. subtract 2^(n+m)+y in four steps:
    # - Negate z: |y> |z> -> |y> |2^k-1-z>
    # - Add y: |y> |2^k-1-z> -> |y> |2^k-1-z+y>
    # - Add 2^(n+m) by incrementing the (k-(n+m)) most significant bits
    #   |y> |2^k-1-z+y> -> |y> |2^k-1-z+y+2^(n+m)>
    # - Negate z again: |y> |2^k-1-z+y+2^(n+m)> -> |y> |z-y-2^(n+m)>
    # The third step only is needed if k>n+m, otherwise those bits to increment do not exist.
    _ = [X(w) for w in output_wires]
    SemiAdder(y_wires, output_wires, work_wires)
    if k > n + m:
        increment_wires = output_wires[: k - n - m]
        _increment(increment_wires, work_wires)
    _ = [X(w) for w in output_wires]

    # Add 2^n y if 2^k > 2^n (otherwise it just vanishes in the modulus)
    if k > n:
        SemiAdder(y_wires, output_wires[: k - n], work_wires)


add_decomps(
    OutMultiplier,
    _out_multiplier_with_qft,
    _out_multiplier_with_adder,
    _out_multiplier_with_caddsub,
)
