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
from pennylane.ops import SWAP, MultiControlledX, X, Z, change_op_basis, ctrl
from pennylane.templates.subroutines.arithmetic import CAddSub, SemiAdder
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
                qml.OutMultiplier(x_wires, y_wires, output_wires, mod, work_wires)
                return qml.sample(wires=output_wires)

        >>> print(circuit())
        [[0 0 1 1]]

        The result :math:`[[0 0 1 1]]`, is the binary representation of
        :math:`2 \cdot 7 + 1\; \text{modulo} \; 12 = 3`.

        The fourth set of wires is ``work_wires`` which consist of the auxiliary qubits used to perform the modular multiplication operation.

        - If :math:`mod = 2^{\text{len(output_wires)}}`, there will be no need for ``work_wires``, hence ``work_wires=()``. This is the case by default.

        - If :math:`mod \neq 2^{\text{len(output_wires)}}`, two ``work_wires`` have to be provided.

        Note that the ``OutMultiplier`` template allows us to perform modular multiplication in the computational basis. However if one just wants to perform
        standard multiplication (with no modulo), that would be equivalent to setting the modulo :math:`mod` to a large enough value to ensure that :math:`x \cdot k < mod`.

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

        - The third implementation uses :class:`~.CAddSub`\ s to replace the controllled
          ``SemiAdder``\ s from the previous implementation, based on
          `arXiv:2410.00899 <https://arxiv.org/abs/2410.00899>`__.
          For :math:`n` ``x_wires``, :math:`m` ``y_wires`` and :math:`k`
          ``output_wires``, we need :math:`L=\min(k, n)` controlled add/subtract operations
          of usually varying size :math:`\min(k + 1 - i, m + 1)` for :math:`0\leq i<L,
          three ``SemiAdder``\ s of sizes :math:`\min(k + 1 - m, n + 1)`,
          :math:`\min(k + 1 - n, m + 1)` and :math:`k+1`, as well as an incrementer on
          :math:`\min(k + 1, n + m)` qubits and Pauli gates.

    """

    grad_method = None

    resource_keys = {"num_output_wires", "num_x_wires", "num_y_wires", "num_work_wires", "mod"}

    def __init__(
        self,
        x_wires: WiresLike,
        y_wires: WiresLike,
        output_wires: WiresLike,
        mod=None,
        work_wires: WiresLike = (),
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
        x_wires: WiresLike, y_wires: WiresLike, output_wires: WiresLike, mod, work_wires: WiresLike
    ):  # pylint: disable=arguments-differ
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
            work_wire = work_wires[1:]
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
    num_output_wires, num_x_wires, num_y_wires, mod, num_work_wires
) -> dict:
    # pylint: disable=unused-argument
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
    return mod in (None, 2**num_output_wires) or num_work_wires >= 1


@register_condition(_out_multiplier_with_qft_condition)
@register_resources(_out_multiplier_with_qft_resources)
def _out_multiplier_with_qft(
    x_wires: WiresLike,
    y_wires: WiresLike,
    output_wires: WiresLike,
    mod,
    work_wires: WiresLike,
    **__,
):
    if mod != 2 ** len(output_wires):
        qft_output_wires = work_wires[:1] + output_wires
        work_wire = work_wires[1:]
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


def _out_multiplier_with_adder_resources(
    num_output_wires, num_x_wires, num_y_wires, num_work_wires, mod
) -> dict:
    # pylint: disable=unused-argument

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


def _out_multiplier_with_adder_condition(num_output_wires, num_y_wires, mod, num_work_wires, **_):
    k = num_output_wires
    m = num_y_wires
    # Controlled adder takes as many work wires as the output register size. The largest controlled
    # adder is the first one in the loop, with size min(k, m+1)
    min_num_work_wires = min(k, m + 1)
    return mod in (None, 2**num_output_wires) and num_work_wires >= min_num_work_wires


@register_condition(_out_multiplier_with_adder_condition)
@register_resources(_out_multiplier_with_adder_resources)
def _out_multiplier_with_adder(
    x_wires: WiresLike,
    y_wires: WiresLike,
    output_wires: WiresLike,
    mod,
    work_wires: WiresLike,
    **__,
):  # pylint: disable=unused-argument
    """We add the y register to the output register, controlled by one bit in the x register,
    and shifted onto the output register by the same shift as the control qubit."""
    m = len(y_wires)
    k = len(output_wires)
    for i, x_wire in enumerate(x_wires[::-1][:k]):
        # Slice the output wires according to the shift in control, and bounded by its own size,
        # and the size of the y_wires
        out_wires = output_wires[max(0, k - (m + 1 + i)) : k - i]
        # Add y wires to shifted output, controlled by current x_wire
        print(f"{y_wires=}")
        print(f"{out_wires =}")
        ctrl(SemiAdder(y_wires, out_wires, work_wires=work_wires), control=x_wire)


def _out_multiplier_with_caddsub_resources(
    num_output_wires, num_x_wires, num_y_wires, num_work_wires, mod
) -> dict:
    # pylint: disable=unused-argument
    n = num_x_wires
    m = num_y_wires
    k = num_output_wires

    resources = defaultdict(int)

    # multiply with 2 on register of size k+1 takes k SWAPs
    resources[resource_rep(SWAP)] += k

    # Controlled add-subtract loop
    for i in range(min(k, n)):
        size = min(k + 1 - i, m + 1)
        resources[resource_rep(CAddSub, num_y_wires=size)] += 1

    # Add 2^m(x+1)
    size = min(k + 1 - m, n + 1)
    resources[resource_rep(SemiAdder, num_y_wires=size)] += 1
    resources[resource_rep(X)] += 3
    resources[resource_rep(X)] += int(size > 1)
    resources[resource_rep(Z)] += 2

    # Subtract y+2^(n+m)
    # First negation
    resources[resource_rep(X)] += k + 1
    # Add y
    resources[resource_rep(SemiAdder, num_y_wires=k + 1)] += 1
    # increment 2^(n+m) bit
    size = min(k + 1, n + m)
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
    resources[resource_rep(X)] += k + 1

    # Add 2^n y
    size = min(k + 1 - n, m + 1)
    resources[resource_rep(SemiAdder, num_y_wires=size)] += 1

    # divide by two on register of size k+1 takes k SWAPs
    resources[resource_rep(SWAP)] += k
    return dict(resources)


def _out_multiplier_with_caddsub_condition(num_output_wires, mod, num_work_wires, **_):
    # Adder sizes are (using n=num_x_wires, m=num_y_wires, k=num_output_wires):
    # - k+1 - max(0, k+1-(m+1+0)), # Largest size occurring in CAddSub loop
    # - k+1-m - max(0, k+1-(m+n+1)), # Add 2^m(x+1)
    # - k+1, # Add y during subtracting 2^(n+m)+y     <-- Largest one
    # - k+1-n - max(0, k+1-(m+n+1)), # Add 2^n y
    largest_adder_size = num_output_wires + 1
    # One work wire for temporarily enlarged output register. Adder takes size-1 work wires.
    min_num_work_wires = 1 + (largest_adder_size - 1)
    return mod in (None, 2**num_output_wires) and num_work_wires >= min_num_work_wires


def _div_by_two(wires):
    _ = [SWAP(pair) for pair in zip(wires[:-1], wires[1:])]


def _mul_with_two(wires):
    wires = wires[::-1]
    _ = [SWAP(pair) for pair in zip(wires[:-1], wires[1:])]


def _increment(wires, work_wires):
    _ = [
        MultiControlledX(wires[::-1][:i], work_wires=work_wires, work_wire_type="zeroed")
        for i in range(len(wires), 1, -1)
    ]
    X(wires[-1])


def _add_plus_one(x_wires, y_wires, work_wires):
    """This qfunc implements ``(x, y, 0) -> (x, (x+y+1) % 2**m, 0)`` for ``m`` the number of
    bits in ``y``. Note that it will produce the right behaviour in a circuit when decomposing
    the right elbows into measurement + CZ, but it will not yield the correct behaviour
    when using the decomposition into unitary operations. We need to resolve this somehow.
    """
    work_wires = work_wires[: len(y_wires) - 1]
    # X(x_wires[-1])
    Z(x_wires[-1])
    X(y_wires[-1])
    if work_wires:
        X(work_wires[-1])
    SemiAdder(x_wires, y_wires, work_wires)
    Z(x_wires[-1])
    X(x_wires[-1])
    X(y_wires[-1])


@register_condition(_out_multiplier_with_caddsub_condition)
@register_resources(_out_multiplier_with_caddsub_resources)
def _out_multiplier_with_caddsub(
    x_wires: WiresLike,
    y_wires: WiresLike,
    output_wires: WiresLike,
    mod,
    work_wires: WiresLike,
    **__,
):  # pylint: disable=unused-argument
    """We add the y register to the output register, controlled by one bit in the x register,
    and shifted onto the output register by the same shift as the control qubit."""
    n = len(x_wires)
    m = len(y_wires)
    k = len(output_wires)
    output_with_cache = output_wires + [work_wires[0]]
    work_wires = work_wires[1:]

    # Multiply by two
    _mul_with_two(output_with_cache)
    # Controlled add-subtract loop
    for i, x_wire in enumerate(x_wires[::-1][:k]):
        # Slice the output wires according to the shift in control, and bounded by its own size,
        # and the size of the y_wires.
        output = output_with_cache[max(0, k + 1 - (m + 1 + i)) : k + 1 - i]
        # Add y wires to shifted output, controlled by current x_wire
        CAddSub(x_wire, y_wires, output, work_wires)
    # Add 2^m(x+1)
    _add_plus_one(x_wires, output_with_cache[max(0, k + 1 - (m + n + 1)) : k + 1 - m], work_wires)
    # Implement |y> |z> -> |y> |z-2^(n+m)-y>, i.e. subtract 2^(n+m)+y
    _ = [X(w) for w in output_with_cache]
    SemiAdder(y_wires, output_with_cache, work_wires)
    increment_wires = output_with_cache[max(0, k + 1 - n - m) :]
    _increment(increment_wires, work_wires)
    _ = [X(w) for w in output_with_cache]
    # Add 2^n y
    SemiAdder(y_wires, output_with_cache[max(0, k + 1 - (n + m + 1)) : k + 1 - n], work_wires)
    # Divide by two
    _div_by_two(output_with_cache)


add_decomps(
    OutMultiplier,
    _out_multiplier_with_qft,
    _out_multiplier_with_adder,
    _out_multiplier_with_caddsub,
)
