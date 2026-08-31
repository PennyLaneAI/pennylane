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
from itertools import combinations

from pennylane import math
from pennylane.core.operator import Operator2, abstractify
from pennylane.core.queuing import AnnotatedQueue, QueuingManager, apply
from pennylane.decomposition import (
    add_decomps,
    register_condition,
    register_resources,
)
from pennylane.decomposition.resources import resource_rep
from pennylane.ops import BasisState, H, Prod, X, adjoint, change_op_basis, ctrl, prod
from pennylane.ops.op_math.change_op_basis import _change_op_basis_abstract
from pennylane.typing import AbstractWires, Bool, Wire
from pennylane.wires import Wires, WiresLike

from ..controlled_sequence import ControlledSequence
from ..qft import QFT
from .incrementer import Incrementer
from .phase_adder import PhaseAdder
from .semi_adder import SemiAdder, _semi_adder, _semi_adder_resources
from .temporary_and import TemporaryAND


def _resolve_mod_and_num_work_wires(num_output_wires, mod, num_work_wires):
    """Resolve default ``mod`` and truncated work wire count."""
    max_mod = 2**num_output_wires
    if mod is None:
        mod = max_mod
    elif mod != max_mod:
        num_work_wires = 2  # After the ≥2 work-wire guard in __init__/__abstract_init__, the truncated count is always 2
    return mod, num_work_wires


class OutMultiplier(Operator2):
    r"""Performs the out-place modular multiplication operation.

    This operator performs the modular multiplication of integers :math:`x` and :math:`y` modulo
    :math:`mod` in the computational basis:

    .. math::
        \text{OutMultiplier}(mod) |x \rangle |y \rangle |z \rangle = |x \rangle |y \rangle |z + x \cdot y \; \text{mod} \; mod \rangle,

    There are three implementations available, which differ in the auxiliary wires
    and in the gate counts they require, and in whether or not they support arbitrary values for
    the modulus ``mod``. See the usage details for more information.

    .. note::

        To obtain the correct result, :math:`x`, :math:`y` and :math:`z` must be smaller than :math:`mod`.

    .. seealso:: :class:`~.Multiplier`, :class:`~.SemiAdder`, and :class:`~.PhaseAdder`.

    Args:
        x_wires (Sequence[int]): wires that store the integer :math:`x`
        y_wires (Sequence[int]): wires that store the integer :math:`y`
        output_wires (Sequence[int]): wires that store the multiplication result. If the
            register is in a non-zero state :math:`z`, the solution will be added to this value
        mod (int): the modulo for performing the multiplication. If not provided, it will be set
            to its maximum value, :math:`2^{\text{len(output_wires)}}`
        work_wires (Sequence[int]): auxiliary wires to use for the multiplication. The needed
            number of work wires depends on the decomposition, the register sizes and
            ``output_wires_zeroed``. Defaults to an empty tuple, i.e., no work wires.
        output_wires_zeroed (bool): Whether the ``output_wires`` are guaranteed to be in state
            :math:`|0\rangle` initially. Setting this argument to ``True`` reduces the cost of
            the operation.

    **Example**

    This example performs the multiplication of two integers :math:`x=2` and :math:`y=7` modulo
    :math:`mod=12`. We'll let :math:`z=0`. See Usage Details for :math:`z \neq 0`.

    .. code-block:: python

        x = 2
        y = 7
        mod = 12

        x_wires = [0, 1]
        y_wires = [2, 3, 4]
        output_wires = [6, 7, 8, 9]
        work_wires = [5, 10]

        dev = qp.device("default.qubit")

        @qp.qnode(dev, shots=1)
        def circuit():
            x_bin = qp.math.int_to_binary(x, len(x_wires))
            y_bin = qp.math.int_to_binary(y, len(y_wires))
            qp.BasisState(x_bin, wires=x_wires)
            qp.BasisState(y_bin, wires=y_wires)
            qp.OutMultiplier(x_wires, y_wires, output_wires, mod, work_wires)
            return qp.sample(wires=output_wires)

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
        to encode the integer :math:`(z+ x \cdot y) \; \text{mod} \; mod` in the computational
        basis. Therefore, it will require at least :math:`\lceil \log_2(mod)\rceil` wires
        Note that these wires can be initialized with any integer :math:`z < mod`.

        The fourth register is ``work_wires`` containing the auxiliary qubits used to
        perform the modular multiplication operation. The number of auxiliary wires determines
        which decomposition is available (also see below).

        **Initial state of output wires**

        As indicated above, the initial state of ``output_wires`` can encode any value
        :math:`z<mod`. The following is an example for :math:`z = 1`.

        .. code-block:: python

            z = 1
            x = 2
            y = 7
            mod = 12

            x_wires = [0, 1]
            y_wires = [2, 3, 4]
            output_wires = [6, 7, 8, 9]
            work_wires = [5, 10]

            dev = qp.device("default.qubit")

            @qp.qnode(dev, shots=1)
            def circuit():
                x_bin = qp.math.int_to_binary(x, len(x_wires))
                qp.BasisState(x_bin, wires=x_wires)
                y_bin = qp.math.int_to_binary(y, len(y_wires))
                qp.BasisState(y_bin, wires=y_wires)
                z_bin = qp.math.int_to_binary(z, len(output_wires))
                qp.BasisState(z_bin, wires=output_wires)
                qp.OutMultiplier(x_wires, y_wires, output_wires, mod, work_wires)
                return qp.sample(wires=output_wires)

        >>> print(circuit())
        [[0 0 1 1]]

        The result :math:`(0011)_2`, is the binary representation of
        :math:`(1 + 2 \cdot 7)\; \text{modulo} \; 12 = 3`:

        If the initial state on the output wires is guaranteed to be :math:`|0\rangle`, this
        can be indicated to ``OutMultiplier`` by setting ``output_wires_zeroed=True``. This
        simplifies some of the available decompositions (also see below), saving quantum resources.

        **Different decompositions**

        There are three decompositions, which differ in the required number of work wires and
        gates, and in whether they support ``mod!=2**len(output_wires)``.

        - The first implementation is based on the quantum Fourier transform (QFT) method presented
          in `arXiv:2311.08555 <https://arxiv.org/abs/2311.08555>`_. We nest
          :class:`~.ControlledSequence` around :class:`~.PhaseAdder` to create doubly-controlled
          in place phase addition in the output register, which is then transformed into
          addition by a basis change using :class:`~.QFT`\ s. It requires zero (two) auxiliary
          wires for ``mod=2**len(output_wires)`` (for other values of ``mod``)
          Any value for ``mod`` is supported, subject to the description above.

        - The second implementation uses controlled :class:`~.SemiAdder`\ s to realize the
          multiplication. For :math:`n` ``x_wires``, :math:`m` ``y_wires`` and :math:`k`
          ``output_wires``, we need :math:`L = \min(k, n)` adders with usually varying sizes
          :math:`\min(k - i, m + 1)` for :math:`0\leq i<L`. The implementation is shown in
          Fig. 2a) (for :math:`k=2m=2n`) and Fig. 2c) (for :math:`k=m=n`) in
          `arXiv:2410.00899 <https://arxiv.org/abs/2410.00899>`__.

        - The third implementation uses controlled addition/subtraction to replace the controlled
          ``SemiAdder``\ s from the previous implementation, based on Litinski's
          `arXiv:2410.00899 <https://arxiv.org/abs/2410.00899>`__.
          For :math:`n` ``x_wires``, :math:`m` ``y_wires`` and :math:`k`
          ``output_wires``, we need :math:`L=\min(k, n)` controlled add/subtract operations
          of usually varying size :math:`\min(k + 1 - i, m + 1)` for :math:`0\leq i<L`,
          three ``SemiAdder``\ s of sizes :math:`\min(k + 1 - m, n + 1)`,
          :math:`\min(k + 1 - n, m + 1)` and :math:`k+1`, as well as an incrementer on
          :math:`\min(k + 1, n + m)` qubits and Pauli gates. For :math:`n=m` and
          :math:`k=2n`, this implementation is shown in Fig. 2b), for :math:`k=n=m` in Fig. 2d).

    """

    wire_argnames = ("x_wires", "y_wires", "output_wires", "work_wires")
    compilable_argnames = ("mod", "output_wires_zeroed")

    arg_specs = {
        "x_wires": Wire[-1],
        "y_wires": Wire[-1],
        "output_wires": Wire[-1],
        "work_wires": Wire[-1],
    }

    def __init__(
        self,
        x_wires: WiresLike,
        y_wires: WiresLike,
        output_wires: WiresLike,
        mod=None,
        work_wires: WiresLike = (),
        output_wires_zeroed: bool = False,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments

        x_wires = Wires(x_wires)
        y_wires = Wires(y_wires)
        output_wires = Wires(output_wires)
        work_wires = Wires([] if work_wires is None else work_wires)
        num_output_wires = len(output_wires)
        num_work_wires = len(work_wires)
        max_mod = 2**num_output_wires

        if mod is not None and mod != max_mod:
            if num_work_wires < 2:
                raise ValueError(
                    f"If mod is not 2^{num_output_wires}, at least two work wires should be provided."
                )
            if mod > max_mod:
                raise ValueError(
                    "OutMultiplier must have enough wires to represent mod. The maximum mod "
                    f"with len(output_wires)={num_output_wires} is {max_mod}, but received {mod}."
                )

        mod, num_work_wires = _resolve_mod_and_num_work_wires(num_output_wires, mod, num_work_wires)
        if mod != max_mod:
            work_wires = Wires(work_wires[:num_work_wires])

        wires_list = [x_wires, y_wires, output_wires, work_wires]
        wires_name = ["x_wires", "y_wires", "output_wires", "work_wires"]

        _wires_are_traced = any(math.is_abstract(w) for ws in wires_list for w in ws)

        if not _wires_are_traced:
            wires_dict = dict(zip(wires_name, wires_list, strict=True))
            for name0, name1 in combinations(wires_name, r=2):
                if wires_dict[name0].intersection(wires_dict[name1]):
                    raise ValueError(f"None of the wires in {name1} should be included in {name0}.")

        super().__init__(
            x_wires,
            y_wires,
            output_wires,
            mod=mod,
            work_wires=work_wires,
            output_wires_zeroed=output_wires_zeroed,
        )

    # pylint: disable=arguments-differ
    def __abstract_init__(
        self,
        x_wires: AbstractWires | WiresLike,
        y_wires: AbstractWires | WiresLike,
        output_wires: AbstractWires | WiresLike,
        mod=None,
        work_wires: AbstractWires | WiresLike = (),
        output_wires_zeroed: bool = False,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        num_output_wires = len(output_wires)
        num_work_wires = len(work_wires)
        max_mod = 2**num_output_wires

        if mod is not None and mod != max_mod:
            if num_work_wires < 2:
                raise ValueError(
                    f"If mod is not 2^{num_output_wires}, at least two work wires should be provided."
                )
            if mod > max_mod:
                raise ValueError(
                    "OutMultiplier must have enough wires to represent mod. The maximum mod "
                    f"with len(output_wires)={num_output_wires} is {max_mod}, but received {mod}."
                )

        mod, num_work_wires = _resolve_mod_and_num_work_wires(num_output_wires, mod, num_work_wires)
        if mod != 2**num_output_wires:
            work_wires = Wire[num_work_wires]

        super().__abstract_init__(
            x_wires,
            y_wires,
            output_wires,
            mod=mod,
            work_wires=work_wires,
            output_wires_zeroed=output_wires_zeroed,
        )

    @property
    def wires(self):
        """All wires involved in the operation."""
        return self.x_wires + self.y_wires + self.output_wires + self.work_wires


def _out_multiplier_with_qft_resources(
    x_wires, y_wires, output_wires, mod, work_wires, output_wires_zeroed=False
):  # pylint: disable=too-many-arguments,unused-argument
    num_output_wires = len(output_wires)
    num_x_wires = len(x_wires)
    num_y_wires = len(y_wires)
    num_qft_wires = num_output_wires + 1 if mod != 2**num_output_wires else num_output_wires

    if output_wires_zeroed:
        compute_rep = resource_rep(Prod, resources={abstractify(H): num_qft_wires})
    else:
        compute_rep = QFT(Wire[num_qft_wires])

    uncompute_rep = adjoint(QFT(Wire[num_qft_wires]))
    target_rep = resource_rep(
        ControlledSequence,
        base_rep=resource_rep(
            ControlledSequence,
            base_rep=resource_rep(PhaseAdder, num_x_wires=num_qft_wires, mod=mod),
            num_control_wires=num_x_wires,
        ),
        num_control_wires=num_y_wires,
    )
    return {
        _change_op_basis_abstract(
            abstractify(compute_rep), abstractify(target_rep), abstractify(uncompute_rep)
        ): 1
    }


def _out_multiplier_with_qft_condition(
    x_wires, y_wires, output_wires, mod, work_wires, output_wires_zeroed=False
):  # pylint: disable=unused-argument, too-many-arguments
    return mod == 2 ** len(output_wires) or len(work_wires) >= 2


@register_condition(_out_multiplier_with_qft_condition)
@register_resources(_out_multiplier_with_qft_resources)
def _out_multiplier_with_qft(
    x_wires: WiresLike,
    y_wires: WiresLike,
    output_wires: WiresLike,
    mod,
    work_wires: WiresLike,
    output_wires_zeroed: bool,
):  # pylint: disable=too-many-arguments, unused-argument
    if mod != 2 ** len(output_wires):
        qft_output_wires = work_wires[:1] + output_wires
        work_wire = work_wires[1:2]
    else:
        qft_output_wires = output_wires
        work_wire = ()

    if output_wires_zeroed:
        compute_op = prod(*(H(w) for w in qft_output_wires))
    else:
        compute_op = QFT(qft_output_wires)
    uncompute_op = adjoint(QFT)(qft_output_wires)

    target_op = ControlledSequence(
        ControlledSequence(PhaseAdder(1, qft_output_wires, mod, work_wire), control=x_wires),
        control=y_wires,
    )
    change_op_basis(compute_op, target_op, uncompute_op)


def _out_multiplier_with_adder_resources(
    x_wires, y_wires, output_wires, mod, work_wires, output_wires_zeroed=False
):  # pylint: disable=too-many-arguments,unused-argument
    """Resources for OutMultiplier decomposition with controlled adders."""
    n = len(x_wires)
    m = len(y_wires)
    k = len(output_wires)
    num_work_wires = len(work_wires)

    resources = defaultdict(int)
    if output_wires_zeroed:
        resources[TemporaryAND] += min(m, k)

    for i in range(int(output_wires_zeroed), min(k, n)):
        if output_wires_zeroed:
            size = min(k - i, m + 1)
        else:
            size = k - i
        resources[ctrl(SemiAdder(Wire[m], Wire[size], Wire[num_work_wires]), Wire[1])] += 1
    return dict(resources)


def _out_multiplier_with_adder_condition(
    x_wires, y_wires, output_wires, mod, work_wires, output_wires_zeroed=False
):  # pylint: disable=unused-argument, too-many-arguments
    k = len(output_wires)
    m = len(y_wires)
    # Controlled adder takes as many work wires as the output register size. The largest controlled
    # adder is the first one in the loop, with size `min(k - 1, m+1)` if output_wires_zeroed=True
    # (because in that case the very first adder is replaced by ctrl(copy)) and size `k` else.
    if output_wires_zeroed:
        min_num_work_wires = min(k - 1, m + 1)
    else:
        min_num_work_wires = k
    return mod == 2 ** k and len(work_wires) >= min_num_work_wires


@register_condition(_out_multiplier_with_adder_condition)
@register_resources(_out_multiplier_with_adder_resources)
def _out_multiplier_with_adder(
    x_wires: WiresLike,
    y_wires: WiresLike,
    output_wires: WiresLike,
    mod,
    work_wires: WiresLike,
    output_wires_zeroed: bool,
):  # pylint: disable=unused-argument, too-many-arguments
    """Implementation of Schoolbook multiplication via controlled adders as sole building block,
    except for a potential simplification for the very first adder.
    The j-th building block adds y⋅x_{n-1-j}⋅2^j to the output register, by controlling the
    addition of y on x_{n-1-j} and shifting the output wires of the addition by j bits.
    Overall, we thus add

    sum_{j=0}^{n-1} 2^j⋅x_{n-1-j}⋅y = x⋅y

    to the output register. Note that the size of the addition output registers as well as the
    upper limit of the sum are adjusted depending on the sizes n, m, and k of the three registers
    x_wires, y_wires, and output_wires.
    """
    m = len(y_wires)
    k = len(output_wires)

    # If the output wires are zeroed, the first controlled adder is just a controlled copy.
    if output_wires_zeroed:
        # We use strict=False here because we only need to copy for as long as both
        # more y_wires and more output_wires exist. zip(strict=False) produces exactly this bound
        for y_wire, out_wire in zip(
            y_wires[::-1], output_wires[max(0, k - (m + 1)) : k][::-1], strict=False
        ):
            TemporaryAND([x_wires[-1], y_wire, out_wire])

    # If the output wires are zeroed, we already did the first controlled adder above
    start = int(output_wires_zeroed)
    for i, x_wire in enumerate(x_wires[::-1][start:k], start=start):
        # Slice the output wires according to the shift in control, and bounded by its own size,
        # and the size of the y_wires
        if output_wires_zeroed:
            out_wires = output_wires[max(0, k - (m + 1 + i)) : k - i]
        else:
            out_wires = output_wires[: k - i]
        # Add y wires to shifted output, controlled by current x_wire
        ctrl(SemiAdder(y_wires, out_wires, work_wires=work_wires), control=x_wire)


def _out_multiplier_with_caddsub_resources(
    x_wires, y_wires, output_wires, mod, work_wires, output_wires_zeroed=False
):  # pylint: disable=unused-argument,too-many-arguments
    n = len(x_wires)
    m = len(y_wires)
    k = len(output_wires) + 1  # augmented output register
    num_work_wires = len(work_wires)
    num_passed_ww = num_work_wires - 1  # One work wire is used by the arithmetic logic itself.

    resources = defaultdict(int)

    # Controlled add-subtract loop
    for i in range(min(k, n)):
        size = min(k - i, m + 1) if output_wires_zeroed else k - i
        for key, value in _c_add_sub_resources(m, size).items():
            resources[key] += value

    # Add 2^m(x+1)
    if k > m:
        adder_resources = _semi_adder_resources(Wire[n], Wire[k - m])
        for key, value in adder_resources.items():
            resources[key] += value
        # bit flips corresponding to input carry activated. Accounts for the fact that
        # we don't need to flip a work wire if k=m+1, in which case there are no work wires.
        has_work_wires = int(k > m + 1)
        resources[X] += 4 + 2 * has_work_wires

    # Subtract y+2^(n+m)
    # First negation
    resources[X] += k
    # Add y
    add_rep = SemiAdder(Wire[m], Wire[k], Wire[num_passed_ww])
    resources[add_rep] += 1

    # increment 2^(n+m) bit
    if k > n + m:
        size = k - n - m
        resources[resource_rep(Incrementer, num_wires=size, num_work_wires=num_work_wires - 1)] = 1

    # Second negation
    resources[X] += k

    # Add 2^n y
    if k > n:
        resources[SemiAdder(Wire[m], Wire[k - n], Wire[num_passed_ww])] += 1

    return dict(resources)


def _out_multiplier_with_caddsub_condition(  # pylint: disable=too-many-arguments
    x_wires, y_wires, output_wires, mod, work_wires, output_wires_zeroed=False
):  # pylint: disable=unused-argument
    # Adder sizes are (using n=len(x_wires), m=len(y_wires), k=len(output_wires)+1):
    # - min(k, m+1) # Largest size occurring in controlled add/sub loop
    # - k-m, # Add 2^m(x+1)
    # - k, # Add y during subtracting 2^(n+m)+y     <-- Largest one
    # - k-n, # Add 2^n y
    largest_adder_size = len(output_wires) + 1
    # One work wire for temporarily enlarged output register. Adder takes size-1 work wires.
    min_num_work_wires = 1 + (largest_adder_size - 1)
    return mod == 2 ** len(output_wires) and len(work_wires) >= min_num_work_wires


def _adder_flipped_first_work_wire(x_wires, y_wires, work_wires, flip_control=None):
    """SemiAdder decomposition with bit flips on the last work wire inserted after the first
    left elbow and before the last right elbow of the adder.

    If flip_control is provided, the bit flips are controlled accordingly, the adder part remains
    unchanged. We only expect this function to be used with two values for `flip_control`: None
    or a tuple ``(c_wire, c_val)`` for a single control wire.
    """

    with AnnotatedQueue() as q:
        _semi_adder(x_wires, y_wires, work_wires)
    adder_ops = q.queue
    if work_wires:
        # We insert work wire bit flips where a carry-in qubit would cause them,
        # i.e., after the very first left elbow and before the last right elbow
        with QueuingManager.stop_recording():
            if flip_control is None:
                work_wire_flip = X(work_wires[-1])
            else:
                c_wire, c_val = flip_control
                work_wire_flip = ctrl(X(work_wires[-1]), control=c_wire, control_values=[c_val])
        adder_ops.insert(1, work_wire_flip)
        adder_ops.insert(-2, work_wire_flip)
    if QueuingManager.recording():
        for op in adder_ops:
            apply(op)


def _add_plus_one(x_wires, y_wires, work_wires):
    """This qfunc implements ``(x, y, 0) -> (x, (x+y+1) % 2**m, 0)`` for ``m`` the number of
    bits in ``y``. This circuit is similar to the one shown in Fig. 1c) in
    `arXiv:2410.00899 <https://arxiv.org/abs/2410.00899>`__, just without the bit flips on the
    ``x_wires`` before and after the adder. We replace the explicit input carry in that figure
    by bit flips on the least significant bits of all three registers, the bit flip on the work
    wire occurring after the first left elbow/before the last right elbow (this insertion
    is done by _adder_flipped_first_work_wire).
    """
    work_wires = work_wires[: len(y_wires) - 1]
    X(x_wires[-1])
    X(y_wires[-1])
    _adder_flipped_first_work_wire(x_wires, y_wires, work_wires)
    X(y_wires[-1])
    X(x_wires[-1])


def _c_add_sub_resources(num_x_wires, num_y_wires):
    """Resources for _c_add_sub."""
    resources = defaultdict(int)
    if num_x_wires > 1:
        ctrl_basis_rep = ctrl(BasisState(Bool[num_x_wires - 1], Wire[num_x_wires - 1]), Wire[1])
        resources[ctrl_basis_rep] += 2

    cnot_on_0_rep = ctrl(X(Wire[1]), control=Wire[1], control_values=[0])
    resources[cnot_on_0_rep] += 2 * (1 + int(num_y_wires > 1))

    for key, value in _semi_adder_resources(Wire[num_x_wires], Wire[num_y_wires]).items():
        resources[key] += value

    return dict(resources)


def _c_add_sub(c_wire, x_wires, y_wires, work_wires):
    r"""Controlled add/subtract operation. If the control wire ``c_wire`` is in the
    state :math:`|1\rangle`, simply adds :math:`x`, the integer stored in ``x_wires``,
    to :math:`y`, the value in ``y_wires``. If the control wire is in
    the state :math:`|0\rangle`, adds :math:`2^n-x` to :math:`y` instead where :math:`n`
    is the length of ``x_wires``. In short:

    |0>|x>|y>  ->  |0>|x>|y+2^n-x>
    |1>|x>|y>  ->  |1>|x>|y+x>

    This is shown in Fig. 1f) in `arXiv:2410.00899 <https://arxiv.org/abs/2410.00899>`__.
    Note that the figure explicitly shows an input carry for the adder, which
    we do not represent here. Instead, we introduce (controlled) bit flips on the least significant
    bits of each register that correspond to an input carry being set to one. The bit flips on
    the least significant work wire occur after the first left elbow/before the last right elbow.
    """
    # We need to control-flip all x_wires in order to achieve subtraction for c_wire=|0>
    # We also need to control-flip the LSB of x_wires (last wire) to achieve addition plus one
    # (c.f. _add_plus_one). The bit flips on the LSB cancel, so that we only control-flip all _but_
    # the LSB
    c_wire = [c_wire]
    if len(x_wires) > 1:
        ctrl(BasisState([1] * (len(x_wires) - 1), x_wires[:-1]), control=c_wire, control_values=[0])

    work_wires = work_wires[: len(y_wires) - 1]
    # Control-flip the LSB of the output register. This is part of achieving addition plus one
    # for c_wire=|0>. (c.f. _add_plus_one).
    ctrl(X(y_wires[-1]), control=c_wire, control_values=[0])

    # Create the operator sequence for an adder and insert (controlled) work wire bit flips
    # We insert controlled work wire bit flips where a carry-in qubit would cause them,
    # i.e., after the very first left elbow and before the last right elbow
    _adder_flipped_first_work_wire(x_wires, y_wires, work_wires, flip_control=(c_wire, 0))

    ctrl(X(y_wires[-1]), control=c_wire, control_values=[0])

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
    output_wires_zeroed: bool,
):  # pylint: disable=unused-argument, too-many-arguments
    """Implementation of improved Schoolbook multiplication via controlled add/subtract blocks,
    combined with some correction steps. After appending a work wire to the output register,
    effectively multiplying it with two, we first have a bulk computation with n steps (where
    n is the size of x_wires):

    The j-th building block adds y⋅(2x_{n-1-j}-1)⋅2^j+2^(j+m)⋅(1-x_{n-1-j}) to the output register,
    by controlling between addition and subtraction of y on x_{n-1-j}, and shifting the output
    wires of the addition by j bits.
    Overall, we thus computed (including the initial multiplication with two)

    2⋅z + sum_{j=0}^{n-1} (2^j⋅(2x_{n-1-j}-1)⋅y + 2^{j+m}⋅(1-x_{n-1-j})
    = 2⋅z + 2⋅x⋅y - (2^n-1)⋅y + 2^{n+m}-2^m⋅(1+x)

    to the output register. Note that the size of the addition output registers as well as the
    upper limit of the sum are adjusted depending on the sizes n, m, and k of the three registers
    x_wires, y_wires, and output_wires.

    Afterwards, we correct for the additional terms in three steps:
    - Add 2^m⋅(x+1)
    - Subtract 2^{n+m}+y
    - Add 2^n⋅y
    We are left with 2⋅(z+x⋅y), an even number, which we can divide by two by splitting off the
    least significant bit (which is exactly the one we appended initially) from the output register.
    """
    # We extend our output by one wire because we need to store 2x*y intermediately, instead
    # of x*y. This also multiplies the value stored in output_wires with two.
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
        output_msb = max(0, k - (m + 1 + i)) if output_wires_zeroed else 0
        output = output_wires[output_msb : k - i]
        _c_add_sub(x_wire, y_wires, output, work_wires)

    # Add 2^m(x+1)
    if k > m:
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
        Incrementer(increment_wires, work_wires)
    _ = [X(w) for w in output_wires]

    # Add (2^n·y) if 2^k > 2^n (otherwise it just vanishes in the modulus)
    if k > n:
        SemiAdder(y_wires, output_wires[: k - n], work_wires)

    # Note that dividing by two does not have to happen explicitly, because the registers are
    # not explicit return values.


def _out_multiplier_with_cache_condition(
    x_wires, y_wires, output_wires, mod, work_wires, output_wires_zeroed=False
):  # pylint: disable=unused-argument, too-many-arguments
    return len(work_wires) >= 2 * len(output_wires) - 1 and not output_wires_zeroed


def _out_multiplier_with_cache_resources(
    x_wires, y_wires, output_wires, mod, work_wires, output_wires_zeroed=False
):  # pylint: disable=unused-argument, too-many-arguments
    num_x_wires = len(x_wires)
    num_y_wires = len(y_wires)
    num_output_wires = len(output_wires)
    new_num_work_wires = len(work_wires) - num_output_wires
    mod, new_num_work_wires = _resolve_mod_and_num_work_wires(
        num_output_wires, mod, new_num_work_wires
    )
    mult_op = OutMultiplier(
        Wire[num_x_wires],
        Wire[num_y_wires],
        Wire[num_output_wires],
        mod=mod,
        work_wires=Wire[new_num_work_wires],
        output_wires_zeroed=True,
    )
    return {
        mult_op: 1,
        SemiAdder(Wire[num_output_wires], Wire[num_output_wires], Wire[new_num_work_wires]): 1,
        adjoint(mult_op): 1,
    }


@register_condition(_out_multiplier_with_cache_condition)
@register_resources(_out_multiplier_with_cache_resources)
def _out_multiplier_with_cache(
    x_wires: WiresLike,
    y_wires: WiresLike,
    output_wires: WiresLike,
    mod: None,
    work_wires: WiresLike,
    output_wires_zeroed,
):  # pylint: disable=unused-argument,too-many-arguments
    r"""Decompose ``OutMultiplier`` with ``output_wires_zeroed=False`` into two ``OutMultiplier``\ s
    with ``output_wires_zeroed=True`` and one ``SemiAdder``, using additional work wires."""
    cache_wires = work_wires[: len(output_wires)]
    work_wires = work_wires[len(output_wires) :]
    OutMultiplier(
        x_wires, y_wires, cache_wires, mod=mod, work_wires=work_wires, output_wires_zeroed=True
    )
    SemiAdder(cache_wires, output_wires, work_wires)
    adjoint(OutMultiplier)(
        x_wires, y_wires, cache_wires, mod=mod, work_wires=work_wires, output_wires_zeroed=True
    )


add_decomps(
    OutMultiplier,
    _out_multiplier_with_qft,
    _out_multiplier_with_adder,
    _out_multiplier_with_caddsub,
    _out_multiplier_with_cache,
)
