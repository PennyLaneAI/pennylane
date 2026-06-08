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
Contains the SignedOutSquare template.
"""

from collections import defaultdict
from itertools import combinations

from pennylane.decomposition import (
    add_decomps,
    register_resources,
    resource_rep,
)
from pennylane.operation import Operation
from pennylane.ops import BasisState, X
from pennylane.queuing import AnnotatedQueue, QueuingManager, apply
from pennylane.templates.subroutines.arithmetic import OutSquare, SemiAdder
from pennylane.wires import Wires, WiresLike

from .semi_adder import _controlled_semi_adder, _controlled_semi_adder_resource


class SignedOutSquare(Operation):
    r"""Performs out-of-place squaring of a signed integer.

    This operator performs the squaring of a signed integer :math:`x` in two's complement
    convention into an unsigned register. The computation is modulo :math:`2^m`,
    where ``m=len(output_wires)``:

    .. math::
        \text{SignedOutSquare} |x \rangle |b \rangle = |x \rangle |(b + x^2) \; \text{mod} \; 2^m \rangle,

    .. seealso:: :class:`~.OutSquare` and :class:`~.OutMultiplier`.

    Args:
        x_wires (WiresLike): wires that store the integer :math:`x`.
        output_wires (WiresLike): wires that store the squaring result. If the
            register is in a non-zero state :math:`b`, the solution will be added to this value.
            If the register is guaranteed to be in the zero state, it is recommended to set
            ``output_wires_zeroed=True`` to reduce the cost of the operation.
        work_wires (WiresLike): the auxiliary wires to use for the squaring.
            ``len(output_wires)`` work wires are required if ``output_wires_zeroed=False``,
            otherwise ``min(len(output_wires), len(x_wires))`` work wires are required.
        output_wires_zeroed (bool): Whether the output wires are guaranteed to be in the state
            :math:`|0\rangle` initially. Defaults to ``False``.

    **Example**

    Let's compute the square of :math:`x=-5` and :math:`x=7` in superposition, added to :math`b=5`
    modulo :math:`2^n=2^6=64`.

    .. code-block:: python

        import pennylane as qp

        x_wires = list(range(4))
        output_wires = list(range(4, 10))
        work_wires = list(range(10, 16))

        dev = qp.device("lightning.qubit", wires=16, seed=295)

        @qp.qnode(dev, shots=1_000)
        def circuit(output_wires):
            # Create a uniform superposition between integers -5 and 7:
            # - Create superposition between -8 and 0
            qp.H(x_wires[0])
            # - Flip 4's bit if sign bit=0 (=> superposition of -8 and 4)
            qp.ctrl(qp.X(x_wires[1]), x_wires[0], control_values=[0])
            # - Add 3 (=> superposition of -5 and 7)
            qp.BasisEmbedding(3, wires=x_wires[2:])
            # Prepare initial state on output wires
            qp.BasisEmbedding(5, wires=output_wires)
            # Signed square
            qp.SignedOutSquare(x_wires, output_wires, work_wires)
            return qp.counts(wires=output_wires)

    >>> counts = circuit(output_wires)
    >>> counts = {int(k, 2): val for k, val in counts.items()}
    >>> print(counts)
    {30: np.int64(498), 54: np.int64(502)}

    We correctly obtain the squared numbers added to :math:`b=5`, namely
    :math:`5+(-5)^2=30` and :math:`5+7^2=54`.

    Note that reducing the size of the output register (here from ``m=6`` to ``m=4``)
    changes the computed numbers via the reduced modulus:

    >>> output_wires = list(range(4, 8))
    >>> counts = circuit(output_wires)
    >>> counts = {int(k, 2): val for k, val in counts.items()}
    >>> print(counts)
    {6: np.int64(501), 14: np.int64(499)}

    The new results are consistent with the previous ones: the smaller output :math:`30` is
    changed to :math:`30\!\mod\!2^4=14` and :math:`54` is changed to :math:`54\!\mod\!2^4=6`.

    Note that the keyword argument ``output_wires_zeroed`` is passed on to the :class:`~.OutSquare`
    used in the decomposition, leading to smaller cost of this main component. See the usage
    details of ``OutSquare`` for details.

    .. details::
        :title: Theoretical background
        :href: theory

        We compute the square of the signed input register in three steps. First, note that
        :math:`x=x_u-2^{n-1}x_s`, where :math:`x_u` is the unsigned part of :math:`x`
        and :math:`x_s` is the sign bit. As example, consider the bit string
        :math:`(1101)_{\text{2's compl}}=(101)_2 - 2^3 = 5-8=-3`. We have :math:`x_u=5`
        and :math:`x_s=1`.

        Consider the binomial expansion of :math:`x^2`:

        .. math::

            x^2 = (x_u-2^{n-1}x_s)^2=x_u^2 - x_s (2^{n}x_u) + 2^{2n-2}x_s^2.
            =x_u^2 + x_s 2^{n} (2^{n-2}-x_u)

        In order to arrive at steps that we can easily compute, we rewrite this as

        .. math::

            x^2 = x_u^2 + x_s 2^n (2^{n-1}-x_u) - x_s 2^{2n-2}

        The three steps in our calculation directly correspond to these three terms. That is,
        we first compute the square of :math:`x_u` into the output register, then we add
        :math:`2^n(2^{n-1}-x_u)`, controlled on the sign bit :math:`x_s`. For this, we simply
        combine some simple bit flips with a controlled :class:`~.SemiAdder`. Finally, we subtract
        :math:`x_s 2^{2n-2}` from the output, by wrapping an
        adder onto the most significant bits in bit flips of the same output subregister.

    """

    resource_keys = {"num_x_wires", "num_output_wires", "num_work_wires", "output_wires_zeroed"}

    def __init__(
        self,
        x_wires: WiresLike,
        output_wires: WiresLike,
        work_wires: WiresLike,
        output_wires_zeroed: bool = False,
    ):

        x_wires = Wires(x_wires)
        output_wires = Wires(output_wires)
        work_wires = Wires(work_wires)

        n = len(x_wires)
        m = len(output_wires)

        # Work wires required for the unsigned square (its input is reduced by one)
        num_required_work_wires = min(n, m) if output_wires_zeroed else m
        # Work wires required for the first correction adder are `min(m-n, n)-1`, which is smaller
        # than `m` and smaller than `n`, so that the unsigned square always needs more work wires.
        if len(work_wires) < num_required_work_wires:
            raise ValueError(
                f"SignedOutSquare requires at least {num_required_work_wires} work wires for "
                f"{n} input wires, {m} output wires and {output_wires_zeroed=}."
                f"Got {len(work_wires)} work wires instead."
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
        all_wires = x_wires + output_wires + work_wires
        super().__init__(wires=all_wires)

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

        return SignedOutSquare(
            new_dict["x_wires"],
            new_dict["output_wires"],
            new_dict["work_wires"],
            self.hyperparameters["output_wires_zeroed"],
        )

    @classmethod
    def _primitive_bind_call(cls, *args, **kwargs):
        return cls._primitive.bind(*args, **kwargs)


def _c_subtract_then_add_one_resources(n, m, num_work_wires, output_wires_zeroed):
    size = min(m - n, n) if output_wires_zeroed else m - n
    add_base_params = {"num_x_wires": n - 1, "num_y_wires": size, "num_work_wires": size - 1}
    cadd_params = {
        "num_control_wires": 1,
        "num_zero_control_values": 0,
        "num_work_wires": num_work_wires - size + 1,
        "work_wire_type": "zeroed",
    }
    cadd_resources = _controlled_semi_adder_resource(add_base_params, SemiAdder, **cadd_params)

    # Bit flips on input register
    if n - 1 > 1:
        basis_rep = resource_rep(BasisState, num_wires=n - 2)
        cadd_resources[basis_rep] = cadd_resources.get(basis_rep, 0) + 2

    # Bit flips on output and work registers
    x_rep = resource_rep(X)
    cadd_resources[x_rep] = cadd_resources.get(x_rep, 0) + (2 + 2 * (num_work_wires > 0))
    return cadd_resources


def _c_subtract_then_add_one(c_wire, x_wires, y_wires, work_wires):
    """Subtract x from y, controlled on c_wire."""
    # Flip input bits (except for the LSB, which would be flipped back by the input carry set)
    if len(x_wires) > 1:
        BasisState([1] * (len(x_wires) - 1), x_wires[:-1])
    # Flip LSB of output register, due to input carry being set
    X(y_wires[-1])

    # Create C(SemiAdder) decomposition and inject work wire bit flips required for input carry
    m = len(y_wires)
    with QueuingManager.stop_recording():
        base = SemiAdder(x_wires, y_wires, work_wires[: m - 1])
    with AnnotatedQueue() as q:
        _controlled_semi_adder(
            base, control_wires=[c_wire], work_wires=work_wires[m - 1 :], work_wire_type="zeroed"
        )
    cadder_ops = q.queue

    if work_wires:
        # We insert work wire bit flips where a carry-in qubit would cause them,
        # i.e., after the very first left elbow and before the last right elbow
        with QueuingManager.stop_recording():
            work_wire_flip = X(work_wires[m - 2])
        cadder_ops.insert(1, work_wire_flip)
        cadder_ops.insert(-2, work_wire_flip)

    if QueuingManager.recording():
        for op in cadder_ops:
            apply(op)

    # Flip LSB of output register, due to input carry being set
    X(y_wires[-1])
    # Flip input bits (except for the LSB, which would be flipped back by the input carry set)
    if len(x_wires) > 1:
        BasisState([1] * (len(x_wires) - 1), x_wires[:-1])


def _signed_out_square_resources(
    num_x_wires, num_output_wires, num_work_wires, output_wires_zeroed
) -> dict:
    # pylint: disable=unused-argument
    n = num_x_wires
    m = num_output_wires
    resources = defaultdict(int)

    size = min(m, 2 * n - 2) if output_wires_zeroed else m
    square_rep = resource_rep(
        OutSquare,
        num_x_wires=n - 1,
        num_output_wires=size,
        num_work_wires=num_work_wires,
        output_wires_zeroed=output_wires_zeroed,
    )
    resources[square_rep] += 1

    if n < m:
        # Add x_s 2^n (2^{n-1}-x)
        _res = _c_subtract_then_add_one_resources(n, m, num_work_wires, output_wires_zeroed)
        for key, value in _res.items():
            resources[key] += value

        if m >= 2 * n - 1:
            # Subtract x_s 2^{2n-2}
            size = min(m - (2 * n - 2), 2) if output_wires_zeroed else m - (2 * n - 2)
            x_rep = resource_rep(X)
            resources[x_rep] += 2 * size
            add_rep = resource_rep(
                SemiAdder, num_x_wires=1, num_y_wires=size, num_work_wires=num_work_wires
            )
            resources[add_rep] += 1

    return dict(resources)


@register_resources(_signed_out_square_resources)
def _signed_out_square(
    x_wires: WiresLike,
    output_wires: WiresLike,
    work_wires: WiresLike,
    output_wires_zeroed: bool,
    **_,
):
    """Implement signed squaring in three steps: Unsigned squaring, controlled subtraction
    (with input carry), and a single-bit subtraction.
    See the documentation of SignedOutSquare for details."""
    n = len(x_wires)
    m = len(output_wires)
    # Compute (x_u)^2 into output register
    OutSquare(x_wires[1:], output_wires, work_wires, output_wires_zeroed=output_wires_zeroed)
    if m > n:
        # Add x_s * 2^n (2^(n-1) - x_u)
        # For output_wires_zeroed=True, note that there are n-1 bits in x_u, so the squaring
        # above can only have produced numbers smaller than 2^(2n-2). We here add a number that
        # is at most 2^(2n-1), so that the output is guaranteed to be smaller than 2^(2n) and
        # we can restrict the subroutine to the 2n LSB of the output.
        # As the wire ordering is big-endian, slicing off the n LSBs (to multiply by 2^n) is done
        # by setting the upper limit of the output wires to `m-n`.
        output_msb = max(0, m - 2 * n) if output_wires_zeroed else 0
        output = output_wires[output_msb : m - n]
        _c_subtract_then_add_one(x_wires[0], x_wires[1:], output, work_wires)

        if m >= 2 * n - 1:
            # Subtract x_s * 2^(2n-2).
            # As the wire ordering is big-endian, slicing off the n LSBs (to multiply by 2^(2n-2))
            # is done by setting the upper limit of the output wires to `m-(2n-2)`.
            output = output_wires[output_msb : m - (2 * n - 2)]
            _ = [X(w) for w in output]
            SemiAdder(x_wires[:1], output, work_wires)
            _ = [X(w) for w in output]


add_decomps(SignedOutSquare, _signed_out_square)
