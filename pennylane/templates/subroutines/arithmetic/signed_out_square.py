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
from pennylane.ops import BasisState, X, ctrl
from pennylane.ops.mid_measure import MidMeasure, measure
from pennylane.queuing import AnnotatedQueue, QueuingManager, apply
from pennylane.templates.subroutines.arithmetic import OutSquare, SemiAdder
from pennylane.wires import Wires, WiresLike


class SignedOutSquare(Operation):
    r"""Performs out-of-place squaring of a signed integer.

    This operator performs the squaring of a signed integer :math:`x` in two's complement
    convention into an unsigned register. The computation is  modulo :math:`2^n`,
    where ``n=len(output_wires)``:

    .. math::
        \text{SignedOutSquare} |x \rangle |b \rangle = |x \rangle |(b + x^2) \; \text{mod} \; 2^n \rangle,

    .. seealso:: :class:`~.OutSquare` and :class:`~.OutMultiplier`.

    Args:
        x_wires (WiresLike): wires that store the integer :math:`x`.
        output_wires (WiresLike): wires that store the squaring result. If the
            register is in a non-zero state :math:`b`, the solution will be added to this value.
            If the register is guaranteed to be in the zero state, it is recommended to set
            ``output_wires_zeroed=True``.
        work_wires (WiresLike): the auxiliary wires to use for the squaring.
            ``len(output_wires)`` work wires are required if ``output_wires_zeroed=False``,
            otherwise ``min(len(output_wires), len(x_wires)+1)`` work wires are required.
        output_wires_zeroed (bool): Whether the output wires are guaranteed to be in the state
            :math:`|0\rangle` initially. Defaults to ``False``.

    **Example**

    # TODO
    Let's compute the square of :math:`x=3` and :math:`x=7` in superposition, added to :math`b=5`
    modulo :math:`2^n=2^6=64`.

    .. code-block:: python

        import pennylane as qp

        x_wires = list(range(3))
        output_wires = list(range(3, 9))
        work_wires = list(range(9, 15))

        dev = qp.device("lightning.qubit", wires=15, seed=295)

        @qp.qnode(dev, shots=1_000)
        def circuit(output_wires):
            # Create a uniform superposition between integers 3 and 7
            qp.H(x_wires[0]) # Superposition between 0 and 4
            qp.BasisEmbedding(3, wires=x_wires[1:]) # Add 3, by preparing lower-precision wires
            # Prepare initial state on output wires
            qp.BasisEmbedding(5, wires=output_wires)
            # Square
            qp.OutSquare(x_wires, output_wires, work_wires)
            return qp.counts(wires=output_wires)

    >>> counts = circuit(output_wires)
    >>> counts = {int(k, 2): val for k, val in counts.items()}
    >>> print(counts)
    {14: np.int64(498), 54: np.int64(502)}

    We correctly obtain the squared numbers added to :math:`b=5`, namely
    :math:`5+3^2=14` and :math:`5+7^2=54`.

    Note that reducing the size of the output register changes the computed numbers via the reduced
    modulus:

    >>> output_wires = list(range(3, 6))
    >>> counts = circuit(output_wires)
    >>> counts = {int(k, 2): val for k, val in counts.items()}
    >>> print(counts)
    {6: np.int64(1000)}

    Why do we obtain a single result? This is simply because :math:`(5+3^2)\!\mod\!8=6` and
    :math:`(5+7^2)\!\mod\!8=6` happen to be equal.


    .. details::
        :title: Usage Details

        This template takes as input three wire registers.

        The first one is ``x_wires`` which is used to encode the integer :math:`x` in the
        computational basis. Therefore, ``x_wires`` must contain
        at least :math:`\lceil \log_2(x)\rceil` wires to represent :math:`x`.

        The second one is ``output_wires``, which is used to encode the integer
        :math:`b+ x^2 \; \text{mod} \; 2^n` in the computational basis, where :math:`n`
        denotes the length of ``output_wires``.

        The third register is ``work_wires``, which consists of the auxiliary qubits used to
        perform the modular squaring operation. The required number of work wires depends
        on whether we are guaranteed that :math:`b=0` in the ``output_wires`` before the
        computation, which needs to be passed via ``output_wires_zeroed`` (see below for an
        example). If ``output_wires_zeroed=False`` (the default), :math:`n` work wires are
        required. If ``output_wires_zeroed=True``, :math:`min(n, k+1)` work wires are required,
        where :math:`k` denotes the length of the first register ``x_wires``.

        **Cheaper decomposition for zeroed output state**

        If we know that the qubits in ``output_wires`` are in the state
        :math:`|0\rangle^{\otimes n}` before ``OutSquare`` is applied, we can pass this information
        to the template via ``output_wires_zeroed``, leading to a cheaper decomposition.
        Consider the following example, where we control this information with the ``QNode``
        argument ``zeroed``:

        .. code-block:: python

            x_wires = list(range(4))
            x = 13
            output_wires = list(range(4, 12))
            work_wires = list(range(12, 20))

            dev = qp.device("lightning.qubit", wires=20, seed=295)

            @qp.decompose(max_expansion=1) # To see resources easily
            @qp.qnode(dev, shots=1_000)
            def circuit(zeroed):
                qp.BasisEmbedding(x, wires=x_wires)
                qp.OutSquare(x_wires, output_wires, work_wires, output_wires_zeroed=zeroed)
                return qp.counts(wires=output_wires)

        We can compute the required resources with ``zeroed=False``, i.e., when not passing
        the information to the template:

        >>> specs_false = qp.specs(circuit)(False)["resources"].gate_types
        >>> print(specs_false)
        {'PauliX': 3, 'CNOT': 8, 'C(SemiAdder)': 4}

        When we do pass the information, we save a controlled :class:`~.SemiAdder` and some of
        the other adders become smaller (depending on the register sizes):

        >>> specs_true = qp.specs(circuit)(True)["resources"].gate_types
        >>> print(specs_true)
        {'PauliX': 3, 'CNOT': 7, 'TemporaryAND': 3, 'C(SemiAdder)': 3}

        Of course, both decompositions are correctly implementing the squaring operation:

        >>> print(circuit(False))
        {np.str_('10101001'): np.int64(1000)}
        >>> print(circuit(True))
        {np.str_('10101001'): np.int64(1000)}

        Here, :math:`(10101001)_2=128 + 32 + 8 + 1=169` is the expected result of
        :math:`13^2`.
        To conclude, we draw the two circuit variants:

        >>> print(qp.draw(circuit)(False))
         0: ──X────╭SemiAdder───────╭SemiAdder───────╭SemiAdder────╭●─╭SemiAdder─╭●─┤
         1: ──X────├SemiAdder───────├SemiAdder────╭●─├SemiAdder─╭●─│──├SemiAdder─│──┤
         2: ───────├SemiAdder────╭●─├SemiAdder─╭●─│──├SemiAdder─│──│──├SemiAdder─│──┤
         3: ──X─╭●─├SemiAdder─╭●─│──├SemiAdder─│──│──├SemiAdder─│──│──├SemiAdder─│──┤
         4: ────│──├SemiAdder─│──│──├SemiAdder─│──│──├SemiAdder─│──│──├SemiAdder─│──┤ ╭Counts
         5: ────│──├SemiAdder─│──│──├SemiAdder─│──│──├SemiAdder─│──│──├SemiAdder─│──┤ ├Counts
         6: ────│──├SemiAdder─│──│──├SemiAdder─│──│──├SemiAdder─│──│──├SemiAdder─│──┤ ├Counts
         7: ────│──├SemiAdder─│──│──├SemiAdder─│──│──├SemiAdder─│──│──├SemiAdder─│──┤ ├Counts
         8: ────│──├SemiAdder─│──│──├SemiAdder─│──│──├SemiAdder─│──│──├SemiAdder─│──┤ ├Counts
         9: ────│──├SemiAdder─│──│──├SemiAdder─│──│──├SemiAdder─│──│──│──────────│──┤ ├Counts
        10: ────│──├SemiAdder─│──│──├SemiAdder─│──│──│──────────│──│──│──────────│──┤ ├Counts
        11: ────│──├SemiAdder─│──│──│──────────│──│──│──────────│──│──│──────────│──┤ ╰Counts
        12: ────╰X─├●─────────╰X─╰X─├●─────────╰X─╰X─├●─────────╰X─╰X─├●─────────╰X─┤
        13: ───────├SemiAdder───────├SemiAdder───────├SemiAdder───────├SemiAdder────┤
        14: ───────├SemiAdder───────├SemiAdder───────├SemiAdder───────├SemiAdder────┤
        15: ───────├SemiAdder───────├SemiAdder───────├SemiAdder───────├SemiAdder────┤
        16: ───────├SemiAdder───────├SemiAdder───────├SemiAdder───────├SemiAdder────┤
        17: ───────├SemiAdder───────├SemiAdder───────├SemiAdder───────├SemiAdder────┤
        18: ───────├SemiAdder───────├SemiAdder───────├SemiAdder───────├SemiAdder────┤
        19: ───────╰SemiAdder───────╰SemiAdder───────╰SemiAdder───────╰SemiAdder────┤

        >>> print(qp.draw(circuit)(True))
         0: ──X──────────╭●────╭SemiAdder───────╭SemiAdder────╭●─╭SemiAdder─╭●─┤
         1: ──X───────╭●─│─────├SemiAdder────╭●─├SemiAdder─╭●─│──├SemiAdder─│──┤
         2: ───────╭●─│──│──╭●─├SemiAdder─╭●─│──├SemiAdder─│──│──├SemiAdder─│──┤
         3: ──X─╭●─├●─├●─├●─│──├SemiAdder─│──│──├SemiAdder─│──│──├SemiAdder─│──┤
         4: ────│──│──│──│──│──│──────────│──│──│──────────│──│──├SemiAdder─│──┤ ╭Counts
         5: ────│──│──│──│──│──│──────────│──│──├SemiAdder─│──│──├SemiAdder─│──┤ ├Counts
         6: ────│──│──│──│──│──├SemiAdder─│──│──├SemiAdder─│──│──├SemiAdder─│──┤ ├Counts
         7: ────│──│──│──│──│──├SemiAdder─│──│──├SemiAdder─│──│──├SemiAdder─│──┤ ├Counts
         8: ────│──│──│──╰⊕─│──├SemiAdder─│──│──├SemiAdder─│──│──├SemiAdder─│──┤ ├Counts
         9: ────│──│──╰⊕────│──├SemiAdder─│──│──├SemiAdder─│──│──│──────────│──┤ ├Counts
        10: ────│──╰⊕───────│──├SemiAdder─│──│──│──────────│──│──│──────────│──┤ ├Counts
        11: ────╰X──────────│──│──────────│──│──│──────────│──│──│──────────│──┤ ╰Counts
        12: ────────────────╰X─├●─────────╰X─╰X─├●─────────╰X─╰X─├●─────────╰X─┤
        13: ───────────────────├SemiAdder───────├SemiAdder───────├SemiAdder────┤
        14: ───────────────────├SemiAdder───────├SemiAdder───────├SemiAdder────┤
        15: ───────────────────├SemiAdder───────├SemiAdder───────├SemiAdder────┤
        16: ───────────────────├SemiAdder───────├SemiAdder───────├SemiAdder────┤
        17: ───────────────────├SemiAdder───────├SemiAdder───────├SemiAdder────┤
        18: ───────────────────├SemiAdder───────├SemiAdder───────├SemiAdder────┤
        19: ───────────────────╰SemiAdder───────╰SemiAdder───────╰SemiAdder────┤

    """

    grad_method = None

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

        if output_wires_zeroed:
            num_required_work_wires = min(n, m)
        else:
            num_required_work_wires = m
        sign_correc_adder = min(m - n, n)
        num_required_work_wires = max(num_required_work_wires, sign_correc_adder - 1)
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
            x_wires (WiresLike): wires that store the integer :math:`x`.
            output_wires (WiresLike): the wires that store the squaring result. If the register
                is in a non-zero state :math:`b`, the solution will be added to this value.
                If the register is guaranteed to be in the zero state, it is recommended to set
                ``output_wires_zeroed=True``.
            work_wires (WiresLike): the auxiliary wires to use for the squaring.
                ``len(output_wires)`` work wires are required if ``output_wires_zeroed=False``,
                otherwise ``min(len(output_wires), len(x_wires)+1)`` work wires are required.
            output_wires_zeroed (bool): Whether the output wires are guaranteed to be in the state
                :math:`|0\rangle` initially. Defaults to ``False``.

        Returns:
            list[.Operator]: Decomposition of the operator

        **Example**

        >>> all_wires = ([0, 1], [2, 3], [4, 5])
        >>> qp.SignedOutSquare.compute_decomposition(*all_wires, output_wires_zeroed=True)
        [CNOT(wires=[1, 3]), TemporaryAND(wires=Wires([1, 0, 2])), CNOT(wires=[0, 4]), Controlled(SemiAdder(wires=[0, 1, 2, 5]), control_wires=[4]), CNOT(wires=[0, 4])]
        """
        with AnnotatedQueue() as q:
            _signed_out_square(x_wires, output_wires, work_wires, output_wires_zeroed)

        if QueuingManager.recording():
            for op in q.queue:
                apply(op)

        return q.queue


def _c_subtract_then_add_one(c_wire, x_wires, y_wires, work_wires):
    if len(x_wires) > 1:
        BasisState([1] * (len(x_wires) - 1), x_wires[:-1])

    m = len(y_wires)
    X(y_wires[-1])
    if work_wires:
        X(work_wires[m - 2])
    ctrl(
        SemiAdder(x_wires, y_wires, work_wires[: m - 1]),
        control=c_wire,
        work_wires=work_wires[m - 1 :],
        work_wire_type="zeroed",
    )
    X(y_wires[-1])
    if work_wires:
        # In principle, we could just apply a bit flip here to reset the work wire.
        # However, the SemiAdder uses a decomposition with a right elbow, which has an assumption
        # about its output state attached to it, and we violate this assumption here.
        # In order to obtain a correct behaviour both for unitary decompositions of the right elbow
        # and for its implementation relying on the assumption, we replace the bit flip with
        # a measurement + reset.
        measure(work_wires[m - 2], reset=True)

    if len(x_wires) > 1:
        BasisState([1] * (len(x_wires) - 1), x_wires[:-1])


def _signed_out_square_resources(
    num_x_wires, num_output_wires, num_work_wires, output_wires_zeroed
) -> dict:
    # pylint: disable=unused-argument
    n = num_x_wires
    m = num_output_wires
    resources = defaultdict(int)
    square_rep = resource_rep(
        OutSquare,
        num_x_wires=n - 1,
        num_output_wires=m,
        num_work_wires=num_work_wires,
        output_wires_zeroed=output_wires_zeroed,
    )
    resources[square_rep] += 1

    # Add (2^n-1-x) + 1
    if n < m:
        if n - 1 > 1:
            resources[resource_rep(BasisState, num_wires=n - 2)] += 2

        resources[resource_rep(X)] += 3
        resources[resource_rep(MidMeasure)] += 1
        size = min(m - (n + 1), n + 1)
        resources[
            resource_rep(
                SemiAdder, num_x_wires=n - 1, num_y_wires=size, num_work_wires=num_work_wires
            )
        ] += 1

    return dict(resources)


@register_resources(_signed_out_square_resources)
def _signed_out_square(
    x_wires: WiresLike,
    output_wires: WiresLike,
    work_wires: WiresLike,
    output_wires_zeroed: bool,
    **_,
):
    n = len(x_wires)
    m = len(output_wires)
    # Compute (x_u)^2 into output register
    OutSquare(x_wires[1:], output_wires, work_wires, output_wires_zeroed=output_wires_zeroed)
    # Add 2^(n+1)(2^(n-1) - x_u)
    if m > n:
        output_msb = max(0, m - (2 * n + 2))  # redo
        _c_subtract_then_add_one(
            x_wires[0], x_wires[1:], output_wires[output_msb : m - n], work_wires
        )

        if m >= 2 * n - 1:
            output = output_wires[: m - (2 * n - 2)]
            _ = [X(w) for w in output]
            SemiAdder(x_wires[:1], output, work_wires)
            _ = [X(w) for w in output]


add_decomps(OutSquare, _signed_out_square)
