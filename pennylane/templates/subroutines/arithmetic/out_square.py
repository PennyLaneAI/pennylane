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
    adjoint_resource_rep,
    controlled_resource_rep,
    register_condition,
    register_resources,
)
from pennylane.decomposition.resources import resource_rep
from pennylane.operation import Operation
from pennylane.ops import CNOT, BasisState, X, adjoint, ctrl
from pennylane.ops.mid_measure import MidMeasure, measure
from pennylane.queuing import AnnotatedQueue, QueuingManager, apply
from pennylane.templates.subroutines.arithmetic import SemiAdder, TemporaryAND
from pennylane.wires import Wires, WiresLike


class OutSquare(Operation):
    r"""Performs out-of-place modular squaring.

    This operator performs the modular squaring of integers :math:`x` modulo
    :math:`2^n` in the computational basis, where ``n=len(output_wires)``:

    .. math::
        \text{OutSquare} |x \rangle |b \rangle = |x \rangle |(b + x^2) \; \text{mod} \; 2^n \rangle,

    .. seealso:: :class:`~.SemiAdder`, :class:`~.Multiplier` , and :class:`~.OutMultiplier`.

    Args:
        x_wires (WiresLike): wires that store the integer :math:`x`.
        output_wires (WiresLike): the wires that store the squaring result. If the
            register is in a non-zero state :math:`b`, the solution will be added to this value.
            If the register is guaranteed to be in the zero state, it is recommended to set
            ``zeroed_output_wires=True``.
        work_wires (WiresLike): the auxiliary wires to use for the squaring.
            ``len(output_wires)`` work wires are required if ``zeroed_output_wires=False``,
            otherwise ``min(len(output_wires), len(x_wires)+1)`` work wires are required.
        zeroed_output_wires (bool): Whether the output wires are guaranteed to be in the state
            :math:`|0\rangle` initially. Defaults to ``False``.

    **Example**

    Let's compute the square of :math:`x=3` and :math:`x=7` in superposition, added to :math`b=5`
    modulo :math:`2^n=2^6=64`.

    .. code-block:: python

        import pennylane as qml

        x = 2
        y = 7
        mod = 12

        x_wires = list(range(3))
        output_wires = list(range(3, 9))
        work_wires = list(range(9, 15))

        dev = qml.device("lightning.qubit", wires=15, seed=295)

        @qml.qnode(dev, shots=1_000)
        def circuit(output_wires):
            # Create a uniform superposition between integers 3 and 7
            qml.H(x_wires[0]) # Superposition between 0 and 4
            qml.BasisEmbedding(3, wires=x_wires[1:]) # Add 3, by embedding in lower-precision wires
            # Prepare output state
            qml.BasisEmbedding(5, wires=output_wires)
            # Square
            qml.templates.subroutines.arithmetic.OutSquare(x_wires, output_wires, work_wires)
            return qml.counts(wires=output_wires)

    >>> counts = circuit(output_wires)
    >>> counts = {int(k, 2):val for k, val in counts.items()}
    >>> print(counts)
    {14: np.int64(498), 54: np.int64(502)}

    We correctly obtain the squared numbers added to :math:`b=5`, namely
    :math:`5+3^2=14` and :math:`5+7^2=54`.

    Note that reducing the size of the output register changes the computed numbers via the reduced
    modulus:

    >>> output_wires = list(range(3, 6))
    >>> counts = circuit(output_wires)
    >>> counts = {int(k, 2):val for k, val in counts.items()}
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
        computation, which needs to be passed via ``zeroed_output_wires`` (see below for an
        example). If ``zeroed_output_wires=False`` (the default), :math:`n` work wires are
        required. If ``zeroed_output_wires=True``, :math:`min(n, k+1)` work wires are required,
        where :math:`k` denotes the length of the first register ``x_wires``.

        **Cheaper decomposition for zeroed output state**

        If we know that the qubits in ``output_wires`` are in the state
        :math:`|0\rangle^{\otimes n}` before ``OutSquare`` is applied, we can pass this information
        to the template via ``zeroed_output_wires``, leading to a cheaper decomposition.
        Consider the following example, where we control this information with the ``QNode``
        argument ``zeroed``:

        .. code-block:: python

            x_wires = list(range(4))
            x = 13
            output_wires = list(range(4, 12))
            work_wires = list(range(12, 20))

            dev = qml.device("lightning.qubit", wires=20, seed=295)

            @qml.decompose(max_expansion=1) # To see resources easily
            @qml.qnode(dev, shots=1_000)
            def circuit(zeroed):
                qml.BasisEmbedding(x, wires=x_wires)
                qml.templates.subroutines.arithmetic.OutSquare(x_wires, output_wires, work_wires, zeroed_output_wires=zeroed)
                return qml.counts(wires=output_wires)

        We can compute the required resources with ``zeroed=False``, i.e., when not passing
        the information to the template:

        >>> specs_false = qml.specs(circuit)(False)["resources"].gate_types
        >>> print(specs_false)
        {'PauliX': 3, 'CNOT': 8, 'C(SemiAdder)': 4}

        When we do pass the information, we save a controlled :class:`~.SemiAdder` and some of
        the other adders become smaller (depending on the register sizes):

        >>> specs_true = qml.specs(circuit)(True)["resources"].gate_types
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

        >>> print(qml.draw(circuit)(False))
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

        >>> print(qml.draw(circuit)(True))
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

    resource_keys = {"num_x_wires", "num_output_wires", "num_work_wires", "zeroed_output_wires"}

    def __init__(
        self,
        x_wires: WiresLike,
        output_wires: WiresLike,
        work_wires: WiresLike,
        zeroed_output_wires: bool = False,
    ):

        x_wires = Wires(x_wires)
        output_wires = Wires(output_wires)
        work_wires = Wires(work_wires)

        n = len(x_wires)
        k = len(output_wires)

        if zeroed_output_wires:
            num_required_work_wires = min(n + 1, k)
        else:
            num_required_work_wires = k
        if len(work_wires) < num_required_work_wires:
            raise ValueError(
                f"OutSquare requires at least {num_required_work_wires} work wires for "
                f"{n} input wires, {k} output wires "
                f"and {zeroed_output_wires=}."
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

        self.hyperparameters["zeroed_output_wires"] = zeroed_output_wires
        all_wires = x_wires + output_wires + work_wires
        super().__init__(wires=all_wires)

    @property
    def resource_params(self) -> dict:
        return {
            "num_x_wires": len(self.hyperparameters["x_wires"]),
            "num_output_wires": len(self.hyperparameters["output_wires"]),
            "num_work_wires": len(self.hyperparameters["work_wires"]),
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
            for key in ["x_wires", "output_wires", "work_wires"]
        }

        return OutSquare(
            new_dict["x_wires"],
            new_dict["output_wires"],
            new_dict["work_wires"],
            self.hyperparameters["zeroed_output_wires"],
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
        zeroed_output_wires: bool,
    ):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a product of other operators.

        Args:
            x_wires (WiresLike): wires that store the integer :math:`x`.
            output_wires (WiresLike): the wires that store the squaring result. If the register
                is in a non-zero state :math:`b`, the solution will be added to this value.
                If the register is guaranteed to be in the zero state, it is recommended to set
                ``zeroed_output_wires=True``.
            work_wires (WiresLike): the auxiliary wires to use for the squaring.
                ``len(output_wires)`` work wires are required if ``zeroed_output_wires=False``,
                otherwise ``min(len(output_wires), len(x_wires)+1)`` work wires are required.
            zeroed_output_wires (bool): Whether the output wires are guaranteed to be in the state
                :math:`|0\rangle` initially. Defaults to ``False``.

        Returns:
            list[.Operator]: Decomposition of the operator

        **Example**

        >>> all_wires = ([0, 1], [2, 3], [4, 5])
        >>> qml.OutSquare.compute_decomposition(*all_wires, zeroed_output_wires=True)
        [CNOT(wires=[1, 3]), TemporaryAND(wires=Wires([1, 0, 2])), CNOT(wires=[0, 4]), Controlled(SemiAdder(wires=[0, 1, 2, 5]), control_wires=[4]), CNOT(wires=[0, 4])]
        """
        with AnnotatedQueue() as q:
            _out_square_with_adder(x_wires, output_wires, work_wires, zeroed_output_wires)

        if QueuingManager.recording():
            for op in q.queue:
                apply(op)

        return q.queue


def _out_square_with_adder_condition(
    num_x_wires, num_output_wires, num_work_wires, zeroed_output_wires
) -> bool:
    n = num_x_wires
    k = num_output_wires
    if zeroed_output_wires:
        largest_adder = min(k, n + 1)
    else:
        largest_adder = k
    # work wires: one for control cache, largest_adder-1 for adder
    min_num_work_wires = 1 + (largest_adder - 1)
    return num_work_wires >= min_num_work_wires


def _out_square_with_adder_resources(
    num_x_wires, num_output_wires, num_work_wires, zeroed_output_wires
) -> dict:
    # pylint: disable=unused-argument
    n = num_x_wires
    m = num_output_wires
    resources = defaultdict(int)
    if zeroed_output_wires:
        # Copying of first bit is a CNOT, all other bits require a TemporaryAND
        resources[resource_rep(CNOT)] += 1
        resources[resource_rep(TemporaryAND)] = zeroed_output_wires * (min(n, m) - 1)

    # Controlled adders, includes the one for copying if zeroed_output_wires=False
    for i in range(zeroed_output_wires, min(n, m)):
        num_out = min(m - i, n + 1) if zeroed_output_wires else m - i
        resources[resource_rep(CNOT)] += 2
        resources[
            controlled_resource_rep(
                base_class=SemiAdder,
                base_params={
                    "num_x_wires": n,
                    "num_y_wires": num_out,
                    "num_work_wires": num_work_wires - 1,
                },
                num_control_wires=1,
                # num_work_wires=num_work_wires-num_out,
                # work_wire_type="zeroed",
            )
        ] += 1
    return dict(resources)


@register_condition(_out_square_with_adder_condition)
@register_resources(_out_square_with_adder_resources)
def _out_square_with_adder(
    x_wires: WiresLike,
    output_wires: WiresLike,
    work_wires: WiresLike,
    zeroed_output_wires: bool,
    **_,
):
    n = len(x_wires)
    m = len(output_wires)

    if zeroed_output_wires:
        # Copy x, controlled on the least significant bit (LSB) of x, to the output register,
        # which is in |0>. This can be reduced to a CNOT for the LSB and TemporaryANDs for
        # the other bits.
        CNOT([x_wires[-1], output_wires[-1]])  # First control-copy is a CNOT
        for x_wire, out_wire in zip(x_wires[-2::-1], output_wires[-2::-1]):
            TemporaryAND([x_wires[-1], x_wire, out_wire])  # Subsequent control-copies
        # Mark that the copying has happened and does not have to happen via an adder below
        x_wires_to_multiply = x_wires[-m:-1]
        start = 1
    else:
        x_wires_to_multiply = x_wires[-m:]
        start = 0

    for i, x_wire in enumerate(reversed(x_wires_to_multiply), start=start):
        # Add x to the output register, controlled on x_wire via the work_wires[0] and
        # shifted by i bit positions. For zeroed_output_wires=False, includes the initial copy
        # The output wires of the adder need to take all of the output register of square
        # into account due to carry values. For zeroed_output_wires=True, we can reduce to
        # a fixed size (`n`) instead, because we know at each step how large the value stored
        # in the output register can have grown by then.
        output_msb = max(0, m - n - i - 1) if zeroed_output_wires else 0
        output = output_wires[output_msb : m - i]
        CNOT([x_wire, work_wires[0]])
        ctrl(
            SemiAdder(x_wires=x_wires, y_wires=output, work_wires=work_wires[1:]),
            control=work_wires[:1],
        )
        CNOT([x_wire, work_wires[0]])


def _out_square_with_caddsub_condition(
    num_x_wires, num_output_wires, num_work_wires, zeroed_output_wires
) -> bool:
    n = num_x_wires
    k = num_output_wires + 1
    if zeroed_output_wires:
        largest_adder = min(k, n + 1)
    else:
        largest_adder = k
    # work wires: one for control cache, one for augmented output, largest_adder-1 for adder
    min_num_work_wires = 1 + 1 + (largest_adder - 1)
    largest_correction_adder = k
    # work wires: one for augmented output, largest_correction_adder-1 for adder
    min_num_work_wires = max(min_num_work_wires, 1 + (largest_correction_adder - 1))
    return num_work_wires >= min_num_work_wires


def _out_square_with_caddsub_resources(
    num_x_wires, num_output_wires, num_work_wires, zeroed_output_wires
) -> dict:
    # pylint: disable=unused-argument
    n = num_x_wires
    k = num_output_wires + 1
    resources = defaultdict(int)

    cnot_rep = resource_rep(CNOT)
    cnot_on_0_kwargs = {"base_params": {}, "num_control_wires": 1, "num_zero_control_values": 1}
    cnot_on_0_rep = controlled_resource_rep(X, **cnot_on_0_kwargs)
    x_rep = resource_rep(X)
    meas_rep = resource_rep(MidMeasure)

    # Controlled add-subtract loop
    loop_size = min(k, n)
    # Bit flips on the y_wires, controlled on |0>: two per ctrl-add-subtract
    if n > 1:
        c_flips = controlled_resource_rep(
            BasisState,
            base_params={"num_wires": n - 1},
            num_control_wires=1,
            num_zero_control_values=1,
        )
        resources[c_flips] += 2 * loop_size

    # Caching of bit in x onto c_wire, to control on it.
    resources[cnot_rep] += 2 * loop_size
    # Bit flip of LSB output wire, controlled on |0>: two per ctrl-add-subtract
    resources[cnot_on_0_rep] += 2 * loop_size
    # Bit flip on LSB work wire, controlled on |0>: one per ctrl-add-subtract that has work wires
    c_add_subs_with_work_wires = min(n, k - 1)
    resources[cnot_on_0_rep] += c_add_subs_with_work_wires
    # Bit reset on LSB work wire: one per ctrl-add-subtract that has work wires
    resources[meas_rep] += c_add_subs_with_work_wires

    # SemiAdder of x_wires onto output_wires: One per ctrl-add-subtract, varying size
    for i in range(loop_size):
        size = min(k - i, n + 1) if zeroed_output_wires else k - i
        resources[
            resource_rep(SemiAdder, num_x_wires=n, num_y_wires=size, num_work_wires=size - 1)
        ] += 1

    # Subtract 2^(2n)
    size = k - 2 * n
    if size > 0:
        if size > 1:
            resources[resource_rep(TemporaryAND)] += size - 2
            resources[adjoint_resource_rep(TemporaryAND)] += size - 2
        resources[resource_rep(CNOT)] += size - 1
        resources[x_rep] += 1 + 2 * size

    # Add (2^n-1-x) + 1
    if n > 1:
        resources[resource_rep(BasisState, num_wires=n - 1)] += 2

    resources[x_rep] += 3
    resources[meas_rep] += 1
    resources[resource_rep(SemiAdder, num_x_wires=n, num_y_wires=k, num_work_wires=k - 1)] += 1

    # Add 2^(n+1) x if 2^k > 2^(n+1) (otherwise it just vanishes in the modulus)
    if k > n + 1:
        resources[
            resource_rep(SemiAdder, num_x_wires=n, num_y_wires=k - n - 1, num_work_wires=k - n - 2)
        ] += 1

    return dict(resources)


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
        # In principle, we could just apply a bit flip here to reset the work wire, controlled
        # on the `c_wire` being in state |0>.
        # However, we want to use a measurement+reset instead
        # for addition + 1. In case `c_wire` is in state |1>, we just add a reset of a work
        # wire that anyways is returned in state |0> by `SemiAdder`, so there is no harm done.
        measure(work_wires[-1], reset=True)

    if len(x_wires) > 1:
        ctrl(BasisState([1] * (len(x_wires) - 1), x_wires[:-1]), control=c_wire, control_values=[0])


def _sub_then_add_one(x_wires, y_wires, work_wires):
    if len(x_wires) > 1:
        BasisState([1] * (len(x_wires) - 1), x_wires[:-1])

    work_wires = work_wires[: len(y_wires) - 1]
    X(y_wires[-1])
    if work_wires:
        X(work_wires[-1])
    SemiAdder(x_wires, y_wires, work_wires)
    X(y_wires[-1])
    if work_wires:
        # In principle, we could just apply a bit flip here to reset the work wire.
        # However, the SemiAdder uses a decomposition with a right elbow, which has an assumption
        # about its output state attached to it, and we violate this assumption here.
        # In order to obtain a correct behaviour both for unitary decompositions of the right elbow
        # and for its implementation relying on the assumption, we replace the bit flip with
        # a measurement + reset.
        measure(work_wires[-1], reset=True)

    if len(x_wires) > 1:
        BasisState([1] * (len(x_wires) - 1), x_wires[:-1])


def _increment(wires, work_wires):
    """Increment the input `wires` by one, using zeroed `work_wires`.
    We use a left elbow ladder together with a CNOT+right elbow uncompute ladder.
    This is a manually reduced decomposition of the standard incrementer via MCX gates if
    work wires are available:

    Generic decomposition:
    0: ─╭X────────────────┤
    1: ─├●─╭X─────────────┤
    2: ─├●─├●─╭X──────────┤
    3: ─├●─├●─├●─╭X───────┤
    4: ─├●─├●─├●─├●─╭X────┤
    5: ─╰●─╰●─╰●─╰●─╰●──X─┤

    Decompose all MCX gates into elbows and CNOTs:
       0: ─────────────╭X──────────────────────────────────────────────────────────────────────────┤
       1: ──────────╭●─│───●╮──────────────────────╭X──────────────────────────────────────────────┤
       2: ───────╭●─│──│────│──●╮───────────────╭●─│───●╮───────────────╭X─────────────────────────┤
       3: ────╭●─│──│──│────│───│──●╮────────╭●─│──│────│──●╮────────╭●─│───●╮────────╭X───────────┤
       4: ─╭●─│──│──│──│────│───│───│──●╮─╭●─│──│──│────│───│──●╮─╭●─│──│────│──●╮─╭●─│───●╮─╭X────┤
       5: ─├●─│──│──│──│────│───│───│──●┤─├●─│──│──│────│───│──●┤─├●─│──│────│──●┤─├●─│───●┤─╰●──X─┤
    aux0: ─│──│──├⊕─├●─│───●┤──⊕┤───│───│─│──│──│──│────│───│───│─│──│──│────│───│─│──│────│───────┤
    aux1: ─│──├⊕─╰●─│──│────│──●╯──⊕┤───│─│──├⊕─├●─│───●┤──⊕┤───│─│──│──│────│───│─│──│────│───────┤
    aux2: ─╰⊕─╰●────│──│────│──────●╯──⊕╯─╰⊕─╰●─│──│────│──●╯──⊕╯─╰⊕─├●─│───●┤──⊕╯─│──│────│───────┤
    aux3: ──────────╰⊕─╰●──⊕╯───────────────────╰⊕─╰●──⊕╯────────────╰⊕─╰●──⊕╯─────╰⊕─╰●──⊕╯───────┤

    Cancel neighbouring right and left elbows (moving some work wire usage around in the process)
       0: ─────────────╭X───────────────────────────────┤
       1: ──────────╭●─│───●╮─╭X────────────────────────┤
       2: ───────╭●─│──│────│─│──●╮──╭X─────────────────┤
       3: ────╭●─│──│──│────│─│───│──│──●╮─╭X───────────┤
       4: ─╭●─│──│──│──│────│─│───│──│───│─│───●╮─╭X────┤
       5: ─├●─│──│──│──│────│─│───│──│───│─│───●┤─╰●──X─┤
    aux0: ─│──│──├⊕─├●─│───●┤─╰●─⊕┤──│───│─│────│───────┤
    aux1: ─│──├⊕─╰●─│──│────│────●╯──╰●─⊕┤─│────│───────┤
    aux2: ─╰⊕─╰●────│──│────│───────────●╯─╰●──⊕╯───────┤
    aux3: ──────────╰⊕─╰●──⊕╯───────────────────────────┤

    We see a leading ladder of left elbows and a backwards ladder of CNOT+right elbow pairs.
    """
    wires = wires[::-1]
    if len(wires) > 1:
        # Construct the wires on which the ladder will act.
        all_wires = wires[:1] + list(sum(zip(wires[1:], work_wires), start=tuple()))
        # Forward ladder
        for k in range(len(wires) - 2):
            TemporaryAND(all_wires[2 * k : 2 * k + 3])
        # Backward ladder
        for k in range(len(wires) - 3, -1, -1):
            CNOT([all_wires[2 * k + 2], all_wires[2 * k + 3]])
            adjoint(TemporaryAND)(all_wires[2 * k : 2 * k + 3])
        # Trailing CNOT
        CNOT(wires[:2])
    X(wires[0])


def _sub_two_to_the_two_n(n, output_wires, work_wires):
    if len(output_wires) > 2 * n:
        _output = output_wires[: -2 * n]
        _ = [X(w) for w in _output]
        _increment(_output, work_wires)
        _ = [X(w) for w in _output]


@register_condition(_out_square_with_caddsub_condition)
@register_resources(_out_square_with_caddsub_resources)
def _out_square_with_caddsub(
    x_wires: WiresLike,
    output_wires: WiresLike,
    work_wires: WiresLike,
    zeroed_output_wires: bool,
    **_,
):
    # First work wire is used for caching control bits
    c_wire = work_wires[0]
    # Second work wire is used to augment the output wires because we compute
    # twice the desired output at first, and then divide by 2.
    output_wires = output_wires + [work_wires[1]]
    work_wires = work_wires[2:]
    n = len(x_wires)
    k = len(output_wires)

    for i, x_wire in enumerate(x_wires[::-1][:k]):
        output_msb = max(0, k - (n + 1 + i)) if zeroed_output_wires else 0
        output = output_wires[output_msb : k - i]
        CNOT([x_wire, c_wire])
        _c_add_sub(c_wire, x_wires, output, work_wires)
        CNOT([x_wire, c_wire])

    # Corrections - no need for control wire any more
    work_wires = [c_wire] + work_wires
    # Subtract 2^(2n)
    _sub_two_to_the_two_n(n, output_wires, work_wires)
    # Add (2^n-1-x) + 1
    _sub_then_add_one(x_wires, output_wires, work_wires)

    # Add 2^(n+1) x if 2^k > 2^(n+1) (otherwise it just vanishes in the modulus)
    if k > n + 1:
        SemiAdder(x_wires, output_wires[: k - n - 1], work_wires[: k - n - 2])


add_decomps(OutSquare, _out_square_with_adder, _out_square_with_caddsub)
