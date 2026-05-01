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
from pennylane.queuing import AnnotatedQueue, QueuingManager, apply
from pennylane.wires import Wires, WiresLike

from .out_multiplier import _add_plus_one, _c_add_sub, _increment
from .semi_adder import SemiAdder, _semiadder_resources
from .temporary_and import TemporaryAND


class OutSquare(Operation):
    r"""Performs out-of-place squaring.

    This operator performs the squaring of an :math:`n`-qubit integer :math:`x` modulo
    :math:`2^m` into an :math:`m`-qubit output register:

    .. math::
        \text{OutSquare} |x \rangle |y \rangle = |x \rangle |(y + x^2) \; \text{mod} \; 2^m \rangle,

    .. seealso:: :class:`~.OutMultiplier`, :class:`~.SemiAdder` , and :class:`~.Multiplier`.

    Args:
        x_wires (WiresLike): wires that store the integer :math:`x`.
        output_wires (WiresLike): wires that store the squaring result. If the register initially
            encodes a non-zero value :math:`y`, the solution will be added to this value.
            If the register is guaranteed to be in the zero state, it is recommended to set
            ``output_wires_zeroed=True``.
        work_wires (WiresLike): the auxiliary wires to use for the squaring.
            :math:`m` work wires are required if ``output_wires_zeroed=False``,
            otherwise :math:`\min(m, n+1)` work wires are required.
        output_wires_zeroed (bool): Whether the output wires are guaranteed to be in the state
            :math:`|0\rangle` initially. Defaults to ``False``.

    **Example**

    Let's compute the square of :math:`x=3` and :math:`x=7` in superposition, added to a
    :math:`m=6`-qubit register that holds the value :math`y=5` initially.
    The computation will be modulo :math:`2^m=2^6=64`.

    .. code-block:: python

        import pennylane as qp

        n = 3
        m = 6
        x_wires = list(range(n))
        output_wires = list(range(n, n + m))
        work_wires = list(range(n + m, n + 2 * m))

        dev = qp.device("lightning.qubit", wires=n + 2 * m, seed=295)

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

    We correctly obtain the squared numbers added to :math:`y=5`, namely
    :math:`5+3^2=14` and :math:`5+7^2=54`.

    Note that reducing the size of the output register changes the computed numbers via the reduced
    modulus:

    >>> m = 3
    >>> output_wires = list(range(n, n + m))
    >>> counts = circuit(output_wires)
    >>> counts = {int(k, 2): val for k, val in counts.items()}
    >>> print(counts)
    {6: np.int64(1000)}

    Why do we obtain a single result? This is simply because :math:`(5+3^2)\!\mod\!8=6` and
    :math:`(5+7^2)\!\mod\!8=6` happen to be equal.


    .. details::
        :title: Usage Details

        This template takes as input three wire registers.

        The first one is ``x_wires`` which encodes the integer :math:`x` to be squared.

        The second one is ``output_wires``, which encodes the integer
        :math:`y+ x^2 \; \text{mod} \; 2^m`, where :math:`m` denotes the length of the register.

        The third register is ``work_wires``, which consists of the auxiliary qubits used to
        perform the squaring operation. The required number of work wires depends
        on whether we are guaranteed that :math:`y=0` in the ``output_wires`` before the
        computation, which needs to be passed via ``output_wires_zeroed`` (see below for an
        example). If ``output_wires_zeroed=False`` (the default), :math:`m` work wires are
        required. If ``output_wires_zeroed=True``, :math:`min(m, n+1)` work wires are required,
        where :math:`n` denotes the length of the first register ``x_wires``.

        **Cheaper decomposition for zeroed output state**

        If we know that the qubits in ``output_wires`` are in the state
        :math:`|0\rangle^{\otimes m}` before ``OutSquare`` is applied, we can pass this information
        to the template via ``output_wires_zeroed``, leading to a cheaper decomposition.
        Consider the following example, where we control this information with the ``QNode``
        argument ``zeroed``:

        .. code-block:: python

            n = 4
            m = 8
            x_wires = list(range(n))
            output_wires = list(range(n, n + m))
            work_wires = list(range(n + m, n + 2 * m))

            dev = qp.device("lightning.qubit", wires=20, seed=295)

            @qp.decompose(max_expansion=1) # To see resources easily
            @qp.qnode(dev, shots=1_000)
            def circuit(zeroed):
                qp.BasisEmbedding(13, wires=x_wires)
                qp.OutSquare(x_wires, output_wires, work_wires, output_wires_zeroed=zeroed)
                return qp.counts(wires=output_wires)

        We can compute the required resources with ``zeroed=False``, i.e., when not passing
        the information to the template:

        >>> specs_false = qp.specs(circuit)(False)["resources"].gate_types
        >>> print(specs_false)
        {'PauliX': 3, 'CNOT': 8, 'C(SemiAdder)': 4}

        When we do pass the information, we replace one controlled :class:`~.SemiAdder` by
        some :class:`~.TemporaryAND` gates and some of
        the other adders become smaller (depending on the register sizes):

        >>> specs_true = qp.specs(circuit)(True)["resources"].gate_types
        >>> print(specs_true)
        {'PauliX': 3, 'CNOT': 7, 'TemporaryAND': 3, 'C(SemiAdder)': 3}

        Of course, both decompositions are correctly implementing the squaring operation:

        >>> print(circuit(False))
        {np.str_('10101001'): np.int64(1000)}
        >>> print(circuit(True))
        {np.str_('10101001'): np.int64(1000)}

        Here, :math:`(10101001)_2=128 + 32 + 8 + 1=169` is the expected result of :math:`13^2`.
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

        num_required_work_wires = min(n + 1, m) if output_wires_zeroed else m
        if len(work_wires) < num_required_work_wires:
            raise ValueError(
                f"OutSquare requires at least {num_required_work_wires} work wires for "
                f"{n} input wires, {m} output wires "
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
    def _unflatten(cls, _, metadata):
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
            x_wires (WiresLike): wires that store the integer :math:`x`.
            output_wires (WiresLike): wires that store the squaring result. If the register
                initially encodes a non-zero value :math:`y`, the solution will be added to this
                value. If the register is guaranteed to be in the zero state, it is recommended to
                set ``output_wires_zeroed=True``.
            work_wires (WiresLike): the auxiliary wires to use for the squaring.
                :math:`m` work wires are required if ``output_wires_zeroed=False``,
                otherwise :math:`\min(m, n+1)` work wires are required.
            output_wires_zeroed (bool): Whether the output wires are guaranteed to be in the state
                :math:`|0\rangle` initially. Defaults to ``False``.

        Returns:
            list[.Operator]: Decomposition of the operator

        **Example**

        >>> all_wires = ([0, 1], [2, 3], [4, 5])
        >>> qp.OutSquare.compute_decomposition(*all_wires, output_wires_zeroed=True)
        [CNOT(wires=[1, 3]), TemporaryAND(wires=Wires([1, 0, 2])), CNOT(wires=[0, 4]), Controlled(SemiAdder(wires=[0, 1, 2, 5]), control_wires=[4]), CNOT(wires=[0, 4])]
        """
        with AnnotatedQueue() as q:
            _out_square_with_adder(x_wires, output_wires, work_wires, output_wires_zeroed)

        if QueuingManager.recording():
            for op in q.queue:
                apply(op)

        return q.queue


def _out_square_with_adder_condition(
    num_x_wires, num_output_wires, num_work_wires, output_wires_zeroed
) -> bool:
    n = num_x_wires
    m = num_output_wires
    largest_adder = min(m, n + 1) if output_wires_zeroed else m
    # work wires: one for control cache, largest_adder-1 for adder
    min_num_work_wires = 1 + (largest_adder - 1)
    return num_work_wires >= min_num_work_wires


def _out_square_with_adder_resources(
    num_x_wires, num_output_wires, num_work_wires, output_wires_zeroed
) -> dict:
    # pylint: disable=unused-argument
    n = num_x_wires
    m = num_output_wires
    resources = defaultdict(int)
    if output_wires_zeroed:
        # Copying of first bit is a CNOT, all other bits require a TemporaryAND
        resources[resource_rep(CNOT)] += 1
        resources[resource_rep(TemporaryAND)] = output_wires_zeroed * (min(n, m) - 1)

    # Controlled adders, includes the one for copying if output_wires_zeroed=False
    for i in range(output_wires_zeroed, min(n, m)):
        num_out = min(m - i, n + 1) if output_wires_zeroed else m - i
        resources[resource_rep(CNOT)] += 2
        add_params = {"num_x_wires": n, "num_y_wires": num_out, "num_work_wires": num_out - 1}
        ctrl_params = {
            "num_control_wires": 1,
            "num_work_wires": num_work_wires - num_out,
            "work_wire_type": "zeroed",
        }
        c_add_rep = controlled_resource_rep(SemiAdder, base_params=add_params, **ctrl_params)
        resources[c_add_rep] += 1
    return dict(resources)


@register_condition(_out_square_with_adder_condition)
@register_resources(_out_square_with_adder_resources)
def _out_square_with_adder(
    x_wires: WiresLike,
    output_wires: WiresLike,
    work_wires: WiresLike,
    output_wires_zeroed: bool,
    **_,
):
    n = len(x_wires)
    m = len(output_wires)

    if output_wires_zeroed:
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
        # shifted by i bit positions. For output_wires_zeroed=False, includes the initial copy
        # The output wires of the adder need to take all of the output register of square
        # into account due to carry values. For output_wires_zeroed=True, we can reduce to
        # a fixed size (`n`) instead, because we know at each step how large the value stored
        # in the output register can have grown by then.
        output_msb = max(0, m - n - i - 1) if output_wires_zeroed else 0
        output = output_wires[output_msb : m - i]
        CNOT([x_wire, work_wires[0]])
        ctrl(
            SemiAdder(x_wires=x_wires, y_wires=output, work_wires=work_wires[1 : len(output)]),
            control=work_wires[:1],
            work_wires=work_wires[len(output) :],
            work_wire_type="zeroed",
        )
        CNOT([x_wire, work_wires[0]])


def _out_square_with_caddsub_condition(
    num_x_wires, num_output_wires, num_work_wires, output_wires_zeroed
) -> bool:
    n = num_x_wires
    m = num_output_wires + 1
    # We have a series of controlled adders, each of which has a control cache qubit, an
    # output augmentation qubit, and (output size - 1) work wires. Note that while the controlled
    # adder needs one more work wire to decompose Toffolis to elbows, this is not strictly needed
    # to make the decomposition admissible. The largest output size of the series is
    # ╭min(m, n+1) if output_wires_zeroed
    # ╰m           else.
    # There are multiple correction steps, the largest work wire demand is by the addition of
    # 2^n-x, which uses an adder of size m+1 and thus needs m work wires.
    min_num_work_wires = max(min(m, n + 1) + 1, m) if output_wires_zeroed else m + 1
    return num_work_wires >= min_num_work_wires


def _out_square_with_caddsub_resources(
    num_x_wires, num_output_wires, num_work_wires, output_wires_zeroed
) -> dict:
    # pylint: disable=unused-argument
    n = num_x_wires
    m = num_output_wires + 1
    resources = defaultdict(int)

    cnot_rep = resource_rep(CNOT)
    cnot_on_0_kwargs = {"base_params": {}, "num_control_wires": 1, "num_zero_control_values": 1}
    cnot_on_0_rep = controlled_resource_rep(X, **cnot_on_0_kwargs)
    x_rep = resource_rep(X)

    # Controlled add-subtract loop
    loop_size = min(m, n)
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
    c_add_subs_with_work_wires = min(n, m - 1)
    resources[cnot_on_0_rep] += 2 * c_add_subs_with_work_wires

    # SemiAdder of x_wires onto output_wires: One per ctrl-add-subtract, varying size
    for i in range(loop_size):
        size = min(m - i, n + 1) if output_wires_zeroed else m - i
        adder_resources = _semiadder_resources(num_x_wires=n, num_y_wires=size)
        for key, value in adder_resources.items():
            resources[key] += value

    # Subtract 2^(2n)
    size = m - 2 * n
    if size > 0:
        if size > 1:
            resources[resource_rep(TemporaryAND)] += size - 2
            resources[adjoint_resource_rep(TemporaryAND)] += size - 2
        resources[resource_rep(CNOT)] += size - 1
        resources[x_rep] += 1

    # Add (2^n-1-x) + 1
    resources[x_rep] += 6 + 2 * n
    adder_resources = _semiadder_resources(num_x_wires=n, num_y_wires=m)
    for key, value in adder_resources.items():
        resources[key] += value

    # Add 2^(n+1) x if 2^m > 2^(n+1) (otherwise it just vanishes in the modulus)
    if m > n + 1:
        resources[
            resource_rep(SemiAdder, num_x_wires=n, num_y_wires=m - n - 1, num_work_wires=m - n - 2)
        ] += 1

    return dict(resources)


@register_condition(_out_square_with_caddsub_condition)
@register_resources(_out_square_with_caddsub_resources)
def _out_square_with_caddsub(
    x_wires: WiresLike,
    output_wires: WiresLike,
    work_wires: WiresLike,
    output_wires_zeroed: bool,
    **_,
):
    # First work wire is used for caching control bits
    c_wire = work_wires[0]
    # Second work wire is used to augment the output wires because we compute
    # twice the desired output at first, and then divide by 2.
    output_wires = output_wires + [work_wires[1]]
    work_wires = work_wires[2:]
    n = len(x_wires)
    m = len(output_wires)

    for i, x_wire in enumerate(x_wires[::-1][:m]):
        output_msb = max(0, m - (n + 1 + i)) if output_wires_zeroed else 0
        output = output_wires[output_msb : m - i]
        CNOT([x_wire, c_wire])
        _c_add_sub(c_wire, x_wires, output, work_wires)
        CNOT([x_wire, c_wire])

    # Corrections - no need for control wire any more
    work_wires = [c_wire] + work_wires

    # Subtract 2^(2n)
    if len(output_wires) > 2 * n:
        _output = output_wires[: m - 2 * n]
        adjoint(_increment, lazy=False)(_output, work_wires)

    # Add (2^n-1-x) + 1
    _ = [X(w) for w in x_wires]
    _add_plus_one(x_wires, output_wires, work_wires)
    _ = [X(w) for w in x_wires]

    # Add 2^(n+1) x if 2^m > 2^(n+1) (otherwise it just vanishes in the modulus)
    if m > n + 1:
        SemiAdder(x_wires, output_wires[: m - n - 1], work_wires[: m - n - 2])


add_decomps(OutSquare, _out_square_with_adder, _out_square_with_caddsub)
