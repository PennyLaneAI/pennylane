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

from pennylane.core.operator import Operation
from pennylane.decomposition import (
    add_decomps,
    adjoint_resource_rep,
    controlled_resource_rep,
    register_condition,
    register_resources,
)
from pennylane.decomposition.resources import resource_rep
from pennylane.ops import CNOT, X, adjoint, ctrl
from pennylane.wires import Wires, WiresLike

from .out_multiplier import _c_add_sub, _c_add_sub_resources
from .semi_adder import (
    SemiAdder,
    _ctrl_right_block,
    _ctrl_right_block_zeroed,
    _left_block,
    _left_block_zeroed,
    _right_block,
    _right_block_zeroed,
)
from .temporary_and import TemporaryAND


class OutSquare(Operation):
    r"""Performs out-of-place squaring.

    This operator performs the squaring of an :math:`n`-qubit integer :math:`x` modulo
    :math:`2^m` into an :math:`m`-qubit output register:

    .. math::
        \text{OutSquare} |x \rangle |y \rangle = |x \rangle |(y + x^2) \; \text{mod} \; 2^m \rangle.

    There are two implementations available, differing in their :class:`~.Toffoli` and auxiliary
    qubit counts. The first is based on Schoolbook multiplication, using a cache qubit and
    controlled addition. The second uses controlled add-subtract blocks that also are used by
    Litinski in `arXiv:2410.00899 <https://arxiv.org/abs/2410.00899>`__ to reduce the
    cost of multiplication.

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
    :math:`m=6`-qubit register that holds the value :math:`y=5` initially.
    The computation will be modulo :math:`2^m=2^6=64`.

    .. code-block:: python

        import pennylane as qp

        n = 3
        m = 6
        wires = qp.registers({"x": n, "out": m, "work": m})

        dev = qp.device("lightning.qubit", wires=n + 2 * m, seed=295)

        @qp.qnode(dev, shots=1_000)
        def circuit(output_wires):
            # Create a uniform superposition between integers 3 and 7
            qp.H(wires["x"][0]) # Superposition between 0 and 4
            qp.BasisEmbedding(3, wires=wires["x"][1:]) # Add 3, by preparing lower-precision wires
            # Prepare initial state on output wires
            qp.BasisEmbedding(5, wires=output_wires)
            # Square
            qp.OutSquare(wires["x"], output_wires, wires["work"])
            return qp.counts(wires=output_wires)

    >>> counts = circuit(wires["out"])
    >>> counts = {int(k, 2): val for k, val in counts.items()}
    >>> print(counts)
    {14: np.int64(498), 54: np.int64(502)}

    We correctly obtain the squared numbers added to :math:`y=5`, namely
    :math:`5+3^2=14` and :math:`5+7^2=54`.

    Note that reducing the size of the output register (here from ``m=6`` to ``m=4``)
    changes the computed numbers via the reduced modulus:

    >>> counts = circuit(wires["out"][:4])
    >>> counts = {int(k, 2): val for k, val in counts.items()}
    >>> print(counts)
    {6: np.int64(501), 14: np.int64(499)}

    The new results are consistent with the previous ones: the smaller output :math:`14` remains
    unchanged because :math:`14 < 16=2^4`, and :math:`54` is changed to :math:`54\!\mod\!2^4=6`.

    .. details::
        :title: Usage Details

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

        >>> specs_false = qp.specs(circuit)(False).resources.quantum_operations
        >>> print(specs_false)
        {'BasisEmbedding': 1, 'C(BasisState)': 4, 'MultiControlledX': 12, 'TemporaryAND': 19, 'CNOT': 49, 'Adjoint(TemporaryAND)': 19, 'PauliX': 28, 'SemiAdder': 2}

        When we do pass the information, we reduce the required resources by a lot:

        >>> specs_true = qp.specs(circuit)(True).resources.quantum_operations
        >>> print(specs_true)
        {'BasisEmbedding': 1, 'TemporaryAND': 11, 'CNOT': 22, 'MultiControlledX': 8, 'Adjoint(TemporaryAND)': 8}

        Of course, both decompositions are correctly implementing the squaring operation:

        >>> print(circuit(False))
        {np.str_('10101001'): np.int64(1000)}
        >>> print(circuit(True))
        {np.str_('10101001'): np.int64(1000)}

        Here, :math:`(10101001)_2=128 + 32 + 8 + 1=169` is the expected result of :math:`13^2`.
        To conclude, we draw the more efficient circuit variant:

        >>> print(qp.draw(circuit)(True))
         0: ─╭|Ψ⟩───────╭●───────────────────╭X────╭●───────────●╮─╭●────╭X─────────────────────────── ···
         1: ─├|Ψ⟩────╭●─│────────╭X────╭●────│─────│─────────────│─│─────│──────●╮─╭●────╭X─────────── ···
         2: ─├|Ψ⟩─╭●─│──│──╭●────│─────│─────│─────│─────╭●──────│─├●────│───────│─├●────│──────╭●──●╮ ···
         3: ─╰|Ψ⟩─├●─├●─├●─│─────│─────│─────│─────│─────│───────│─│─────│───────│─│─────│──────│────│ ···
         4: ──────│──│──│──│─────│─────│─────│─────│─────│───────│─│─────│───────│─│─────│──────│────│ ···
         5: ──────│──│──│──│─────│─────│─────│─────│─────├X──────│─│─────│───────│─│─────│──────│────│ ···
         6: ──────│──│──│──│─────│─────│─────│──╭X─├●────│──────●┤─╰X─╭X─│───────│─│─────│──────│────│ ···
         7: ──────│──│──╰⊕─│─────│──╭X─├●────│──│──│─────│───────│────│──│──────●┤─╰X─╭X─│──────│────│ ···
         8: ──────│──╰⊕────│──╭●─│──│──│─────│──│──│─────│───────│────│──│───────│────│──│───●╮─├X───│ ···
         9: ──────╰⊕───────├●─│──│──│──│─────│──│──│─────│───────│────│──│───────│────│──│────│─│───●┤ ···
        10: ───────────────│──│──│──│──│─────│──│──│─────│───────│────│──│───────│────│──│────│─│────│ ···
        11: ───────────────│──│──│──│──│─────│──│──│─────│───────│────│──│───────│────│──│────│─│────│ ···
        12: ───────────────╰⊕─├●─│──│──│─────│──│──│─────│───────│────│──│───────│────│──│───●┤─╰●──⊕╯ ···
        13: ──────────────────╰⊕─╰●─╰●─│──╭●─│──│──│─────│───────│────│──│──╭●───│────╰●─╰●──⊕╯─────── ···
        14: ───────────────────────────╰⊕─╰X─╰●─╰●─│──╭●─│──╭●───│────╰●─╰●─╰X──⊕╯──────────────────── ···
        15: ───────────────────────────────────────╰⊕─╰X─╰●─╰X──⊕╯──────────────────────────────────── ···
        <BLANKLINE>
         0: ··· ──────────╭X────╭●───────────●╮─╭●────╭X───────────────╭●─╭●──●╮─╭●────┤
         1: ··· ────╭●────│─────│─────╭●──────│─├●────│──────╭●──●╮─╭●─│──│────│─│─────┤
         2: ··· ─╭●─│─────│─────│─────│───────│─│─────│──────│────│─│──│──│────│─│─────┤
         3: ··· ─│──│─────│─────│─────│───────│─│─────│──────│────│─│──│──│────│─│──╭●─┤
         4: ··· ─│──│─────│─────│─────├X──────│─│─────│──────│────│─│──│──├X───│─│──│──┤ ╭Counts
         5: ··· ─│──│─────│──╭X─├●────│──────●┤─╰X─╭X─│──────│────│─│──├●─│───●┤─╰X─│──┤ ├Counts
         6: ··· ─│──│──╭●─│──│──│─────│───────│────│──│───●╮─├X───│─│──│──│────│────│──┤ ├Counts
         7: ··· ─│──├●─│──│──│──│─────│───────│────│──│────│─│───●┤─╰X─│──│────│────│──┤ ├Counts
         8: ··· ─│──│──│──│──│──│─────│───────│────│──│────│─│────│────│──│────│────│──┤ ├Counts
         9: ··· ─╰X─│──│──│──│──│─────│───────│────│──│────│─│────│────│──│────│────│──┤ ├Counts
        10: ··· ────│──│──│──│──│─────│───────│────│──│────│─│────│────│──│────│────│──┤ ├Counts
        11: ··· ────│──│──│──│──│─────│───────│────│──│────│─│────│────│──│────│────╰X─┤ ╰Counts
        12: ··· ────╰⊕─├●─│──│──│─────│───────│────│──│───●┤─╰●──⊕╯────╰⊕─╰●──⊕╯───────┤
        13: ··· ───────╰⊕─╰●─╰●─│──╭●─│──╭●───│────╰●─╰●──⊕╯───────────────────────────┤
        14: ··· ────────────────╰⊕─╰X─╰●─╰X──⊕╯────────────────────────────────────────┤
        15: ··· ───────────────────────────────────────────────────────────────────────┤

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

        num_required_work_wires = min(n - 1, m - 4) if output_wires_zeroed else m
        if len(work_wires) < num_required_work_wires:
            raise ValueError(
                f"OutSquare requires at least {num_required_work_wires} work wires for "
                f"{n} input wires, {m} output wires and {output_wires_zeroed=}."
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

    @classmethod
    def _primitive_bind_call(cls, *args, **kwargs):
        return cls._primitive.bind(*args, **kwargs)


def _out_square_with_adder_zeroed_condition(
    num_x_wires, num_output_wires, num_work_wires, output_wires_zeroed
) -> bool:
    if not output_wires_zeroed:
        return False
    return num_work_wires >= min(num_x_wires - 1, num_output_wires - 4)


def _self_ctrl_one_sparse_add_resources(num_x_wires, num_y_wires, num_work_wires):
    """Resources for _self_ctrl_one_sparse_add below."""
    if num_y_wires == 1:
        return {resource_rep(CNOT): 1}

    zeroed = _effective_zeros(num_x_wires, num_y_wires, zeroed=[1])
    num_elbows = num_y_wires - 1
    num_blocks = num_y_wires - 2
    num_zeroed_blocks = len(zeroed) - int(num_y_wires - 1 in zeroed)
    # each _left_block uses 3 CNOTs, each _left_block_zeroed none
    # each _ctrl_right_block uses 3 CNOTs, each _right_block_zeroed none
    # each _ctrl_right_block or _ctrl_right_block_zeroed uses 1 ctrl(CNOT)
    # the remaining construction uses 1 CNOT and 1 + int(num_y_wires-1 not in zeroed) ctrl(CNOT)s
    ccnot_rep = controlled_resource_rep(
        CNOT, {}, 1, 0, num_work_wires - num_elbows, work_wire_type="zeroed"
    )
    return {
        resource_rep(TemporaryAND): num_elbows,
        adjoint_resource_rep(TemporaryAND, {}): num_elbows,
        CNOT: 6 * (num_blocks - num_zeroed_blocks) + 1,
        ccnot_rep: num_blocks + (1 + int(num_y_wires - 1 not in zeroed)),
    }


def _self_ctrl_one_sparse_add(x_wires, y_wires, work_wires):
    """Specialized arithmetic unit: Addition controlled on the first qubit of the first addend,
    with a classically fixed bit in state |0> injected into first addend register at the position
    of the 2's bit. All other input bits are shifted in position.

    Effectively, we are adding :math:`x_0 * (x_{n-1} x_{n-2} ... x_1 0 x_0)_2` to ``y_wires``.
    """
    num_y_wires = len(y_wires)
    if num_y_wires == 1:
        CNOT([x_wires[0], y_wires[0]])
        return

    # Set up control structure for controlled ops within decomposition
    ctrl_kwargs = {
        "control": x_wires[:1],
        "control_values": [1],
        "work_wires": work_wires[num_y_wires - 1 :],  # Pass only additional work qubits
        "work_wire_type": "zeroed",
    }

    num_x_wires = len(x_wires)
    # We use static zeroed=[1] for this subroutine
    zeroed = _effective_zeros(num_x_wires, num_y_wires, zeroed=[1])
    work_wires = work_wires[: num_y_wires - 1]

    TemporaryAND([x_wires[0], y_wires[0], work_wires[0]])
    x_pos = 1
    for i in range(1, num_y_wires - 1):
        if i in zeroed:
            _left_block_zeroed([work_wires[i - 1], y_wires[i], work_wires[i]])
        else:
            _left_block([work_wires[i - 1], x_wires[x_pos], y_wires[i], work_wires[i]])
            x_pos += 1

    ctrl(CNOT([work_wires[-1], y_wires[-1]]), **ctrl_kwargs)
    if num_y_wires - 1 not in zeroed:
        ctrl(CNOT([x_wires[-1], y_wires[-1]]), **ctrl_kwargs)

    x_pos -= 1
    for i in range(num_y_wires - 2, 0, -1):
        if i in zeroed:
            _ctrl_right_block_zeroed([work_wires[i - 1], y_wires[i], work_wires[i]], **ctrl_kwargs)
        else:
            _wires = [work_wires[i - 1], x_wires[x_pos], y_wires[i], work_wires[i]]
            _ctrl_right_block(_wires, **ctrl_kwargs)
            x_pos -= 1

    adjoint(TemporaryAND([x_wires[0], y_wires[0], work_wires[0]]))
    CNOT([x_wires[0], y_wires[0]])


def _out_square_with_adder_zeroed_resources(
    num_x_wires, num_output_wires, num_work_wires, **_
) -> dict:
    # pylint: disable=unused-argument
    n = num_x_wires
    m = num_output_wires
    resources = defaultdict(int)
    # Copying of first bit is a CNOT, all other bits require a TemporaryAND
    resources[resource_rep(CNOT)] += 1
    resources[resource_rep(TemporaryAND)] = min(n - 1, m - 2)
    num_work_wires += 2  # Using the 1s and 2s output bits as work wires
    p = min(n, m // 2 + m % 2) - 1
    for i in range(1, p + 1):
        x_size = num_x_wires - i
        y_size = min(m - 2 * i, n + 2 - i)
        for k, val in _self_ctrl_one_sparse_add_resources(x_size, y_size, num_work_wires).items():
            resources[k] += val

    return dict(resources)


@register_condition(_out_square_with_adder_zeroed_condition)
@register_resources(_out_square_with_adder_zeroed_resources, name="out_square_with_adder")
def _out_square_with_adder_zeroed(
    x_wires: WiresLike,
    output_wires: WiresLike,
    work_wires: WiresLike,
    *_,
    **__,
):
    n = len(x_wires)
    m = len(output_wires)
    x_wires = x_wires[::-1]
    output_wires = output_wires[::-1]
    work_wires = Wires.all_wires([work_wires, output_wires[:2]])

    # Copy x, controlled on the least significant bit (LSB) of x, to the output register,
    # which is in |0>. This can be reduced to a CNOT for the LSB and TemporaryANDs for
    # the other bits. The CNOT is performed at the very end, to use the 1s bit of the output as
    # work wire until then.
    num_elbows = min(n - 1, m - 2)
    copy_input = x_wires[1 : num_elbows + 1]
    copy_output = output_wires[2 : num_elbows + 2]
    for x_wire, out_wire in zip(copy_input, copy_output, strict=True):
        TemporaryAND([x_wires[0], x_wire, out_wire])

    p = min(n, m // 2 + (m % 2)) - 1
    for i in range(1, p + 1):
        # Perform specialized "self"-controlled addition with zeroed 2s input bit, using
        # sliced x_wires and output_wires.
        _self_ctrl_one_sparse_add(x_wires[i:], output_wires[2 * i : min(m, n + 2 + i)], work_wires)

    CNOT([x_wires[0], output_wires[0]])  # First control-copy, delayed until end of decomp.


def _out_square_with_caddsub_condition(
    num_x_wires, num_output_wires, num_work_wires, output_wires_zeroed
) -> bool:
    # pylint: disable=unused-argument
    return num_work_wires >= num_output_wires - 1


def _out_square_with_caddsub_resources(
    num_x_wires, num_output_wires, num_work_wires, output_wires_zeroed
) -> dict:
    # pylint: disable=unused-argument
    n = num_x_wires
    m = num_output_wires
    p = min(n - 1, m // 2)

    resources = defaultdict(int)

    # Controlled add-subtract loop
    for i in range(p):
        size = min(n - i, m - 2 * i - 1) if output_wires_zeroed else m - 2 * i - 1
        for key, value in _c_add_sub_resources(n - i - 1, size).items():
            resources[key] += value

    for key, value in _sparse_adder_resources(n, m, [1] + [2 * j for j in range(1, n)]).items():
        resources[key] += value

    # Subtract 2 x_{[1:]}
    resources[resource_rep(X)] += 2 * (m - 1)
    resources[
        resource_rep(SemiAdder, num_x_wires=n - 1, num_y_wires=m - 1, num_work_wires=num_work_wires)
    ] += 1

    if m > n:
        # Shifted addition
        resources[resource_rep(X)] += 2 * (m - n + n - 1)
        resources[
            resource_rep(
                SemiAdder, num_x_wires=n - 1, num_y_wires=m - n, num_work_wires=num_work_wires
            )
        ] += 1

    return dict(resources)


def _effective_zeros(num_x_wires, num_y_wires, zeroed):
    used_x = 0
    new_zeroed = []
    for i in range(num_y_wires):
        if i in zeroed or used_x >= num_x_wires:
            new_zeroed.append(i)
        else:
            used_x += 1
    return new_zeroed


def _sparse_adder_resources(num_x_wires, num_y_wires, zeroed):
    if num_y_wires == 1:
        return {CNOT: 1}

    zeroed = _effective_zeros(num_x_wires, num_y_wires, zeroed)
    num_elbows = num_y_wires - 1
    num_zeroed_blocks = len(zeroed) - int(num_y_wires - 1 in zeroed)
    num_nonzeroed_blocks = (num_y_wires - 2) - num_zeroed_blocks
    # each _left_block uses 3 CNOTs, each _left_block_zeroed none
    # each _right_block uses 3 CNOTs, each _right_block_zeroed just 1 CNOT
    # the remaining construction uses 2 + int(num_y_wires-1 not in zeroed) CNOTs
    return {
        resource_rep(TemporaryAND): num_elbows,
        adjoint_resource_rep(TemporaryAND, {}): num_elbows,
        CNOT: num_zeroed_blocks + 6 * num_nonzeroed_blocks + 2 + int(num_y_wires - 1 not in zeroed),
    }


def _sparse_adder(x_wires, y_wires, work_wires, zeroed):
    """Perform sparse addition, i.e., addition of ``x_wires`` interlaced with zeroed bits
    specified by ``zeroed``. It is assumed that 0 is not in ``zeroed``.
    This function assumes little endian ordering!
    """
    num_y_wires = len(y_wires)
    if num_y_wires == 1:
        CNOT([x_wires[0], y_wires[0]])
        return
    assert zeroed[0] != 0

    num_x_wires = len(x_wires)
    zeroed = _effective_zeros(num_x_wires, num_y_wires, zeroed)
    work_wires = work_wires[: num_y_wires - 1]

    TemporaryAND([x_wires[0], y_wires[0], work_wires[0]])

    x_pos = 1
    for i in range(1, num_y_wires - 1):
        if i in zeroed:
            _left_block_zeroed([work_wires[i - 1], y_wires[i], work_wires[i]])
        else:
            _left_block([work_wires[i - 1], x_wires[x_pos], y_wires[i], work_wires[i]])
            x_pos += 1

    CNOT([work_wires[-1], y_wires[-1]])

    if num_y_wires - 1 not in zeroed:
        CNOT([x_wires[x_pos], y_wires[-1]])

    x_pos -= 1
    for i in range(num_y_wires - 2, 0, -1):
        if i in zeroed:
            _right_block_zeroed([work_wires[i - 1], y_wires[i], work_wires[i]])
        else:
            _right_block([work_wires[i - 1], x_wires[x_pos], y_wires[i], work_wires[i]])
            x_pos -= 1

    adjoint(TemporaryAND([x_wires[0], y_wires[0], work_wires[0]]))
    CNOT([x_wires[0], y_wires[0]])


def _shifted_adder(x_wires, output_wires, work_wires):
    """Perform shifted addition y -> y + x - 2^n + 1."""
    _ = [X(w) for w in x_wires]
    _ = [X(w) for w in output_wires]
    SemiAdder(x_wires[::-1], output_wires[::-1], work_wires)
    _ = [X(w) for w in output_wires]
    _ = [X(w) for w in x_wires]


@register_condition(_out_square_with_caddsub_condition)
@register_resources(_out_square_with_caddsub_resources)
def _out_square_with_caddsub(
    x_wires: WiresLike,
    output_wires: WiresLike,
    work_wires: WiresLike,
    output_wires_zeroed: bool,
    **_,
):
    r"""This decomposition uses controlled add-subtract blocks.
    We start with augmenting the output register by one auxiliary qubit that we add as the new
    least significant bit (This basically multiplies with two) and applying n controlled
    add-subtract operations. The j-th operation (starting at j=0) caches x_{n-1-j} in a cache qubit
    ``c_wire`` (which we take from the work wires) and then adds or subtracts x into a subregister
    of y, starting at the j-th least significant bit, controlled on the cache. This computes

    ..math :: A = \sum_{j=0}^{n-1} \left(2^j (2x_j-1)x + 2^{n+j}(1-x_j)\right)

    ..math :: =2x^2-x\left(\sum_{j=0}^{n-1}2^j\right) + 2^n \left(\sum_{j=0}^{n-1} 2^j\right) - 2^n x

    ..math :: =2x^2-2^{n}x+x+2^n(2^n-1-x)

    ..math :: =2x^2-2^{n+1}x-(2^n-x)+2^{2n}.

    Then, we subtract 2^{2n}, add (2^n-x), and add 2^{n+1}x to arrive at 2x^2 overall.
    In a last step we divide by two simply by splitting off the auxiliary qubit we had added as
    LSB in the beginning (we do not have to do anything for this in code).
    This qubit is guaranteed to be zeroed because we computed an even number before.
    """
    x_wires = x_wires[::-1]
    output_wires = output_wires[::-1]
    n = len(x_wires)
    m = len(output_wires)
    p = min(n - 1, m // 2)

    for i, x_wire in enumerate(x_wires[:p]):
        _in_reg = x_wires[i + 1 :]
        _out_reg = (
            output_wires[2 * i + 1 : n + 1 + i]
            if output_wires_zeroed
            else output_wires[2 * i + 1 :]
        )
        _c_add_sub(x_wire, _in_reg[::-1], _out_reg[::-1], work_wires)

    _sparse_adder(x_wires, output_wires, work_wires, zeroed=[1] + [2 * j for j in range(1, n)])

    if n > 1 and m > 1:
        _ = [X(w) for w in output_wires[1:]]
        SemiAdder(x_wires[1:][::-1], output_wires[1:][::-1], work_wires)
        _ = [X(w) for w in output_wires[1:]]

    # shifted addition
    if m > n > 1:
        _shifted_adder(x_wires[:-1], output_wires[n:], work_wires)


add_decomps(
    OutSquare,
    _out_square_with_adder_zeroed,
    _out_square_with_caddsub,
)
