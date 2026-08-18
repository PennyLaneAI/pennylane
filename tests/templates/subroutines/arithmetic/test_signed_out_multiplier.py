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
Tests for the SignedOutMultiplier template.
"""

from collections import Counter
from functools import reduce

import numpy as np
import pytest

import pennylane as qp
from pennylane import SignedOutMultiplier, device, qnode
from pennylane.core.operator import abstractify
from pennylane.decomposition import list_decomps
from pennylane.decomposition.resources import controlled_resource_rep
from pennylane.measurements import sample, state
from pennylane.ops import CNOT
from pennylane.ops.functions.assert_valid import _test_decomposition_rule, assert_valid
from pennylane.templates import BasisEmbedding
from pennylane.templates.subroutines.arithmetic.incrementer import Incrementer
from pennylane.templates.subroutines.arithmetic.out_multiplier import OutMultiplier
from pennylane.templates.subroutines.arithmetic.semi_adder import SemiAdder
from pennylane.templates.subroutines.arithmetic.signed_out_multiplier import (
    _not_zeroed_signed_out_multiplier_resources,
    _twos_complement_helper,
    _zeroed_signed_out_multiplier_resources,
)
from pennylane.typing import Wire


def bin_to_int(bits):
    """Converts a binary array to an integer."""
    return int("".join(map(str, bits)), 2)


def int_to_bin(integer, pd=""):
    """Converts an integer to a binary array."""
    if integer < 0:
        bin_str = format(integer, f"#0{pd}b")[3:]
    else:
        bin_str = format(integer, f"#0{pd}b")[2:]
    return list(reduce(lambda acc, nxt: acc + [int(nxt)], bin_str, []))


def twos_complement_value(bits):
    """Calculates the value of a number encoded as a twos complement."""
    sum = 0
    for i, bit in enumerate(bits[1:][::-1]):
        sum += (2**i) * bit
    sum -= (2 ** (len(bits) - 1)) * bits[0]
    return sum


@pytest.mark.parametrize(
    (
        "x_wires",
        "y_wires",
        "output_wires",
        "work_wires",
        "output_wires_zeroed",
        "expected_num_work_wires",
    ),
    [
        ((0, 1, 2), (3, 4, 5), (10, 11, 12, 13, 14, 15), (6, 7, 8, 9), True, 4),
    ],
)
def test_abstract_init(
    x_wires, y_wires, output_wires, work_wires, output_wires_zeroed, expected_num_work_wires
):  # pylint: disable=too-many-arguments
    """Test that abstract init mirrors concrete init."""
    abstract_op = SignedOutMultiplier(
        Wire[len(x_wires)],
        Wire[len(y_wires)],
        Wire[len(output_wires)],
        Wire[len(work_wires)],
        output_wires_zeroed=output_wires_zeroed,
    )
    assert abstract_op.arguments["output_wires_zeroed"] is output_wires_zeroed
    assert len(abstract_op.work_wires) == expected_num_work_wires

    concrete_op = SignedOutMultiplier(
        x_wires, y_wires, output_wires, work_wires, output_wires_zeroed=output_wires_zeroed
    )
    assert abstractify(concrete_op) == abstract_op


def test_wires_property():
    """Test that wires includes all registers, including work wires."""
    op = SignedOutMultiplier([0], [1], [2, 3], [4, 5])
    assert op.wires == qp.wires.Wires([0, 1, 2, 3, 4, 5])


def test_signed_out_multiplier_resources():
    """Test resource functions declare expected abstract operator and gate counts."""
    x_wires = [0, 1]
    y_wires = [2, 3]
    output_wires = [4, 5, 6]
    work_wires = [7, 8, 9, 10]
    num_incrementer_work_wires = len(work_wires) - 2

    zeroed_resources = _zeroed_signed_out_multiplier_resources(
        x_wires, y_wires, output_wires, work_wires, output_wires_zeroed=True
    )
    mult_ops = [key for key in zeroed_resources if isinstance(key, OutMultiplier)]
    assert len(mult_ops) == 1
    mult_op = mult_ops[0]
    assert mult_op.arguments["mod"] == 2 ** (len(output_wires) - 1)
    assert len(mult_op.work_wires) == num_incrementer_work_wires
    assert mult_op.arguments["output_wires_zeroed"] is True
    assert zeroed_resources[CNOT] == 6 + (len(x_wires) + len(y_wires)) * 2 + (len(output_wires) - 1)

    expected_incrementers = Counter()
    for num_wires, count in (
        (len(x_wires), 2),
        (len(output_wires) - 1, 1),
        (len(y_wires), 2),
    ):
        inc_rep = controlled_resource_rep(
            Incrementer,
            {"num_wires": num_wires, "num_work_wires": num_incrementer_work_wires},
            num_control_wires=1,
        )
        expected_incrementers[inc_rep] += count

    for inc_rep, count in expected_incrementers.items():
        assert zeroed_resources[inc_rep] == count

    not_zeroed_resources = _not_zeroed_signed_out_multiplier_resources(
        x_wires, y_wires, output_wires, work_wires
    )
    nested_ops = [key for key in not_zeroed_resources if isinstance(key, SignedOutMultiplier)]
    assert len(nested_ops) == 1
    assert nested_ops[0].arguments["output_wires_zeroed"] is True

    semi_adder_rep = SemiAdder(
        Wire[len(output_wires)],
        Wire[len(output_wires)],
        Wire[len(output_wires) - 1],
    )
    assert not_zeroed_resources[semi_adder_rep] == 1


@pytest.mark.jax
@pytest.mark.parametrize(
    "x_wires, y_wires, work_wires, output_wires, zeroed",
    [
        ((0, 1, 2), (3, 4, 5), (6, 7, 8, 9), (10, 11, 12, 13, 14, 15), True),
        ((0, 1), (2, 3), (4, 5, 6, 7, 8), (9, 10), False),
    ],
)
@pytest.mark.usefixtures("enable_and_disable_capture")
def test_assert_valid(x_wires, y_wires, work_wires, output_wires, zeroed):
    op = SignedOutMultiplier(x_wires, y_wires, output_wires, work_wires, zeroed)
    assert_valid(op)


@pytest.mark.parametrize(
    ("x_wires", "y_wires", "output_wires", "work_wires", "msg_match"),
    [
        (
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [1, 10],
            "None of the wires in work_wires should be included in x_wires.",
        ),
        (
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [3, 10],
            "None of the wires in work_wires should be included in y_wires.",
        ),
        (
            [0, 1, 2],
            [2, 4, 5],
            [6, 7, 8],
            [9, 10],
            "None of the wires in y_wires should be included in x_wires.",
        ),
        (
            [0, 1, 2],
            [3, 7, 5],
            [6, 7, 8],
            [9, 10],
            "None of the wires in output_wires should be included in y_wires.",
        ),
        (
            [0, 1, 7],
            [3, 4, 5],
            [6, 7, 8],
            [9, 10],
            "None of the wires in output_wires should be included in x_wires.",
        ),
    ],
)
def test_wires_error(x_wires, y_wires, output_wires, work_wires, msg_match):
    """Test an error is raised when some work_wires don't meet the requirements"""
    with pytest.raises(ValueError, match=msg_match):
        SignedOutMultiplier(x_wires, y_wires, output_wires, work_wires)


@pytest.mark.parametrize(
    "x_wires, y_wires, work_wires, output_wires, zeroed",
    [
        ((0, 1, 2), (3, 4, 5), (6, 7, 8, 9), (10, 11, 12, 13, 14, 15), True),
        ((0, 1), (2, 3), (4, 5, 6, 7, 8), (9, 10), False),
    ],
)
@pytest.mark.usefixtures("enable_and_disable_capture")
def test_decomposition(x_wires, y_wires, work_wires, output_wires, zeroed):
    op = SignedOutMultiplier(x_wires, y_wires, output_wires, work_wires, zeroed)

    for rule in list_decomps(SignedOutMultiplier):
        _test_decomposition_rule(op, rule)


@pytest.mark.capture
@pytest.mark.parametrize(
    "rule_name, registers, expected_primitives",
    [
        (
            "_signed_out_multiplier_decomposition_zeroed",
            ([0, 1, 2], [3, 4, 5], [10, 11, 12, 13, 14, 15], [6, 7, 8, 9]),
            {"for_loop": 5, "OutMultiplier": 1},
        ),
        (
            "_signed_out_multiplier_decomposition_not_zeroed",
            ([0, 1], [2, 3], [9, 10], [4, 5, 6, 7, 8]),
            {"concatenate": 2, "SignedOutMultiplier": 1},
        ),
    ],
)
def test_decomposition_with_abstract_wires(rule_name, registers, expected_primitives):
    """Test the decomposition rules with every register passed as an abstract wire argument."""
    jnp = pytest.importorskip("jax.numpy")
    rule = list_decomps(SignedOutMultiplier)[rule_name]

    def decomposition(x_wires, y_wires, output_wires, work_wires):
        rule(
            x_wires=x_wires,
            y_wires=y_wires,
            output_wires=output_wires,
            work_wires=work_wires,
        )

    plxpr = qp.capture.make_plxpr(decomposition, autograph=False)(
        *(jnp.array(register) for register in registers)
    )
    primitive_names = [eqn.primitive.name for eqn in plxpr.jaxpr.eqns]
    for eqn in plxpr.jaxpr.eqns:
        if eqn.primitive.name == "operator":
            primitive_names.append(eqn.params["op_cls"].__name__)

    for primitive, expected_count in expected_primitives.items():
        assert primitive_names.count(primitive) == expected_count


@pytest.mark.parametrize(
    "x_wires, y_wires, work_wires, output_wires, init_state, zeroed",
    [
        (
            (0, 1),
            (2, 3),
            (4, 5, 6, 7, 8),
            (9, 10),
            [1, 1]  # operand one: -1
            + [0, 1]  # operand two: 1
            + [0, 0, 0, 0, 0]  # work wires are zeroed
            + [0, 1],  # output register starts in non-zero state!
            False,
        ),
        (
            (0, 1),
            (2, 3),
            (4, 5, 6, 7, 8),
            (9, 10),
            [0, 1]  # operand one: 1
            + [0, 1]  # operand two: 1
            + [0, 0, 0, 0, 0]  # work wires are zeroed
            + [1, 1],  # output register starts in negative non-zero state!
            False,
        ),
        (
            (0, 1, 2),
            (3, 4),
            (6, 7, 8, 9),
            (10, 11, 12, 13, 14, 15),
            [1, 0, 1]  # operand one: -3
            + [1, 1]  # operand two: -1
            + [0, 0, 0, 0]  # work wires are zeroed
            + [0, 0, 0, 0, 0, 0],  # output register starts in |0>
            True,
        ),
        (
            (0, 1, 2),
            (3, 4, 5),
            (6, 7, 8, 9),
            (10, 11, 12, 13, 14, 15),
            [1, 0, 1]  # operand one: -3
            + [0, 1, 1]  # operand two: 3
            + [0, 0, 0, 0]  # work wires are zeroed
            + [0, 0, 0, 0, 0, 0],  # output register starts in |0>
            True,
        ),
        (
            (0, 1, 2),
            (3, 4, 5),
            (6, 7, 8, 9),
            (10, 11, 12, 13, 14, 15),
            [1, 1, 1]  # operand one: -1
            + [1, 0, 1]  # operand two: -3
            + [0, 0, 0, 0]  # work wires are zeroed
            + [0, 0, 0, 0, 0, 0],  # output register starts in |0>
            True,
        ),
        (
            (0, 1, 2),
            (3, 4, 5),
            (6, 7, 8, 9),
            (10, 11, 12, 13, 14, 15),
            [1, 0, 0]  # operand one: -4
            + [0, 1, 1]  # operand two: 3
            + [0, 0, 0, 0]  # work wires are zeroed
            + [0, 0, 0, 0, 0, 0],  # output register starts in |0>
            True,
        ),
    ],
)
def test_signed_out_multiplier_correct(
    x_wires, y_wires, work_wires, output_wires, init_state, zeroed
):  # pylint: disable=too-many-arguments
    """Tests with a few examples that the Template yields correct results."""

    dev = device("default.qubit", wires=x_wires + y_wires + work_wires + output_wires)

    @qnode(dev)
    def signed_multiply(
        x_wires, y_wires, work_wires, output_wires, init_state, zeroed
    ):  # pylint: disable=too-many-arguments
        BasisEmbedding(
            init_state,
            x_wires + y_wires + work_wires + output_wires,
        )
        SignedOutMultiplier(x_wires, y_wires, output_wires, work_wires, output_wires_zeroed=zeroed)
        return state()

    # get the initial state of our inputs
    x_state = [init_state[x] for x in x_wires]
    y_state = [init_state[y] for y in y_wires]

    # get the integer value of the x input
    if init_state[0] == 1:
        # get the value encoded using twos complement if it is negative
        x = twos_complement_value(x_state)
    else:
        # otherwise just convert from binary to int
        x = bin_to_int(x_state)

    # get the integer value of the y input
    if init_state[3] == 1:
        # get the value encoded using twos complement if it is negative
        y = twos_complement_value(y_state)
    else:
        # otherwise just convert from binary to int
        y = bin_to_int(y_state)

    # get initial output register value
    if zeroed:
        z = twos_complement_value(init_state[-6:])
    else:
        z = twos_complement_value(init_state[-2:])

    # calculate the expected result
    expected = x * y + z

    # execute the quantum signed out multiplier circuit
    result = signed_multiply(x_wires, y_wires, work_wires, output_wires, init_state, zeroed)

    # convert to bitstring
    # isclose will not match entries with wrong phase
    bin_result = int_to_bin(
        np.where(np.isclose(result, 1.0))[0][0] % (2 ** len(output_wires)), pd=len(output_wires)
    )

    # get the value encoded as a twos complement if the result is negative
    if bin_result[0] == 1:
        result = twos_complement_value(bin_result)
    else:
        result = bin_to_int(bin_result)

    assert result == expected


@pytest.mark.parametrize(
    "aux, wires, init_state, work_wires, expected",
    [
        (3, [0, 1, 2], [1, 1, 1], [4, 5], [0, 0, 1]),  # -1
        (3, [0, 1, 2], [1, 1, 0], [4, 5], [0, 1, 0]),  # -2
        (3, [0, 1, 2], [1, 0, 1], [4, 5], [0, 1, 1]),  # -3
        (3, [0, 1, 2], [1, 0, 0], [4, 5], [1, 0, 0]),  # -4
    ],
)
def test_twos_complement_helper(aux, wires, init_state, work_wires, expected):
    """Tests that the twos complement helper works correctly."""

    dev = device("default.qubit")

    @qnode(dev, shots=1)
    def twos_complement(aux, wires, init_state, work_wires):
        # load value
        BasisEmbedding(init_state, wires)

        # sign extend
        CNOT([wires[0], aux])

        # calculate twos complement
        _twos_complement_helper(wires, aux, work_wires)

        # measure
        return sample(wires=wires)

    expected_calc = -twos_complement_value(init_state)
    assert expected_calc == bin_to_int(expected)

    result = twos_complement(aux, wires, init_state, work_wires)[0]
    assert np.all(result == expected)
