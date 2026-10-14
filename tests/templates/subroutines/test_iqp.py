# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

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
Unit tests for the :func:`pennylane.template.subroutines.iqp` class.
"""

import re
from itertools import combinations

import numpy as np
import pytest

import pennylane as qp
from pennylane import math
from pennylane.core.operator import abstractify
from pennylane.decomposition import list_decomps
from pennylane.ops import PPR_4, H, MultiRZ
from pennylane.ops.functions.assert_valid import _test_decomposition_rule, assert_valid
from pennylane.templates.subroutines.iqp import IQP
from pennylane.typing import AbstractArray, AbstractWires


def local_gates(n_qubits: int, max_weight=2):
    """
    Generates a gate list for containing all gates whose generators have Pauli weight
    less or equal than max_weight.
    :param n_qubits: The number of qubits in the gate list
    :param max_weight: maximum Pauli weight of gate generators
    :return (list[list[list[int]]]): gate list
    """
    gates = []
    for weight in math.arange(1, max_weight + 1):
        for gate in combinations(math.arange(n_qubits), weight):
            gates.append([list(gate)])
    return gates


@pytest.mark.parametrize(
    ("params", "error", "match"),
    [
        (
            ([0], [0, 1], [[0, 1], [0]], False),
            ValueError,
            "Number of gates and number of parameters for an Instantaneous Quantum Polynomial circuit must be the same",
        ),
        (
            ([0, 1], [], [[0, 1], [0]], False),
            ValueError,
            "At least one valid wire",
        ),
    ],
)
def test_raises(params, error, match):
    with pytest.raises(error, match=re.escape(match)):
        IQP(*params)


@pytest.mark.parametrize(
    ("weights", "pattern", "spin_sym", "wires"),
    [
        (
            math.random.uniform(0, 2 * np.pi, 4),
            local_gates(4, 1),
            False,
            [0, 1, 2, 3],
        ),
        (
            math.random.uniform(0, 2 * np.pi, 6),
            local_gates(6, 1),
            True,
            range(6),
        ),
        # multi-qubit (Pauli weight 2) generators exercise the multi-wire ``MultiRZ`` path
        (
            math.random.uniform(0, 2 * np.pi, len(local_gates(4, 2))),
            local_gates(4, 2),
            False,
            [0, 1, 2, 3],
        ),
        (
            math.random.uniform(0, 2 * np.pi, len(local_gates(4, 2))),
            local_gates(4, 2),
            True,
            [0, 1, 2, 3],
        ),
    ],
)
@pytest.mark.usefixtures("enable_and_disable_capture")
def test_decomposition_new(weights, pattern, spin_sym, wires):  # pylint: disable=too-many-arguments
    op = IQP(weights, wires, pattern, spin_sym)

    for rule in list_decomps(IQP):
        _test_decomposition_rule(op, rule)


@pytest.mark.parametrize(
    ("weights", "pattern", "spin_sym", "wires", "expected_circuit"),
    [
        (
            math.random.uniform(0, 2 * np.pi, 4),
            local_gates(4, 1),
            True,
            ["a", "b", "c", "d"],
            [PPR_4, H, H, H, H, MultiRZ, MultiRZ, MultiRZ, MultiRZ, H, H, H, H],
        ),
        (
            math.random.uniform(0, 2 * np.pi, 4),
            local_gates(4, 1),
            False,
            ["a", "b", "c", "d"],
            [H, H, H, H, MultiRZ, MultiRZ, MultiRZ, MultiRZ, H, H, H, H],
        ),
    ],
)
def test_decomposition_contents(
    weights, pattern, spin_sym, wires, expected_circuit
):  # pylint: disable=too-many-arguments
    op = IQP(weights, wires, pattern, spin_sym)
    decomp = op.decomposition()

    assert [type(o) for o in decomp] == expected_circuit


@pytest.mark.usefixtures("enable_and_disable_capture")
@pytest.mark.parametrize("spin_sym", [False, True])
@pytest.mark.parametrize("max_weight", [1, 2])
def test_standard_validity(spin_sym, max_weight):
    """Test that IQP satisfies the standard ``Operator2`` validity checks."""
    pattern = local_gates(4, max_weight)
    weights = math.random.uniform(0, 2 * np.pi, len(pattern))
    op = IQP(weights, [0, 1, 2, 3], pattern, spin_sym)
    assert_valid(op, skip_differentiation=True)


class TestAttributes:
    """Tests for the argument classification and stored data of the migrated operator."""

    def test_data(self):
        """Test that weights are exposed as the operator's (trainable) data."""
        op = IQP([0.1, 0.2], [0, 1], [[[0]], [[1]]], spin_sym=False)
        assert len(op.data) == 1
        assert math.allclose(op.data[0], [0.1, 0.2])

    def test_pattern_canonicalized_to_tuples(self):
        """Test that pattern is stored as nested tuples (required for pytree metadata)."""
        op = IQP([0.1, 0.2], [0, 1], [[[0]], [[1]]], spin_sym=False)
        assert op.arguments["pattern"] == (((0,),), ((1,),))

    def test_abstractify(self):
        """Test that abstractify replaces the dynamic weights and wires with abstract types while
        preserving the compilable pattern and spin_sym arguments."""
        op = IQP([0.1, 0.2], [0, 1], [[[0]], [[1]]], spin_sym=True)
        abstract_op = abstractify(op)

        assert isinstance(abstract_op, IQP)
        assert isinstance(abstract_op.arguments["weights"], AbstractArray)
        assert abstract_op.arguments["weights"].shape == (2,)
        assert abstract_op.arguments["wires"] == AbstractWires(2)
        assert abstract_op.arguments["pattern"] == (((0,),), ((1,),))
        assert abstract_op.arguments["spin_sym"] is True
        assert abstract_op.is_fully_abstract


class TestMatrix:
    """Tests for ``IQP.compute_matrix`` against independent references."""

    def test_single_qubit_generator(self):
        """Test that a single-qubit generator equals ``RX(2 * theta)``."""
        theta = 0.7
        op = IQP([theta], [0], [[[0]]], spin_sym=False)
        assert math.allclose(op.matrix(), qp.RX(2 * theta, 0).matrix())

    def test_two_qubit_generator(self):
        """Test that a two-qubit generator equals ``PauliRot(2 * theta, 'XX')``."""
        theta = 0.7
        op = IQP([theta], [0, 1], [[[0, 1]]], spin_sym=False)
        assert math.allclose(op.matrix(), qp.PauliRot(2 * theta, "XX", [0, 1]).matrix())

    def test_multiple_generators(self):
        """Test that commuting single-qubit generators equal the product of ``RX`` rotations."""
        a, b = 0.3, 0.9
        op = IQP([a, b], [0, 1], [[[0]], [[1]]], spin_sym=False)
        reference = qp.matrix(
            qp.tape.QuantumScript([qp.RX(2 * a, 0), qp.RX(2 * b, 1)]), wire_order=[0, 1]
        )
        assert math.allclose(op.matrix(), reference)

    def test_spin_sym_prepends_ppr(self):
        """Test that ``spin_sym=True`` multiplies the ``spin_sym=False`` matrix by the PPR factor."""
        num_wires = 3
        weights = math.random.uniform(0, 2 * np.pi, num_wires)
        pattern = [[[i]] for i in range(num_wires)]
        without = IQP(weights, range(num_wires), pattern, spin_sym=False).matrix()
        ppr = PPR_4.compute_matrix(1, "Y" + "X" * (num_wires - 1))
        with_spin_sym = IQP(weights, range(num_wires), pattern, spin_sym=True).matrix()
        assert math.allclose(with_spin_sym, without @ ppr)

    def test_matrix_respects_wire_order(self):
        """Test that ``matrix`` embeds the operator according to a provided wire order."""
        theta = 0.6
        # generator [[0]] targets ``wires[0]`` (wire 1 here), so this is ``RX(2 * theta)`` on wire 1
        op = IQP([theta], [1], [[[0]]], spin_sym=False)
        reference = qp.matrix(qp.tape.QuantumScript([qp.RX(2 * theta, 1)]), wire_order=[0, 1])
        assert math.allclose(op.matrix(wire_order=[0, 1]), reference)


def test_lower_to_mlir():
    """Test that IQP can be lowered to MLIR through ``qjit``."""
    pytest.importorskip("catalyst")

    @qp.qjit(capture=True, target="mlir")
    @qp.qnode(qp.device("lightning.qubit", wires=2))
    def circuit():
        qp.IQP([0.1, 0.2], wires=[0, 1], pattern=[[[0]], [[1]]], spin_sym=False)
        return qp.state()

    resources = qp.specs(circuit, level=0)()["resources"]
    assert resources.counts == {"IQP": 1}
