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
Tests for the SelectSwap template.
"""

from collections import Counter
from functools import partial
from itertools import product
from unittest import mock

import jax
import numpy as np
import pytest

import pennylane as qp
from pennylane.core.operator import abstractify
from pennylane.decomposition import DecompositionGraph
from pennylane.templates.subroutines import qrom_load as qrom_load_module
from pennylane.templates.subroutines.select_swap import (
    _adjoint_select_swap_measurement,
    _select_swap,
    _select_swap_sizes,
)
from pennylane.testing import assert_decomposable
from pennylane.testing.decompositions import _assert_counts_match
from pennylane.typing import Bool, Wire

BITSTRINGS = (
    (0, 0, 0),
    (0, 0, 1),
    (1, 1, 1),
    (0, 1, 1),
    (0, 0, 0),
    (1, 0, 1),
    (1, 1, 0),
    (1, 1, 1),
)


@pytest.mark.usefixtures("enable_and_disable_capture")
@pytest.mark.parametrize(
    "work_wires",
    [
        [],  # only loads one bitstring at a time
        [6, 7, 8],  # Select-SWAP without a work wire for the CSWAPs
        [6, 7, 8, 9, 10, 11],  # Select-SWAP with a work wire for the CSWAPs
    ],
)
def test_assert_valid_select_swap(work_wires, mocker):
    """Run standard validity tests."""
    # Under capture, assert_valid cannot convert the classical processing of measurement outcomes
    # into a tape. The rule is tested in TestAdjointSelectSwapMeasurement instead.
    mocker.patch.object(_adjoint_select_swap_measurement, "is_applicable", return_value=False)
    op = qp.SelectSwap(BITSTRINGS, [0, 1, 2], [3, 4, 5], work_wires)
    qp.testing.assert_valid(op, skip_differentiation=True)


@pytest.mark.parametrize(
    ("num_bitstrings", "num_targets", "num_work_wires"),
    [(8, 3, 0), (8, 3, 3), (8, 3, 6), (5, 2, 3), (32, 1, 5)],
)
def test_load_and_unload(num_bitstrings, num_targets, num_work_wires, seed):
    """Test that SelectSwap loads the right bitstring into the target register and that its
    adjoint returns the target and work wires to the zero state."""
    bitstrings = np.random.default_rng(seed).integers(0, 2, size=(num_bitstrings, num_targets))
    num_control_wires = qp.math.ceil_log2(num_bitstrings)
    wires = qp.registers(
        {
            "control_wires": num_control_wires,
            "target_wires": num_targets,
            "work_wires": num_work_wires,
        }
    )
    dev = qp.device("default.qubit")

    @qp.set_shots(1)
    @qp.qnode(dev)
    def circuit(j, unload):
        qp.BasisState(qp.math.int_to_binary(j, num_control_wires), wires=wires["control_wires"])
        qp.SelectSwap(bitstrings, **wires)
        if unload:
            qp.adjoint(qp.SelectSwap(bitstrings, **wires))
        return qp.sample(wires=wires["target_wires"] + wires["work_wires"])

    for j in range(2**num_control_wires):
        expected = bitstrings[j] if j < num_bitstrings else [0] * num_targets
        assert np.allclose(circuit(j, False)[0][:num_targets], expected)
        assert np.allclose(circuit(j, True)[0], 0)


def test_decomposition():
    """Test that the decomposition loads four bitstrings in parallel and swaps the right one into
    the target register, without cleaning up afterwards."""
    bitstrings = [[(i // 3) % 2] for i in range(32)]
    decomposition = qp.SelectSwap(
        bitstrings,
        control_wires=[0, 1, 2, 3, 4],
        target_wires=[5],
        work_wires=[6, 7, 8, 9, 10],
    ).decomposition()

    # Four bitstrings are loaded in parallel, into the target wire and three work wires
    columns = [sum(bitstrings[4 * i : 4 * i + 4], []) for i in range(8)]
    qrom_load = qp.QROMLoad(
        columns, control_wires=[0, 1, 2], target_wires=[5, 6, 7, 8], work_wires=[9, 10]
    )
    cswaps = [
        qp.ctrl(qp.SWAP(wires), control=c, work_wires=[9], work_wire_type="zeroed")
        for c, wires in [(3, [6, 8]), (3, [5, 7]), (4, [5, 6])]
    ]
    expected_gates = [qrom_load] + cswaps

    assert len(decomposition) == len(expected_gates)
    for op1, op2 in zip(decomposition, expected_gates):
        qp.assert_equal(op1, op2)


@pytest.mark.usefixtures("enable_and_disable_capture")
@pytest.mark.parametrize(
    ("num_bitstrings", "control_wires", "target_wires", "work_wires"),
    [
        (1, [], [0], [1]),
        (2, [0], [1], []),
        (3, [0, 1], [2], [3, 4]),
        (4, [2, 3], [0, 1], [4, 5, 6, 7, 8, 9]),
        (8, [0, 1, 2], [3, 4], [5]),
        (7, [0, 1, 2], [3, 4], [5, 6, 7, 8, 9, 10, 11]),
        (11, [0, 1, 2, 3], [4, 5], [6, 7]),
        (32, [0, 1, 2, 3, 4], [5], [6, 7, 8, 9, 10]),
    ],
)
def test_decomposition_new(num_bitstrings, control_wires, target_wires, work_wires, seed):
    """Tests the decomposition rule implemented with the new system."""
    rng = np.random.default_rng(seed)
    bitstrings = rng.integers(0, 2, size=(num_bitstrings, len(target_wires)))
    op = qp.SelectSwap(bitstrings, control_wires, target_wires, work_wires)
    assert_decomposable(op)


@pytest.mark.parametrize("sizes", [(32, 5, 1, 5), (64, 6, 1, 9), (64, 6, 2, 20), (256, 8, 2, 11)])
def test_t_count(sizes, seed):
    """Test that SelectSwap uses less than half as many T gates as QROMLoad, which has to clean
    up its work wires."""
    num_bitstrings, n_ctrl, n_target, n_work = sizes
    bitstrings = np.random.default_rng(seed).integers(0, 2, size=(num_bitstrings, n_target))
    wires = qp.registers({"control_wires": n_ctrl, "target_wires": n_target, "work_wires": n_work})
    # Weigh T gates heavily so that the graph minimizes the T count, which is what we compare
    gate_set = {"T": 100, "Adjoint(T)": 100, "H": 1, "S": 1, "Adjoint(S)": 1, "CNOT": 1, "X": 1}
    gate_set |= {"Z": 1, "CZ": 1, "GlobalPhase": 1, "MidMeasure": 1}

    def t_count(op):
        counts = DecompositionGraph([op], gate_set).solve().resource_estimate(op).gate_counts
        return sum(num for gate, num in counts.items() if str(gate) in {"T", "Adjoint(T)"})

    select_swap_t_count = t_count(qp.SelectSwap(bitstrings, **wires))
    assert 0 < 2 * select_swap_t_count < t_count(qp.QROMLoad(bitstrings, **wires))


def _adjoint_select_swap_setup(num_bitstrings, num_targets, num_work_wires, num_extra_controls):
    rng = np.random.default_rng(7 * num_bitstrings + 3 * num_targets + num_work_wires)
    num_control_wires = max(1, qp.math.ceil_log2(num_bitstrings)) + num_extra_controls
    wires = qp.registers(
        {
            "control_wires": num_control_wires,
            "target_wires": num_targets,
            "work_wires": num_work_wires,
        }
    )
    bitstrings = rng.integers(0, 2, size=(num_bitstrings, num_targets))
    return qp.SelectSwap(bitstrings, **wires), rng


def _num_data_wires(op):
    """Number of target and work wires that hold loaded bitstrings after ``SelectSwap``."""
    depth = _select_swap_sizes(**op.dynamic_args, **op.wire_args)[-1]
    return depth * len(op.target_wires)


# (num_bitstrings, num_targets, num_work_wires, num_extra_control_wires)
ADJOINT_SELECT_SWAP_CASES = [
    (4, 2, 0, 0),  # one bitstring at a time, without work wires
    (8, 2, 2, 0),  # one bitstring at a time, with work wires
    (8, 1, 3, 0),  # two in parallel
    (5, 1, 3, 0),  # two in parallel, with padding
    (6, 2, 3, 0),
    (16, 2, 4, 0),
    (11, 1, 4, 0),  # four in parallel, with padding
    (16, 1, 5, 0),
    (4, 1, 3, 1),  # extra control wire
]


class TestAdjointSelectSwapMeasurement:
    """Tests for the measurement-based uncomputation of SelectSwap, which measures the target
    wires together with the garbage on the work wires, following Appendix C of arXiv:1902.02134."""

    def test_registered(self):
        """Test that the rule is registered for ``Adjoint(SelectSwap)``."""
        assert _adjoint_select_swap_measurement in qp.list_decomps("Adjoint(SelectSwap)")

    def test_condition(self):
        """Test that the rule requires program capture and at least one control wire."""
        op = qp.SelectSwap(Bool[4, 2], Wire[2], Wire[2], Wire[2])
        assert not _adjoint_select_swap_measurement.is_applicable(base=op)
        with qp.capture.toggle_ctx(True):
            assert _adjoint_select_swap_measurement.is_applicable(base=op)
            op = qp.SelectSwap(Bool[1, 2], Wire[0], Wire[2], Wire[1])
            assert not _adjoint_select_swap_measurement.is_applicable(base=op)

    @pytest.mark.parametrize("case", ADJOINT_SELECT_SWAP_CASES)
    def test_resources(self, case):
        """Test that the resource function matches the decomposition when all measurement
        outcomes are 1, so that all conditional corrections are applied."""
        op, _ = _adjoint_select_swap_setup(*case)

        def measure(pauli_word, wire):
            qp.pauli_measure(pauli_word, wire)
            return 1

        with mock.patch.object(qrom_load_module, "pauli_measure", measure):
            with qp.queuing.AnnotatedQueue() as q:
                _adjoint_select_swap_measurement(base=op)

        tape = qp.tape.QuantumScript.from_queue(q)
        actual = Counter(
            abstractify(o.base if isinstance(o, qp.ops.Conditional) else o) for o in tape.operations
        )
        expected = _adjoint_select_swap_measurement.compute_resources(base=op).gate_counts
        _assert_counts_match({k: v for k, v in expected.items() if v > 0}, actual)

    @pytest.mark.parametrize(
        ("case", "chooses_measurement"),
        [
            # Wide bitstrings: skipping the inverted SWAP network pays off.
            ((64, 8, 24, 0), True),
            # Narrow bitstrings: the smaller fixup table after the inverted SWAP network wins.
            ((64, 1, 7, 0), False),
        ],
    )
    def test_graph_choice(self, case, chooses_measurement):
        """Test that the graph chooses between this rule and the adjoint of the Select-SWAP
        decomposition, whose ``Adjoint(QROMLoad)`` also uncomputes with measurements, by their
        T counts."""
        op = qp.adjoint(_adjoint_select_swap_setup(*case)[0])
        measurement_rule = _adjoint_select_swap_measurement
        inverse_rule = qp.list_decomps(op)["adjoint(_select_swap)"]
        gate_set = {"T": 100, "Adjoint(T)": 100, "H": 1, "S": 1, "Adjoint(S)": 1, "CNOT": 1, "X": 1}
        gate_set |= {"Z": 1, "CZ": 1, "GlobalPhase": 1, "MidMeasure": 1, "PauliMeasure": 1}

        def solve(**kwargs):
            solution = DecompositionGraph([op], gate_set, **kwargs).solve()
            counts = solution.resource_estimate(op).gate_counts
            t_count = sum(n for gate, n in counts.items() if str(gate) in {"T", "Adjoint(T)"})
            return solution, t_count

        if chooses_measurement:
            chosen_rule, other_rule = measurement_rule, inverse_rule
        else:
            chosen_rule, other_rule = inverse_rule, measurement_rule
        with qp.capture.toggle_ctx(True):
            solution, t_count = solve()
            _, other_t_count = solve(fixed_decomps={"Adjoint(SelectSwap)": other_rule})

        assert solution.decomposition(op).name == chosen_rule.name
        assert t_count < other_t_count

    @pytest.mark.usefixtures("enable_graph_decomposition")
    @pytest.mark.parametrize("case", ADJOINT_SELECT_SWAP_CASES)
    def test_correctness(self, case):
        """Test that applying SelectSwap followed by the measurement-based uncomputation acts as
        the identity on a random address state, including addresses without a bitstring, for
        outcomes of the X-basis measurements on the target wires and the garbage.

        The measurements are replaced by postselected measurements, so that the
        measurement-dependent fixup table can be computed from concrete outcomes without capture.
        """
        op, rng = _adjoint_select_swap_setup(*case)
        control_wires = list(op.control_wires)
        state = rng.random(2 ** len(control_wires)) + 1j * rng.random(2 ** len(control_wires))
        state /= np.linalg.norm(state)

        @qp.qnode(qp.device("default.qubit"), mcm_method="deferred")
        def circuit():
            qp.StatePrep(state, wires=control_wires)
            _select_swap(**op.dynamic_args, **op.wire_args)
            _adjoint_select_swap_measurement(base=op)
            qp.adjoint(qp.StatePrep(state, wires=control_wires))
            return qp.probs(wires=op.wires)

        def postselected_measure(outcome_iter, pauli_word, wire):
            if pauli_word == "Z":
                # Exercise the conditional reset of the measured wire.
                qp.measure(wire, postselect=1)
                return 1
            outcome = next(outcome_iter)
            qp.H(wire)
            qp.measure(wire, postselect=outcome)
            qp.H(wire)
            return outcome

        all_outcomes = list(product([0, 1], repeat=_num_data_wires(op)))
        if len(all_outcomes) > 8:
            all_outcomes = [all_outcomes[i] for i in rng.choice(len(all_outcomes), 8, False)]
        for outcomes in all_outcomes:
            measure = partial(postselected_measure, iter(outcomes))
            with mock.patch.object(qrom_load_module, "pauli_measure", measure):
                probs = circuit()
            assert np.isclose(probs[0] / np.sum(probs), 1.0), f"outcomes={outcomes}"

    @pytest.mark.usefixtures("enable_capture")
    @pytest.mark.parametrize("case", ADJOINT_SELECT_SWAP_CASES)
    def test_capture(self, case):
        """Test that the decomposition can be captured, with an X-basis measurement and a Z-basis
        reset measurement per wire that holds a loaded bitstring."""
        op, _ = _adjoint_select_swap_setup(*case)
        jaxpr = jax.make_jaxpr(partial(_adjoint_select_swap_measurement, base=op))()
        measurements = [eqn for eqn in jaxpr.eqns if eqn.primitive.name == "pauli_measure"]
        assert [eqn.params["pauli_word"] for eqn in measurements] == ["X", "Z"] * _num_data_wires(
            op
        )

    @pytest.mark.usefixtures("enable_graph_decomposition")
    @pytest.mark.catalyst
    @pytest.mark.parametrize("case", [(8, 1, 3, 0), (11, 1, 4, 0), (6, 2, 3, 0), (4, 1, 3, 1)])
    def test_qjit(self, case, mocker):
        """Test that the graph applies the rule under ``qjit`` with program capture, and that
        loading and uncomputing a SelectSwap leaves the target and work wires in the zero state,
        without introducing relative phases on a uniform superposition of addresses."""
        op, _ = _adjoint_select_swap_setup(*case)
        control_wires = list(op.control_wires)
        shots = 20
        spy = mocker.spy(_adjoint_select_swap_measurement, "_impl")

        @qp.qjit(capture=True)
        @qp.decompose(
            gate_set=qp.gate_sets.CLIFFORD_T_PLUS_RZ | {"PauliMeasure"},
            fixed_decomps={"Adjoint(SelectSwap)": _adjoint_select_swap_measurement},
        )
        @qp.set_shots(shots)
        @qp.qnode(qp.device("lightning.qubit", wires=len(op.wires)))
        def circuit():
            for wire in control_wires:
                qp.H(wire)
            _select_swap(**op.dynamic_args, **op.wire_args)
            qp.adjoint(qp.SelectSwap(**op.dynamic_args, **op.wire_args))
            for wire in control_wires:
                qp.H(wire)
            return qp.sample(wires=op.wires)

        samples = circuit()
        assert spy.call_count == 1
        assert samples.shape == (shots, len(op.wires))
        assert np.allclose(samples, 0)
