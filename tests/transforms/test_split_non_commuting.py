# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the transform ``qml.transform.split_non_commuting`` """

# pylint: disable=import-outside-toplevel,unnecessary-lambda,too-many-arguments

import itertools
from functools import partial

import numpy as np
import pytest

import pennylane as qml
from pennylane.transforms import split_non_commuting

# Two commuting groups: [[0, 3], [1, 2, 4]]
# Four groups based on wire overlaps: [[0, 2], [1], [3], [4]]
single_term_obs_list = [
    qml.X(0),
    qml.Y(0),
    qml.Z(1),
    qml.X(0) @ qml.Y(1),
    qml.Y(0) @ qml.Z(1),
]

single_term_qwc_groups = [
    [qml.X(0), qml.X(0) @ qml.Y(1)],
    [qml.Y(0), qml.Z(1), qml.Y(0) @ qml.Z(1)],
]

single_term_wires_groups = [
    [qml.X(0), qml.Z(1)],
    [qml.Y(0)],
    [qml.X(0) @ qml.Y(1)],
    [qml.Y(0) @ qml.Z(1)],
]

# contains the following observables: X(0), Y(0), Y(0) @ Z(1), X(1), Z(1), X(0) @ Y(1)
# qwc groups: [[0, 5], [1, 3], [2, 4]]
# wires groups: [[0, 3], [1, 4], [2], [5]]
complex_obs_list = [
    qml.X(0),  # single observable
    0.5 * qml.Y(0),  # scalar product
    qml.X(0) + qml.Y(0) @ qml.Z(1) + 2.0 * qml.X(1) + qml.I(),  # sum
    qml.Hamiltonian(
        [0.1, 0.2, 0.3, 0.4], [qml.Z(1), qml.X(0) @ qml.Y(1), qml.Y(0) @ qml.Z(1), qml.I()]
    ),
    1.5 * qml.I(),  # identity
]

complex_no_grouping_obs = [
    qml.X(0),
    qml.Y(0),
    qml.Y(0) @ qml.Z(1),
    qml.X(1),
    qml.Z(1),
    qml.X(0) @ qml.Y(1),
]


def _convert_obs_to_legacy_opmath(obs):
    """Convert single-term observables to legacy opmath"""

    if isinstance(obs, qml.ops.Prod):
        return qml.operation.Tensor(*obs.operands)

    if isinstance(obs, list):
        return [_convert_obs_to_legacy_opmath(o) for o in obs]

    return obs


def complex_no_grouping_processing_fn(results):
    """The expected processing function without grouping of complex_obs_list"""

    return (
        results[0],
        0.5 * results[1],
        results[0] + results[2] + 2.0 * results[3] + 1.0,
        0.1 * results[4] + 0.2 * results[5] + 0.3 * results[2] + 0.4,
        1.5,
    )


complex_qwc_groups = [
    [qml.X(0), qml.X(0) @ qml.Y(1)],
    [qml.Y(0), qml.Y(0) @ qml.Z(1), qml.Z(1)],
    [qml.X(1)],
]


def complex_qwc_processing_fn(results):
    """The expected processing function for qwc grouping of complex_obs_list"""
    group0, group1, group2 = results
    return (
        group0[0],
        0.5 * group1[0],
        group0[0] + group1[1] + 2.0 * group2[0] + 1.0,
        0.1 * group1[2] + 0.2 * group0[1] + 0.3 * group1[1] + 0.4,
        1.5,
    )


complex_wires_groups = [
    [qml.X(0), qml.X(1)],
    [qml.Y(0), qml.Z(1)],
    [qml.Y(0) @ qml.Z(1)],
    [qml.X(0) @ qml.Y(1)],
]


def complex_wires_processing_fn(results):
    """The expected processing function for wires grouping of complex_obs_list"""

    group0, group1, group2, group3 = results
    return (
        group0[0],
        0.5 * group1[0],
        group0[0] + group2 + 2.0 * group0[1] + 1.0,
        0.1 * group1[1] + 0.2 * group3 + 0.3 * group2 + 0.4,
        1.5,
    )


# Measurements that accept observables as arguments
obs_measurements = [qml.expval, qml.var, qml.probs, qml.counts, qml.sample]

# measurements that accept wires as arguments
wire_measurements = [qml.probs, qml.counts, qml.sample]


class TestUnits:
    """Unit tests for components of the ``split_non_commuting`` transform"""

    @pytest.mark.parametrize("measure_fn", obs_measurements)
    @pytest.mark.parametrize(
        "grouping_strategy, n_tapes", [(None, 5), ("default", 2), ("qwc", 2), ("wires", 4)]
    )
    def test_number_of_tapes(self, measure_fn, grouping_strategy, n_tapes):
        """Tests that the correct number of tapes is returned"""

        measurements = [measure_fn(op=o) for o in single_term_obs_list]
        tape = qml.tape.QuantumScript([qml.X(0), qml.CNOT([0, 1])], measurements, shots=100)
        tapes, _ = split_non_commuting(tape, grouping_strategy=grouping_strategy)
        assert len(tapes) == n_tapes
        assert all(t.operations == [qml.X(0), qml.CNOT([0, 1])] for t in tapes)
        assert all(t.shots == tape.shots for t in tapes)

    @pytest.mark.parametrize(
        "grouping_strategy, n_tapes", [(None, 5), ("default", 2), ("qwc", 2), ("wires", 4)]
    )
    @pytest.mark.parametrize(
        "make_H",
        [
            lambda coeffs, obs: qml.Hamiltonian(coeffs, obs),
            lambda coeffs, obs: qml.sum(*(qml.s_prod(c, o) for c, o in zip(coeffs, obs))),
        ],
    )
    def test_number_of_tapes_single_hamiltonian(self, grouping_strategy, n_tapes, make_H):
        """Tests that the correct number of tapes is returned for a single Hamiltonian"""

        obs_list = single_term_obs_list
        if not qml.operation.active_new_opmath():
            obs_list = _convert_obs_to_legacy_opmath(obs_list)

        obs_list = obs_list + [qml.Y(0), qml.X(0) @ qml.Y(1)]  # add duplicate terms
        coeffs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

        H = make_H(coeffs, obs_list)
        tape = qml.tape.QuantumScript([qml.X(0), qml.CNOT([0, 1])], [qml.expval(H)], shots=100)
        tapes, _ = split_non_commuting(tape, grouping_strategy=grouping_strategy)
        assert len(tapes) == n_tapes
        assert all(t.operations == [qml.X(0), qml.CNOT([0, 1])] for t in tapes)
        assert all(t.shots == tape.shots for t in tapes)

    @pytest.mark.parametrize(
        "grouping_strategy, n_tapes", [(None, 6), ("default", 4), ("qwc", 3), ("wires", 4)]
    )
    def test_number_of_tapes_complex_obs(self, grouping_strategy, n_tapes):
        """Tests number of tapes with mixed types of observables"""

        measurements = [qml.expval(o) for o in complex_obs_list]
        tape = qml.tape.QuantumScript([qml.X(0), qml.CNOT([0, 1])], measurements, shots=100)
        tapes, _ = split_non_commuting(tape, grouping_strategy=grouping_strategy)
        assert len(tapes) == n_tapes
        assert all(t.operations == [qml.X(0), qml.CNOT([0, 1])] for t in tapes)
        assert all(t.shots == tape.shots for t in tapes)

    @pytest.mark.parametrize("grouping_strategy", [None, "default", "qwc", "wires"])
    def test_state_measurement_in_separate_tape(self, grouping_strategy):
        """Tests that a state measurement is in a separate tape"""

        measurements = [qml.expval(qml.Z(0) @ qml.Z(1)), qml.state()]
        tape = qml.tape.QuantumScript([qml.X(0), qml.CNOT([0, 1])], measurements, shots=100)
        tapes, _ = split_non_commuting(tape, grouping_strategy=grouping_strategy)
        assert len(tapes) == 2
        assert all(t.operations == [qml.X(0), qml.CNOT([0, 1])] for t in tapes)
        assert all(t.shots == tape.shots for t in tapes)

    @pytest.mark.parametrize("grouping_strategy", [None, "default", "qwc", "wires"])
    @pytest.mark.parametrize(
        "make_H",
        [
            lambda obs_list: qml.Hamiltonian([0.1, 0.2, 0.3, 0.4, 0.5], obs_list),
            lambda obs_list: qml.sum(
                *(qml.s_prod(c, o) for c, o in zip([0.1, 0.2, 0.3, 0.4, 0.5], obs_list))
            ),
        ],
    )
    def test_existing_grouping_used_for_single_hamiltonian(self, grouping_strategy, make_H):
        """Tests that if a Hamiltonian has an existing grouping, it is used regardless of
        what is requested through the ``grouping_strategy`` argument."""

        obs_list = single_term_obs_list
        if not qml.operation.active_new_opmath():
            obs_list = _convert_obs_to_legacy_opmath(obs_list)

        H = make_H(obs_list)
        H.compute_grouping()

        tape = qml.tape.QuantumScript([qml.X(0), qml.CNOT([0, 1])], [qml.expval(H)], shots=100)
        tapes, _ = split_non_commuting(tape, grouping_strategy=grouping_strategy)
        assert len(tapes) == 2
        assert all(t.operations == [qml.X(0), qml.CNOT([0, 1])] for t in tapes)
        assert all(t.shots == tape.shots for t in tapes)

    @pytest.mark.parametrize("measure_fn", obs_measurements)
    def test_single_group(self, measure_fn):
        """Tests when all measurements can be taken at the same time"""

        with qml.queuing.AnnotatedQueue() as q:
            qml.PauliZ(0)
            qml.Hadamard(0)
            qml.CNOT((0, 1))
            measure_fn(op=qml.X(0))
            measure_fn(op=qml.Y(1))
            measure_fn(op=qml.Z(2))
            measure_fn(op=qml.X(0) @ qml.Y(1))
            measure_fn(op=qml.Y(1) @ qml.Z(2))

        tape = qml.tape.QuantumScript.from_queue(q, shots=100)
        tapes, fn = split_non_commuting(tape)

        assert len(tapes) == 1
        assert fn([[0.1, 0.2, 0.3, 0.4, 0.5]]) == (0.1, 0.2, 0.3, 0.4, 0.5)

    @pytest.mark.parametrize("grouping_strategy", [None, "default", "qwc", "wires"])
    def test_single_observable(self, grouping_strategy):
        """Tests a circuit that contains a single observable"""

        tape = qml.tape.QuantumScript([], [qml.expval(qml.X(0))])
        tapes, fn = split_non_commuting(tape, grouping_strategy=grouping_strategy)
        assert len(tapes) == 1
        assert fn([0.1]) == 0.1

    @pytest.mark.parametrize("grouping_strategy", [None, "default", "qwc", "wires"])
    def test_single_hamiltonian_single_observable(self, grouping_strategy):
        """Tests a circuit that contains a single observable"""

        tape = qml.tape.QuantumScript([], [qml.expval(qml.Hamiltonian([0.1], [qml.X(0)]))])
        tapes, fn = split_non_commuting(tape, grouping_strategy=grouping_strategy)
        assert len(tapes) == 1
        assert qml.math.allclose(fn([0.1]), 0.01)

    @pytest.mark.parametrize("measure_fn", wire_measurements)
    def test_all_wire_measurements(self, measure_fn):
        """Tests that measurements based on wires don't need to be split"""

        with qml.queuing.AnnotatedQueue() as q:
            qml.PauliZ(0)
            qml.Hadamard(0)
            qml.CNOT((0, 1))
            measure_fn()
            measure_fn(wires=[0])
            measure_fn(wires=[1])
            measure_fn(wires=[0, 1])
            measure_fn(op=qml.PauliZ(0))
            measure_fn(op=qml.PauliZ(0) @ qml.PauliZ(2))

        tape = qml.tape.QuantumScript.from_queue(q)
        tapes, fn = split_non_commuting(tape)

        assert len(tapes) == 1
        assert fn([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]) == (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)

    @pytest.mark.parametrize("obs_meas_1, obs_meas_2", itertools.combinations(obs_measurements, 2))
    @pytest.mark.parametrize(
        "wire_meas_1, wire_meas_2", itertools.combinations(wire_measurements, 2)
    )
    def test_mix_measurement_types(self, obs_meas_1, obs_meas_2, wire_meas_1, wire_meas_2):
        """Tests that tapes mixing different measurement types is handled correctly"""

        with qml.queuing.AnnotatedQueue() as q:
            obs_meas_1(op=qml.PauliX(0))
            obs_meas_2(op=qml.PauliZ(1))
            obs_meas_1(op=qml.PauliZ(0))
            wire_meas_1(wires=[0])
            wire_meas_2(wires=[1])
            wire_meas_1(wires=[0, 1])

        tape = qml.tape.QuantumScript.from_queue(q)
        tapes, _ = split_non_commuting(tape)
        assert len(tapes) == 2
        assert tapes[0].measurements == [
            obs_meas_1(op=qml.PauliX(0)),
            obs_meas_2(op=qml.PauliZ(1)),
            wire_meas_2(wires=[1]),
        ]
        assert tapes[1].measurements == [
            obs_meas_1(op=qml.PauliZ(0)),
            wire_meas_1(wires=[0]),
            wire_meas_1(wires=[0, 1]),
        ]

    def test_grouping_strategies(self):
        """Tests that the tape is split correctly for different grouping strategies"""

        measurements = [
            qml.expval(c * o) for c, o in zip([0.1, 0.2, 0.3, 0.4, 0.5], single_term_obs_list)
        ]
        tape = qml.tape.QuantumScript([], measurements, shots=100)

        expected_tapes_no_grouping = [
            qml.tape.QuantumScript([], [qml.expval(o)], shots=100) for o in single_term_obs_list
        ]

        # qwc grouping produces [[0, 3], [1, 2, 4]]
        expected_tapes_qwc_grouping = [
            qml.tape.QuantumScript([], [qml.expval(o) for o in group], shots=100)
            for group in single_term_qwc_groups
        ]

        # wires grouping produces [[0, 2], [1], [3], [4]]
        expected_tapes_wires_grouping = [
            qml.tape.QuantumScript([], [qml.expval(o) for o in group], shots=100)
            for group in single_term_wires_groups
        ]

        tapes, fn = split_non_commuting(tape, grouping_strategy=None)
        for actual_tape, expected_tape in zip(tapes, expected_tapes_no_grouping):
            qml.assert_equal(actual_tape, expected_tape)
        assert qml.math.allclose(fn([0.1, 0.2, 0.3, 0.4, 0.5]), [0.01, 0.04, 0.09, 0.16, 0.25])

        tapes, fn = split_non_commuting(tape, grouping_strategy="default")
        # When new opmath is disabled, c * o gives Hamiltonians, which leads to wires grouping
        if qml.operation.active_new_opmath():
            for actual_tape, expected_tape in zip(tapes, expected_tapes_qwc_grouping):
                qml.assert_equal(actual_tape, expected_tape)
            assert qml.math.allclose(
                fn([[0.1, 0.2], [0.3, 0.4, 0.5]]), [0.01, 0.06, 0.12, 0.08, 0.25]
            )
        else:
            for actual_tape, expected_tape in zip(tapes, expected_tapes_wires_grouping):
                qml.assert_equal(actual_tape, expected_tape)
            assert qml.math.allclose(
                fn([[0.1, 0.2], 0.3, 0.4, 0.5]), [0.01, 0.06, 0.06, 0.16, 0.25]
            )

        tapes, fn = split_non_commuting(tape, grouping_strategy="qwc")
        for actual_tape, expected_tape in zip(tapes, expected_tapes_qwc_grouping):
            qml.assert_equal(actual_tape, expected_tape)
        assert qml.math.allclose(fn([[0.1, 0.2], [0.3, 0.4, 0.5]]), [0.01, 0.06, 0.12, 0.08, 0.25])

        tapes, fn = split_non_commuting(tape, grouping_strategy="wires")
        for actual_tape, expected_tape in zip(tapes, expected_tapes_wires_grouping):
            qml.assert_equal(actual_tape, expected_tape)
        assert qml.math.allclose(fn([[0.1, 0.2], 0.3, 0.4, 0.5]), [0.01, 0.06, 0.06, 0.16, 0.25])

    @pytest.mark.parametrize(
        "make_H",
        [
            lambda coeffs, obs_list: qml.Hamiltonian(coeffs, obs_list),
            lambda coeffs, obs_list: qml.sum(*(qml.s_prod(c, o) for c, o in zip(coeffs, obs_list))),
        ],
    )
    def test_grouping_strategies_single_hamiltonian(self, make_H):
        """Tests that a single Hamiltonian or Sum is split correctly"""

        coeffs, obs_list = [0.1, 0.2, 0.3, 0.4, 0.5], single_term_obs_list
        qwc_groups = single_term_qwc_groups

        if not qml.operation.active_new_opmath():
            obs_list = _convert_obs_to_legacy_opmath(obs_list)
            qwc_groups = _convert_obs_to_legacy_opmath(single_term_qwc_groups)

        expected_tapes_no_grouping = [
            qml.tape.QuantumScript([], [qml.expval(o)], shots=100) for o in obs_list
        ]

        expected_tapes_qwc_grouping = [
            qml.tape.QuantumScript([], [qml.expval(o) for o in group], shots=100)
            for group in qwc_groups
        ]

        if qml.operation.active_new_opmath():
            coeffs, obs_list = coeffs + [0.6], obs_list + [qml.I()]

        H = make_H(coeffs, obs_list)  # Tests that constant offset is handled

        if not qml.operation.active_new_opmath() and isinstance(H, qml.ops.Sum):
            pytest.skip("Sum is not part of legacy opmath")

        tape = qml.tape.QuantumScript([], [qml.expval(H)], shots=100)

        tapes, fn = split_non_commuting(tape, grouping_strategy=None)
        for actual_tape, expected_tape in zip(tapes, expected_tapes_no_grouping):
            qml.assert_equal(actual_tape, expected_tape)
        expected = 0.55 if not qml.operation.active_new_opmath() else 1.15
        assert qml.math.allclose(fn([0.1, 0.2, 0.3, 0.4, 0.5]), expected)

        tapes, fn = split_non_commuting(tape, grouping_strategy="default")
        for actual_tape, expected_tape in zip(tapes, expected_tapes_qwc_grouping):
            qml.assert_equal(actual_tape, expected_tape)
        expected = 0.52 if not qml.operation.active_new_opmath() else 1.12
        assert qml.math.allclose(fn([[0.1, 0.2], [0.3, 0.4, 0.5]]), expected)

    @pytest.mark.parametrize(
        "H",
        [
            qml.sum(qml.X(0), qml.Hadamard(1) @ qml.Z(0), qml.Y(1)),
            qml.Hamiltonian([1, 2, 3], [qml.X(0), qml.Hadamard(1) @ qml.Z(0), qml.Y(1)]),
        ],
    )
    def test_single_hamiltonian_non_pauli_words(self, H):
        """Tests that a single Hamiltonian with non-pauli words is split correctly"""

        tape = qml.tape.QuantumScript([], [qml.expval(H)], shots=100)
        tapes, _ = split_non_commuting(tape)
        expected_tapes = [
            qml.tape.QuantumScript([], [qml.expval(qml.X(0)), qml.expval(qml.Y(1))], shots=100),
            qml.tape.QuantumScript([], [qml.expval(qml.Hadamard(1) @ qml.Z(0))], shots=100),
        ]
        for actual_tape, expected_tape in zip(tapes, expected_tapes):
            qml.assert_equal(actual_tape, expected_tape)

    @pytest.mark.parametrize(
        "grouping_strategy, expected_tapes, processing_fn, mock_results",
        [
            (
                None,
                [
                    qml.tape.QuantumScript([], [qml.expval(o)], shots=100)
                    for o in complex_no_grouping_obs
                ],
                complex_no_grouping_processing_fn,
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            ),
            (
                "wires",
                [
                    qml.tape.QuantumScript([], [qml.expval(o) for o in group], shots=100)
                    for group in complex_wires_groups
                ],
                complex_wires_processing_fn,
                [[0.1, 0.2], [0.3, 0.4], 0.5, 0.6],
            ),
            (
                "qwc",
                [
                    qml.tape.QuantumScript([], [qml.expval(o) for o in group], shots=100)
                    for group in complex_qwc_groups
                ],
                complex_qwc_processing_fn,
                [[0.1, 0.2], [0.3, 0.5, 0.6], [0.4]],
            ),
        ],
    )
    def test_grouping_strategies_complex(
        self, grouping_strategy, expected_tapes, processing_fn, mock_results
    ):
        """Tests that the tape is split correctly when containing more complex observables"""

        obs_list = complex_obs_list
        if not qml.operation.active_new_opmath():
            obs_list = obs_list[:-1]  # exclude the identity term

        measurements = [qml.expval(o) for o in obs_list]
        tape = qml.tape.QuantumScript([], measurements, shots=100)
        tapes, fn = split_non_commuting(tape, grouping_strategy=grouping_strategy)

        for actual_tape, expected_tape in zip(tapes, expected_tapes):
            qml.assert_equal(actual_tape, expected_tape)

        expected = processing_fn(mock_results)
        if not qml.operation.active_new_opmath():
            expected = expected[:-1]  # exclude the identity term

        assert qml.math.allclose(fn(mock_results), expected)

    @pytest.mark.parametrize("batch_type", (tuple, list))
    def test_batch_of_tapes(self, batch_type):
        """Test that `split_non_commuting` can transform a batch of tapes"""

        tape_batch = batch_type(
            [
                qml.tape.QuantumScript(
                    [qml.RX(1.2, 0)],
                    [qml.expval(qml.X(0)), qml.expval(qml.Y(0)), qml.expval(qml.X(1))],
                ),
                qml.tape.QuantumScript(
                    [qml.RY(0.5, 0)], [qml.expval(qml.Z(0)), qml.expval(qml.Y(0))]
                ),
            ]
        )
        tapes, fn = split_non_commuting(tape_batch)

        expected_tapes = [
            qml.tape.QuantumScript([qml.RX(1.2, 0)], [qml.expval(qml.X(0)), qml.expval(qml.X(1))]),
            qml.tape.QuantumScript([qml.RX(1.2, 0)], [qml.expval(qml.Y(0))]),
            qml.tape.QuantumScript([qml.RY(0.5, 0)], [qml.expval(qml.Z(0))]),
            qml.tape.QuantumScript([qml.RY(0.5, 0)], [qml.expval(qml.Y(0))]),
        ]
        for actual_tape, expected_tape in zip(tapes, expected_tapes):
            qml.assert_equal(actual_tape, expected_tape)

        result = ([0.1, 0.2], 0.2, 0.3, 0.4)
        assert fn(result) == ((0.1, 0.2, 0.2), (0.3, 0.4))

    @pytest.mark.parametrize(
        "non_pauli_obs",
        [
            [
                qml.Projector([0], wires=[1]),
                qml.Projector([1, 1, 0, 1], wires=[0, 1]),
            ],
            [
                qml.Hadamard(wires=[1]),
                qml.Hadamard(wires=[0]) @ qml.PauliX(wires=[1]),
            ],
        ],
    )
    def test_tape_with_non_pauli_obs(self, non_pauli_obs):
        """Tests that the tape is split correctly when containing non-Pauli observables"""

        obs_list = single_term_obs_list + non_pauli_obs

        if not qml.operation.active_new_opmath():
            non_pauli_obs = _convert_obs_to_legacy_opmath(non_pauli_obs)
            obs_list = _convert_obs_to_legacy_opmath(obs_list)

        measurements = [
            qml.expval(c * o) for c, o in zip([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], obs_list)
        ]
        tape = qml.tape.QuantumScript([], measurements, shots=100)

        expected_tapes_no_grouping = [
            qml.tape.QuantumScript([], [qml.expval(o)], shots=100) for o in obs_list
        ]

        tapes, fn = split_non_commuting(tape, grouping_strategy=None)
        for actual_tape, expected_tape in zip(tapes, expected_tapes_no_grouping):
            qml.assert_equal(actual_tape, expected_tape)
        assert qml.math.allclose(
            fn([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]), [0.01, 0.04, 0.09, 0.16, 0.25, 0.36, 0.49]
        )

        wires_groups = [
            [qml.X(0), qml.Z(1)],
            [qml.Y(0), non_pauli_obs[0]],
            [qml.X(0) @ qml.Y(1)],
            [qml.Y(0) @ qml.Z(1)],
            [non_pauli_obs[1]],
        ]

        # wires grouping produces [[0, 2], [1, 5], [3], [4], [6]]
        expected_tapes_wires_grouping = [
            qml.tape.QuantumScript([], [qml.expval(o) for o in group], shots=100)
            for group in wires_groups
        ]

        tapes, fn = split_non_commuting(tape)
        for actual_tape, expected_tape in zip(tapes, expected_tapes_wires_grouping):
            qml.assert_equal(actual_tape, expected_tape)
        assert qml.math.allclose(
            fn([[0.1, 0.2], [0.3, 0.6], 0.4, 0.5, 0.7]), [0.01, 0.06, 0.06, 0.16, 0.25, 0.36, 0.49]
        )

    @pytest.mark.parametrize("grouping_strategy", [None, "default", "qwc", "wires"])
    def test_no_measurements(self, grouping_strategy):
        """Test that if the tape contains no measurements, the transform doesn't
        modify it"""

        tape = qml.tape.QuantumScript([qml.X(0)])
        tapes, post_processing_fn = split_non_commuting(tape, grouping_strategy=grouping_strategy)
        assert len(tapes) == 1
        assert tapes[0] == tape
        assert post_processing_fn(tapes) == tape

    @pytest.mark.parametrize(
        "observable",
        [
            qml.X(0) + qml.Y(1),
            2 * (qml.X(0) + qml.Y(1)),
            3 * (2 * (qml.X(0) + qml.Y(1)) + qml.X(1)),
        ],
    )
    def test_splitting_sums_in_unsupported_mps_raises_error(self, observable):

        tape = qml.tape.QuantumScript([qml.X(0)], measurements=[qml.counts(observable)])
        with pytest.raises(
            RuntimeError, match="Cannot split up terms in sums for MeasurementProcess"
        ):
            _, _ = split_non_commuting(tape)


class TestIntegration:
    """Tests the ``split_non_commuting`` transform performed on a QNode"""

    @pytest.mark.parametrize("grouping_strategy", [None, "default", "qwc", "wires"])
    @pytest.mark.parametrize("shots", [None, 20000, [20000, 30000, 40000]])
    @pytest.mark.parametrize(
        "params, expected_results",
        [
            (
                [np.pi / 4, 3 * np.pi / 4],
                [
                    0.5,
                    -np.cos(np.pi / 4),
                    -0.5,
                    -0.5 * np.cos(np.pi / 4),
                    0.5 * np.cos(np.pi / 4),
                ],
            ),
            (
                [[np.pi / 4, 3 * np.pi / 4], [3 * np.pi / 4, 3 * np.pi / 4]],
                [
                    [0.5, -0.5],
                    [-np.cos(np.pi / 4), -np.cos(np.pi / 4)],
                    [-0.5, 0.5],
                    [-0.5 * np.cos(np.pi / 4), 0.5 * np.cos(np.pi / 4)],
                    [0.5 * np.cos(np.pi / 4), -0.5 * np.cos(np.pi / 4)],
                ],
            ),
        ],
    )
    def test_single_expval(self, grouping_strategy, shots, params, expected_results):
        """Tests that a QNode with a single expval measurement is executed correctly"""

        coeffs, obs = [0.1, 0.2, 0.3, 0.4, 0.5], single_term_obs_list

        if not qml.operation.active_new_opmath():
            obs = _convert_obs_to_legacy_opmath(obs)

        if qml.operation.active_new_opmath():
            # test constant offset with new opmath
            coeffs, obs = coeffs + [0.6], obs + [qml.I()]

        dev = qml.device("default.qubit", wires=2, shots=shots)

        @qml.qnode(dev)
        def circuit(angles):
            qml.RX(angles[0], wires=0)
            qml.RY(angles[1], wires=0)
            qml.RX(angles[0], wires=1)
            qml.RY(angles[1], wires=1)
            return qml.expval(qml.Hamiltonian(coeffs, obs))

        circuit = split_non_commuting(circuit, grouping_strategy=grouping_strategy)
        res = circuit(params)

        if qml.operation.active_new_opmath():
            identity_results = [1] if len(np.shape(params)) == 1 else [[1, 1]]
            expected_results = expected_results + identity_results

        expected = np.dot(coeffs, expected_results)

        if isinstance(shots, list):
            assert qml.math.shape(res) == (3,) if len(np.shape(res)) == 1 else (3, 2)
            for i in range(3):
                assert qml.math.allclose(res[i], expected, atol=0.05)
        else:
            assert qml.math.allclose(res, expected, atol=0.05)

    @pytest.mark.parametrize("grouping_strategy", [None, "default", "qwc", "wires"])
    @pytest.mark.parametrize("shots", [None, 20000, [20000, 30000, 40000]])
    @pytest.mark.parametrize(
        "params, expected_results",
        [
            (
                [np.pi / 4, 3 * np.pi / 4],
                [
                    0.5,
                    -0.5 * np.cos(np.pi / 4),
                    0.5 + np.cos(np.pi / 4) * 0.5 + 2.0 * 0.5 + 1,
                    np.dot(
                        [0.1, 0.2, 0.3, 0.4],
                        [-0.5, -0.5 * np.cos(np.pi / 4), 0.5 * np.cos(np.pi / 4), 1],
                    ),
                    1.5,
                ],
            ),
            (
                [[np.pi / 4, 3 * np.pi / 4], [3 * np.pi / 4, 3 * np.pi / 4]],
                [
                    [0.5, -0.5],
                    [-0.5 * np.cos(np.pi / 4), -0.5 * np.cos(np.pi / 4)],
                    [
                        0.5 + np.cos(np.pi / 4) * 0.5 + 2.0 * 0.5 + 1,
                        -0.5 - np.cos(np.pi / 4) * 0.5 - 2.0 * 0.5 + 1,
                    ],
                    [
                        np.dot(
                            [0.1, 0.2, 0.3, 0.4],
                            [-0.5, -0.5 * np.cos(np.pi / 4), 0.5 * np.cos(np.pi / 4), 1],
                        ),
                        np.dot(
                            [0.1, 0.2, 0.3, 0.4],
                            [0.5, 0.5 * np.cos(np.pi / 4), -0.5 * np.cos(np.pi / 4), 1],
                        ),
                    ],
                    [1.5, 1.5],
                ],
            ),
        ],
    )
    def test_multiple_expval(self, grouping_strategy, shots, params, expected_results, seed):
        """Tests that a QNode with multiple expval measurements is executed correctly"""

        dev = qml.device("default.qubit", wires=2, shots=shots, seed=seed)

        obs_list = complex_obs_list
        if not qml.operation.active_new_opmath():
            obs_list = obs_list[:-1]  # exclude the identity term

        @qml.qnode(dev)
        def circuit(angles):
            qml.RX(angles[0], wires=0)
            qml.RY(angles[1], wires=0)
            qml.RX(angles[0], wires=1)
            qml.RY(angles[1], wires=1)
            return [qml.expval(obs) for obs in obs_list]

        circuit = split_non_commuting(circuit, grouping_strategy=grouping_strategy)
        res = circuit(params)

        if not qml.operation.active_new_opmath():
            expected_results = expected_results[:-1]  # exclude the identity term

        if isinstance(shots, list):
            assert qml.math.shape(res) == (3, *np.shape(expected_results))
            for i in range(3):
                assert qml.math.allclose(res[i], expected_results, atol=0.05)
        else:
            assert qml.math.allclose(res, expected_results, atol=0.05)

    @pytest.mark.parametrize("grouping_strategy", [None, "default", "qwc", "wires"])
    @pytest.mark.parametrize("shots", [20000, [20000, 30000, 40000]])
    @pytest.mark.parametrize(
        "params, expected_results",
        [
            (
                [np.pi / 4, 3 * np.pi / 4],
                [
                    0.5,
                    -0.5 * np.cos(np.pi / 4),
                    0.5 + np.cos(np.pi / 4) * 0.5 + 2.0 * 0.5 + 1,
                    np.dot(
                        [0.1, 0.2, 0.3, 0.4],
                        [-0.5, -0.5 * np.cos(np.pi / 4), 0.5 * np.cos(np.pi / 4), 1],
                    ),
                    1.5,
                ],
            ),
            (
                [[np.pi / 4, 3 * np.pi / 4], [3 * np.pi / 4, 3 * np.pi / 4]],
                [
                    [0.5, -0.5],
                    [-0.5 * np.cos(np.pi / 4), -0.5 * np.cos(np.pi / 4)],
                    [
                        0.5 + np.cos(np.pi / 4) * 0.5 + 2.0 * 0.5 + 1,
                        -0.5 - np.cos(np.pi / 4) * 0.5 - 2.0 * 0.5 + 1,
                    ],
                    [
                        np.dot(
                            [0.1, 0.2, 0.3, 0.4],
                            [-0.5, -0.5 * np.cos(np.pi / 4), 0.5 * np.cos(np.pi / 4), 1],
                        ),
                        np.dot(
                            [0.1, 0.2, 0.3, 0.4],
                            [0.5, 0.5 * np.cos(np.pi / 4), -0.5 * np.cos(np.pi / 4), 1],
                        ),
                    ],
                    [1.5, 1.5],
                ],
            ),
        ],
    )
    def test_mixed_measurement_types(self, grouping_strategy, shots, params, expected_results):
        """Tests that a QNode with mixed measurement types is executed correctly"""

        dev = qml.device("default.qubit", wires=2, shots=shots)

        obs_list = complex_obs_list
        if not qml.operation.active_new_opmath():
            obs_list = obs_list[:-1]  # exclude the identity term

        @qml.qnode(dev)
        def circuit(angles):
            qml.RX(angles[0], wires=0)
            qml.RY(angles[1], wires=0)
            qml.RX(angles[0], wires=1)
            qml.RY(angles[1], wires=1)
            return (
                qml.probs(wires=0),
                qml.probs(wires=[0, 1]),
                qml.counts(wires=0),
                qml.sample(wires=0),
                *[qml.expval(obs) for obs in obs_list],
            )

        circuit = split_non_commuting(circuit, grouping_strategy=grouping_strategy)
        res = circuit(params)

        if not qml.operation.active_new_opmath():
            expected_results = expected_results[:-1]  # exclude the identity term

        if isinstance(shots, list):
            assert len(res) == 3
            for i in range(3):

                prob_res_0 = res[i][0]
                prob_res_1 = res[i][1]
                counts_res = res[i][2]
                sample_res = res[i][3]
                if len(qml.math.shape(params)) == 1:
                    assert qml.math.shape(prob_res_0) == (2,)
                    assert qml.math.shape(prob_res_1) == (4,)
                    assert isinstance(counts_res, dict)
                    assert qml.math.shape(sample_res) == (shots[i],)
                else:
                    assert qml.math.shape(prob_res_0) == (2, 2)
                    assert qml.math.shape(prob_res_1) == (2, 4)
                    assert all(isinstance(_res, dict) for _res in counts_res)
                    assert qml.math.shape(sample_res) == (2, shots[i])

                expval_res = res[i][4:]
                assert qml.math.allclose(expval_res, expected_results, atol=0.05)
        else:
            prob_res_0 = res[0]
            prob_res_1 = res[1]
            counts_res = res[2]
            sample_res = res[3]
            if len(qml.math.shape(params)) == 1:
                assert qml.math.shape(prob_res_0) == (2,)
                assert qml.math.shape(prob_res_1) == (4,)
                assert isinstance(counts_res, dict)
                assert qml.math.shape(sample_res) == (shots,)
            else:
                assert qml.math.shape(prob_res_0) == (2, 2)
                assert qml.math.shape(prob_res_1) == (2, 4)
                assert all(isinstance(_res, dict) for _res in counts_res)
                assert qml.math.shape(sample_res) == (2, shots)

            expval_res = res[4:]
            assert qml.math.allclose(expval_res, expected_results, atol=0.05)

    @pytest.mark.parametrize("grouping_strategy", [None, "default", "qwc", "wires"])
    def test_single_hamiltonian_only_constant_offset(self, grouping_strategy):
        """Tests that split_non_commuting can handle a single Identity observable"""

        dev = qml.device("default.qubit", wires=2)
        H = qml.Hamiltonian([1.5, 2.5], [qml.I(), qml.I()])

        @partial(split_non_commuting, grouping_strategy=grouping_strategy)
        @qml.qnode(dev)
        def circuit():
            return qml.expval(H)

        with dev.tracker:
            res = circuit()
        assert dev.tracker.totals == {}
        assert qml.math.allclose(res, 4.0)

    @pytest.mark.parametrize("grouping_strategy", [None, "default", "qwc", "wires"])
    def test_no_obs_tape(self, grouping_strategy):
        """Tests tapes with only constant offsets (only measurements on Identity)"""

        _dev = qml.device("default.qubit", wires=1)

        @qml.qnode(_dev)
        def circuit():
            return qml.expval(1.5 * qml.I(0))

        circuit = split_non_commuting(circuit, grouping_strategy=grouping_strategy)

        with _dev.tracker:
            res = circuit()

        assert _dev.tracker.totals == {}
        assert qml.math.allclose(res, 1.5)

    @pytest.mark.parametrize("grouping_strategy", [None, "default", "qwc", "wires"])
    def test_no_obs_tape_multi_measurement(self, grouping_strategy):
        """Tests tapes with only constant offsets (only measurements on Identity)"""

        _dev = qml.device("default.qubit", wires=1)

        @qml.qnode(_dev)
        def circuit():
            return qml.expval(1.5 * qml.I()), qml.expval(2.5 * qml.I())

        circuit = split_non_commuting(circuit, grouping_strategy=grouping_strategy)

        with _dev.tracker:
            res = circuit()

        assert _dev.tracker.totals == {}
        assert qml.math.allclose(res, [1.5, 2.5])

    def test_non_pauli_obs_in_circuit(self):
        """Tests that the tape is executed correctly with non-pauli observables"""

        _dev = qml.device("default.qubit", wires=1)

        @qml.transforms.split_non_commuting
        @qml.qnode(_dev)
        def circuit():
            qml.Hadamard(0)
            return (
                qml.expval(qml.Projector([0], wires=[0])),
                qml.expval(qml.Projector([1], wires=[0])),
            )

        with _dev.tracker:
            res = circuit()
        assert _dev.tracker.totals["simulations"] == 2
        assert qml.math.allclose(res, [0.5, 0.5])


expected_grad_param_0 = [
    0.125,
    0.125,
    0.125,
    -0.375,
    -0.5,
    0.5 * -np.cos(np.pi / 4),
    -0.5 - 2.0 * 0.5,
    0.1 * 0.5,
    0,
]

expected_grad_param_1 = [
    -0.125,
    -0.125,
    -0.125,
    0.375,
    -0.5,
    0,
    -0.5 + np.cos(np.pi / 4) / 2 - 2.0 * 0.5,
    np.dot([0.1, 0.2, 0.3], [-0.5, np.cos(np.pi / 4) / 2, np.cos(np.pi / 4) / 2]),
    0,
]


class TestDifferentiability:
    """Tests the differentiability of the ``split_non_commuting`` transform"""

    @pytest.mark.autograd
    @pytest.mark.parametrize("grouping_strategy", [None, "default", "qwc", "wires"])
    def test_autograd(self, grouping_strategy):
        """Tests that the output of ``split_non_commuting`` is differentiable with autograd"""

        import pennylane.numpy as pnp

        dev = qml.device("default.qubit", wires=2)

        obs_list = complex_obs_list
        if not qml.operation.active_new_opmath():
            obs_list = obs_list[:-1]  # exclude the identity term

        @partial(split_non_commuting, grouping_strategy=grouping_strategy)
        @qml.qnode(dev)
        def circuit(theta, phi):
            qml.RX(theta, wires=0)
            qml.RY(phi, wires=0)
            qml.RX(theta, wires=1)
            qml.RY(phi, wires=1)
            return qml.probs(wires=[0, 1]), *[qml.expval(o) for o in obs_list]

        def cost(theta, phi):
            res = circuit(theta, phi)
            return qml.math.concatenate([res[0], qml.math.stack(res[1:])], axis=0)

        params = pnp.array(pnp.pi / 4), pnp.array(3 * pnp.pi / 4)
        grad1, grad2 = qml.jacobian(cost)(*params)

        expected_grad_1 = expected_grad_param_0
        expected_grad_2 = expected_grad_param_1
        if not qml.operation.active_new_opmath():
            expected_grad_1 = expected_grad_param_0[:-1]
            expected_grad_2 = expected_grad_param_1[:-1]

        assert qml.math.allclose(grad1, expected_grad_1)
        assert qml.math.allclose(grad2, expected_grad_2)

    @pytest.mark.autograd
    @pytest.mark.parametrize("grouping_strategy", [None, "default", "qwc", "wires"])
    def test_trainable_hamiltonian_autograd(self, grouping_strategy):
        """Tests that measurements of trainable Hamiltonians are differentiable"""

        import pennylane.numpy as pnp

        dev = qml.device("default.qubit", wires=2, shots=50000)

        @partial(split_non_commuting, grouping_strategy=grouping_strategy)
        @qml.qnode(dev)
        def circuit(coeff1, coeff2):
            qml.RX(np.pi / 4, wires=0)
            qml.RY(np.pi / 4, wires=1)
            return qml.expval(qml.Hamiltonian([coeff1, coeff2], [qml.Y(0) @ qml.Z(1), qml.X(1)]))

        params = pnp.array(pnp.pi / 4), pnp.array(3 * pnp.pi / 4)
        actual = qml.jacobian(circuit)(*params)

        assert qml.math.allclose(actual, [-0.5, np.cos(np.pi / 4)], rtol=0.05)

    @pytest.mark.autograd
    @pytest.mark.parametrize("grouping_strategy", [None, "default", "qwc", "wires"])
    def test_non_trainable_obs_autograd(self, grouping_strategy):
        """Test that we can measure a hamiltonian with non-trainable autograd coefficients."""

        dev = qml.device("default.qubit")

        @partial(split_non_commuting, grouping_strategy=grouping_strategy)
        @qml.qnode(dev, diff_method="adjoint")
        def circuit(x):
            qml.RX(x, 0)
            c1 = qml.numpy.array(0.1, requires_grad=False)
            c2 = qml.numpy.array(0.2, requires_grad=False)
            H = c1 * qml.Z(0) + c2 * qml.X(0) + c2 * qml.I(0)
            return qml.expval(H)

        x = qml.numpy.array(0.5)
        actual = qml.grad(circuit)(x)

        assert qml.math.allclose(actual, -0.1 * np.sin(x))

    @pytest.mark.jax
    @pytest.mark.parametrize("use_jit", [False, True])
    @pytest.mark.parametrize("grouping_strategy", [None, "default", "qwc", "wires"])
    def test_jax(self, grouping_strategy, use_jit):
        """Tests that the output of ``split_non_commuting`` is differentiable with jax"""

        import jax
        import jax.numpy as jnp

        dev = qml.device("default.qubit", wires=2)

        obs_list = complex_obs_list
        if not qml.operation.active_new_opmath():
            obs_list = obs_list[:-1]  # exclude the identity term

        @partial(split_non_commuting, grouping_strategy=grouping_strategy)
        @qml.qnode(dev)
        def circuit(theta, phi):
            qml.RX(theta, wires=0)
            qml.RY(phi, wires=0)
            qml.RX(theta, wires=1)
            qml.RY(phi, wires=1)
            return qml.probs(wires=[0, 1]), *[qml.expval(o) for o in obs_list]

        if use_jit:
            circuit = jax.jit(circuit)

        def cost(theta, phi):
            res = circuit(theta, phi)
            return qml.math.concatenate([res[0], qml.math.stack(res[1:])], axis=0)

        params = jnp.array(jnp.pi / 4), jnp.array(3 * jnp.pi / 4)
        grad1, grad2 = jax.jacobian(cost, argnums=[0, 1])(*params)

        expected_grad_1 = expected_grad_param_0
        expected_grad_2 = expected_grad_param_1
        if not qml.operation.active_new_opmath():
            expected_grad_1 = expected_grad_param_0[:-1]
            expected_grad_2 = expected_grad_param_1[:-1]

        assert qml.math.allclose(grad1, expected_grad_1)
        assert qml.math.allclose(grad2, expected_grad_2)

    @pytest.mark.jax
    @pytest.mark.parametrize("use_jit", [False, True])
    @pytest.mark.parametrize("grouping_strategy", [None, "default", "qwc", "wires"])
    def test_trainable_hamiltonian_jax(self, grouping_strategy, use_jit):
        """Tests that measurements of trainable Hamiltonians are differentiable with jax"""

        import jax
        import jax.numpy as jnp

        dev = qml.device("default.qubit", wires=2, shots=50000)

        @partial(split_non_commuting, grouping_strategy=grouping_strategy)
        @qml.qnode(dev)
        def circuit(coeff1, coeff2):
            qml.RX(np.pi / 4, wires=0)
            qml.RY(np.pi / 4, wires=1)
            return qml.expval(qml.Hamiltonian([coeff1, coeff2], [qml.Y(0) @ qml.Z(1), qml.X(1)]))

        if use_jit:
            circuit = jax.jit(circuit)

        params = jnp.array(np.pi / 4), jnp.array(3 * np.pi / 4)
        actual = jax.jacobian(circuit, argnums=[0, 1])(*params)

        assert qml.math.allclose(actual, [-0.5, np.cos(np.pi / 4)], rtol=0.05)

    @pytest.mark.torch
    @pytest.mark.parametrize("grouping_strategy", [None, "default", "qwc", "wires"])
    def test_torch(self, grouping_strategy):
        """Tests that the output of ``split_non_commuting`` is differentiable with torch"""

        import torch
        from torch.autograd.functional import jacobian

        dev = qml.device("default.qubit", wires=2)

        obs_list = complex_obs_list
        if not qml.operation.active_new_opmath():
            obs_list = obs_list[:-1]  # exclude the identity term

        @partial(split_non_commuting, grouping_strategy=grouping_strategy)
        @qml.qnode(dev)
        def circuit(theta, phi):
            qml.RX(theta, wires=0)
            qml.RY(phi, wires=0)
            qml.RX(theta, wires=1)
            qml.RY(phi, wires=1)
            return qml.probs(wires=[0, 1]), *[qml.expval(o) for o in obs_list]

        def cost(theta, phi):
            res = circuit(theta, phi)
            return qml.math.concatenate([res[0], qml.math.stack(res[1:])], axis=0)

        params = torch.tensor(np.pi / 4), torch.tensor(3 * np.pi / 4)
        grad1, grad2 = jacobian(cost, params)

        expected_grad_1 = expected_grad_param_0
        expected_grad_2 = expected_grad_param_1
        if not qml.operation.active_new_opmath():
            expected_grad_1 = expected_grad_param_0[:-1]
            expected_grad_2 = expected_grad_param_1[:-1]

        assert qml.math.allclose(grad1, expected_grad_1, atol=1e-5)
        assert qml.math.allclose(grad2, expected_grad_2, atol=1e-5)

    @pytest.mark.torch
    @pytest.mark.parametrize("grouping_strategy", [None, "default", "qwc", "wires"])
    def test_trainable_hamiltonian_torch(self, grouping_strategy):
        """Tests that measurements of trainable Hamiltonians are differentiable with torch"""

        import torch
        from torch.autograd.functional import jacobian

        dev = qml.device("default.qubit", wires=2, shots=50000)

        @partial(split_non_commuting, grouping_strategy=grouping_strategy)
        @qml.qnode(dev)
        def circuit(coeff1, coeff2):
            qml.RX(np.pi / 4, wires=0)
            qml.RY(np.pi / 4, wires=1)
            return qml.expval(qml.Hamiltonian([coeff1, coeff2], [qml.Y(0) @ qml.Z(1), qml.X(1)]))

        params = torch.tensor(np.pi / 4), torch.tensor(3 * np.pi / 4)
        actual = jacobian(circuit, params)

        assert qml.math.allclose(actual, [-0.5, np.cos(np.pi / 4)], rtol=0.05)

    @pytest.mark.tf
    @pytest.mark.parametrize("grouping_strategy", [None, "default", "qwc", "wires"])
    def test_tensorflow(self, grouping_strategy):
        """Tests that the output of ``split_non_commuting`` is differentiable with tensorflow"""

        import tensorflow as tf

        dev = qml.device("default.qubit", wires=2)

        obs_list = complex_obs_list
        if not qml.operation.active_new_opmath():
            obs_list = obs_list[:-1]  # exclude the identity term

        @qml.qnode(dev)
        def circuit(theta, phi):
            qml.RX(theta, wires=0)
            qml.RY(phi, wires=0)
            qml.RX(theta, wires=1)
            qml.RY(phi, wires=1)
            return qml.probs(wires=[0, 1]), *[qml.expval(o) for o in obs_list]

        params = tf.Variable(np.pi / 4), tf.Variable(3 * np.pi / 4)

        with tf.GradientTape() as tape:
            res = split_non_commuting(circuit, grouping_strategy=grouping_strategy)(*params)
            cost = qml.math.concatenate([res[0], qml.math.stack(res[1:])], axis=0)

        grad1, grad2 = tape.jacobian(cost, params)

        expected_grad_1 = expected_grad_param_0
        expected_grad_2 = expected_grad_param_1
        if not qml.operation.active_new_opmath():
            expected_grad_1 = expected_grad_param_0[:-1]
            expected_grad_2 = expected_grad_param_1[:-1]

        assert qml.math.allclose(grad1, expected_grad_1, atol=1e-5)
        assert qml.math.allclose(grad2, expected_grad_2, atol=1e-5)

    @pytest.mark.tf
    @pytest.mark.parametrize("grouping_strategy", [None, "default", "qwc", "wires"])
    def test_trainable_hamiltonian_tensorflow(self, grouping_strategy):
        """Tests that measurements of trainable Hamiltonians are differentiable with tensorflow"""

        import tensorflow as tf

        dev = qml.device("default.qubit", wires=2, shots=50000)

        @qml.qnode(dev)
        def circuit(coeff1, coeff2):
            qml.RX(np.pi / 4, wires=0)
            qml.RY(np.pi / 4, wires=1)
            return qml.expval(qml.Hamiltonian([coeff1, coeff2], [qml.Y(0) @ qml.Z(1), qml.X(1)]))

        params = tf.Variable(np.pi / 4), tf.Variable(3 * np.pi / 4)

        with tf.GradientTape() as tape:
            cost = split_non_commuting(circuit, grouping_strategy=grouping_strategy)(*params)

        actual = tape.jacobian(cost, params)

        assert qml.math.allclose(actual, [-0.5, np.cos(np.pi / 4)], rtol=0.05)
