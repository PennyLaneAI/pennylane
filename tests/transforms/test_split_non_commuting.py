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

"""Tests for the transform ``qml.transforms.split_non_commuting``"""
from functools import partial
from unittest.mock import MagicMock

import numpy as np
import pytest

import pennylane as qml
from pennylane.transforms import split_non_commuting
from pennylane.transforms.split_non_commuting import ShotDistFunction

# Two qubit-wise commuting groups: [[0, 3], [1, 2, 4]]
# Four groups based on wire overlaps: [[0, 2], [1], [3], [4]]
single_term_obs_list = [
    qml.X(0),
    qml.Y(0),
    qml.Z(1),
    qml.X(0) @ qml.Y(1),
    qml.Y(0) @ qml.Z(1),
    qml.I(0),
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
# qwc groups: [[0, 5], [1, 3, 4], [2]]
# wires groups: [[0, 3], [1, 4], [2], [5]]
complex_obs_list = [
    qml.X(0),  # single observable
    0.5 * qml.Y(0),  # scalar product
    qml.X(0) + qml.Y(0) @ qml.Z(1) + 2.0 * qml.X(1) + qml.I(),  # sum
    qml.Hamiltonian(
        [0.1, 0.2, 0.3, 0.4], [qml.Z(1), qml.X(0) @ qml.Y(1), qml.Y(0) @ qml.Z(1), qml.I()]
    ),
    1.5 * qml.I(0),  # identity
]

complex_no_grouping_obs = [
    qml.X(0),
    qml.Y(0),
    qml.Y(0) @ qml.Z(1),
    qml.X(1),
    qml.Z(1),
    qml.X(0) @ qml.Y(1),
]


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
        group0[0] + group1[1] + 2.0 * group2 + 1.0,
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


# contains the following observables: X(0), Y(0), Y(0) @ Z(1), X(1), H(0) @ X(1), Projector(1)
# wires groups: ((0, 3), (1, 5), (2,), (4,))
non_pauli_obs_list = [
    qml.X(0),
    0.5 * qml.Y(0),
    qml.X(0) + qml.Y(0) @ qml.Z(1) + 2.0 * qml.X(1) + qml.H(0) @ qml.PauliX(wires=[1]),
    qml.Projector([0, 1], wires=1),
]

non_pauli_obs_wires_groups = [
    [qml.X(0), qml.X(1)],
    [qml.Y(0), qml.Projector([0, 1], wires=1)],
    [qml.Y(0) @ qml.Z(1)],
    [qml.H(0) @ qml.X(1)],
]


def non_pauli_obs_processing_fn(results):
    """The expected processing function for non-Pauli observables"""

    group0, group1, group2, group3 = results
    res = (group0[0], 0.5 * group1[0], group0[0] + group2 + 2.0 * group0[1] + group3, group1[1])
    return res


@pytest.mark.integration
class TestSplitNonCommuting:
    """Tests the basic functionality of the split_non_commuting transform

    The ``split_non_commuting`` transform supports three different grouping strategies:
     - wires: groups observables based on wire overlaps
     - qwc: groups observables based on qubit-wise commuting groups
     - None: no grouping (each observable is measured in a separate tape)

    The unit tests below test each grouping strategy separately. The tests in this test class
    focus on whether the transform produces the correct tapes, and whether the processing function
    is able to recombine the results correctly.

    """

    def test_tape_no_measurements(self):
        """Tests that a tape with no measurements is returned unchanged."""

        initial_tape = qml.tape.QuantumScript([qml.Z(0)], [])
        tapes, fn = split_non_commuting(initial_tape)
        assert tapes == [initial_tape]
        assert fn([0]) == 0

    @pytest.mark.parametrize(
        "H",
        [
            qml.Hamiltonian([0.1, 0.2, 0.3, 0.4, 0.5, 2.5], single_term_obs_list),
            qml.dot([0.1, 0.2, 0.3, 0.4, 0.5, 2.5], single_term_obs_list),
        ],
    )
    @pytest.mark.parametrize(
        "grouping_indices, results, expected_result",
        [
            (
                [[0, 2, 5], [1], [3], [4]],
                [[0.6, 0.7], [0.8], [0.9], [1]],
                0.6 * 0.1 + 0.7 * 0.3 + 0.8 * 0.2 + 0.9 * 0.4 + 0.5 + 2.5,
            ),
            (
                [[0, 3, 5], [1, 2, 4]],
                [[0.6, 0.7], [0.8, 0.9, 1]],
                0.6 * 0.1 + 0.7 * 0.4 + 0.8 * 0.2 + 0.9 * 0.3 + 0.5 + 2.5,
            ),
        ],
    )
    def test_single_hamiltonian_precomputed_grouping(
        self, H, grouping_indices, results, expected_result
    ):
        """Tests that precomputed grouping of a single Hamiltonian is used."""

        H.grouping_indices = grouping_indices  # pylint: disable=protected-access
        initial_tape = qml.tape.QuantumScript([qml.X(0)], [qml.expval(H)], shots=100)
        tapes, fn = split_non_commuting(initial_tape)
        assert len(tapes) == len(grouping_indices)

        for group_idx, group in enumerate(grouping_indices):
            ob_group = [single_term_obs_list[i] for i in group]
            obs_no_identity = [obs for obs in ob_group if not isinstance(obs, qml.Identity)]
            expected_measurements = [qml.expval(obs) for obs in obs_no_identity]
            assert tapes[group_idx].measurements == expected_measurements
            assert tapes[group_idx].shots.total_shots == 100
            assert tapes[group_idx].operations == [qml.X(0)]

        assert fn(results) == expected_result

    @pytest.mark.parametrize(
        "H",
        [
            # A duplicate term is added to the tests below to verify that it is handled correctly.
            qml.Hamiltonian([0.1, 0.2, 0.3, 0.4, 0.5, 2.5, 0.8], single_term_obs_list + [qml.X(0)]),
            qml.dot([0.1, 0.2, 0.3, 0.4, 0.5, 2.5, 0.8], single_term_obs_list + [qml.X(0)]),
        ],
    )
    @pytest.mark.parametrize("grouping_strategy", ["qwc", "default"])
    def test_single_hamiltonian_grouping(self, H, grouping_strategy):
        """Tests that a single Hamiltonian is split correctly."""

        initial_tape = qml.tape.QuantumScript([qml.X(0)], [qml.expval(H)], shots=100)
        tapes, fn = split_non_commuting(initial_tape, grouping_strategy=grouping_strategy)
        assert H.grouping_indices is not None  # H.grouping_indices should be computed by now.

        groups_without_duplicates = []
        for group in H.grouping_indices:
            # The actual groups should not contain the added duplicate item
            groups_without_duplicates.append([i for i in group if i != 6])

        obs_list = single_term_obs_list + [qml.X(0)]
        for group_idx, group in enumerate(groups_without_duplicates):
            ob_group = [obs_list[i] for i in group]
            obs_no_identity = [obs for obs in ob_group if not isinstance(obs, qml.Identity)]
            expected_measurements = [qml.expval(obs) for obs in obs_no_identity]
            assert tapes[group_idx].measurements == expected_measurements
            assert tapes[group_idx].shots.total_shots == 100
            assert tapes[group_idx].operations == [qml.X(0)]

        assert (
            fn([[0.6, 0.7], [0.8, 0.9, 1]])
            == 0.6 * 0.1 + 0.7 * 0.4 + 0.8 * 0.2 + 0.9 * 0.3 + 0.5 + 2.5 + 0.8 * 0.6
        )

    @pytest.mark.parametrize(
        "obs, terms, results, expected_result",
        [
            (
                [qml.Hamiltonian([0.1, 0.2, 0.3, 0.4, 0.5, 2.5], single_term_obs_list)],
                single_term_obs_list[:-1],
                [0.6, 0.7, 0.8, 0.9, 1],
                0.6 * 0.1 + 0.7 * 0.2 + 0.8 * 0.3 + 0.9 * 0.4 + 0.5 + 2.5,
            ),
            (
                single_term_obs_list,
                single_term_obs_list[:-1],
                [0.1, 0.2, 0.3, 0.4, 0.5],
                [0.1, 0.2, 0.3, 0.4, 0.5, 1.0],
            ),
            (
                complex_obs_list,
                complex_no_grouping_obs,
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                complex_no_grouping_processing_fn([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            ),
        ],
    )
    def test_no_grouping(self, obs, terms, results, expected_result):
        """Tests splitting each observable into a separate tape."""

        initial_tape = qml.tape.QuantumScript(
            [qml.X(0)], measurements=[qml.expval(o) for o in obs], shots=100
        )
        tapes, fn = split_non_commuting(initial_tape, grouping_strategy=None)
        assert len(tapes) == len(terms)
        for tape, term in zip(tapes, terms):
            assert tape.measurements == [qml.expval(term)]
            assert tape.shots.total_shots == 100
            assert tape.operations == [qml.X(0)]

        assert qml.math.allclose(fn(results), expected_result)

    @pytest.mark.parametrize(
        "obs, groups, results, expected_result, grouping_strategy",
        [
            # The following tests should route to wire-based grouping.
            (
                [qml.Hamiltonian([0.1, 0.2, 0.3, 0.4, 0.5, 2.5], single_term_obs_list)],
                single_term_wires_groups,
                [[0.6, 0.7], 0.8, 0.9, 1],
                0.6 * 0.1 + 0.7 * 0.3 + 0.8 * 0.2 + 0.9 * 0.4 + 0.5 + 2.5,
                "wires",
            ),
            (
                single_term_obs_list,
                single_term_wires_groups,
                [[0.1, 0.2], 0.3, 0.4, 0.5],
                [0.1, 0.3, 0.2, 0.4, 0.5, 1.0],
                "wires",
            ),
            (
                complex_obs_list,
                complex_wires_groups,  # [[0, 3], [1, 4], [2], [5]]
                [[0.1, 0.2], [0.3, 0.4], 0.5, 0.6],
                complex_wires_processing_fn([[0.1, 0.2], [0.3, 0.4], 0.5, 0.6]),
                "wires",
            ),
            (
                non_pauli_obs_list,
                non_pauli_obs_wires_groups,
                [[0.1, 0.2], [0.3, 0.4], 0.5, 0.6],
                non_pauli_obs_processing_fn([[0.1, 0.2], [0.3, 0.4], 0.5, 0.6]),
                "default",  # wire-based grouping should be automatically chosen
            ),
            # The following tests should route to qwc grouping.
            (
                [qml.Hamiltonian([0.1, 0.2, 0.3, 0.4, 0.5, 2.5], single_term_obs_list)],
                single_term_qwc_groups,  # [[0, 3], [1, 2, 4]]
                [[0.6, 0.7], [0.8, 0.9, 1]],
                0.6 * 0.1 + 0.7 * 0.4 + 0.8 * 0.2 + 0.9 * 0.3 + 0.5 + 2.5,
                "default",  # qwc grouping should be the default in this case.
            ),
            (
                [qml.Hamiltonian([0.1, 0.2, 0.3, 0.4, 0.5, 2.5], single_term_obs_list)],
                single_term_qwc_groups,  # [[0, 3], [1, 2, 4]]
                [[0.6, 0.7], [0.8, 0.9, 1]],
                0.6 * 0.1 + 0.7 * 0.4 + 0.8 * 0.2 + 0.9 * 0.3 + 0.5 + 2.5,
                "qwc",
            ),
            (
                single_term_obs_list,
                single_term_qwc_groups,
                [[0.6, 0.7], [0.8, 0.9, 1]],
                [0.6, 0.8, 0.9, 0.7, 1.0, 1.0],
                "default",  # qwc grouping should be the default in this case.
            ),
            (
                single_term_obs_list,
                single_term_qwc_groups,
                [[0.6, 0.7], [0.8, 0.9, 1]],
                [0.6, 0.8, 0.9, 0.7, 1.0, 1.0],
                "qwc",
            ),
            (
                complex_obs_list,
                complex_qwc_groups,  # [[0, 5], [1, 3, 4], [2]]
                [[0.1, 0.2], [0.3, 0.4, 0.5], 0.6],
                complex_qwc_processing_fn([[0.1, 0.2], [0.3, 0.4, 0.5], 0.6]),
                "qwc",
            ),
            (
                complex_obs_list,
                complex_qwc_groups,
                [[0.1, 0.2], [0.3, 0.4, 0.5], 0.6],
                complex_qwc_processing_fn([[0.1, 0.2], [0.3, 0.4, 0.5], 0.6]),
                "default",  # qwc grouping should be automatically chosen for this case.
            ),
        ],
    )
    def test_grouping_strategies(
        self, obs, groups, results, expected_result, grouping_strategy
    ):  # pylint: disable=too-many-arguments
        """Tests wire-based grouping and qwc grouping."""

        initial_tape = qml.tape.QuantumScript(
            [qml.X(0)], measurements=[qml.expval(o) for o in obs], shots=100
        )
        tapes, fn = split_non_commuting(initial_tape, grouping_strategy=grouping_strategy)
        assert len(tapes) == len(groups)
        for tape, group in zip(tapes, groups):
            assert tape.measurements == [qml.expval(term) for term in group]
            assert tape.shots.total_shots == 100
            assert tape.operations == [qml.X(0)]

        assert qml.math.allclose(fn(results), expected_result)

    @pytest.mark.parametrize("grouping_strategy", ["wires", "qwc"])
    def test_single_group(self, grouping_strategy):
        """Tests when all measurements can be taken at the same time."""

        initial_tape = qml.tape.QuantumScript(
            [qml.X(0)],
            measurements=[qml.expval(qml.X(0)), qml.expval(qml.Y(1) + qml.Z(2))],
            shots=100,
        )
        tapes, fn = split_non_commuting(initial_tape, grouping_strategy=grouping_strategy)
        assert len(tapes) == 1
        assert tapes[0].measurements == [
            qml.expval(qml.X(0)),
            qml.expval(qml.Y(1)),
            qml.expval(qml.Z(2)),
        ]
        assert tapes[0].shots.total_shots == 100
        assert tapes[0].operations == [qml.X(0)]
        assert qml.math.allclose(fn([[0.1, 0.2, 0.3]]), [0.1, 0.2 + 0.3])

    @pytest.mark.parametrize("grouping_strategy", ["wires", "qwc", None])
    def test_single_observable(self, grouping_strategy):
        """Tests a tape containing measurements of a single observable."""

        initial_tape = qml.tape.QuantumScript(
            [qml.X(0)], measurements=[qml.expval(qml.Z(0)), qml.expval(0.5 * qml.Z(0))], shots=100
        )
        tapes, fn = split_non_commuting(initial_tape, grouping_strategy=grouping_strategy)
        assert len(tapes) == 1
        assert tapes[0].measurements == [qml.expval(qml.Z(0))]
        assert tapes[0].shots.total_shots == 100
        assert tapes[0].operations == [qml.X(0)]
        assert qml.math.allclose(fn([0.1]), [0.1, 0.05])

    def test_mix_measurement_types_qwc(self):
        """Tests multiple measurement types can be handled by qwc grouping"""

        initial_tape = qml.tape.QuantumScript(
            [qml.X(0)],
            measurements=[
                # The observables in the following list of measurements are listed here.
                # Note that wire-based measurements are assumed to be in the Z-basis, and
                # therefore is assigned a dummy observable of Z
                # [Z(0), X(0), Z(0), X(0), Y(1), Z(1), Z(0) @ Z(1), Z(1), Z(0)]
                qml.expval(qml.Z(0)),
                qml.expval(qml.Z(0) + qml.X(0)),
                qml.var(qml.Z(0)),
                qml.probs(op=qml.X(0)),
                qml.counts(qml.Y(1)),
                qml.sample(qml.Z(1)),
                qml.probs(wires=[0, 1]),
                qml.sample(wires=[1]),
                qml.counts(wires=[0]),
            ],
        )

        # The list of observables in the comment above can be placed in two groups:
        # ((0, 2, 5, 6, 7, 8), (1, 3, 4))
        tapes, fn = split_non_commuting(initial_tape, grouping_strategy="qwc")
        assert len(tapes) == 2
        assert tapes[0].measurements == [
            qml.expval(qml.Z(0)),
            qml.var(qml.Z(0)),
            qml.sample(qml.Z(1)),
            qml.probs(wires=[0, 1]),
            qml.sample(wires=[1]),
            qml.counts(wires=[0]),
        ]
        assert tapes[1].measurements == [
            qml.expval(qml.X(0)),
            qml.probs(op=qml.X(0)),
            qml.counts(qml.Y(1)),
        ]
        results = fn(
            [
                [
                    0.1,
                    0.2,
                    np.array([1.0, 1.0]),
                    np.array([1.0, 0.0, 0.0, 0.0]),
                    np.array([0, 0]),
                    {"0": 2},
                ],
                [0.5, np.array([0.5, 0.5]), {"1": 2}],
            ]
        )
        expected = (
            0.1,
            0.1 + 0.5,
            0.2,
            np.array([0.5, 0.5]),
            {"1": 2},
            np.array([1.0, 1.0]),
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([0, 0]),
            {"0": 2},
        )
        assert len(results) == len(expected)
        for res, _expected in zip(results, expected):
            if isinstance(res, np.ndarray):
                assert np.allclose(res, _expected)
            else:
                assert res == _expected

    def test_mix_measurement_types_wires(self):
        """Tests multiple measurement types can be handled by wire-based grouping"""

        initial_tape = qml.tape.QuantumScript(
            [qml.X(0)],
            measurements=[
                # [Z(0), X(0), Z(0), X(0), Y(1), Z(1), Z(0) @ Z(1), Z(1), Z(0)]
                qml.expval(qml.Z(0)),
                qml.expval(qml.Z(0) + qml.X(0)),
                qml.var(qml.Z(0)),
                qml.probs(op=qml.X(0)),
                qml.counts(qml.Y(1)),
                qml.sample(qml.Z(1)),
                qml.probs(wires=[0, 1]),
                qml.sample(wires=[1]),
                qml.counts(wires=[0]),
            ],
        )

        # ((0, 4), (1, 5), (2, 7), (3, ), (6, ), (8, ))
        tapes, fn = split_non_commuting(initial_tape, grouping_strategy="wires")
        assert len(tapes) == 6
        expected_groups = [
            [qml.expval(qml.Z(0)), qml.counts(qml.Y(1))],
            [qml.expval(qml.X(0)), qml.sample(qml.Z(1))],
            [qml.var(qml.Z(0)), qml.sample(wires=[1])],
            [qml.probs(op=qml.X(0))],
            [qml.probs(wires=[0, 1])],
            [qml.counts(wires=[0])],
        ]
        for tape, group in zip(tapes, expected_groups):
            assert tape.measurements == group

        results = fn(
            [
                [0.1, {"0": 2}],
                [0.3, np.array([1.0, 1.0])],
                [0.9, np.array([-1.0, -1.0])],
                np.array([0.5, 0.5]),
                np.array([1.0, 0.0, 0.0, 0.0]),
                {"1": 2},
            ]
        )
        expected = (
            0.1,
            0.1 + 0.3,
            0.9,
            np.array([0.5, 0.5]),
            {"0": 2},
            np.array([1.0, 1.0]),
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([-1.0, -1.0]),
            {"1": 2},
        )

        assert len(results) == len(expected)
        for res, _expected in zip(results, expected):
            if isinstance(res, np.ndarray):
                assert np.allclose(res, _expected)
            else:
                assert res == _expected

    @pytest.mark.parametrize("grouping_strategy", ["qwc", "wires"])
    def test_state_measurement_in_separate_tape(self, grouping_strategy):
        """Tests that a state measurement is in a separate tape

        The legacy device does not support state measurements combined with any other
        measurement, so each state measurement must be in its own tape.

        """

        measurements = [qml.expval(qml.Z(0)), qml.state()]
        initial_tape = qml.tape.QuantumScript([qml.X(0)], measurements, shots=100)
        tapes, _ = split_non_commuting(initial_tape, grouping_strategy=grouping_strategy)
        assert len(tapes) == 2
        for tape, meas in zip(tapes, measurements):
            assert tape.measurements == [meas]
            assert tape.shots == initial_tape.shots
            assert tape.operations == [qml.X(0)]

    @pytest.mark.parametrize(
        "obs",
        [
            qml.X(0) + qml.Y(1),
            2 * (qml.X(0) + qml.Y(1)),
            (qml.X(0) + qml.Y(1)) @ qml.X(1),
        ],
    )
    def test_unsupported_mps_of_sum(self, obs):
        """Tests a measurement of Sum other than expval raises an error."""

        initial_tape = qml.tape.QuantumScript([], measurements=[qml.counts(obs)])
        with pytest.raises(RuntimeError, match="Cannot split up terms in sums"):
            _, __ = split_non_commuting(initial_tape)


@pytest.mark.system
class TestQNodeIntegration:
    """Tests that split_non_commuting is correctly applied to a QNode.

    This test class focuses on testing the ``split_non_commuting`` transform applied to a QNode.
    These are end-to-end tests for how the transform integrates with the full execution workflow.
    Here we include tests for different combinations of shot vectors and parameter broadcasting,
    as well as some edge cases.

    It's typically unnecessary to test numerical correctness here, since the unit tests in the
    test above should have caught any mathematical errors that the transform might make. Here we
    simply want to make sure that the workflow executes without error, and that the results
    have the correct shapes. Note that the differentiation tests in the test class below will
    test the numerical correctness of the derivatives, which acts as a safeguard.

    """

    @pytest.mark.parametrize("grouping_strategy", [None, "qwc", "wires"])
    @pytest.mark.parametrize("shots", [None, 10, [10, 20, 30]])
    @pytest.mark.parametrize(
        "params",
        [
            [0.1, 0.2, 0.3],
            [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]],
        ],
    )
    def test_measurement_of_single_hamiltonian(self, grouping_strategy, shots, params):
        """Tests executing a QNode returning a single measurement of a Hamiltonian."""

        coeffs, obs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], single_term_obs_list

        dev = qml.device("default.qubit", wires=3)

        @partial(split_non_commuting, grouping_strategy=grouping_strategy)
        @qml.set_shots(shots)
        @qml.qnode(dev)
        def circuit(angles):
            qml.RX(angles[0], wires=0)
            qml.RY(angles[1], wires=1)
            qml.RZ(angles[2], wires=2)
            return qml.expval(qml.Hamiltonian(coeffs, obs))

        res = circuit(params)

        shot_dimension = (3,) if isinstance(shots, list) else ()
        parameter_dimension = (2,) if len(qml.math.shape(params)) > 1 else ()
        expected_dimension = shot_dimension + parameter_dimension
        assert qml.math.shape(res) == expected_dimension

    @pytest.mark.parametrize("grouping_strategy", [None, "qwc", "wires"])
    @pytest.mark.parametrize("shots", [10, [10, 20, 30]])
    @pytest.mark.parametrize(
        "params",
        [
            [0.1, 0.2, 0.3],
            [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]],
        ],
    )
    def test_general_circuits(self, grouping_strategy, shots, params):
        """Tests executing a QNode with different grouping strategies on a typical circuit."""

        dev = qml.device("default.qubit", wires=3)

        @partial(split_non_commuting, grouping_strategy=grouping_strategy)
        @qml.set_shots(shots)
        @qml.qnode(dev)
        def circuit(angles):
            qml.RX(angles[0], wires=0)
            qml.RY(angles[1], wires=1)
            qml.RZ(angles[2], wires=2)
            return (
                qml.expval(qml.X(0) + qml.Y(0) @ qml.Z(1) + 2.0 * qml.X(1) + qml.I()),
                qml.expval(qml.X(0)),
                qml.expval(0.5 * qml.Y(0)),
                qml.expval(1.5 * qml.I(0)),
                qml.var(qml.Z(0)),
                qml.probs(op=qml.X(0)),
                qml.counts(qml.Y(1)),
                qml.sample(qml.Z(1)),
                qml.probs(wires=[0, 1]),
                qml.sample(wires=[1]),
                qml.counts(wires=[0]),
            )

        shot_dimension = (3,) if isinstance(shots, list) else ()
        measurements_dimension = (11,)
        parameter_dimension = (2,) if len(qml.math.shape(params)) > 1 else ()
        expected_shape = shot_dimension + measurements_dimension + parameter_dimension

        result = circuit(params)

        def _recursively_check_shape(_result, _expected_shape):
            """Recursively check the shape of _result and _expected_shape.

            ``qml.math.shape`` will not work in this case because there are arrays of different
            shapes and dictionaries nested in the results.

            """

            if not _expected_shape:
                return

            assert len(_result) == _expected_shape[0]
            for res in _result:
                _recursively_check_shape(res, _expected_shape[1:])

        _recursively_check_shape(result, expected_shape)

    @pytest.mark.parametrize("grouping_strategy", [None, "qwc", "wires"])
    @pytest.mark.parametrize(
        "obs, expected_result",
        [
            ([qml.Hamiltonian([1.5, 2.5], [qml.I(0), qml.I(1)])], 4.0),
            ([1.5 * qml.I(), 2.5 * qml.I()], [1.5, 2.5]),
        ],
    )
    def test_only_constant_offset(self, grouping_strategy, obs, expected_result):
        """Tests that split_non_commuting can handle a circuit only measuring Identity."""

        dev = qml.device("default.qubit", wires=2)

        @partial(split_non_commuting, grouping_strategy=grouping_strategy)
        @qml.qnode(dev)
        def circuit():
            return tuple(qml.expval(ob) for ob in obs)

        with dev.tracker:
            res = circuit()
        assert dev.tracker.totals == {}
        assert qml.math.allclose(res, expected_result)


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


def circuit_to_test(device, grouping_strategy):
    """The test circuit used in differentiation tests."""

    @partial(split_non_commuting, grouping_strategy=grouping_strategy)
    @qml.qnode(device)
    def circuit(theta, phi):
        qml.RX(theta, wires=0)
        qml.RY(phi, wires=0)
        qml.RX(theta, wires=1)
        qml.RY(phi, wires=1)
        return qml.probs(wires=[0, 1]), *[qml.expval(o) for o in complex_obs_list]

    return circuit


def circuit_with_trainable_H(device, grouping_strategy):
    """Test circuit with trainable Hamiltonian."""

    @partial(split_non_commuting, grouping_strategy=grouping_strategy)
    @qml.qnode(device)
    def circuit(coeff1, coeff2):
        qml.RX(np.pi / 4, wires=0)
        qml.RY(np.pi / 4, wires=1)
        return qml.expval(qml.Hamiltonian([coeff1, coeff2], [qml.Y(0) @ qml.Z(1), qml.X(1)]))

    return circuit


class TestDifferentiability:
    """Tests the differentiability of the ``split_non_commuting`` transform"""

    @pytest.mark.autograd
    @pytest.mark.parametrize("grouping_strategy", [None, "qwc", "wires"])
    def test_autograd(self, grouping_strategy):
        """Tests that the output of ``split_non_commuting`` is differentiable with autograd"""

        import pennylane.numpy as pnp

        dev = qml.device("default.qubit", wires=2)
        circuit = circuit_to_test(dev, grouping_strategy=grouping_strategy)

        def cost(theta, phi):
            res = circuit(theta, phi)
            return qml.math.concatenate([res[0], qml.math.stack(res[1:])], axis=0)

        params = pnp.array(pnp.pi / 4), pnp.array(3 * pnp.pi / 4)
        grad1, grad2 = qml.jacobian(cost)(*params)

        expected_grad_1 = expected_grad_param_0
        expected_grad_2 = expected_grad_param_1

        assert qml.math.allclose(grad1, expected_grad_1)
        assert qml.math.allclose(grad2, expected_grad_2)

    @pytest.mark.autograd
    @pytest.mark.parametrize("grouping_strategy", [None, "qwc", "wires"])
    def test_trainable_hamiltonian_autograd(self, grouping_strategy):
        """Tests that measurements of trainable Hamiltonians are differentiable"""

        import pennylane.numpy as pnp

        dev = qml.device("default.qubit", wires=2)
        circuit = circuit_with_trainable_H(dev, grouping_strategy=grouping_strategy)

        params = pnp.array(pnp.pi / 4), pnp.array(3 * pnp.pi / 4)
        actual = qml.jacobian(circuit)(*params)

        assert qml.math.allclose(actual, [-0.5, np.cos(np.pi / 4)])

    @pytest.mark.autograd
    @pytest.mark.parametrize("grouping_strategy", [None, "qwc", "wires"])
    def test_non_trainable_obs_autograd(self, grouping_strategy):
        """Test that we can measure a hamiltonian with non-trainable autograd coefficients."""

        dev = qml.device("default.qubit")

        @partial(split_non_commuting, grouping_strategy=grouping_strategy)
        @qml.qnode(dev)
        def circuit(param):
            qml.RX(param, 0)
            c1 = qml.numpy.array(0.1, requires_grad=False)
            c2 = qml.numpy.array(0.2, requires_grad=False)
            H = c1 * qml.Z(0) + c2 * qml.X(0) + c2 * qml.I(0)
            return qml.expval(H)

        x = qml.numpy.array(0.5)
        actual = qml.grad(circuit)(x)

        assert qml.math.allclose(actual, -0.1 * np.sin(x))

    @pytest.mark.jax
    @pytest.mark.parametrize("use_jit", [False, True])
    @pytest.mark.parametrize("grouping_strategy", [None, "qwc", "wires"])
    def test_jax(self, grouping_strategy, use_jit):
        """Tests that the output of ``split_non_commuting`` is differentiable with jax"""

        import jax
        import jax.numpy as jnp

        dev = qml.device("default.qubit", wires=2)
        circuit = circuit_to_test(dev, grouping_strategy=grouping_strategy)

        if use_jit:
            circuit = jax.jit(circuit)

        def cost(theta, phi):
            res = circuit(theta, phi)
            return qml.math.concatenate([res[0], qml.math.stack(res[1:])], axis=0)

        params = jnp.array(jnp.pi / 4), jnp.array(3 * jnp.pi / 4)
        grad1, grad2 = jax.jacobian(cost, argnums=[0, 1])(*params)

        expected_grad_1 = expected_grad_param_0
        expected_grad_2 = expected_grad_param_1

        assert qml.math.allclose(grad1, expected_grad_1)
        assert qml.math.allclose(grad2, expected_grad_2)

    @pytest.mark.jax
    @pytest.mark.parametrize("use_jit", [False, True])
    @pytest.mark.parametrize("grouping_strategy", [None, "qwc", "wires"])
    def test_trainable_hamiltonian_jax(self, grouping_strategy, use_jit):
        """Tests that measurements of trainable Hamiltonians are differentiable with jax"""

        import jax
        import jax.numpy as jnp

        dev = qml.device("default.qubit", wires=2)
        circuit = circuit_with_trainable_H(dev, grouping_strategy=grouping_strategy)

        if use_jit:
            circuit = jax.jit(circuit)

        params = jnp.array(np.pi / 4), jnp.array(3 * np.pi / 4)
        actual = jax.jacobian(circuit, argnums=[0, 1])(*params)

        assert qml.math.allclose(actual, [-0.5, np.cos(np.pi / 4)])

    @pytest.mark.torch
    @pytest.mark.parametrize("grouping_strategy", [None, "qwc", "wires"])
    def test_torch(self, grouping_strategy):
        """Tests that the output of ``split_non_commuting`` is differentiable with torch"""

        import torch
        from torch.autograd.functional import jacobian

        dev = qml.device("default.qubit", wires=2)
        circuit = circuit_to_test(dev, grouping_strategy=grouping_strategy)

        def cost(theta, phi):
            res = circuit(theta, phi)
            return qml.math.concatenate([res[0], qml.math.stack(res[1:])], axis=0)

        params = torch.tensor(np.pi / 4), torch.tensor(3 * np.pi / 4)
        grad1, grad2 = jacobian(cost, params)

        expected_grad_1 = expected_grad_param_0
        expected_grad_2 = expected_grad_param_1

        assert qml.math.allclose(grad1, expected_grad_1, atol=1e-7)
        assert qml.math.allclose(grad2, expected_grad_2, atol=1e-7)

    @pytest.mark.torch
    @pytest.mark.parametrize("grouping_strategy", [None, "qwc", "wires"])
    def test_trainable_hamiltonian_torch(self, grouping_strategy):
        """Tests that measurements of trainable Hamiltonians are differentiable with torch"""

        import torch
        from torch.autograd.functional import jacobian

        dev = qml.device("default.qubit", wires=2)
        circuit = circuit_with_trainable_H(dev, grouping_strategy=grouping_strategy)

        params = torch.tensor(np.pi / 4), torch.tensor(3 * np.pi / 4)
        actual = jacobian(circuit, params)

        assert qml.math.allclose(actual, [-0.5, np.cos(np.pi / 4)])

    @pytest.mark.tf
    @pytest.mark.parametrize("grouping_strategy", [None, "qwc", "wires"])
    def test_tensorflow(self, grouping_strategy):
        """Tests that the output of ``split_non_commuting`` is differentiable with tensorflow"""

        import tensorflow as tf

        dev = qml.device("default.qubit", wires=2)
        circuit = circuit_to_test(dev, grouping_strategy=grouping_strategy)

        params = tf.Variable(np.pi / 4), tf.Variable(3 * np.pi / 4)

        with tf.GradientTape() as tape:
            res = circuit(*params)
            cost = qml.math.concatenate([res[0], qml.math.stack(res[1:])], axis=0)

        grad1, grad2 = tape.jacobian(cost, params)

        expected_grad_1 = expected_grad_param_0
        expected_grad_2 = expected_grad_param_1

        assert qml.math.allclose(grad1, expected_grad_1, atol=1e-7)
        assert qml.math.allclose(grad2, expected_grad_2, atol=1e-7)

    @pytest.mark.tf
    @pytest.mark.parametrize("grouping_strategy", [None, "qwc", "wires"])
    def test_trainable_hamiltonian_tensorflow(self, grouping_strategy):
        """Tests that measurements of trainable Hamiltonians are differentiable with tensorflow"""

        import tensorflow as tf

        dev = qml.device("default.qubit", wires=2)
        circuit = circuit_with_trainable_H(dev, grouping_strategy=grouping_strategy)

        params = tf.Variable(np.pi / 4), tf.Variable(3 * np.pi / 4)

        with tf.GradientTape() as tape:
            cost = split_non_commuting(circuit, grouping_strategy=grouping_strategy)(*params)

        actual = tape.jacobian(cost, params)

        assert qml.math.allclose(actual, [-0.5, np.cos(np.pi / 4)])


# Single hamiltonian expval measurement to test shot distribution
ham = qml.Hamiltonian(
    coeffs=[10, 0.1, 20, 100, 0.2],
    observables=[
        qml.X(0) @ qml.Y(1),
        qml.Z(0) @ qml.Z(2),
        qml.Y(1),
        qml.X(1) @ qml.X(2),
        qml.Z(0) @ qml.Z(1) @ qml.Z(2),
    ],
)


class TestShotDistribution:
    """
    Test shot distribution for the `split_non_commuting` transform.
    At the moment, this feature is available only for the single hamiltonian
    measurement case and with "default" or "qwc" grouping_strategy.
    """

    @pytest.mark.parametrize("grouping_strategy", ["default", "qwc", "wires", None])
    def test_none_shot_dist(self, grouping_strategy):
        """Test standard behaviour with no shot distribution strategy."""

        total_shots = 1000
        initial_tape = qml.tape.QuantumScript(measurements=[qml.expval(ham)], shots=total_shots)

        tapes, _ = split_non_commuting(
            initial_tape,
            grouping_strategy=grouping_strategy,
            shot_dist=None,
        )

        assert sum(tape.shots.total_shots for tape in tapes) == total_shots * len(tapes)

        for tape in tapes:
            assert tape.shots.total_shots == total_shots

    @pytest.mark.parametrize("shot_dist", [0, 1.2, [], {}])
    def test_type_error_shot_dist(self, shot_dist):
        """Test an error is raised for shot_dist incorrect type."""

        initial_tape = qml.tape.QuantumScript(measurements=[qml.expval(ham)], shots=10)

        with pytest.raises(TypeError, match="shot_dist must be a callable or str or None,"):
            _ = split_non_commuting(initial_tape, shot_dist=shot_dist)

    def test_value_error_shot_dist(self):
        """Test an error is raised for an unknown shot_dist strategy."""

        initial_tape = qml.tape.QuantumScript(measurements=[qml.expval(ham)], shots=10)

        with pytest.raises(ValueError, match="Unknown shot_dist='unknown'. Available options are"):
            _ = split_non_commuting(initial_tape, shot_dist="unknown")

    @pytest.mark.parametrize("grouping_strategy", ["default", "qwc", "wires", None])
    @pytest.mark.parametrize("shot_dist", ["uniform", "weighted", "weighted_random"])
    def test_warning_multiple_measurements(self, grouping_strategy, shot_dist):
        """Test a warning is raised for the multiple measurements case."""

        initial_tape = qml.tape.QuantumScript(
            measurements=[qml.expval(ham), qml.expval(ham)], shots=10
        )

        with pytest.warns(
            UserWarning,
            match=f"shot_dist='{shot_dist}' is not supported for multiple measurements.",
        ):
            _ = split_non_commuting(
                initial_tape, grouping_strategy=grouping_strategy, shot_dist=shot_dist
            )

    @pytest.mark.parametrize("grouping_strategy", ["default", "qwc"])
    @pytest.mark.parametrize(
        ["shot_dist", "expected_shots"],
        [
            ("uniform", (334, 333, 333)),
            ("weighted", (231, 2, 767)),
        ],
    )
    def test_single_hamiltonian_sampling_strategy(
        self, grouping_strategy, shot_dist, expected_shots
    ):
        """Test built-in deterministic shot distribution strategies for the single hamiltonian case."""

        total_shots = 1000
        initial_tape = qml.tape.QuantumScript(measurements=[qml.expval(ham)], shots=total_shots)

        tapes, _ = split_non_commuting(
            initial_tape, grouping_strategy=grouping_strategy, shot_dist=shot_dist
        )

        # check that the original total number of shots is conserved
        assert sum(tape.shots.total_shots for tape in tapes) == total_shots

        # check that for all output tapes the number of shots is computed as expected
        for tape, shots in zip(tapes, expected_shots, strict=True):
            assert tape.shots.total_shots == shots

    @pytest.mark.parametrize("grouping_strategy", ["default", "qwc"])
    def test_single_hamiltonian_random_sampling_strategy(self, grouping_strategy, seed):
        """Test built-in random shot distribution strategy for the single hamiltonian case."""

        total_shots = 1000
        initial_tape = qml.tape.QuantumScript(measurements=[qml.expval(ham)], shots=total_shots)

        tapes, _ = split_non_commuting(
            initial_tape,
            grouping_strategy=grouping_strategy,
            shot_dist="weighted_random",
            seed=seed,
        )

        # check that the original total number of shots is conserved
        assert sum(tape.shots.total_shots for tape in tapes) == total_shots

        shots_per_tape = [tape.shots.total_shots for tape in tapes]
        expected_shots = [231, 2, 767]

        # check that the number of shots for each tape is close enough to the expected number
        assert np.allclose(shots_per_tape, expected_shots, rtol=0.1, atol=5)

    @pytest.mark.parametrize("grouping_strategy", ["default", "qwc"])
    @pytest.mark.parametrize("seed", [42, None])
    def test_single_hamiltonian_mock_function(self, grouping_strategy, seed):
        """Test shot distribution mock function for the single hamiltonian case."""

        total_shots = 1000
        initial_tape = qml.tape.QuantumScript(measurements=[qml.expval(ham)], shots=total_shots)

        mock_shot_dist = MagicMock(spec=ShotDistFunction)
        mock_shot_dist.return_value = [334, 333, 333]

        _ = split_non_commuting(
            initial_tape,
            grouping_strategy=grouping_strategy,
            shot_dist=mock_shot_dist,
            seed=seed,
        )

        # check that the shot distribution function gets called exactly once with the expected signature
        coeffs_per_group = [[10, 20], [0.1, 0.2], [100]]
        mock_shot_dist.assert_called_once_with(total_shots, coeffs_per_group, seed)

    @pytest.mark.parametrize("grouping_strategy", ["default", "qwc"])
    def test_single_hamiltonian_custom_function(self, grouping_strategy):
        """Test shot distribution custom function for the single hamiltonian case."""

        initial_tape = qml.tape.QuantumScript(measurements=[qml.expval(ham)], shots=10)

        def custom_shot_dist(_, coeffs_per_group, __):
            return [1] * len(coeffs_per_group)

        tapes, _ = split_non_commuting(
            initial_tape,
            grouping_strategy=grouping_strategy,
            shot_dist=custom_shot_dist,
        )

        for tape in tapes:
            assert tape.shots.total_shots == 1

    @pytest.mark.parametrize("grouping_strategy", ["default", "qwc"])
    def test_drop_tape_with_zero_shots(self, grouping_strategy):
        """Test that a tape with zero shots gets dropped."""

        h = qml.Hamiltonian(coeffs=[1.0, 1.0, 1.0], observables=[qml.X(0), qml.Y(0), qml.Z(0)])
        initial_tape = qml.tape.QuantumScript(measurements=[qml.expval(h)], shots=100)

        mock_shot_dist = MagicMock(spec=ShotDistFunction)
        mock_shot_dist.return_value = [20, 0, 80]

        tapes, _ = split_non_commuting(
            initial_tape,
            grouping_strategy=grouping_strategy,
            shot_dist=mock_shot_dist,
        )

        assert len(tapes) == 2

        assert tapes[0].measurements[0] == qml.expval(qml.X(0))
        assert tapes[0].shots.total_shots == 20

        assert tapes[1].measurements[0] == qml.expval(qml.Z(0))
        assert tapes[1].shots.total_shots == 80
