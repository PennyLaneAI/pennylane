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

"""Tests for the transform ``qml.transform.split_to_single_terms``"""

# pylint: disable=import-outside-toplevel,unnecessary-lambda

from functools import partial
from typing import Optional

import numpy as np
import pytest

import pennylane as qml
from pennylane.transforms import split_to_single_terms
from pennylane.transforms.split_to_single_terms import null_postprocessing

single_term_obs_list = [
    qml.X(0),
    qml.Y(0),
    qml.Z(1),
    qml.X(0) @ qml.Y(1),
    qml.Y(0) @ qml.Z(1),
]


# contains the following observables: X(0), Y(0), Y(0) @ Z(1), X(1), Z(1), X(0) @ Y(1)
complex_obs_list = [
    qml.X(0),  # single observable
    0.5 * qml.Y(0),  # scalar product
    qml.X(0) + qml.Y(0) @ qml.Z(1) + 2.0 * qml.X(1) + qml.I(),  # sum
    qml.Hamiltonian(
        [0.1, 0.2, 0.3, 0.4], [qml.Z(1), qml.X(0) @ qml.Y(1), qml.Y(0) @ qml.Z(1), qml.I()]
    ),
    1.5 * qml.I(),  # identity
]


# pylint: disable=too-few-public-methods
class NoTermsDevice(qml.devices.DefaultQubit):
    """A device that builds on default.qubit, but won't accept LinearCombination or Sum"""

    def execute(self, circuits, execution_config: Optional[qml.devices.ExecutionConfig] = None):
        for t in circuits:
            for mp in t.measurements:
                if mp.obs and isinstance(mp.obs, qml.ops.Sum):
                    raise ValueError(
                        "no terms device does not accept observables with multiple terms"
                    )
        return super().execute(circuits, execution_config)


class TestUnits:
    """Unit tests for components of the ``split_to_single_terms`` transform"""

    def test_single_term_observable(self):
        """Test that the transform does not affect a circuit that
        contains only an observable with a single term"""

        tape = qml.tape.QuantumScript([], [qml.expval(qml.X(0))])
        tapes, fn = split_to_single_terms(tape)

        assert len(tapes) == 1
        assert tapes[0] == tape
        assert fn is null_postprocessing

    def test_no_measurements(self):
        """Test that if the tape contains no measurements, the transform doesn't
        modify it"""

        tape = qml.tape.QuantumScript([qml.X(0)])
        tapes, fn = split_to_single_terms(tape)
        assert len(tapes) == 1
        assert tapes[0] == tape
        assert fn(tapes) == tape

    @pytest.mark.parametrize("measure_fn", [qml.probs, qml.counts, qml.sample])
    def test_all_wire_measurements(self, measure_fn):
        """Tests that measurements based on wires don't need to be split, so the
        transform does nothing"""

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
        tapes, fn = split_to_single_terms(tape)

        assert len(tapes) == 1
        assert tapes[0] == tape
        assert fn == null_postprocessing

    def test_single_sum(self):
        """Test that the transform works as expected for a circuit that
        returns a single sum"""
        tape = qml.tape.QuantumScript([], [qml.expval(qml.X(0) + qml.Y(1))])
        tapes, fn = split_to_single_terms(tape)
        assert len(tapes) == 1
        assert tapes[0].measurements == [qml.expval(qml.X(0)), qml.expval(qml.Y(1))]
        assert np.allclose(fn(([0.1, 0.2],)), 0.3)

    def test_multiple_sums(self):
        """Test that the transform works as expected for a circuit that
        returns multiple sums"""
        tape = qml.tape.QuantumScript(
            [], [qml.expval(qml.X(0) + qml.Y(1)), qml.expval(qml.X(2) + qml.Z(3))]
        )
        tapes, fn = split_to_single_terms(tape)
        assert len(tapes) == 1
        assert tapes[0].measurements == [
            qml.expval(qml.X(0)),
            qml.expval(qml.Y(1)),
            qml.expval(qml.X(2)),
            qml.expval(qml.Z(3)),
        ]
        assert fn(([0.1, 0.2, 0.3, 0.4],)) == (0.1 + 0.2, 0.3 + 0.4)

    def test_multiple_sums_overlapping(self):
        """Test that the transform works as expected for a circuit that
        returns multiple sums, where some terms are included in multiple sums"""
        tape = qml.tape.QuantumScript(
            [], [qml.expval(qml.X(0) + qml.Y(1)), qml.expval(qml.X(2) + qml.Y(1))]
        )
        tapes, fn = split_to_single_terms(tape)
        assert len(tapes) == 1
        assert tapes[0].measurements == [
            qml.expval(qml.X(0)),
            qml.expval(qml.Y(1)),
            qml.expval(qml.X(2)),
        ]
        assert fn(([0.1, 0.2, 0.3],)) == (0.1 + 0.2, 0.3 + 0.2)

    def test_multiple_sums_duplicated(self):
        """Test that the transform works as expected for a circuit that returns multiple sums, where each
        sum includes the same term more than once"""
        tape = qml.tape.QuantumScript(
            [], [qml.expval(qml.X(0) + qml.X(0)), qml.expval(qml.X(1) + qml.Y(1) + qml.Y(1))]
        )
        tapes, fn = split_to_single_terms(tape)
        assert len(tapes) == 1
        assert tapes[0].measurements == [
            qml.expval(qml.X(0)),
            qml.expval(qml.X(1)),
            qml.expval(qml.Y(1)),
        ]
        assert fn(([0.1, 0.2, 0.3],)) == (0.1 + 0.1, 0.2 + 0.3 + 0.3)

    @pytest.mark.parametrize("batch_type", (tuple, list))
    def test_batch_of_tapes(self, batch_type):
        """Test that `split_to_single_terms` can transform a batch of tapes with multi-term observables"""

        tape_batch = [
            qml.tape.QuantumScript([qml.RX(1.2, 0)], [qml.expval(qml.X(0) + qml.Y(0) + qml.X(1))]),
            qml.tape.QuantumScript([qml.RY(0.5, 0)], [qml.expval(qml.Z(0) + qml.Y(0))]),
        ]
        tape_batch = batch_type(tape_batch)

        tapes, fn = split_to_single_terms(tape_batch)

        expected_tapes = [
            qml.tape.QuantumScript(
                [qml.RX(1.2, 0)],
                [qml.expval(qml.X(0)), qml.expval(qml.Y(0)), qml.expval(qml.X(1))],
            ),
            qml.tape.QuantumScript([qml.RY(0.5, 0)], [qml.expval(qml.Z(0)), qml.expval(qml.Y(0))]),
        ]
        for actual_tape, expected_tape in zip(tapes, expected_tapes):
            qml.assert_equal(actual_tape, expected_tape)

        result = ([0.1, 0.2, 0.3], [0.4, 0.2])
        assert fn(result) == ((0.1 + 0.2 + 0.3), (0.4 + 0.2))

    @pytest.mark.parametrize(
        "non_pauli_obs", [qml.Projector([0], wires=[1]), qml.Hadamard(wires=[1])]
    )
    def test_tape_with_non_pauli_obs(self, non_pauli_obs):
        """Tests that the tape is split correctly when containing non-Pauli observables"""

        measurements = [
            qml.expval(qml.X(0) + qml.Y(0) + qml.X(1)),
            qml.expval(non_pauli_obs + qml.Z(3)),
        ]
        tape = qml.tape.QuantumScript([qml.RX(1.2, 0)], measurements=measurements)

        tapes, fn = split_to_single_terms(tape)

        expected_tape = qml.tape.QuantumScript(
            [qml.RX(1.2, 0)],
            [
                qml.expval(qml.X(0)),
                qml.expval(qml.Y(0)),
                qml.expval(qml.X(1)),
                qml.expval(non_pauli_obs),
                qml.expval(qml.Z(3)),
            ],
        )

        qml.assert_equal(tapes[0], expected_tape)

        result = [[0.1, 0.2, 0.3, 0.4, 0.5]]
        assert fn(result) == ((0.1 + 0.2 + 0.3), (0.4 + 0.5))

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
            _, _ = split_to_single_terms(tape)


class TestIntegration:
    """Tests the ``split_to_single_terms`` transform performed on a QNode. In these tests,
    the supported observables of ``default_qubit`` are mocked to make the device reject Sum
    and LinearCombination, to ensure the transform works as intended."""

    def test_splitting_sums(self):
        """Test that the transform takes a tape that is not executable on a device that
        doesn't support Sum, and turns it into one that is"""

        coeffs, obs = [0.1, 0.2, 0.3, 0.4, 0.5], single_term_obs_list

        dev = NoTermsDevice(wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.RX(1.2, wires=0)
            return qml.expval(qml.Hamiltonian(coeffs, obs))

        @split_to_single_terms
        @qml.qnode(dev)
        def circuit_split():
            qml.RX(1.2, wires=0)
            return qml.expval(qml.Hamiltonian(coeffs, obs))

        with pytest.raises(ValueError, match="does not accept observables with multiple terms"):
            circuit()

        with dev.tracker:
            circuit_split()
        assert dev.tracker.totals["simulations"] == 1

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
    def test_single_expval(self, shots, params, expected_results, seed):
        """Tests that a QNode with a single expval measurement is executed correctly"""

        coeffs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        obs = single_term_obs_list + [qml.I()]  # test constant offset

        dev = NoTermsDevice(wires=2, seed=seed)

        @qml.qnode(dev, shots=shots)
        def circuit(angles):
            qml.RX(angles[0], wires=0)
            qml.RY(angles[1], wires=0)
            qml.RX(angles[0], wires=1)
            qml.RY(angles[1], wires=1)
            return qml.expval(qml.Hamiltonian(coeffs, obs))

        circuit = split_to_single_terms(circuit)
        res = circuit(params)

        identity_results = [1] if len(np.shape(params)) == 1 else [[1, 1]]
        expected_results = expected_results + identity_results

        expected = np.dot(coeffs, expected_results)

        if isinstance(shots, list):
            assert qml.math.shape(res) == (3,) if len(np.shape(res)) == 1 else (3, 2)
            for i in range(3):
                assert qml.math.allclose(res[i], expected, atol=0.05)
        else:
            assert qml.math.allclose(res, expected, atol=0.05)

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
    def test_multiple_expval(self, shots, params, expected_results, seed):
        """Tests that a QNode with multiple expval measurements is executed correctly"""

        dev = NoTermsDevice(wires=2, seed=seed)

        obs_list = complex_obs_list

        @qml.qnode(dev, shots=shots)
        def circuit(angles):
            qml.RX(angles[0], wires=0)
            qml.RY(angles[1], wires=0)
            qml.RX(angles[0], wires=1)
            qml.RY(angles[1], wires=1)
            return [qml.expval(obs) for obs in obs_list]

        circuit = split_to_single_terms(circuit)
        res = circuit(params)

        if isinstance(shots, list):
            assert qml.math.shape(res) == (3, *np.shape(expected_results))
            for i in range(3):
                assert qml.math.allclose(res[i], expected_results, atol=0.05)
        else:
            assert qml.math.allclose(res, expected_results, atol=0.05)

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
    def test_mixed_measurement_types(self, shots, params, expected_results, seed):
        """Tests that a QNode with mixed measurement types is executed correctly"""

        dev = NoTermsDevice(wires=2, seed=seed)

        obs_list = complex_obs_list

        @qml.qnode(dev, shots=shots)
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

        circuit = split_to_single_terms(circuit)
        res = circuit(params)

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
                    assert qml.math.shape(sample_res) == (shots[i], 1)
                else:
                    assert qml.math.shape(prob_res_0) == (2, 2)
                    assert qml.math.shape(prob_res_1) == (2, 4)
                    assert all(isinstance(_res, dict) for _res in counts_res)
                    assert qml.math.shape(sample_res) == (2, shots[i], 1)

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
                assert qml.math.shape(sample_res) == (shots, 1)
            else:
                assert qml.math.shape(prob_res_0) == (2, 2)
                assert qml.math.shape(prob_res_1) == (2, 4)
                assert all(isinstance(_res, dict) for _res in counts_res)
                assert qml.math.shape(sample_res) == (2, shots, 1)

            expval_res = res[4:]
            assert qml.math.allclose(expval_res, expected_results, atol=0.05)

    @pytest.mark.parametrize("shots", [None, 20000, [20000, 30000, 40000]])
    def test_sum_with_only_identity(self, shots, seed):
        """Tests that split_to_single_terms can handle Identity observables (these
        are treated separately as offsets in the transform)"""

        dev = NoTermsDevice(wires=2, seed=seed)
        H = qml.Hamiltonian([1.5, 2.5], [qml.I(), qml.I()])

        @split_to_single_terms
        @qml.qnode(dev, shots=shots)
        def circuit():
            return qml.expval(H)

        res = circuit()
        assert qml.math.allclose(res, 1.5 + 2.5)

    @pytest.mark.parametrize("shots", [None, 20000, [20000, 30000, 40000]])
    def test_sum_with_identity_and_observable(self, shots, seed):
        """Tests that split_to_single_terms can handle a combination Identity observables (these
        are treated separately as offsets in the transform) and other observables"""

        dev = NoTermsDevice(wires=2, seed=seed)
        H = qml.Hamiltonian([1.5, 2.5], [qml.I(0), qml.Y(0)])

        @split_to_single_terms
        @qml.qnode(dev, shots=shots)
        def circuit():
            qml.RX(-np.pi / 2, 0)
            return qml.expval(H)

        res = circuit()
        assert qml.math.allclose(res, 4.0)

    def test_non_pauli_obs_in_circuit(self):
        """Tests that the tape is executed correctly with non-pauli observables"""

        dev = NoTermsDevice(wires=1)

        @split_to_single_terms
        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(0)
            return qml.expval(qml.Projector([0], wires=[0]) + qml.Projector([1], wires=[0]))

        with dev.tracker:
            res = circuit()
        assert dev.tracker.totals["simulations"] == 1
        assert qml.math.allclose(res, 1, atol=0.01)


class TestDifferentiability:
    """Tests the differentiability of the ``split_to_single_terms`` transform"""

    @pytest.mark.autograd
    def test_trainable_hamiltonian_autograd(self, seed):
        """Tests that measurements of trainable Hamiltonians are differentiable"""

        import pennylane.numpy as pnp

        dev = NoTermsDevice(wires=2, seed=seed)

        @split_to_single_terms
        @qml.qnode(dev, shots=50000)
        def circuit(coeff1, coeff2):
            qml.RX(np.pi / 4, wires=0)
            qml.RY(np.pi / 4, wires=1)
            return qml.expval(qml.Hamiltonian([coeff1, coeff2], [qml.Y(0) @ qml.Z(1), qml.X(1)]))

        params = pnp.array(pnp.pi / 4), pnp.array(3 * pnp.pi / 4)
        actual = qml.jacobian(circuit)(*params)

        assert qml.math.allclose(actual, [-0.5, np.cos(np.pi / 4)], rtol=0.05)

    @pytest.mark.jax
    @pytest.mark.parametrize("use_jit", [False, True])
    def test_trainable_hamiltonian_jax(self, use_jit, seed):
        """Tests that measurements of trainable Hamiltonians are differentiable with jax"""

        import jax
        import jax.numpy as jnp

        dev = NoTermsDevice(wires=2, seed=seed)

        @partial(split_to_single_terms)
        @qml.qnode(dev, shots=50000)
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
    def test_trainable_hamiltonian_torch(self, seed):
        """Tests that measurements of trainable Hamiltonians are differentiable with torch"""

        import torch
        from torch.autograd.functional import jacobian

        dev = NoTermsDevice(wires=2, seed=seed)

        @split_to_single_terms
        @qml.qnode(dev, shots=50000)
        def circuit(coeff1, coeff2):
            qml.RX(np.pi / 4, wires=0)
            qml.RY(np.pi / 4, wires=1)
            return qml.expval(qml.Hamiltonian([coeff1, coeff2], [qml.Y(0) @ qml.Z(1), qml.X(1)]))

        params = torch.tensor(np.pi / 4), torch.tensor(3 * np.pi / 4)
        actual = jacobian(circuit, params)

        assert qml.math.allclose(actual, [-0.5, np.cos(np.pi / 4)], rtol=0.05)

    @pytest.mark.tf
    def test_trainable_hamiltonian_tensorflow(self, seed):
        """Tests that measurements of trainable Hamiltonians are differentiable with tensorflow"""

        import tensorflow as tf

        dev = NoTermsDevice(wires=2, seed=seed)

        @qml.qnode(dev, shots=50000)
        def circuit(coeff1, coeff2):
            qml.RX(np.pi / 4, wires=0)
            qml.RY(np.pi / 4, wires=1)
            return qml.expval(qml.Hamiltonian([coeff1, coeff2], [qml.Y(0) @ qml.Z(1), qml.X(1)]))

        params = tf.Variable(np.pi / 4), tf.Variable(3 * np.pi / 4)

        with tf.GradientTape() as tape:
            cost = split_to_single_terms(circuit)(*params)

        actual = tape.jacobian(cost, params)

        assert qml.math.allclose(actual, [-0.5, np.cos(np.pi / 4)], rtol=0.05)
