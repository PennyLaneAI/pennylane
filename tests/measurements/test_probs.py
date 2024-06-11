# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the probs module"""
from typing import Sequence

import numpy as np
import pytest

import pennylane as qml
from pennylane import numpy as pnp
from pennylane.measurements import MeasurementProcess, Probability, ProbabilityMP, Shots
from pennylane.queuing import AnnotatedQueue

# make the test deterministic
np.random.seed(42)


@pytest.fixture(name="init_state")
def fixture_init_state():
    """Fixture that creates an initial state"""

    def _init_state(n):
        """An initial state over n wires"""
        state = np.random.random([2**n]) + np.random.random([2**n]) * 1j
        state /= np.linalg.norm(state)
        return state

    return _init_state


class TestProbs:
    """Tests for the probs function"""

    # pylint:disable=too-many-public-methods

    def test_queue(self):
        """Test that the right measurement class is queued."""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            return qml.probs(wires=0)

        circuit()

        assert isinstance(circuit.tape[0], ProbabilityMP)

    def test_numeric_type(self):
        """Test that the numeric type is correct."""
        res = qml.probs(wires=0)
        assert res.numeric_type is float

    @pytest.mark.parametrize("wires", [[0], [2, 1], ["a", "c", 3]])
    @pytest.mark.parametrize("shots", [None, 10])
    def test_shape(self, wires, shots):
        """Test that the shape is correct."""
        dev = qml.device("default.qubit", wires=3, shots=shots)
        res = qml.probs(wires=wires)
        assert res.shape(dev, Shots(shots)) == (2 ** len(wires),)

    def test_shape_empty_wires(self):
        """Test that shape works when probs is broadcasted onto all available wires."""
        dev = qml.device("default.qubit", wires=(1, 2, 3))
        res = qml.probs()
        assert res.shape(dev, Shots(None)) == (8,)

        dev2 = qml.device("default.qubit")
        res = qml.probs()
        assert res.shape(dev2, Shots(None)) == (1,)

    @pytest.mark.parametrize("wires", [[0], [2, 1], ["a", "c", 3]])
    def test_shape_shot_vector(self, wires):
        """Test that the shape is correct with the shot vector too."""
        res = qml.probs(wires=wires)
        shot_vector = (1, 2, 3)
        dev = qml.device("default.qubit", wires=3, shots=shot_vector)
        assert res.shape(dev, Shots(shot_vector)) == (
            (2 ** len(wires),),
            (2 ** len(wires),),
            (2 ** len(wires),),
        )

    @pytest.mark.parametrize("wires", [[0], [0, 1], [1, 0, 2]])
    def test_annotating_probs(self, wires):
        """Test annotating probs"""
        with AnnotatedQueue() as q:
            qml.probs(wires)

        assert len(q.queue) == 1

        meas_proc = q.queue[0]
        assert isinstance(meas_proc, MeasurementProcess)
        assert meas_proc.return_type == Probability

    def test_probs_empty_wires(self):
        """Test that using ``qml.probs`` with an empty wire list raises an error."""
        with pytest.raises(ValueError, match="Cannot set an empty list of wires."):
            qml.probs(wires=[])

    @pytest.mark.parametrize("shots", [None, 100])
    def test_probs_no_arguments(self, shots):
        """Test that using ``qml.probs`` with no arguments returns the probabilities of all wires."""
        dev = qml.device("default.qubit", wires=3, shots=shots)

        @qml.qnode(dev)
        def circuit():
            return qml.probs()

        res = circuit()

        assert qml.math.allequal(res, [1, 0, 0, 0, 0, 0, 0, 0])

    def test_full_prob(self, init_state, tol):
        """Test that the correct probability is returned."""
        dev = qml.device("default.qubit", wires=4)

        state = init_state(4)

        @qml.qnode(dev)
        def circuit():
            qml.StatePrep(state, wires=list(range(4)))
            return qml.probs(wires=range(4))

        res = circuit()
        expected = np.abs(state) ** 2
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_marginal_prob(self, init_state, tol):
        """Test that the correct marginal probability is returned."""
        dev = qml.device("default.qubit", wires=4)

        state = init_state(4)

        @qml.qnode(dev)
        def circuit():
            qml.StatePrep(state, wires=list(range(4)))
            return qml.probs(wires=[1, 3])

        res = circuit()
        expected = np.reshape(np.abs(state) ** 2, [2] * 4)
        expected = np.einsum("ijkl->jl", expected).flatten()
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_marginal_prob_more_wires(self, init_state, tol):
        """Test that the correct marginal probability is returned, when the
        states_to_binary method is used for probability computations."""
        dev = qml.device("default.qubit", wires=4)
        state = init_state(4)

        @qml.qnode(dev)
        def circuit():
            qml.StatePrep(state, wires=list(range(4)))
            return qml.probs(wires=[1, 0, 3])

        res = circuit()

        expected = np.reshape(np.abs(state) ** 2, [2] * 4)
        expected = np.einsum("ijkl->jil", expected).flatten()
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("interface", ["numpy", "jax", "torch", "tensorflow"])
    @pytest.mark.parametrize(
        "subset_wires,expected",
        [
            ([0, 1], [0.25, 0.25, 0.25, 0.25]),
            ([1, 2], [0.5, 0, 0.5, 0]),
            ([0, 2], [0.5, 0, 0.5, 0]),
            ([2, 0], [0.5, 0.5, 0, 0]),
            ([2, 1], [0.5, 0.5, 0, 0]),
            ([1, 2, 0], [0.25, 0.25, 0, 0, 0.25, 0.25, 0, 0]),
        ],
    )
    def test_process_state(self, interface, subset_wires, expected):
        """Tests that process_state functions as expected with all interfaces."""
        state = qml.math.array([1 / 2, 0] * 4, like=interface)
        wires = qml.wires.Wires(range(3))
        subset_probs = qml.probs(wires=subset_wires).process_state(state, wires)
        assert subset_probs.shape == (len(expected),)
        assert qml.math.allclose(subset_probs, expected)

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("interface", ["numpy", "jax", "torch", "tensorflow"])
    @pytest.mark.parametrize(
        "subset_wires,expected",
        [
            ([1, 2], [[0.5, 0, 0.5, 0], [0, 0.5, 0, 0.5]]),
            ([2, 0], [[0.5, 0.5, 0, 0], [0, 0, 0.5, 0.5]]),
            (
                [1, 2, 0],
                [[0.25, 0.25, 0, 0, 0.25, 0.25, 0, 0], [0, 0, 0.25, 0.25, 0, 0, 0.25, 0.25]],
            ),
        ],
    )
    def test_process_state_batched(self, interface, subset_wires, expected):
        """Tests that process_state functions as expected with all interfaces with batching."""
        states = qml.math.array([[1 / 2, 0] * 4, [0, 1 / 2] * 4], like=interface)
        wires = qml.wires.Wires(range(3))
        subset_probs = qml.probs(wires=subset_wires).process_state(states, wires)
        assert subset_probs.shape == qml.math.shape(expected)
        assert qml.math.allclose(subset_probs, expected)

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("interface", ["numpy", "jax", "torch", "tensorflow"])
    def test_process_density_matrix_basic(self, interface):
        """Test that process_density_matrix returns correct probabilities from a density matrix."""
        dm = qml.math.array([[0.5, 0], [0, 0.5]], like=interface)
        wires = qml.wires.Wires(range(1))
        expected = qml.math.array([0.5, 0.5], like=interface)
        calculated_probs = qml.probs().process_density_matrix(dm, wires)
        assert qml.math.allclose(calculated_probs, expected)

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("interface", ["numpy", "jax", "torch", "tensorflow"])
    @pytest.mark.parametrize(
        "subset_wires, expected",
        [
            ([0], [0.5, 0.5]),
            ([1], [0.25, 0.75]),
            ([1, 0], [0.15, 0.1, 0.35, 0.4]),
            ([0, 1], [0.15, 0.35, 0.1, 0.4]),
        ],
    )
    def test_process_density_matrix_subsets(self, interface, subset_wires, expected):
        """Test processing of density matrix with subsets of wires."""
        dm = qml.math.array(
            [[0.15, 0, 0.1, 0], [0, 0.35, 0, 0.4], [0.1, 0, 0.1, 0], [0, 0.4, 0, 0.4]],
            like=interface,
        )
        wires = qml.wires.Wires(range(2))
        subset_probs = qml.probs(wires=subset_wires).process_density_matrix(dm, wires)
        assert subset_probs.shape == qml.math.shape(expected)
        assert qml.math.allclose(subset_probs, expected)

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("interface", ["numpy", "jax", "torch", "tensorflow"])
    @pytest.mark.parametrize(
        "subset_wires, expected",
        [
            ([0], [[1, 0], [0.5, 0.5], [0.5, 0.5]]),
            ([1], [[1, 0], [0.5, 0.5], [0.5, 0.5]]),
            ([1, 0], [[1, 0, 0, 0], [0.25, 0.25, 0.25, 0.25], [0.5, 0, 0, 0.5]]),
            ([0, 1], [[1, 0, 0, 0], [0.25, 0.25, 0.25, 0.25], [0.5, 0, 0, 0.5]]),
        ],
    )
    def test_process_density_matrix_batched(self, interface, subset_wires, expected):
        """Test processing of a batch of density matrices."""
        # Define a batch of density matrices
        dm_batch = qml.math.array(
            [
                # Pure state |00⟩
                np.outer([1, 0, 0, 0], [1, 0, 0, 0]),
                # Maximally mixed state
                np.identity(4) / 4,
                # Bell state |Φ+⟩ = 1/√2(|00⟩ + |11⟩)
                0.5 * np.outer([1, 0, 0, 1], [1, 0, 0, 1]),
            ],
            like=interface,
        )

        wires = qml.wires.Wires(range(2))
        # Process the entire batch of density matrices
        subset_probs = qml.probs(wires=subset_wires).process_density_matrix(dm_batch, wires)

        expected = qml.math.array(expected, like=interface)
        # Check if the calculated probabilities match the expected values
        assert (
            subset_probs.shape == expected.shape
        ), f"Shape mismatch: expected {expected.shape}, got {subset_probs.shape}"
        assert qml.math.allclose(
            subset_probs, expected
        ), f"Value mismatch: expected {expected.tolist()}, got {subset_probs.tolist()}"

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("interface", ["numpy", "jax", "torch", "tensorflow"])
    @pytest.mark.parametrize(
        "subset_wires, expected",
        [
            (
                [3, 1, 0],
                [
                    0.16527324,
                    0.13982314,
                    0.13847902,
                    0.04774645,
                    0.08946633,
                    0.16450224,
                    0.17995419,
                    0.07475538,
                ],
            )
        ],
    )
    def test_process_density_matrix_medium(self, interface, subset_wires, expected):
        """Test processing of a batch of density matrices."""
        # Define a batch of density matrices
        dm_batch = qml.math.array(
            [
                [
                    0.080989,
                    0.05054025,
                    0.06706085,
                    0.05616603,
                    0.07001544,
                    0.05997686,
                    0.0726786,
                    0.03272224,
                    0.06646674,
                    0.06662265,
                    0.04900118,
                    0.06608795,
                    0.06139745,
                    0.08252963,
                    0.0693334,
                    0.05244662,
                ],
                [
                    0.05054025,
                    0.0039525,
                    0.03570121,
                    0.08530672,
                    0.0778656,
                    0.05587372,
                    0.02017853,
                    0.01982865,
                    0.05265699,
                    0.05702363,
                    0.02701604,
                    0.05136687,
                    0.01316993,
                    0.06475533,
                    0.05319626,
                    0.0704644,
                ],
                [
                    0.06706085,
                    0.03570121,
                    0.08428424,
                    0.05353046,
                    0.05727886,
                    0.04475616,
                    0.05065271,
                    0.06764512,
                    0.06899573,
                    0.07183764,
                    0.06956846,
                    0.05121532,
                    0.04466637,
                    0.05048799,
                    0.07175026,
                    0.05161699,
                ],
                [
                    0.05616603,
                    0.08530672,
                    0.05353046,
                    0.08551383,
                    0.08470469,
                    0.06411541,
                    0.04257544,
                    0.04969202,
                    0.03552623,
                    0.04336215,
                    0.02066428,
                    0.04676245,
                    0.03336671,
                    0.07039378,
                    0.04521083,
                    0.06038351,
                ],
                [
                    0.07001544,
                    0.0778656,
                    0.05727886,
                    0.08470469,
                    0.03851449,
                    0.05794007,
                    0.07131056,
                    0.05296177,
                    0.09827191,
                    0.03160001,
                    0.05309798,
                    0.07633288,
                    0.03661442,
                    0.08934421,
                    0.06731681,
                    0.00329895,
                ],
                [
                    0.05997686,
                    0.05587372,
                    0.04475616,
                    0.06411541,
                    0.05794007,
                    0.09090044,
                    0.08186336,
                    0.05834778,
                    0.09545584,
                    0.04661815,
                    0.06149736,
                    0.04129336,
                    0.05284938,
                    0.04272225,
                    0.04229479,
                    0.04766869,
                ],
                [
                    0.0726786,
                    0.02017853,
                    0.05065271,
                    0.04257544,
                    0.07131056,
                    0.08186336,
                    0.09996453,
                    0.02002902,
                    0.06213783,
                    0.03136972,
                    0.02329919,
                    0.07221994,
                    0.06348929,
                    0.04521518,
                    0.06373298,
                    0.07950719,
                ],
                [
                    0.03272224,
                    0.01982865,
                    0.06764512,
                    0.04969202,
                    0.05296177,
                    0.05834778,
                    0.02002902,
                    0.08905375,
                    0.09785625,
                    0.0525481,
                    0.02573775,
                    0.07713713,
                    0.06029518,
                    0.03442947,
                    0.06029504,
                    0.02708854,
                ],
                [
                    0.06646674,
                    0.05265699,
                    0.06899573,
                    0.03552623,
                    0.09827191,
                    0.09545584,
                    0.06213783,
                    0.09785625,
                    0.04838075,
                    0.02735249,
                    0.03710709,
                    0.02440981,
                    0.05473816,
                    0.0381057,
                    0.02480048,
                    0.06388189,
                ],
                [
                    0.06662265,
                    0.05702363,
                    0.07183764,
                    0.04336215,
                    0.03160001,
                    0.04661815,
                    0.03136972,
                    0.0525481,
                    0.02735249,
                    0.09374992,
                    0.01808993,
                    0.09396117,
                    0.06277499,
                    0.039946,
                    0.05005397,
                    0.04559161,
                ],
                [
                    0.04900118,
                    0.02701604,
                    0.06956846,
                    0.02066428,
                    0.05309798,
                    0.06149736,
                    0.02329919,
                    0.02573775,
                    0.03710709,
                    0.01808993,
                    0.0914424,
                    0.05776955,
                    0.03587103,
                    0.06108309,
                    0.06652214,
                    0.02442639,
                ],
                [
                    0.06608795,
                    0.05136687,
                    0.05121532,
                    0.04676245,
                    0.07633288,
                    0.04129336,
                    0.07221994,
                    0.07713713,
                    0.02440981,
                    0.09396117,
                    0.05776955,
                    0.07075232,
                    0.06369993,
                    0.06199866,
                    0.04284378,
                    0.06532901,
                ],
                [
                    0.06139745,
                    0.01316993,
                    0.04466637,
                    0.03336671,
                    0.03661442,
                    0.05284938,
                    0.06348929,
                    0.06029518,
                    0.05473816,
                    0.06277499,
                    0.03587103,
                    0.06369993,
                    0.03473495,
                    0.06439239,
                    0.06228124,
                    0.04379324,
                ],
                [
                    0.08252963,
                    0.06475533,
                    0.05048799,
                    0.07039378,
                    0.08934421,
                    0.04272225,
                    0.04521518,
                    0.03442947,
                    0.0381057,
                    0.039946,
                    0.06108309,
                    0.06199866,
                    0.06439239,
                    0.00036181,
                    0.08717339,
                    0.06031977,
                ],
                [
                    0.0693334,
                    0.05319626,
                    0.07175026,
                    0.04521083,
                    0.06731681,
                    0.04229479,
                    0.06373298,
                    0.06029504,
                    0.02480048,
                    0.05005397,
                    0.06652214,
                    0.04284378,
                    0.06228124,
                    0.08717339,
                    0.0130115,
                    0.06950743,
                ],
                [
                    0.05244662,
                    0.0704644,
                    0.05161699,
                    0.06038351,
                    0.00329895,
                    0.04766869,
                    0.07950719,
                    0.02708854,
                    0.06388189,
                    0.04559161,
                    0.02442639,
                    0.06532901,
                    0.04379324,
                    0.06031977,
                    0.06950743,
                    0.07439358,
                ],
            ],
            like=interface,
        )

        wires = qml.wires.Wires(range(4))
        # Process the entire batch of density matrices
        subset_probs = qml.probs(wires=subset_wires).process_density_matrix(dm_batch, wires)

        expected = qml.math.array(expected, like=interface)
        # Check if the calculated probabilities match the expected values
        assert (
            subset_probs.shape == expected.shape
        ), f"Shape mismatch: expected {expected.shape}, got {subset_probs.shape}"
        assert qml.math.allclose(
            subset_probs, expected
        ), f"Value mismatch: expected {expected.tolist()}, got {subset_probs.tolist()}"

    def test_integration(self, tol):
        """Test the probability is correct for a known state preparation."""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[0, 1])

        # expected probability, using [00, 01, 10, 11]
        # ordering, is [0.5, 0.5, 0, 0]

        res = circuit()
        expected = np.array([0.5, 0.5, 0, 0])
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.jax
    @pytest.mark.parametrize("shots", (None, 500))
    @pytest.mark.parametrize("obs", ([0, 1], qml.PauliZ(0) @ qml.PauliZ(1)))
    @pytest.mark.parametrize("params", ([np.pi / 2], [np.pi / 2, np.pi / 2, np.pi / 2]))
    def test_integration_jax(self, tol_stochastic, shots, obs, params):
        """Test the probability is correct for a known state preparation when jitted with JAX."""
        jax = pytest.importorskip("jax")

        dev = qml.device("default.qubit", wires=2, shots=shots, seed=jax.random.PRNGKey(0))
        params = jax.numpy.array(params)

        @qml.qnode(dev, diff_method=None)
        def circuit(x):
            qml.PhaseShift(x, wires=1)
            qml.RX(x, wires=1)
            qml.PhaseShift(x, wires=1)
            qml.CNOT(wires=[0, 1])
            if isinstance(obs, Sequence):
                return qml.probs(wires=obs)
            return qml.probs(op=obs)

        # expected probability, using [00, 01, 10, 11]
        # ordering, is [0.5, 0.5, 0, 0]

        assert "pure_callback" not in str(jax.make_jaxpr(circuit)(params))

        res = jax.jit(circuit)(params)
        expected = np.array([0.5, 0.5, 0, 0])
        assert np.allclose(res, expected, atol=tol_stochastic, rtol=0)

    @pytest.mark.parametrize("shots", [100, [1, 10, 100]])
    def test_integration_analytic_false(self, tol, shots):
        """Test the probability is correct for a known state preparation when the
        analytic attribute is set to False."""
        dev = qml.device("default.qubit", wires=3, shots=shots)

        @qml.qnode(dev)
        def circuit():
            qml.PauliX(0)
            return qml.probs(wires=dev.wires)

        res = circuit()
        expected = np.array([0, 0, 0, 0, 1, 0, 0, 0])
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("shots", [None, 1111, [1111, 1111]])
    @pytest.mark.parametrize("phi", np.arange(0, 2 * np.pi, np.pi / 3))
    def test_observable_is_measurement_value(
        self, shots, phi, tol, tol_stochastic
    ):  # pylint: disable=too-many-arguments
        """Test that probs for mid-circuit measurement values
        are correct for a single measurement value."""
        dev = qml.device("default.qubit", wires=2, shots=shots)

        @qml.qnode(dev)
        def circuit(phi):
            qml.RX(phi, 0)
            m0 = qml.measure(0)
            return qml.probs(op=m0)

        atol = tol if shots is None else tol_stochastic
        expected = np.array([np.cos(phi / 2) ** 2, np.sin(phi / 2) ** 2])

        for func in [circuit, qml.defer_measurements(circuit)]:
            res = func(phi)
            if not isinstance(shots, list):
                assert np.allclose(np.array(res), expected, atol=atol, rtol=0)
            else:
                for r in res:  # pylint: disable=not-an-iterable
                    assert np.allclose(r, expected, atol=atol, rtol=0)

    @pytest.mark.parametrize("shots", [None, 1111, [1111, 1111]])
    @pytest.mark.parametrize("phi", [0.0, np.pi / 3, np.pi])
    def test_observable_is_measurement_value_list(
        self, shots, phi, tol, tol_stochastic
    ):  # pylint: disable=too-many-arguments
        """Test that probs for mid-circuit measurement values
        are correct for a measurement value list."""
        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit(phi):
            qml.RX(phi, 0)
            m0 = qml.measure(0)
            qml.RX(0.5 * phi, 1)
            m1 = qml.measure(1)
            qml.RX(2.0 * phi, 2)
            m2 = qml.measure(2)
            return qml.probs(op=[m0, m1, m2])

        res = circuit(phi, shots=shots)

        @qml.qnode(dev)
        def expected_circuit(phi):
            qml.RX(phi, 0)
            qml.RX(0.5 * phi, 1)
            qml.RX(2.0 * phi, 2)
            return qml.probs(wires=[0, 1, 2])

        expected = expected_circuit(phi)

        atol = tol if shots is None else tol_stochastic

        if not isinstance(shots, list):
            assert np.allclose(np.array(res), expected, atol=atol, rtol=0)
        else:
            for r in res:  # pylint: disable=not-an-iterable
                assert np.allclose(r, expected, atol=atol, rtol=0)

    def test_composite_measurement_value_not_allowed(self):
        """Test that measuring composite mid-circuit measurement values raises
        an error."""
        m0 = qml.measure(0)
        m1 = qml.measure(1)

        with pytest.raises(ValueError, match=r"Cannot use qml.probs\(\) when measuring multiple"):
            _ = qml.probs(op=m0 + m1)

    def test_mixed_lists_as_op_not_allowed(self):
        """Test that passing a list not containing only measurement values raises an error."""
        m0 = qml.measure(0)

        with pytest.raises(
            qml.QuantumFunctionError,
            match="Only sequences of single MeasurementValues can be passed with the op argument",
        ):
            _ = qml.probs(op=[m0, qml.PauliZ(0)])

    def test_composed_measurement_value_lists_not_allowed(self):
        """Test that passing a list containing measurement values composed with arithmetic
        raises an error."""
        m0 = qml.measure(0)
        m1 = qml.measure(1)
        m2 = qml.measure(2)

        with pytest.raises(
            qml.QuantumFunctionError,
            match="Only sequences of single MeasurementValues can be passed with the op argument",
        ):
            _ = qml.probs(op=[m0 + m1, m2])

    @pytest.mark.parametrize("shots", [None, 100])
    def test_batch_size(self, shots):
        """Test the probability is correct for a batched input."""
        dev = qml.device("default.qubit", wires=1, shots=shots)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, 0)
            return qml.probs(wires=dev.wires)  # TODO: Use ``qml.probs()`` when supported

        x = np.array([0, np.pi / 2])
        res = circuit(x)
        expected = [[1.0, 0.0], [0.5, 0.5]]
        assert np.allclose(res, expected, atol=0.1, rtol=0.1)

    @pytest.mark.tf
    @pytest.mark.parametrize(
        "prep_op, expected",
        [
            (None, [1, 0]),
            (qml.StatePrep([[0, 1], [1, 0]], 0), [[0, 1], [1, 0]]),
        ],
    )
    def test_process_state_tf_autograph(self, prep_op, expected):
        """Test that process_state passes when decorated with tf.function."""
        import tensorflow as tf

        wires = qml.wires.Wires([0])

        @tf.function
        def probs_from_state(state):
            return qml.probs(wires=wires).process_state(state, wires)

        state = qml.devices.qubit.create_initial_state(wires, prep_operation=prep_op)
        assert np.allclose(probs_from_state(state), expected)

    @pytest.mark.autograd
    def test_numerical_analytic_diff_agree(self, tol):
        """Test that the finite difference and parameter shift rule
        provide the same Jacobian."""
        w = 4
        dev = qml.device("default.qubit", wires=w)

        def circuit(x, y, z):
            for i in range(w):
                qml.RX(x, wires=i)
                qml.PhaseShift(z, wires=i)
                qml.RY(y, wires=i)

            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[2, 3])

            return qml.probs(wires=[1, 3])

        params = pnp.array([0.543, -0.765, -0.3], requires_grad=True)

        circuit_F = qml.QNode(circuit, dev, diff_method="finite-diff")
        circuit_A = qml.QNode(circuit, dev, diff_method="parameter-shift")
        res_F = qml.jacobian(circuit_F)(*params)
        res_A = qml.jacobian(circuit_A)(*params)

        # Both jacobians should be of shape (2**prob.wires, num_params)
        assert isinstance(res_F, tuple) and len(res_F) == 3
        assert all(_r.shape == (2**2,) for _r in res_F)
        assert isinstance(res_A, tuple) and len(res_A) == 3
        assert all(_r.shape == (2**2,) for _r in res_A)

        # Check that they agree up to numeric tolerance
        assert all(np.allclose(_rF, _rA, atol=tol, rtol=0) for _rF, _rA in zip(res_F, res_A))

    @pytest.mark.parametrize("hermitian", [1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])])
    def test_prob_generalize_param_one_qubit(self, hermitian, tol):
        """Test that the correct probability is returned."""
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit(x):
            qml.RZ(x, wires=0)
            return qml.probs(op=qml.Hermitian(hermitian, wires=0))

        res = circuit(0.56)

        def circuit_rotated(x):
            qml.RZ(x, wires=0)
            qml.Hermitian(hermitian, wires=0).diagonalizing_gates()

        state = np.array([1, 0])
        matrix = qml.matrix(circuit_rotated, wire_order=[0])(0.56)
        state = np.dot(matrix, state)
        expected = np.reshape(np.abs(state) ** 2, [2] * 1)
        expected = expected.flatten()

        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("hermitian", [1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])])
    def test_prob_generalize_param(self, hermitian, tol):
        """Test that the correct probability is returned."""
        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def circuit(x, y):
            qml.RZ(x, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RY(y, wires=1)
            qml.CNOT(wires=[0, 2])
            return qml.probs(op=qml.Hermitian(hermitian, wires=0))

        res = circuit(0.56, 0.1)

        def circuit_rotated(x, y):
            qml.RZ(x, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RY(y, wires=1)
            qml.CNOT(wires=[0, 2])
            qml.Hermitian(hermitian, wires=0).diagonalizing_gates()

        state = np.array([1, 0, 0, 0, 0, 0, 0, 0])
        matrix = qml.matrix(circuit_rotated, wire_order=[0, 1, 2])(0.56, 0.1)
        state = np.dot(matrix, state)
        expected = np.reshape(np.abs(state) ** 2, [2] * 3)
        expected = np.einsum("ijk->i", expected).flatten()
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("hermitian", [1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])])
    def test_prob_generalize_param_multiple(self, hermitian, tol):
        """Test that the correct probability is returned."""
        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def circuit(x, y):
            qml.RZ(x, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RY(y, wires=1)
            qml.CNOT(wires=[0, 2])
            return (
                qml.probs(op=qml.Hermitian(hermitian, wires=0)),
                qml.probs(wires=[1]),
                qml.probs(wires=[2]),
            )

        res = circuit(0.56, 0.1)
        res = np.reshape(res, (3, 2))

        def circuit_rotated(x, y):
            qml.RZ(x, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RY(y, wires=1)
            qml.CNOT(wires=[0, 2])
            qml.Hermitian(hermitian, wires=0).diagonalizing_gates()

        state = np.array([1, 0, 0, 0, 0, 0, 0, 0])
        matrix = qml.matrix(circuit_rotated, wire_order=[0, 1, 2])(0.56, 0.1)
        state = np.dot(matrix, state)

        expected = np.reshape(np.abs(state) ** 2, [2] * 3)
        expected_0 = np.einsum("ijk->i", expected).flatten()
        expected_1 = np.einsum("ijk->j", expected).flatten()
        expected_2 = np.einsum("ijk->k", expected).flatten()

        assert np.allclose(res[0], expected_0, atol=tol, rtol=0)
        assert np.allclose(res[1], expected_1, atol=tol, rtol=0)
        assert np.allclose(res[2], expected_2, atol=tol, rtol=0)

    @pytest.mark.parametrize("hermitian", [1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])])
    @pytest.mark.parametrize("wire", [0, 1, 2, 3])
    def test_prob_generalize_initial_state(self, hermitian, wire, init_state, tol):
        """Test that the correct probability is returned."""
        # pylint:disable=too-many-arguments
        dev = qml.device("default.qubit", wires=4)

        state = init_state(4)

        @qml.qnode(dev)
        def circuit():
            qml.StatePrep(state, wires=list(range(4)))
            qml.PauliX(wires=0)
            qml.PauliX(wires=1)
            qml.PauliX(wires=2)
            qml.PauliX(wires=3)
            return qml.probs(op=qml.Hermitian(hermitian, wires=wire))

        res = circuit()

        def circuit_rotated():
            qml.PauliX(wires=0)
            qml.PauliX(wires=1)
            qml.PauliX(wires=2)
            qml.PauliX(wires=3)
            qml.Hermitian(hermitian, wires=wire).diagonalizing_gates()

        matrix = qml.matrix(circuit_rotated, wire_order=[0, 1, 2, 3])()
        state = np.dot(matrix, state)
        expected = np.reshape(np.abs(state) ** 2, [2] * 4)

        if wire == 0:
            expected = np.einsum("ijkl->i", expected).flatten()
        elif wire == 1:
            expected = np.einsum("ijkl->j", expected).flatten()
        elif wire == 2:
            expected = np.einsum("ijkl->k", expected).flatten()
        elif wire == 3:
            expected = np.einsum("ijkl->l", expected).flatten()

        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("operation", [qml.PauliX, qml.PauliY, qml.Hadamard])
    @pytest.mark.parametrize("wire", [0, 1, 2, 3])
    def test_operation_prob(self, operation, wire, init_state, tol):
        "Test the rotated probability with different wires and rotating operations."
        # pylint:disable=too-many-arguments
        dev = qml.device("default.qubit", wires=4)

        state = init_state(4)

        @qml.qnode(dev)
        def circuit():
            qml.StatePrep(state, wires=list(range(4)))
            qml.PauliX(wires=0)
            qml.PauliZ(wires=1)
            qml.PauliY(wires=2)
            qml.PauliZ(wires=3)
            return qml.probs(op=operation(wires=wire))

        res = circuit()

        def circuit_rotated():
            qml.PauliX(wires=0)
            qml.PauliZ(wires=1)
            qml.PauliY(wires=2)
            qml.PauliZ(wires=3)
            operation(wires=wire).diagonalizing_gates()

        matrix = qml.matrix(circuit_rotated, wire_order=[0, 1, 2, 3])()
        state = np.dot(matrix, state)
        expected = np.reshape(np.abs(state) ** 2, [2] * 4)

        if wire == 0:
            expected = np.einsum("ijkl->i", expected).flatten()
        elif wire == 1:
            expected = np.einsum("ijkl->j", expected).flatten()
        elif wire == 2:
            expected = np.einsum("ijkl->k", expected).flatten()
        elif wire == 3:
            expected = np.einsum("ijkl->l", expected).flatten()

        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("observable", [(qml.PauliX, qml.PauliY)])
    def test_observable_tensor_prob(self, observable, init_state, tol):
        "Test the rotated probability with a tensor observable."
        dev = qml.device("default.qubit", wires=4)

        state = init_state(4)

        @qml.qnode(dev)
        def circuit():
            qml.StatePrep(state, wires=list(range(4)))
            qml.PauliX(wires=0)
            qml.PauliZ(wires=1)
            qml.PauliY(wires=2)
            qml.PauliZ(wires=3)
            return qml.probs(op=observable[0](wires=0) @ observable[1](wires=1))

        res = circuit()

        def circuit_rotated():
            qml.PauliX(wires=0)
            qml.PauliZ(wires=1)
            qml.PauliY(wires=2)
            qml.PauliZ(wires=3)
            observable[0](wires=0).diagonalizing_gates()
            observable[1](wires=1).diagonalizing_gates()

        matrix = qml.matrix(circuit_rotated, wire_order=[0, 1, 2, 3])()
        state = np.dot(matrix, state)
        expected = np.reshape(np.abs(state) ** 2, [2] * 4)

        expected = np.einsum("ijkl->ij", expected).flatten()

        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize("coeffs, obs", [([1, 1], [qml.PauliX(wires=0), qml.PauliX(wires=1)])])
    def test_hamiltonian_error(self, coeffs, obs, init_state):
        "Test that an error is returned for hamiltonians."
        H = qml.Hamiltonian(coeffs, obs)

        dev = qml.device("default.qubit", wires=4)
        state = init_state(4)

        @qml.qnode(dev)
        def circuit():
            qml.StatePrep(state, wires=list(range(4)))
            qml.PauliX(wires=0)
            qml.PauliZ(wires=1)
            qml.PauliY(wires=2)
            qml.PauliZ(wires=3)
            return qml.probs(op=H)

        with pytest.raises(
            qml.QuantumFunctionError,
            match="Hamiltonians are not supported for rotating probabilities.",
        ):
            circuit()

    @pytest.mark.parametrize(
        "operation", [qml.SingleExcitation, qml.SingleExcitationPlus, qml.SingleExcitationMinus]
    )
    def test_generalize_prob_not_hermitian(self, operation):
        """Test that Operators that do not have a diagonalizing_gates representation cannot
        be used in probability measurements."""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.PauliX(wires=0)
            qml.PauliZ(wires=1)
            return qml.probs(op=operation(0.56, wires=[0, 1]))

        with pytest.raises(
            qml.QuantumFunctionError,
            match="does not define diagonalizing gates : cannot be used to rotate the probability",
        ):
            circuit()

    @pytest.mark.parametrize("hermitian", [1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])])
    def test_prob_wires_and_hermitian(self, hermitian):
        """Test that we can cannot give simultaneously wires and a hermitian."""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.PauliX(wires=0)
            return qml.probs(op=qml.Hermitian(hermitian, wires=0), wires=1)

        with pytest.raises(
            qml.QuantumFunctionError,
            match="Cannot specify the wires to probs if an observable is "
            "provided. The wires for probs will be determined directly from the observable.",
        ):
            circuit()

    @pytest.mark.parametrize(
        "wires, expected",
        [
            (
                [0],
                [
                    [[0, 0, 0.5], [1, 1, 0.5]],
                    [[0.5, 0.5, 0], [0.5, 0.5, 1]],
                    [[0, 0.5, 1], [1, 0.5, 0]],
                ],
            ),
            (
                [0, 1],
                [
                    [[0, 0, 0], [0, 0, 0.5], [0.5, 0, 0], [0.5, 1, 0.5]],
                    [[0.5, 0.5, 0], [0, 0, 0], [0, 0, 0], [0.5, 0.5, 1]],
                    [[0, 0.5, 0.5], [0, 0, 0.5], [0.5, 0, 0], [0.5, 0.5, 0]],
                ],
            ),
        ],
    )
    def test_estimate_probability_with_binsize_with_broadcasting(self, wires, expected):
        """Tests the estimate_probability method with a bin size and parameter broadcasting"""
        samples = np.array(
            [
                [[1, 0], [1, 1], [1, 1], [1, 1], [1, 1], [0, 1]],
                [[0, 0], [1, 1], [1, 1], [0, 0], [1, 1], [1, 1]],
                [[1, 0], [1, 1], [1, 1], [0, 0], [0, 1], [0, 0]],
            ]
        )

        res = qml.probs(wires=wires).process_samples(
            samples=samples, wire_order=wires, shot_range=None, bin_size=2
        )

        assert np.allclose(res, expected)

    @pytest.mark.parametrize(
        "wires, expected",
        [
            (
                (0, 1, 2),
                [0.1, 0.2, 0.0, 0.1, 0.0, 0.2, 0.1, 0.3],
            ),
            (
                (0, 1),
                [0.3, 0.1, 0.2, 0.4],
            ),
        ],
    )
    def test_estimate_probability_with_counts(self, wires, expected):
        """Tests the estimate_probability method with sampling information in the form of a counts dictionary"""
        counts = {"101": 2, "100": 2, "111": 3, "000": 1, "011": 1, "110": 1}

        wire_order = qml.wires.Wires((2, 1, 0))

        res = qml.probs(wires=wires).process_counts(counts=counts, wire_order=wire_order)

        assert np.allclose(res, expected)

    def test_non_commuting_probs_does_not_raises_error(self):
        """Tests that non-commuting probs with expval does not raise an error."""
        dev = qml.device("default.qubit", wires=5)

        @qml.qnode(dev)
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliX(1)), qml.probs(wires=[0, 1])

        res = circuit(1, 2)
        assert isinstance(res, tuple) and len(res) == 2

    def test_commuting_probs_in_computational_basis(self):
        """Test that `qml.probs` can be used in the computational basis with other commuting observables."""
        dev = qml.device("default.qubit", wires=5)

        @qml.qnode(dev)
        def circuit(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.probs(wires=[0, 1])

        res = circuit(1, 2)

        @qml.qnode(dev)
        def circuit2(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        @qml.qnode(dev)
        def circuit3(x, y):
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[0, 1])

        res2 = circuit2(1, 2)
        res3 = circuit3(1, 2)

        assert res[0] == res2
        assert qml.math.allequal(res[1:], res3)
