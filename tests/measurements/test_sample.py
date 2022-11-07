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
"""Unit tests for the sample module"""

import numpy as np
import pytest

import pennylane as qml
from pennylane.measurements import Sample


class TestSample:
    """Tests for the sample function"""

    def test_sample_dimension(self):
        """Test that the sample function outputs samples of the right size"""
        n_sample = 10

        dev = qml.device("default.qubit", wires=2, shots=n_sample)

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.54, wires=0)
            return qml.sample(qml.PauliZ(0)), qml.sample(qml.PauliX(1))

        sample = circuit()

        assert np.array_equal(sample.shape, (2, n_sample))

    @pytest.mark.filterwarnings("ignore:Creating an ndarray from ragged nested sequences")
    def test_sample_combination(self):
        """Test the output of combining expval, var and sample"""
        n_sample = 10

        dev = qml.device("default.qubit", wires=3, shots=n_sample)

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit():
            qml.RX(0.54, wires=0)

            return qml.sample(qml.PauliZ(0)), qml.expval(qml.PauliX(1)), qml.var(qml.PauliY(2))

        result = circuit()

        assert len(result) == 3
        assert np.array_equal(result[0].shape, (n_sample,))
        assert isinstance(result[1], np.ndarray)
        assert isinstance(result[2], np.ndarray)

    def test_single_wire_sample(self):
        """Test the return type and shape of sampling a single wire"""
        n_sample = 10

        dev = qml.device("default.qubit", wires=1, shots=n_sample)

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.54, wires=0)

            return qml.sample(qml.PauliZ(0))

        result = circuit()

        assert isinstance(result, np.ndarray)
        assert np.array_equal(result.shape, (n_sample,))

    def test_multi_wire_sample_regular_shape(self):
        """Test the return type and shape of sampling multiple wires
        where a rectangular array is expected"""
        n_sample = 10

        dev = qml.device("default.qubit", wires=3, shots=n_sample)

        @qml.qnode(dev)
        def circuit():
            return qml.sample(qml.PauliZ(0)), qml.sample(qml.PauliZ(1)), qml.sample(qml.PauliZ(2))

        result = circuit()

        # If all the dimensions are equal the result will end up to be a proper rectangular array
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result.shape, (3, n_sample))
        assert result.dtype == np.dtype("int")

    @pytest.mark.filterwarnings("ignore:Creating an ndarray from ragged nested sequences")
    def test_sample_output_type_in_combination(self):
        """Test the return type and shape of sampling multiple works
        in combination with expvals and vars"""
        n_sample = 10

        dev = qml.device("default.qubit", wires=3, shots=n_sample)

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit():
            return qml.expval(qml.PauliZ(0)), qml.var(qml.PauliZ(1)), qml.sample(qml.PauliZ(2))

        result = circuit()

        # If all the dimensions are equal the result will end up to be a proper rectangular array
        assert len(result) == 3
        assert isinstance(result[0], np.ndarray)
        assert isinstance(result[1], np.ndarray)
        assert result[2].dtype == np.dtype("int")
        assert np.array_equal(result[2].shape, (n_sample,))

    def test_not_an_observable(self):
        """Test that a UserWarning is raised if the provided
        argument might not be hermitian."""
        dev = qml.device("default.qubit", wires=2, shots=10)

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.52, wires=0)
            return qml.sample(qml.prod(qml.PauliX(0), qml.PauliZ(0)))

        with pytest.warns(UserWarning, match="Prod might not be hermitian."):
            _ = circuit()

    def test_observable_return_type_is_sample(self):
        """Test that the return type of the observable is :attr:`ObservableReturnTypes.Sample`"""
        n_shots = 10
        dev = qml.device("default.qubit", wires=1, shots=n_shots)

        @qml.qnode(dev)
        def circuit():
            res = qml.sample(qml.PauliZ(0))
            assert res.return_type is Sample
            return res

        circuit()

    def test_providing_observable_and_wires(self):
        """Test that a ValueError is raised if both an observable is provided and wires are specified"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            return qml.sample(qml.PauliZ(0), wires=[0, 1])

        with pytest.raises(
            ValueError,
            match="Cannot specify the wires to sample if an observable is provided."
            " The wires to sample will be determined directly from the observable.",
        ):
            _ = circuit()

    def test_providing_no_observable_and_no_wires(self):
        """Test that we can provide no observable and no wires to sample function"""
        dev = qml.device("default.qubit", wires=2, shots=1000)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            res = qml.sample()
            assert res.obs is None
            assert res.wires == qml.wires.Wires([])
            return res

        circuit()

    def test_providing_no_observable_and_no_wires_shot_vector(self):
        """Test that we can provide no observable and no wires to sample
        function when using a shot vector"""
        num_wires = 2

        shots1 = 1
        shots2 = 10
        shots3 = 1000
        dev = qml.device("default.qubit", wires=num_wires, shots=[shots1, shots2, shots3])

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            return qml.sample()

        res = circuit()

        assert isinstance(res, tuple)

        expected_shapes = [(num_wires,), (num_wires, shots2), (num_wires, shots3)]
        assert len(res) == len(expected_shapes)
        assert (r.shape == exp_shape for r, exp_shape in zip(res, expected_shapes))

    def test_providing_no_observable_and_wires(self):
        """Test that we can provide no observable but specify wires to the sample function"""
        wires = [0, 2]
        wires_obj = qml.wires.Wires(wires)
        dev = qml.device("default.qubit", wires=3, shots=1000)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            res = qml.sample(wires=wires)

            assert res.obs is None
            assert res.wires == wires_obj
            return res

        circuit()

    @pytest.mark.parametrize(
        "obs,exp",
        [
            # Single observables
            (None, int),  # comp basis samples
            (qml.PauliX(0), int),
            (qml.PauliY(0), int),
            (qml.PauliZ(0), int),
            (qml.Hadamard(0), int),
            (qml.Identity(0), int),
            (qml.Hermitian(np.diag([1, 2]), 0), float),
            (qml.Hermitian(np.diag([1.0, 2.0]), 0), float),
            # Tensor product observables
            (
                qml.PauliX("c")
                @ qml.PauliY("a")
                @ qml.PauliZ(1)
                @ qml.Hadamard("wire1")
                @ qml.Identity("b"),
                int,
            ),
            (qml.Projector([0, 1], wires=[0, 1]) @ qml.PauliZ(2), float),
            (qml.Hermitian(np.array(np.eye(2)), wires=[0]) @ qml.PauliZ(2), float),
            (
                qml.Projector([0, 1], wires=[0, 1]) @ qml.Hermitian(np.array(np.eye(2)), wires=[2]),
                float,
            ),
        ],
    )
    def test_numeric_type(self, obs, exp):
        """Test that the numeric type is correct."""
        res = qml.sample(obs) if obs is not None else qml.sample()
        assert res.numeric_type is exp

    @pytest.mark.parametrize(
        "obs",
        [
            None,
            qml.PauliZ(0),
            qml.Hermitian(np.diag([1, 2]), 0),
            qml.Hermitian(np.diag([1.0, 2.0]), 0),
        ],
    )
    def test_shape(self, obs):
        """Test that the shape is correct."""
        shots = 10
        dev = qml.device("default.qubit", wires=3, shots=shots)
        res = qml.sample(obs) if obs is not None else qml.sample()
        expected = (1, shots) if obs is not None else (1, shots, 3)
        assert res.shape(dev) == expected

    @pytest.mark.parametrize(
        "obs",
        [qml.PauliZ(0), qml.Hermitian(np.diag([1, 2]), 0), qml.Hermitian(np.diag([1.0, 2.0]), 0)],
    )
    def test_shape_shot_vector(self, obs):
        """Test that the shape is correct with the shot vector too."""
        shot_vector = (1, 2, 3)
        dev = qml.device("default.qubit", wires=3, shots=shot_vector)
        res = qml.sample(obs)
        expected = ((), (2,), (3,))
        assert res.shape(dev) == expected

    def test_shape_shot_vector_no_obs(self):
        """Test that the shape is correct with the shot vector and no observable too."""
        shot_vec = (2, 2)
        dev = qml.device("default.qubit", wires=3, shots=shot_vec)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            qml.PauliZ(0)
            return qml.sample(qml.PauliZ(0))

        binned_samples = circuit()

        assert isinstance(binned_samples, tuple)
        assert len(binned_samples) == len(shot_vec)
        assert binned_samples[0].shape == (shot_vec[0],)
