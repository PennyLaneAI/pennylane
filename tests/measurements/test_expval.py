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
"""Unit tests for the expval module"""
import numpy as np
import pytest

import pennylane as qml
from pennylane.measurements import Expectation


# TODO: Remove this when new CustomMP are the default
def custom_measurement_process(device, spy):

    assert len(spy.call_args_list) > 0  # make sure method is mocked properly

    samples = device._samples  # pylint: disable=protected-access
    state = device._state  # pylint: disable=protected-access
    call_args_list = list(spy.call_args_list)
    for call_args in call_args_list:
        obs = call_args.args[1]
        shot_range, bin_size = (
            call_args.kwargs["shot_range"],
            call_args.kwargs["bin_size"],
        )
        # no need to use op, because the observable has already been applied to ``self.dev._state``
        meas = qml.expval(op=obs)
        old_res = device.expval(obs, shot_range=shot_range, bin_size=bin_size)
        if device.shots is None:
            new_res = meas.process_state(state=state, wire_order=device.wires)
        else:
            new_res = meas.process_samples(
                samples=samples, wire_order=device.wires, shot_range=shot_range, bin_size=bin_size
            )
        assert qml.math.allequal(old_res, new_res)


class TestExpval:
    """Tests for the expval function"""

    @pytest.mark.parametrize("shots", [None, 1000, [1000, 10000]])
    @pytest.mark.parametrize("r_dtype", [np.float32, np.float64])
    def test_value(self, tol, r_dtype, mocker, shots):
        """Test that the expval interface works"""
        dev = qml.device("default.qubit", wires=2, shots=shots)
        dev.R_DTYPE = r_dtype

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliY(0))

        new_dev = circuit.device
        spy = mocker.spy(qml.QubitDevice, "expval")

        x = 0.54
        res = circuit(x)
        expected = -np.sin(x)

        atol = tol if shots is None else 0.05
        rtol = 0 if shots is None else 0.05

        assert np.allclose(res, expected, atol=atol, rtol=rtol)
        assert res.dtype == r_dtype

        custom_measurement_process(new_dev, spy)

    def test_not_an_observable(self, mocker):
        """Test that a warning is raised if the provided
        argument might not be hermitian."""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.52, wires=0)
            return qml.expval(qml.prod(qml.PauliX(0), qml.PauliZ(0)))

        new_dev = circuit.device
        spy = mocker.spy(qml.QubitDevice, "expval")

        with pytest.warns(UserWarning, match="Prod might not be hermitian."):
            _ = circuit()

        custom_measurement_process(new_dev, spy)

    def test_observable_return_type_is_expectation(self, mocker):
        """Test that the return type of the observable is :attr:`ObservableReturnTypes.Expectation`"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            res = qml.expval(qml.PauliZ(0))
            assert res.return_type is Expectation
            return res

        new_dev = circuit.device
        spy = mocker.spy(qml.QubitDevice, "expval")

        circuit()

        custom_measurement_process(new_dev, spy)

    @pytest.mark.parametrize(
        "obs",
        [qml.PauliZ(0), qml.Hermitian(np.diag([1, 2]), 0), qml.Hermitian(np.diag([1.0, 2.0]), 0)],
    )
    def test_numeric_type(self, obs):
        """Test that the numeric type is correct."""
        res = qml.expval(obs)
        assert res.numeric_type is float

    @pytest.mark.parametrize(
        "obs",
        [qml.PauliZ(0), qml.Hermitian(np.diag([1, 2]), 0), qml.Hermitian(np.diag([1.0, 2.0]), 0)],
    )
    def test_shape(self, obs):
        """Test that the shape is correct."""
        res = qml.expval(obs)
        assert res.shape() == (1,)

    @pytest.mark.parametrize(
        "obs",
        [qml.PauliZ(0), qml.Hermitian(np.diag([1, 2]), 0), qml.Hermitian(np.diag([1.0, 2.0]), 0)],
    )
    def test_shape_shot_vector(self, obs):
        """Test that the shape is correct with the shot vector too."""
        res = qml.expval(obs)
        shot_vector = (1, 2, 3)
        dev = qml.device("default.qubit", wires=3, shots=shot_vector)
        assert res.shape(dev) == (len(shot_vector),)

    @pytest.mark.parametrize("shots", [None, 1000, [1000, 10000]])
    def test_projector_expval(self, shots, mocker):
        """Tests that the expectation of a ``Projector`` object is computed correctly."""
        dev = qml.device("default.qubit", wires=3, shots=shots)

        basis_state = np.array([0, 0, 0])

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(0)
            return qml.expval(qml.Projector(basis_state, wires=range(3)))

        new_dev = circuit.device
        spy = mocker.spy(qml.QubitDevice, "expval")

        res = circuit()
        expected = [0.5, 0.5] if isinstance(shots, list) else 0.5

        assert np.allclose(res, expected, atol=0.02, rtol=0.02)

        custom_measurement_process(new_dev, spy)
