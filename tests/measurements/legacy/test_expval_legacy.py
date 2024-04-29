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
from pennylane.devices.qubit.measure import flatten_state
from pennylane.measurements import Expectation, Shots


# TODO: Remove this when new CustomMP are the default
def custom_measurement_process(device, spy):
    assert len(spy.call_args_list) > 0  # make sure method is mocked properly

    samples = device._samples  # pylint: disable=protected-access
    state = flatten_state(device._state, device.num_wires)  # pylint: disable=protected-access
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
        assert qml.math.allclose(old_res, new_res)


class TestExpval:
    """Tests for the expval function"""

    @pytest.mark.parametrize("shots", [None, 10000, [10000, 10000]])
    @pytest.mark.parametrize("r_dtype", [np.float32, np.float64])
    def test_value(self, tol, r_dtype, mocker, shots):
        """Test that the expval interface works"""
        dev = qml.device("default.qubit.legacy", wires=2, shots=shots)
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

        # pylint: disable=no-member, unsubscriptable-object
        if isinstance(res, tuple):
            assert res[0].dtype == r_dtype
            assert res[1].dtype == r_dtype
        else:
            assert res.dtype == r_dtype

        custom_measurement_process(new_dev, spy)

    def test_observable_return_type_is_expectation(self, mocker):
        """Test that the return type of the observable is :attr:`ObservableReturnTypes.Expectation`"""
        dev = qml.device("default.qubit.legacy", wires=2)

        @qml.qnode(dev)
        def circuit():
            res = qml.expval(qml.PauliZ(0))
            assert res.return_type is Expectation
            return res

        new_dev = circuit.device
        spy = mocker.spy(qml.QubitDevice, "expval")

        circuit()

        custom_measurement_process(new_dev, spy)

    @pytest.mark.parametrize("shots", [None, 10000, [10000, 10000]])
    @pytest.mark.parametrize("phi", np.arange(0, 2 * np.pi, np.pi / 3))
    @pytest.mark.parametrize("device_name", ["default.qubit.legacy", "default.mixed"])
    def test_observable_is_measurement_value(
        self, shots, phi, mocker, tol, tol_stochastic, device_name
    ):  # pylint: disable=too-many-arguments
        """Test that expectation values for mid-circuit measurement values
        are correct for a single measurement value."""
        np.random.seed(42)
        dev = qml.device(device_name, wires=2, shots=shots)

        @qml.qnode(dev)
        def circuit(phi):
            qml.RX(phi, 0)
            m0 = qml.measure(0)
            return qml.expval(m0)

        new_dev = circuit.device
        spy = mocker.spy(qml.QubitDevice, "expval")

        res = circuit(phi)

        atol = tol if shots is None else tol_stochastic
        assert np.allclose(np.array(res), np.sin(phi / 2) ** 2, atol=atol, rtol=0)

        if device_name != "default.mixed":
            custom_measurement_process(new_dev, spy)

    @pytest.mark.parametrize("shots", [None, 10000, [10000, 10000]])
    @pytest.mark.parametrize("phi", np.arange(0, 2 * np.pi, np.pi / 3))
    @pytest.mark.parametrize("device_name", ["default.qubit.legacy", "default.mixed"])
    def test_observable_is_composite_measurement_value(
        self, shots, phi, mocker, tol, tol_stochastic, device_name
    ):  # pylint: disable=too-many-arguments
        """Test that expectation values for mid-circuit measurement values
        are correct for a composite measurement value."""
        np.random.seed(42)
        dev = qml.device(device_name, wires=6, shots=shots)

        @qml.qnode(dev)
        def circuit(phi):
            qml.RX(phi, 0)
            m0 = qml.measure(0)
            qml.RX(0.5 * phi, 1)
            m1 = qml.measure(1)
            qml.RX(2 * phi, 2)
            m2 = qml.measure(2)
            return qml.expval(m0 * m1 + m2)

        new_dev = circuit.device
        spy = mocker.spy(qml.QubitDevice, "expval")

        res = circuit(phi, shots=shots)

        @qml.qnode(dev)
        def expected_circuit(phi):
            qml.RX(phi, 0)
            qml.RX(0.5 * phi, 1)
            qml.RX(2 * phi, 2)
            return (
                qml.expval(qml.Projector([1], 0)),
                qml.expval(qml.Projector([1], 1)),
                qml.expval(qml.Projector([1], 2)),
            )

        evals = expected_circuit(phi, shots=None)
        expected = evals[0] * evals[1] + evals[2]

        atol = tol if shots is None else tol_stochastic
        assert np.allclose(np.array(res), expected, atol=atol, rtol=0)

        if device_name != "default.mixed":
            custom_measurement_process(new_dev, spy)

    @pytest.mark.parametrize(
        "obs",
        [qml.PauliZ(0), qml.Hermitian(np.diag([1, 2]), 0), qml.Hermitian(np.diag([1.0, 2.0]), 0)],
    )
    def test_shape(self, obs):
        """Test that the shape is correct."""
        dev = qml.device("default.qubit.legacy", wires=1)

        res = qml.expval(obs)
        # pylint: disable=use-implicit-booleaness-not-comparison
        assert res.shape(dev, Shots(None)) == ()
        assert res.shape(dev, Shots(100)) == ()

    @pytest.mark.parametrize(
        "obs",
        [qml.PauliZ(0), qml.Hermitian(np.diag([1, 2]), 0), qml.Hermitian(np.diag([1.0, 2.0]), 0)],
    )
    def test_shape_shot_vector(self, obs):
        """Test that the shape is correct with the shot vector too."""
        res = qml.expval(obs)
        shot_vector = (1, 2, 3)
        dev = qml.device("default.qubit.legacy", wires=3, shots=shot_vector)
        assert res.shape(dev, Shots(shot_vector)) == ((), (), ())

    @pytest.mark.parametrize("state", [np.array([0, 0, 0]), np.array([1, 0, 0, 0, 0, 0, 0, 0])])
    @pytest.mark.parametrize("shots", [None, 1000, [1000, 10000]])
    def test_projector_expval(self, state, shots, mocker):
        """Tests that the expectation of a ``Projector`` object is computed correctly for both of
        its subclasses."""
        dev = qml.device("default.qubit.legacy", wires=3, shots=shots)
        np.random.seed(42)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(0)
            return qml.expval(qml.Projector(state, wires=range(3)))

        new_dev = circuit.device
        spy = mocker.spy(qml.QubitDevice, "expval")

        res = circuit()
        expected = [0.5, 0.5] if isinstance(shots, list) else 0.5
        assert np.allclose(res, expected, atol=0.02, rtol=0.02)

        custom_measurement_process(new_dev, spy)

    def test_permuted_wires(self, mocker):
        """Test that the expectation value of an operator with permuted wires is the same."""
        obs = qml.prod(qml.PauliZ(8), qml.s_prod(2, qml.PauliZ(10)), qml.s_prod(3, qml.PauliZ("h")))
        obs_2 = qml.prod(
            qml.s_prod(3, qml.PauliZ("h")), qml.PauliZ(8), qml.s_prod(2, qml.PauliZ(10))
        )

        dev = qml.device("default.qubit.legacy", wires=["h", 8, 10])
        spy = mocker.spy(qml.QubitDevice, "expval")

        @qml.qnode(dev)
        def circuit():
            qml.RX(1.23, wires=["h"])
            qml.RY(2.34, wires=[8])
            return qml.expval(obs)

        @qml.qnode(dev)
        def circuit2():
            qml.RX(1.23, wires=["h"])
            qml.RY(2.34, wires=[8])
            return qml.expval(obs_2)

        assert circuit() == circuit2()
        custom_measurement_process(dev, spy)
