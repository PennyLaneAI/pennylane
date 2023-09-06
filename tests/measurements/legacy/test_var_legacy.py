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
"""Unit tests for the var module"""
import numpy as np
import pytest

import pennylane as qml
from pennylane.measurements import Variance, Shots


# TODO: Remove this when new CustomMP are the default
def custom_measurement_process(device, spy):
    assert len(spy.call_args_list) > 0  # make sure method is mocked properly

    # pylint: disable=protected-access
    samples = device._samples
    state = device._state
    call_args_list = list(spy.call_args_list)
    for call_args in call_args_list:
        obs = call_args.args[1]
        shot_range, bin_size = (
            call_args.kwargs["shot_range"],
            call_args.kwargs["bin_size"],
        )
        meas = qml.var(op=obs)
        old_res = device.var(obs, shot_range, bin_size)
        if samples is not None:
            new_res = meas.process_samples(
                samples=samples, wire_order=device.wires, shot_range=shot_range, bin_size=bin_size
            )
        else:
            new_res = meas.process_state(state=state, wire_order=device.wires)
        assert qml.math.allclose(old_res, new_res)


class TestVar:
    """Tests for the var function"""

    @pytest.mark.parametrize("shots", [None, 10000, [10000, 10000]])
    @pytest.mark.parametrize("r_dtype", [np.float32, np.float64])
    def test_value(self, tol, r_dtype, mocker, shots):
        """Test that the var function works"""
        dev = qml.device("default.qubit.legacy", wires=2, shots=shots)
        spy = mocker.spy(qml.QubitDevice, "var")
        dev.R_DTYPE = r_dtype

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.var(qml.PauliZ(0))

        x = 0.54
        res = circuit(x)
        expected = [np.sin(x) ** 2, np.sin(x) ** 2] if isinstance(shots, list) else np.sin(x) ** 2
        atol = tol if shots is None else 0.05
        rtol = 0 if shots is None else 0.05

        assert np.allclose(res, expected, atol=atol, rtol=rtol)
        # pylint: disable=no-member, unsubscriptable-object
        if isinstance(res, tuple):
            assert res[0].dtype == r_dtype
            assert res[1].dtype == r_dtype
        else:
            assert res.dtype == r_dtype

        custom_measurement_process(dev, spy)

    def test_not_an_observable(self, mocker):
        """Test that a UserWarning is raised if the provided
        argument might not be hermitian."""
        dev = qml.device("default.qubit.legacy", wires=2)
        spy = mocker.spy(qml.QubitDevice, "var")

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.52, wires=0)
            return qml.var(qml.prod(qml.PauliX(0), qml.PauliZ(0)))

        with pytest.warns(UserWarning, match="Prod might not be hermitian."):
            _ = circuit()

        custom_measurement_process(dev, spy)

    def test_observable_return_type_is_variance(self, mocker):
        """Test that the return type of the observable is :attr:`ObservableReturnTypes.Variance`"""
        dev = qml.device("default.qubit.legacy", wires=2)
        spy = mocker.spy(qml.QubitDevice, "var")

        @qml.qnode(dev)
        def circuit():
            res = qml.var(qml.PauliZ(0))
            assert res.return_type is Variance
            return res

        circuit()

        custom_measurement_process(dev, spy)

    @pytest.mark.parametrize(
        "obs",
        [qml.PauliZ(0), qml.Hermitian(np.diag([1, 2]), 0), qml.Hermitian(np.diag([1.0, 2.0]), 0)],
    )
    def test_shape(self, obs):
        """Test that the shape is correct."""
        dev = qml.device("default.qubit.legacy", wires=1)
        res = qml.var(obs)
        # pylint: disable=use-implicit-booleaness-not-comparison
        assert res.shape(dev, Shots(None)) == ()
        assert res.shape(dev, Shots(100)) == ()

    @pytest.mark.parametrize(
        "obs",
        [qml.PauliZ(0), qml.Hermitian(np.diag([1, 2]), 0), qml.Hermitian(np.diag([1.0, 2.0]), 0)],
    )
    def test_shape_shot_vector(self, obs):
        """Test that the shape is correct with the shot vector too."""
        res = qml.var(obs)
        shot_vector = (1, 2, 3)
        dev = qml.device("default.qubit.legacy", wires=3, shots=shot_vector)
        assert res.shape(dev, Shots(shot_vector)) == ((), (), ())

    @pytest.mark.parametrize("state", [np.array([0, 0, 0]), np.array([1, 0, 0, 0, 0, 0, 0, 0])])
    @pytest.mark.parametrize("shots", [None, 1000, [1000, 10000]])
    def test_projector_var(self, state, shots, mocker):
        """Tests that the variance of a ``Projector`` object is computed correctly."""
        dev = qml.device("default.qubit.legacy", wires=3, shots=shots)
        spy = mocker.spy(qml.QubitDevice, "var")

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(0)
            return qml.var(qml.Projector(state, wires=range(3)))

        res = circuit()
        expected = [0.25, 0.25] if isinstance(shots, list) else 0.25

        assert np.allclose(res, expected, atol=0.02, rtol=0.02)

        custom_measurement_process(dev, spy)

    def test_permuted_wires(self, mocker):
        """Test that the variance of an operator with permuted wires is the same."""
        obs = qml.prod(qml.PauliZ(8), qml.s_prod(2, qml.PauliZ(10)), qml.s_prod(3, qml.PauliZ("h")))
        obs_2 = qml.prod(
            qml.s_prod(3, qml.PauliZ("h")), qml.PauliZ(8), qml.s_prod(2, qml.PauliZ(10))
        )

        dev = qml.device("default.qubit.legacy", wires=["h", 8, 10])
        spy = mocker.spy(qml.QubitDevice, "var")

        @qml.qnode(dev)
        def circuit():
            qml.RX(1.23, wires=["h"])
            qml.RY(2.34, wires=[8])
            return qml.var(obs)

        @qml.qnode(dev)
        def circuit2():
            qml.RX(1.23, wires=["h"])
            qml.RY(2.34, wires=[8])
            return qml.var(obs_2)

        assert circuit() == circuit2()
        custom_measurement_process(dev, spy)
