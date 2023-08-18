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
import copy
import numpy as np
import pytest

import pennylane as qml
from pennylane.measurements import Expectation, Shots
from pennylane.measurements.expval import ExpectationMP


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
        assert qml.math.allclose(old_res, new_res)


class TestExpval:
    """Tests for the expval function"""

    @pytest.mark.parametrize("shots", [None, 10000, [10000, 10000]])
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

        # pylint: disable=no-member, unsubscriptable-object
        if isinstance(res, tuple):
            assert res[0].dtype == r_dtype
            assert res[1].dtype == r_dtype
        else:
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
        dev = qml.device("default.qubit", wires=1)

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
        dev = qml.device("default.qubit", wires=3, shots=shot_vector)
        assert res.shape(dev, Shots(shot_vector)) == ((), (), ())

    @pytest.mark.parametrize("state", [np.array([0, 0, 0]), np.array([1, 0, 0, 0, 0, 0, 0, 0])])
    @pytest.mark.parametrize("shots", [None, 1000, [1000, 10000]])
    def test_projector_expval(self, state, shots, mocker):
        """Tests that the expectation of a ``Projector`` object is computed correctly for both of
        its subclasses."""
        dev = qml.device("default.qubit", wires=3, shots=shots)
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

        dev = qml.device("default.qubit", wires=["h", 8, 10])
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

    def test_copy_observable(self):
        """Test that the observable is copied if present."""
        m = qml.expval(qml.PauliX(0))
        copied_m = copy.copy(m)
        assert m.obs is not copied_m.obs
        assert qml.equal(m.obs, copied_m.obs)

    def test_copy_eigvals(self):
        """Test that the eigvals value is just assigned to new mp without copying."""
        # pylint: disable=protected-access
        m = ExpectationMP(eigvals=[-0.5, 0.5], wires=qml.wires.Wires(0))
        copied_m = copy.copy(m)
        assert m._eigvals is copied_m._eigvals

    def test_standard_obs(self):
        """Check that the hash of an expectation value of an observable can distinguish different observables."""

        o1 = qml.prod(qml.PauliX(0), qml.PauliY(1))
        o2 = qml.prod(qml.PauliX(0), qml.PauliZ(1))

        assert qml.expval(o1).hash == qml.expval(o1).hash
        assert qml.expval(o2).hash == qml.expval(o2).hash
        assert qml.expval(o1).hash != qml.expval(o2).hash

        o3 = qml.sum(qml.PauliX("a"), qml.PauliY("b"))
        assert qml.expval(o1).hash != qml.expval(o3).hash

    def test_eigvals(self):
        """Test that the eigvals property controls the hash property."""
        m1 = ExpectationMP(eigvals=[-0.5, 0.5], wires=qml.wires.Wires(0))
        m2 = ExpectationMP(eigvals=[-0.5, 0.5], wires=qml.wires.Wires(0), id="something")

        assert m1.hash == m2.hash

        m3 = ExpectationMP(eigvals=[-0.5, 0.5], wires=qml.wires.Wires(1))
        assert m1.hash != m3.hash

        m4 = ExpectationMP(eigvals=[-1, 1], wires=qml.wires.Wires(1))
        assert m1.hash != m4.hash
        assert m3.hash != m4.hash
