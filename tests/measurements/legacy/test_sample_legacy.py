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
from pennylane.measurements import MeasurementShapeError, MeasurementValue, Sample, Shots
from pennylane.operation import Operator

# pylint: disable=protected-access, no-member


# TODO: Remove this when new CustomMP are the default
def custom_measurement_process(device, spy):
    assert len(spy.call_args_list) > 0  # make sure method is mocked properly

    samples = device._samples
    call_args_list = list(spy.call_args_list)
    for call_args in call_args_list:
        meas = call_args.args[1]
        shot_range, bin_size = (call_args.kwargs["shot_range"], call_args.kwargs["bin_size"])
        if isinstance(meas, (Operator, MeasurementValue)):
            meas = qml.sample(op=meas)
        assert qml.math.allequal(
            device.sample(call_args.args[1], **call_args.kwargs),
            meas.process_samples(
                samples=samples,
                wire_order=device.wires,
                shot_range=shot_range,
                bin_size=bin_size,
            ),
        )


class TestSample:
    """Tests for the sample function"""

    @pytest.mark.parametrize("n_sample", (1, 10))
    def test_sample_dimension(self, mocker, n_sample):
        """Test that the sample function outputs samples of the right size"""

        dev = qml.device("default.qubit.legacy", wires=2, shots=n_sample)
        spy = mocker.spy(qml.QubitDevice, "sample")

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.54, wires=0)
            return qml.sample(qml.PauliZ(0)), qml.sample(qml.PauliX(1))

        output = circuit()

        assert len(output) == 2
        assert circuit._qfunc_output[0].shape(dev, Shots(n_sample)) == (
            (n_sample,) if not n_sample == 1 else ()
        )
        assert circuit._qfunc_output[1].shape(dev, Shots(n_sample)) == (
            (n_sample,) if not n_sample == 1 else ()
        )

        custom_measurement_process(dev, spy)

    @pytest.mark.filterwarnings("ignore:Creating an ndarray from ragged nested sequences")
    def test_sample_combination(self, mocker):
        """Test the output of combining expval, var and sample"""
        n_sample = 10

        dev = qml.device("default.qubit.legacy", wires=3, shots=n_sample)
        spy = mocker.spy(qml.QubitDevice, "sample")

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit():
            qml.RX(0.54, wires=0)

            return qml.sample(qml.PauliZ(0)), qml.expval(qml.PauliX(1)), qml.var(qml.PauliY(2))

        result = circuit()

        assert len(result) == 3
        assert np.array_equal(result[0].shape, (n_sample,))
        assert circuit._qfunc_output[0].shape(dev, Shots(n_sample)) == (n_sample,)
        assert isinstance(result[1], np.ndarray)
        assert isinstance(result[2], np.ndarray)

        custom_measurement_process(dev, spy)

    @pytest.mark.parametrize("shots", [5, [5, 5]])
    @pytest.mark.parametrize("phi", np.arange(0, 2 * np.pi, np.pi / 2))
    @pytest.mark.parametrize("device_name", ["default.qubit.legacy", "default.mixed"])
    def test_observable_is_measurement_value(self, shots, phi, mocker, device_name):
        """Test that samples for mid-circuit measurement values
        are correct for a single measurement value."""
        dev = qml.device(device_name, wires=2, shots=shots)

        @qml.qnode(dev)
        def circuit(phi):
            qml.RX(phi, 0)
            m0 = qml.measure(0)
            return qml.sample(m0)

        new_dev = circuit.device
        spy = mocker.spy(qml.QubitDevice, "sample")

        res = circuit(phi)

        if isinstance(shots, list):
            assert len(res) == len(shots)
            assert all(r.shape == (s,) for r, s in zip(res, shots))
        else:
            assert res.shape == (shots,)

        if device_name != "default.mixed":
            custom_measurement_process(new_dev, spy)

    @pytest.mark.parametrize("shots", [5, [5, 5]])
    @pytest.mark.parametrize("phi", np.arange(0, 2 * np.pi, np.pi / 2))
    @pytest.mark.parametrize("device_name", ["default.qubit.legacy", "default.mixed"])
    def test_observable_is_composite_measurement_value(self, shots, phi, mocker, device_name):
        """Test that samples for mid-circuit measurement values
        are correct for a composite measurement value."""
        dev = qml.device(device_name, wires=4, shots=shots)

        @qml.qnode(dev)
        def circuit(phi):
            qml.RX(phi, 0)
            m0 = qml.measure(0)
            qml.RX(phi, 1)
            m1 = qml.measure(1)
            return qml.sample(op=m0 + m1)

        new_dev = circuit.device
        spy = mocker.spy(qml.QubitDevice, "sample")

        res = circuit(phi)

        if isinstance(shots, list):
            assert len(res) == len(shots)
            assert all(r.shape == (s,) for r, s in zip(res, shots))
        else:
            assert res.shape == (shots,)

        if device_name != "default.mixed":
            custom_measurement_process(new_dev, spy)

    @pytest.mark.parametrize("shots", [5, [5, 5]])
    @pytest.mark.parametrize("phi", np.arange(0, 2 * np.pi, np.pi / 2))
    @pytest.mark.parametrize("device_name", ["default.qubit.legacy", "default.mixed"])
    def test_observable_is_measurement_value_list(self, shots, phi, mocker, device_name):
        """Test that samples for mid-circuit measurement values
        are correct for a measurement value list."""
        dev = qml.device(device_name, wires=4, shots=shots)

        @qml.qnode(dev)
        def circuit(phi):
            qml.RX(phi, 0)
            m0 = qml.measure(0)
            m1 = qml.measure(1)
            return qml.sample(op=[m0, m1])

        new_dev = circuit.device
        spy = mocker.spy(qml.QubitDevice, "sample")

        res = circuit(phi)

        if isinstance(shots, list):
            assert len(res) == len(shots)
            assert all(r.shape == (s, 2) for r, s in zip(res, shots))
        else:
            assert res.shape == (shots, 2)

        if device_name != "default.mixed":
            custom_measurement_process(new_dev, spy)

    def test_single_wire_sample(self, mocker):
        """Test the return type and shape of sampling a single wire"""
        n_sample = 10

        dev = qml.device("default.qubit.legacy", wires=1, shots=n_sample)
        spy = mocker.spy(qml.QubitDevice, "sample")

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.54, wires=0)
            return qml.sample(qml.PauliZ(0))

        result = circuit()

        assert isinstance(result, np.ndarray)
        assert np.array_equal(result.shape, (n_sample,))
        assert circuit._qfunc_output.shape(dev, Shots(n_sample)) == (n_sample,)

        custom_measurement_process(dev, spy)

    def test_multi_wire_sample_regular_shape(self, mocker):
        """Test the return type and shape of sampling multiple wires
        where a rectangular array is expected"""
        n_sample = 10

        dev = qml.device("default.qubit.legacy", wires=3, shots=n_sample)
        spy = mocker.spy(qml.QubitDevice, "sample")

        @qml.qnode(dev)
        def circuit():
            return qml.sample(qml.PauliZ(0)), qml.sample(qml.PauliZ(1)), qml.sample(qml.PauliZ(2))

        result = circuit()

        assert circuit._qfunc_output[0].shape(dev, Shots(n_sample)) == (n_sample,)
        assert circuit._qfunc_output[1].shape(dev, Shots(n_sample)) == (n_sample,)
        assert circuit._qfunc_output[2].shape(dev, Shots(n_sample)) == (n_sample,)

        # If all the dimensions are equal the result will end up to be a proper rectangular array
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert result[0].dtype == np.dtype("int")

        custom_measurement_process(dev, spy)

    @pytest.mark.filterwarnings("ignore:Creating an ndarray from ragged nested sequences")
    def test_sample_output_type_in_combination(self, mocker):
        """Test the return type and shape of sampling multiple works
        in combination with expvals and vars"""
        n_sample = 10

        dev = qml.device("default.qubit.legacy", wires=3, shots=n_sample)
        spy = mocker.spy(qml.QubitDevice, "sample")

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

        custom_measurement_process(dev, spy)

    def test_observable_return_type_is_sample(self, mocker):
        """Test that the return type of the observable is :attr:`ObservableReturnTypes.Sample`"""
        n_shots = 10
        dev = qml.device("default.qubit.legacy", wires=1, shots=n_shots)
        spy = mocker.spy(qml.QubitDevice, "sample")

        @qml.qnode(dev)
        def circuit():
            res = qml.sample(qml.PauliZ(0))
            assert res.return_type is Sample
            return res

        circuit()

        custom_measurement_process(dev, spy)

    def test_providing_observable_and_wires(self):
        """Test that a ValueError is raised if both an observable is provided and wires are specified"""
        dev = qml.device("default.qubit.legacy", wires=2)

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

    def test_providing_no_observable_and_no_wires(self, mocker):
        """Test that we can provide no observable and no wires to sample function"""
        dev = qml.device("default.qubit.legacy", wires=2, shots=1000)
        spy = mocker.spy(qml.QubitDevice, "sample")

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            res = qml.sample()
            assert res.obs is None
            assert res.wires == qml.wires.Wires([])
            return res

        circuit()

        custom_measurement_process(dev, spy)

    def test_providing_no_observable_and_no_wires_shot_vector(self, mocker):
        """Test that we can provide no observable and no wires to sample
        function when using a shot vector"""
        num_wires = 2

        shots1 = 1
        shots2 = 10
        shots3 = 1000
        dev = qml.device("default.qubit.legacy", wires=num_wires, shots=[shots1, shots2, shots3])
        spy = mocker.spy(qml.QubitDevice, "sample")

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.sample()

        res = circuit()

        assert isinstance(res, tuple)

        expected_shapes = [(num_wires,), (shots2, num_wires), (shots3, num_wires)]
        assert len(res) == len(expected_shapes)
        assert all(r.shape == exp_shape for r, exp_shape in zip(res, expected_shapes))

        # assert first wire is always the same as second
        # pylint: disable=unsubscriptable-object
        assert np.all(res[0][0] == res[0][1])
        assert np.all(res[1][:, 0] == res[1][:, 1])
        assert np.all(res[2][:, 0] == res[2][:, 1])

        custom_measurement_process(dev, spy)

    def test_providing_no_observable_and_wires(self, mocker):
        """Test that we can provide no observable but specify wires to the sample function"""
        wires = [0, 2]
        wires_obj = qml.wires.Wires(wires)
        dev = qml.device("default.qubit.legacy", wires=3, shots=1000)
        spy = mocker.spy(qml.QubitDevice, "sample")

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            res = qml.sample(wires=wires)

            assert res.obs is None
            assert res.wires == wires_obj
            return res

        circuit()

        custom_measurement_process(dev, spy)

    def test_shape_no_shots_error(self):
        """Test that the appropriate error is raised with no shots are specified"""
        dev = qml.device("default.qubit.legacy", wires=2, shots=None)
        shots = Shots(None)
        mp = qml.sample()

        with pytest.raises(
            MeasurementShapeError, match="Shots are required to obtain the shape of the measurement"
        ):
            _ = mp.shape(dev, shots)

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
        dev = qml.device("default.qubit.legacy", wires=3, shots=shots)
        res = qml.sample(obs) if obs is not None else qml.sample()
        expected = (shots,) if obs is not None else (shots, 3)
        assert res.shape(dev, Shots(shots)) == expected

    @pytest.mark.parametrize("n_samples", (1, 10))
    def test_shape_wires(self, n_samples):
        """Test that the shape is correct when wires are provided."""
        dev = qml.device("default.qubit.legacy", wires=3, shots=n_samples)
        mp = qml.sample(wires=(0, 1))
        assert mp.shape(dev, Shots(n_samples)) == (n_samples, 2) if n_samples != 1 else (2,)

    @pytest.mark.parametrize(
        "obs",
        [
            None,
            qml.PauliZ(0),
            qml.Hermitian(np.diag([1, 2]), 0),
            qml.Hermitian(np.diag([1.0, 2.0]), 0),
        ],
    )
    def test_shape_shot_vector(self, obs):
        """Test that the shape is correct with the shot vector too."""
        shot_vector = (1, 2, 3)
        dev = qml.device("default.qubit.legacy", wires=3, shots=shot_vector)
        res = qml.sample(obs) if obs is not None else qml.sample()
        expected = ((), (2,), (3,)) if obs is not None else ((3,), (2, 3), (3, 3))
        assert res.shape(dev, Shots(shot_vector)) == expected

    def test_shape_shot_vector_obs(self):
        """Test that the shape is correct with the shot vector and a observable too."""
        shot_vec = (2, 2)
        dev = qml.device("default.qubit.legacy", wires=3, shots=shot_vec)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            qml.PauliZ(0)
            return qml.sample(qml.PauliZ(0))

        binned_samples = circuit()

        assert isinstance(binned_samples, tuple)
        assert len(binned_samples) == len(shot_vec)
        # pylint: disable=unsubscriptable-object
        assert binned_samples[0].shape == (shot_vec[0],)

    @pytest.mark.parametrize("shots", [2, 100])
    def test_sample_no_arguments(self, shots):
        """Test that using ``qml.sample`` with no arguments returns the samples of all wires."""
        dev = qml.device("default.qubit.legacy", wires=3, shots=shots)

        @qml.qnode(dev)
        def circuit():
            return qml.sample()

        res = circuit()

        # pylint: disable=comparison-with-callable
        assert res.shape == (shots, 3)


@pytest.mark.jax
@pytest.mark.parametrize("samples", (1, 10))
def test_jitting_with_sampling_on_subset_of_wires(samples):
    """Test case covering bug in Issue #3904.  Sampling should be jit-able
    when sampling occurs on a subset of wires. The bug was occuring due an improperly
    set shape method."""
    import jax

    jax.config.update("jax_enable_x64", True)

    dev = qml.device("default.qubit.legacy", wires=3, shots=samples)

    @qml.qnode(dev, interface="jax")
    def circuit(x):
        qml.RX(x, wires=0)
        return qml.sample(wires=(0, 1))

    results = jax.jit(circuit)(jax.numpy.array(0.123, dtype=jax.numpy.float64))

    expected = (2,) if samples == 1 else (samples, 2)
    assert results.shape == expected
    assert (
        circuit._qfunc_output.shape(dev, Shots(samples)) == (samples, 2) if samples != 1 else (2,)
    )
