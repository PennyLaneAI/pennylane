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
from pennylane.exceptions import EigvalsUndefinedError, MeasurementShapeError, QuantumFunctionError
from pennylane.operation import Operator

# pylint: disable=protected-access, no-member, too-many-public-methods


class TestSample:
    """Tests for the sample function"""

    @pytest.mark.parametrize("n_sample", (1, 10))
    def test_sample_dimension(self, n_sample):
        """Test that the sample function outputs samples of the right size"""

        assert qml.sample(qml.PauliZ(0)).shape(shots=n_sample, num_device_wires=2) == ((n_sample,))
        assert qml.sample(qml.PauliX(1)).shape(shots=n_sample, num_device_wires=2) == ((n_sample,))

    def test_sample_combination(self):
        """Test the output of combining expval, var and sample"""
        n_sample = 10

        dev = qml.device("default.qubit", wires=3)

        @qml.set_shots(n_sample)
        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit():
            qml.RX(0.54, wires=0)

            return qml.sample(qml.PauliZ(0)), qml.expval(qml.PauliX(1)), qml.var(qml.PauliY(2))

        result = circuit()

        assert len(result) == 3
        assert np.array_equal(result[0].shape, (n_sample,))
        assert circuit._qfunc_output[0].shape(shots=n_sample, num_device_wires=3) == (n_sample,)
        assert isinstance(result[1], np.float64)
        assert isinstance(result[2], np.float64)

    def test_single_wire_sample(self):
        """Test the return type and shape of sampling a single wire"""
        n_sample = 10

        dev = qml.device("default.qubit", wires=1)

        @qml.set_shots(n_sample)
        @qml.qnode(dev)
        def circuit():
            qml.RX(0.54, wires=0)
            return qml.sample(qml.PauliZ(0))

        result = circuit()

        assert isinstance(result, np.ndarray)
        assert np.array_equal(result.shape, (n_sample,))
        assert circuit._qfunc_output.shape(shots=n_sample, num_device_wires=1) == (n_sample,)

    def test_multi_wire_sample_regular_shape(self):
        """Test the return type and shape of sampling multiple wires
        where a rectangular array is expected"""
        n_sample = 10

        dev = qml.device("default.qubit", wires=3)

        @qml.set_shots(n_sample)
        @qml.qnode(dev)
        def circuit():
            return qml.sample(qml.PauliZ(0)), qml.sample(qml.PauliZ(1)), qml.sample(qml.PauliZ(2))

        result = circuit()

        assert circuit._qfunc_output[0].shape(shots=n_sample, num_device_wires=3) == (n_sample,)
        assert circuit._qfunc_output[1].shape(shots=n_sample, num_device_wires=3) == (n_sample,)
        assert circuit._qfunc_output[2].shape(shots=n_sample, num_device_wires=3) == (n_sample,)

        # If all the dimensions are equal the result will end up to be a proper rectangular array
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert result[0].dtype == np.dtype("float")

    def test_sample_output_type_in_combination(self):
        """Test the return type and shape of sampling multiple works
        in combination with expvals and vars"""
        n_sample = 10

        dev = qml.device("default.qubit", wires=3)

        @qml.set_shots(n_sample)
        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit():
            return qml.expval(qml.PauliZ(0)), qml.var(qml.PauliZ(1)), qml.sample(qml.PauliZ(2))

        result = circuit()

        # If all the dimensions are equal the result will end up to be a proper rectangular array
        assert len(result) == 3
        assert result[2].dtype == np.dtype("float")
        assert np.array_equal(result[2].shape, (n_sample,))

    @pytest.mark.parametrize("shots", [5, [5, 5]])
    @pytest.mark.parametrize("phi", np.arange(0, 2 * np.pi, np.pi / 2))
    def test_observable_is_measurement_value(self, shots, phi):
        """Test that samples for mid-circuit measurement values
        are correct for a single measurement value."""
        dev = qml.device("default.qubit", wires=2)

        @qml.set_shots(shots)
        @qml.qnode(dev)
        def circuit(phi):
            qml.RX(phi, 0)
            m0 = qml.measure(0)
            return qml.sample(m0)

        for func in [circuit, qml.defer_measurements(circuit)]:
            res = func(phi)
            if isinstance(shots, list):
                assert len(res) == len(shots)
                assert all(r.shape == (s,) for r, s in zip(res, shots))
            else:
                assert res.shape == (shots,)

    @pytest.mark.parametrize("shots", [5, [5, 5]])
    @pytest.mark.parametrize("phi", np.arange(0, 2 * np.pi, np.pi / 2))
    def test_observable_is_composite_measurement_value(self, shots, phi):
        """Test that samples for mid-circuit measurement values
        are correct for a composite measurement value."""
        dev = qml.device("default.qubit")

        @qml.set_shots(shots)
        @qml.qnode(dev)
        def circuit(phi):
            qml.RX(phi, 0)
            m0 = qml.measure(0)
            qml.RX(phi, 1)
            m1 = qml.measure(1)
            return qml.sample(op=m0 + m1)

        for func in [circuit, qml.defer_measurements(circuit)]:
            res = func(phi)
            if isinstance(shots, list):
                assert len(res) == len(shots)
                assert all(r.shape == (s,) for r, s in zip(res, shots))
            else:
                assert res.shape == (shots,)

    @pytest.mark.parametrize("shots", [5, [5, 5]])
    @pytest.mark.parametrize("phi", np.arange(0, 2 * np.pi, np.pi / 2))
    def test_observable_is_measurement_value_list(self, shots, phi):
        """Test that samples for mid-circuit measurement values
        are correct for a measurement value list."""
        dev = qml.device("default.qubit")

        @qml.defer_measurements
        @qml.set_shots(shots)
        @qml.qnode(dev)
        def circuit(phi):
            qml.RX(phi, 0)
            m0 = qml.measure(0)
            m1 = qml.measure(1)
            return qml.sample(op=[m0, m1])

        for func in [circuit, qml.defer_measurements(circuit)]:
            res = func(phi)
            if isinstance(shots, list):
                assert len(res) == len(shots)
                assert all(r.shape == (s, 2) for r, s in zip(res, shots))
            else:
                assert res.shape == (shots, 2)

    def test_mixed_lists_as_op_not_allowed(self):
        """Test that passing a list not containing only measurement values raises an error."""
        m0 = qml.measure(0)

        with pytest.raises(
            QuantumFunctionError,
            match="Only sequences of single MeasurementValues can be passed with the op argument",
        ):
            _ = qml.sample(op=[m0, qml.PauliZ(0)])

    def test_composed_measurement_value_lists_not_allowed(self):
        """Test that passing a list containing measurement values composed with arithmetic
        raises an error."""
        m0 = qml.measure(0)
        m1 = qml.measure(1)
        m2 = qml.measure(2)

        with pytest.raises(
            QuantumFunctionError,
            match="Only sequences of single MeasurementValues can be passed with the op argument",
        ):
            _ = qml.sample(op=[m0 + m1, m2])

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
        dev = qml.device("default.qubit", wires=2)

        @qml.set_shots(1000)
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
        dev = qml.device("default.qubit", wires=num_wires)

        @qml.set_shots([shots1, shots2, shots3])
        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.sample()

        res = circuit()

        assert isinstance(res, tuple)

        expected_shapes = [(shots1, num_wires), (shots2, num_wires), (shots3, num_wires)]
        assert len(res) == len(expected_shapes)
        assert all(r.shape == exp_shape for r, exp_shape in zip(res, expected_shapes))

        # assert first wire is always the same as second
        # pylint: disable=unsubscriptable-object
        assert np.all(res[0][:, 0] == res[0][:, 1])
        assert np.all(res[1][:, 0] == res[1][:, 1])
        assert np.all(res[2][:, 0] == res[2][:, 1])

    def test_providing_no_observable_and_wires(self):
        """Test that we can provide no observable but specify wires to the sample function"""
        wires = [0, 2]
        wires_obj = qml.wires.Wires(wires)
        dev = qml.device("default.qubit", wires=3)

        @qml.set_shots(1000)
        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            res = qml.sample(wires=wires)

            assert res.obs is None
            assert res.wires == wires_obj
            return res

        circuit()

    @pytest.mark.parametrize(
        "obs",
        [
            # Single observables
            (None),  # comp basis samples, expected to be int
            (qml.PauliX(0)),
            (qml.PauliY(0)),
            (qml.PauliZ(0)),
            (qml.Hadamard(0)),
            (qml.Identity(0)),
            (qml.Hermitian(np.diag([1, 2]), 0)),
            (qml.Hermitian(np.diag([1.0, 2.0]), 0)),
            # Tensor product observables
            (
                qml.PauliX("c")
                @ qml.PauliY("a")
                @ qml.PauliZ(1)
                @ qml.Hadamard("wire1")
                @ qml.Identity("b")
            ),
            (qml.Projector([0, 1], wires=[0, 1]) @ qml.PauliZ(2)),
            (qml.Hermitian(np.array(np.eye(2)), wires=[0]) @ qml.PauliZ(2)),
            (qml.Projector([0, 1], wires=[0, 1]) @ qml.Hermitian(np.array(np.eye(2)), wires=[2])),
        ],
    )
    def test_numeric_type(self, obs):
        """Test that the numeric type is correct."""
        eigval_type = type(obs.eigvals()[0]) if obs is not None else np.int64

        res = qml.sample(obs) if obs is not None else qml.sample()
        if res.numeric_type == int:
            expected_type = np.int64
        elif res.numeric_type == float:
            expected_type = np.float64
        elif res.numeric_type == complex:
            expected_type = np.complex64
        else:
            raise ValueError("unexpected numeric type for result")

        assert expected_type == eigval_type

    def test_shape_no_shots_error(self):
        """Test that the appropriate error is raised with no shots are specified"""
        mp = qml.sample()

        with pytest.raises(
            MeasurementShapeError, match="Shots are required to obtain the shape of the measurement"
        ):
            _ = mp.shape(shots=None)

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
        res = qml.sample(obs) if obs is not None else qml.sample()
        expected = (shots,) if obs is not None else (shots, 3)
        assert res.shape(10, 3) == expected

    @pytest.mark.parametrize("n_samples", (1, 10))
    def test_shape_wires(self, n_samples):
        """Test that the shape is correct when wires are provided."""
        mp = qml.sample(wires=(0, 1))
        assert mp.shape(n_samples, 3) == (n_samples, 2) if n_samples != 1 else (2,)

    def test_shape_shot_vector_obs(self):
        """Test that the shape is correct with the shot vector and a observable too."""
        shot_vec = (2, 2)
        dev = qml.device("default.qubit", wires=3)

        @qml.set_shots(shot_vec)
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

    def test_sample_empty_wires(self):
        """Test that using ``qml.sample`` with an empty wire list raises an error."""
        with pytest.raises(ValueError, match="Cannot set an empty list of wires."):
            qml.sample(wires=[])

    @pytest.mark.parametrize("shots", [2, 100])
    def test_sample_no_arguments(self, shots):
        """Test that using ``qml.sample`` with no arguments returns the samples of all wires."""
        dev = qml.device("default.qubit", wires=3)

        @qml.set_shots(shots)
        @qml.qnode(dev)
        def circuit():
            return qml.sample()

        res = circuit()

        # pylint: disable=comparison-with-callable
        assert res.shape == (shots, 3)

    def test_new_sample_with_operator_with_no_eigvals(self):
        """Test that calling process with an operator that has no eigvals defined raises an error."""

        class DummyOp(Operator):  # pylint: disable=too-few-public-methods
            """Dummy operator with no eigenvalues defined."""

            num_wires = 1

        with pytest.raises(EigvalsUndefinedError, match="Cannot compute samples of"):
            qml.sample(op=DummyOp(0)).process_samples(samples=np.array([[1, 0]]), wire_order=[0])

    @pytest.mark.parametrize(
        "coeffs, dtype",
        [
            (1, "int8"),
            (1, "int16"),
            (1, "int32"),
            (1, "int64"),
            (1, "float16"),
            (1, "float32"),
            (1, "float64"),
            (1j, "complex64"),
            (1 + 1j, "complex128"),
        ],
    )
    def test_process_samples_dtype(self, coeffs, dtype):
        """Test that the dtype argument changes the dtype of the returned samples."""
        samples = np.zeros(10, dtype="int64")
        processed_samples = qml.sample(coeffs * qml.X(0), dtype=dtype).process_samples(
            samples, wire_order=[0]
        )
        assert processed_samples.dtype == np.dtype(dtype)

    def test_sample_allowed_with_parameter_shift(self):
        """Test that qml.sample doesn't raise an error with parameter-shift and autograd."""
        dev = qml.device("default.qubit")

        @qml.set_shots(10)
        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(angle):
            qml.RX(angle, wires=0)
            return qml.sample(qml.PauliX(0))

        angle = qml.numpy.array(0.1)
        res = qml.jacobian(circuit)(angle)
        assert qml.math.shape(res) == (10,)
        assert all(r in {-1, 0, 1} for r in np.round(res, 13))

    @pytest.mark.jax
    def test_sample_with_jax_jacobian(self):
        """Test that qml.sample executes with parameter-shift and jax."""
        import jax

        dev = qml.device("default.qubit")

        @qml.set_shots(10)
        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(angle):
            qml.RX(angle, wires=0)
            return qml.sample(qml.PauliX(0))

        angle = jax.numpy.array(0.1)
        _ = jax.jacobian(circuit)(angle)

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("interface", ["autograd", "torch", "jax"])
    @pytest.mark.parametrize(
        "dtype, obs",
        [
            ("int8", None),
            ("int16", None),
            ("int32", None),
            ("int64", None),
            ("float16", qml.Z(0)),
            ("float32", qml.Z(0)),
            ("float64", qml.Z(0)),
        ],
    )
    def test_sample_dtype_combined(self, interface, dtype, obs):
        """Test that the dtype argument changes the dtype of the returned samples,
        both with and without an observable."""

        @qml.set_shots(10)
        @qml.qnode(device=qml.device("default.qubit", wires=1), interface=interface)
        def circuit():
            qml.Hadamard(wires=0)
            return qml.sample(obs, dtype=dtype) if obs is not None else qml.sample(dtype=dtype)

        samples = circuit()
        assert qml.math.get_interface(samples) == interface
        assert qml.math.get_dtype_name(samples) == dtype


@pytest.mark.jax
class TestJAXCompatibility:
    """Tests for JAX compatibility"""

    @pytest.mark.parametrize("samples", (1, 10))
    def test_jitting_with_sampling_on_subset_of_wires(self, samples):
        """Test case covering bug in Issue #3904.  Sampling should be jit-able
        when sampling occurs on a subset of wires. The bug was occurring due an improperly
        set shape method."""
        import jax

        jax.config.update("jax_enable_x64", True)

        dev = qml.device("default.qubit", wires=3)

        @qml.set_shots(samples)
        @qml.qnode(dev, interface="jax")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.sample(wires=(0, 1))

        results = jax.jit(circuit)(jax.numpy.array(0.123, dtype=jax.numpy.float64))

        expected = (samples, 2)
        assert results.shape == expected
        assert circuit._qfunc_output.shape(samples, 3) == (samples, 2) if samples != 1 else (2,)

    def test_sample_with_boolean_tracer(self):
        """Test that qml.sample can be used with Catalyst measurement values (Boolean tracer)."""
        import jax

        def fun(b):
            mp = qml.sample(b)

            assert mp.obs is None
            assert isinstance(mp.mv, jax.interpreters.partial_eval.DynamicJaxprTracer)
            assert mp.mv.dtype == bool
            assert mp.mv.shape == ()
            assert isinstance(mp.wires, qml.wires.Wires)
            assert mp.wires == ()

        jax.make_jaxpr(fun)(True)

    @pytest.mark.parametrize(
        "obs",
        [
            # Single observables
            (qml.PauliX(0)),
            (qml.PauliY(0)),
            (qml.PauliZ(0)),
            (qml.Hadamard(0)),
            (qml.Identity(0)),
        ],
    )
    def test_jitting_with_sampling_on_different_observables(self, obs):
        """Test that jitting works when sampling observables (using their eigvals) rather than returning raw samples"""
        import jax

        jax.config.update("jax_enable_x64", True)

        dev = qml.device("default.qubit", wires=5)

        @qml.set_shots(100)
        @qml.qnode(dev, interface="jax")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.sample(obs)

        results = jax.jit(circuit)(jax.numpy.array(0.123, dtype=jax.numpy.float64))

        assert results.dtype == jax.numpy.float64
        assert np.all([r in [1, -1] for r in results])

    @pytest.mark.parametrize(
        "dtype, obs",
        [
            ("int8", None),
            ("int16", None),
            ("int32", None),
            ("int64", None),
            ("float16", qml.Z(0)),
            ("float32", qml.Z(0)),
            ("float64", qml.Z(0)),
        ],
    )
    def test_jitting_with_dtype(self, dtype, obs):
        """Test that jitting works when the dtype argument is provided"""
        import jax

        @qml.set_shots(10)
        @qml.qnode(device=qml.device("default.qubit", wires=1), interface="jax")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.sample(obs, dtype=dtype) if obs is not None else qml.sample(dtype=dtype)

        samples = jax.jit(circuit)(jax.numpy.array(0.123))
        assert qml.math.get_interface(samples) == "jax"
        assert qml.math.get_dtype_name(samples) == dtype

    def test_process_samples_with_jax_tracer(self):
        """Test that qml.sample can be used when samples is a JAX Tracer"""

        import jax

        def f(samples):
            return qml.sample(op=2 * qml.X(0)).process_samples(
                samples, wire_order=qml.wires.Wires((0, 1))
            )

        samples = jax.numpy.zeros((10, 2), dtype=int)
        jax.jit(f)(samples)


class TestSampleProcessCounts:
    """Tests for the process_counts method in the SampleMP class."""

    def test_process_counts_multiple_wires(self):
        """Test process_counts method with multiple wires."""
        sample_mp = qml.sample(wires=[0, 1])
        counts = {"00": 2, "10": 3}
        wire_order = qml.wires.Wires((0, 1))

        result = sample_mp.process_counts(counts, wire_order)

        assert np.array_equal(result, np.array([[0, 0], [0, 0], [1, 0], [1, 0], [1, 0]]))

    def test_process_counts_single_wire(self):
        """Test process_counts method with a single wire."""
        sample_mp = qml.sample(wires=[0])
        counts = {"00": 2, "10": 3}
        wire_order = qml.wires.Wires((0, 1))

        result = sample_mp.process_counts(counts, wire_order)

        assert np.array_equal(result, np.array([0, 0, 1, 1, 1]))

    @pytest.mark.parametrize(
        "wire_order, expected_result", [((0, 1), [1, 1, -1, -1, -1]), ((1, 0), [1, 1, 1, 1, 1])]
    )
    def test_process_counts_with_eigen_values(self, wire_order, expected_result):
        """Test process_counts method with eigen values."""
        sample_mp = qml.sample(qml.Z(0))
        counts = {"00": 2, "10": 3}
        wire_order = qml.wires.Wires(wire_order)

        result = sample_mp.process_counts(counts, wire_order)

        assert np.array_equal(result, np.array(expected_result))

    @pytest.mark.parametrize(
        "wire_order, expected_result",
        [
            ((0, 1, 2), [1, -1, -1, 1, 1]),
            ((0, 2, 1), [1, 1, -1, -1, -1]),
            ((1, 2, 0), [1, 1, -1, -1, -1]),
            ((2, 0, 1), [1, -1, 1, -1, -1]),
        ],
    )
    def test_process_counts_with_eigen_values_multiple_wires(self, wire_order, expected_result):
        """Test process_counts method with eigen values."""
        sample_mp = qml.sample(qml.Z(0) @ qml.Z(1))
        counts = {"000": 1, "101": 1, "011": 1, "110": 2}
        wire_order = qml.wires.Wires(wire_order)

        result = sample_mp.process_counts(counts, wire_order)

        assert np.array_equal(result, np.array(expected_result))

    def test_process_counts_with_inverted_wire_order(self):
        """Test process_counts method with inverted wire order."""
        sample_mp = qml.sample(wires=[0, 1])
        counts = {"00": 2, "01": 3}
        wire_order = qml.wires.Wires((1, 0))

        result = sample_mp.process_counts(counts, wire_order)

        assert np.array_equal(result, np.array([[0, 0], [0, 0], [1, 0], [1, 0], [1, 0]]))

    def test_process_counts_with_second_single_wire(self):
        """Test process_counts method with the second single wire."""
        sample_mp = qml.sample(wires=[1])
        counts = {"00": 2, "10": 3}
        wire_order = qml.wires.Wires((0, 1))

        result = sample_mp.process_counts(counts, wire_order)

        assert np.array_equal(result, np.array([0, 0, 0, 0, 0]))
