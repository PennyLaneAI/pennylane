# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the QuantumScript"""
import copy
import warnings
from collections import defaultdict

import numpy as np
import pytest
from this import d

import pennylane as qml
from pennylane import CircuitGraph
from pennylane.measurements import (
    MeasurementProcess,
    MeasurementShapeError,
    counts,
    expval,
    sample,
    var,
    probs,
)
from pennylane.tape import QuantumScript


measures = [
    (qml.expval(qml.PauliZ(0)), ()),
    (qml.var(qml.PauliZ(0)), ()),
    (qml.probs(wires=[0]), (2,)),
    (qml.probs(wires=[0, 1]), (4,)),
    (qml.state(), (8,)),  # Assumes 3-qubit device
    (qml.density_matrix(wires=[0, 1]), (4, 4)),
    (
        qml.sample(qml.PauliZ(0)),
        None,
    ),  # Shape is None because the expected shape is in the test case
    (qml.sample(), None),  # Shape is None because the expected shape is in the test case
]

multi_measurements = [
    ([qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))], ((), ())),
    ([qml.probs(wires=[0]), qml.probs(wires=[1])], ((2,), (2,))),
    ([qml.probs(wires=[0]), qml.probs(wires=[1, 2])], ((2,), (4,))),
    ([qml.probs(wires=[0, 2]), qml.probs(wires=[1])], ((4,), (2,))),
    (
        [qml.probs(wires=[0]), qml.probs(wires=[1, 2]), qml.probs(wires=[0, 1, 2])],
        ((2,), (4,), (8,)),
    ),
]


class TestMeasurementProcess:
    """Tests for the shape and numeric type of a measurement process"""

    measurements_no_shots = [
        (qml.expval(qml.PauliZ(0)), ()),
        (qml.var(qml.PauliZ(0)), ()),
        (qml.probs(wires=[0, 1]), (4,)),
        (qml.state(), (8,)),
        (qml.density_matrix(wires=[0, 1]), (4, 4)),
    ]

    measurements_finite_shots = [
        (qml.expval(qml.PauliZ(0)), ()),
        (qml.var(qml.PauliZ(0)), ()),
        (qml.probs(wires=[0, 1]), (4,)),
        (qml.state(), (8,)),
        (qml.density_matrix(wires=[0, 1]), (4, 4)),
        (qml.sample(qml.PauliZ(0)), (10,)),
        (qml.sample(), (10, 3)),
    ]

    measurements_shot_vector = [
        (qml.expval(qml.PauliZ(0)), ((), (), ())),
        (qml.var(qml.PauliZ(0)), ((), (), ())),
        (qml.probs(wires=[0, 1]), ((4,), (4,), (4,))),
        (qml.state(), ((8,), (8,), (8,))),
        (qml.density_matrix(wires=[0, 1]), ((4, 4), (4, 4), (4, 4))),
        (qml.sample(qml.PauliZ(0)), ((10,), (20,), (30,))),
        (qml.sample(), ((10, 3), (20, 3), (30, 3))),
    ]

    @pytest.mark.parametrize("measurement, expected_shape", measurements_no_shots)
    def test_output_shapes_no_shots(self, measurement, expected_shape):
        """Test that the output shape of the measurement process is expected
        when shots=None"""
        num_wires = 3
        dev = qml.device("default.qubit", wires=num_wires, shots=None)

        assert measurement.shape(dev) == expected_shape

    @pytest.mark.parametrize("measurement, expected_shape", measurements_finite_shots)
    def test_output_shapes_finite_shots(self, measurement, expected_shape):
        """Test that the output shape of the measurement process is expected
        when shots is finite"""
        num_wires = 3
        num_shots = 10
        dev = qml.device("default.qubit", wires=num_wires, shots=num_shots)

        assert measurement.shape(dev) == expected_shape

    @pytest.mark.parametrize("measurement, expected_shape", measurements_shot_vector)
    def test_output_shapes_no_shots(self, measurement, expected_shape):
        """Test that the output shape of the measurement process is expected
        when shots is a vector"""
        num_wires = 3
        shot_vector = [10, 20, 30]
        dev = qml.device("default.qubit", wires=num_wires, shots=shot_vector)

        assert measurement.shape(dev) == expected_shape

    @pytest.mark.parametrize("measurement", [qml.probs(wires=[0, 1]), qml.state(), qml.sample()])
    def test_no_device_error(self, measurement):
        """Test that an error is raised when a measurement that requires a device
        is called without a device"""
        msg = "The device argument is required to obtain the shape of the measurement process"

        with pytest.raises(MeasurementShapeError, match=msg):
            measurement.shape()

    def test_undefined_shape_error(self):
        """Test that an error is raised for a measurement with an undefined shape"""
        measurement = qml.counts(wires=[0, 1])
        msg = "Cannot deduce the shape of the measurement process with unrecognized return_type"

        with pytest.raises(qml.QuantumFunctionError, match=msg):
            measurement.shape()


class TestOutputShape:
    """Tests for determining the tape output shape of tapes."""

    @pytest.mark.parametrize("measurement, expected_shape", measures)
    @pytest.mark.parametrize("shots", [None, 1, 10])
    def test_output_shapes_single(self, measurement, expected_shape, shots):
        """Test that the output shape produced by the tape matches the expected
        output shape."""
        if shots is None and measurement.return_type is qml.measurements.Sample:
            pytest.skip("Sample doesn't support analytic computations.")

        num_wires = 3
        dev = qml.device("default.qubit", wires=num_wires, shots=shots)

        a = np.array(0.1)
        b = np.array(0.2)

        ops = [qml.RY(a, 0), qml.RX(b, 0)]
        qs = QuantumScript(ops, [measurement])

        shot_dim = len(shots) if isinstance(shots, tuple) else shots
        if expected_shape is None:
            expected_shape = shot_dim if shot_dim == 1 else (shot_dim,)

        if measurement.return_type is qml.measurements.Sample:
            if measurement.obs is None:
                expected_shape = (num_wires,) if shots == 1 else (shots, num_wires)

            else:
                expected_shape = () if shots == 1 else (shots,)
        assert qs.shape(dev) == expected_shape

    @pytest.mark.parametrize("measurement, expected_shape", measures)
    @pytest.mark.parametrize("shots", [None, 1, 10, (1, 2, 5, 3)])
    def test_output_shapes_single_qnode_check(self, measurement, expected_shape, shots):
        """Test that the output shape produced by the tape matches the output
        shape of a QNode for a single measurement."""
        if shots is None and measurement.return_type is qml.measurements.Sample:
            pytest.skip("Sample doesn't support analytic computations.")

        dev = qml.device("default.qubit", wires=3, shots=shots)

        a = np.array(0.1)
        b = np.array(0.2)

        ops = [qml.RY(a, 0), qml.RX(b, 0)]
        qs = QuantumScript(ops, [measurement])

        if shots is not None and measurement.return_type is qml.measurements.State:
            # this is allowed by the tape but raises a warning
            with pytest.warns(
                UserWarning, match="Requested state or density matrix with finite shots"
            ):
                res = qml.execute([qs], dev, gradient_fn=None)[0]
        else:
            # TODO: test gradient_fn is not None when the interface `execute` functions are implemented
            res = qml.execute([qs], dev, gradient_fn=None)[0]

        if isinstance(shots, tuple):
            res_shape = tuple(r.shape for r in res)
        else:
            res_shape = res.shape

        assert qs.shape(dev) == res_shape

    def test_output_shapes_single_qnode_check_cutoff(self):
        """Test that the tape output shape is correct when computing
        probabilities with a dummy device that defines a cutoff value."""

        class CustomDevice(qml.QubitDevice):
            """A dummy device that has a cutoff value specified and returns
            analytic probabilities in a fashion similar to the
            strawberryfields.fock device.

            Note: this device definition is used as PennyLane-SF is not a
            dependency of PennyLane core and there are no CV device in
            PennyLane core using a cutoff value.
            """

            name = "Device with cutoff"
            short_name = "dummy.device"
            pennylane_requires = "0.1.0"
            version = "0.0.1"
            author = "CV quantum"

            operations = {}
            observables = {"Identity"}

            def __init__(self, shots=None, wires=None, cutoff=None):
                super().__init__(wires=wires, shots=shots)
                self.cutoff = cutoff

            def apply(self, operations, **kwargs):
                pass

            def analytic_probability(self, wires=None):
                if wires is None:
                    wires = self.wires
                return np.zeros(self.cutoff ** len(wires))

        dev = CustomDevice(wires=2, cutoff=13)

        # If PennyLane-SF is installed, the following can be checked e.g., locally:
        # dev = qml.device("strawberryfields.fock", wires=2, cutoff_dim=13)

        qs = QuantumScript(measurements=[qml.probs(wires=[0])])

        res_shape = qml.execute([qs], dev, gradient_fn=None)[0]
        assert qs.shape(dev) == res_shape.shape

    @pytest.mark.autograd
    @pytest.mark.parametrize("measurements, expected", multi_measurements)
    @pytest.mark.parametrize("shots", [None, 1, 10])
    def test_multi_measure(self, measurements, expected, shots):
        """Test that the expected output shape is obtained when using multiple
        expectation value, variance and probability measurements."""
        dev = qml.device("default.qubit", wires=3, shots=shots)

        qs = QuantumScript(measurements=measurements)

        if measurements[0].return_type is qml.measurements.Sample:
            expected[1] = shots
            expected = tuple(expected)

        res = qs.shape(dev)
        assert res == expected

        # TODO: test gradient_fn is not None when the interface `execute` functions are implemented
        res = qml.execute([qs], dev, gradient_fn=None)[0]
        res_shape = tuple(r.shape for r in res)

        assert res_shape == expected

    @pytest.mark.autograd
    @pytest.mark.parametrize("measurements, expected", multi_measurements)
    def test_multi_measure_shot_vector(self, measurements, expected):
        """Test that the expected output shape is obtained when using multiple
        expectation value, variance and probability measurements with a shot
        vector."""
        if measurements[0].return_type is qml.measurements.Probability:
            num_wires = set(len(m.wires) for m in measurements)
            if len(num_wires) > 1:
                pytest.skip(
                    "Multi-probs with varying number of varies when using a shot vector is to be updated in PennyLane."
                )

        shots = (1, 1, 3, 3, 5, 1)
        dev = qml.device("default.qubit", wires=3, shots=shots)

        a = np.array(0.1)
        b = np.array(0.2)
        ops = [qml.RY(a, 0), qml.RX(b, 0)]
        qs = QuantumScript(ops, measurements)

        # Update expected as we're using a shotvector
        expected = tuple(expected for _ in shots)
        res = qs.shape(dev)
        assert res == expected

        # TODO: test gradient_fn is not None when the interface `execute` functions are implemented
        res = qml.execute([qs], dev, gradient_fn=None)[0]
        res_shape = tuple(tuple(r_.shape for r_ in r) for r in res)

        assert res_shape == expected

    @pytest.mark.autograd
    @pytest.mark.parametrize("shots", [1, 10])
    def test_multi_measure_sample(self, shots):
        """Test that the expected output shape is obtained when using multiple
        qml.sample measurements."""
        dev = qml.device("default.qubit", wires=3, shots=shots)

        a = np.array(0.1)
        b = np.array(0.2)

        num_samples = 3
        ops = [qml.RY(a, 0), qml.RX(b, 0)]
        qs = QuantumScript(ops, [qml.sample(qml.PauliZ(i)) for i in range(num_samples)])

        expected = tuple(() if shots == 1 else (shots,) for _ in range(num_samples))

        res = qs.shape(dev)
        assert res == expected

        res = qml.execute([qs], dev, gradient_fn=None)[0]
        res_shape = tuple(r.shape for r in res)

        assert res_shape == expected

    @pytest.mark.autograd
    def test_multi_measure_sample_obs_shot_vector(self):
        """Test that the expected output shape is obtained when using multiple
        qml.sample measurements with an observable with a shot vector."""
        shots = (1, 1, 3, 3, 5, 1)
        dev = qml.device("default.qubit", wires=3, shots=shots)

        a = np.array(0.1)
        b = np.array(0.2)

        num_samples = 3
        ops = [qml.RY(a, 0), qml.RX(b, 0)]
        qs = QuantumScript(ops, [qml.sample(qml.PauliZ(i)) for i in range(num_samples)])

        expected = tuple(tuple(() if s == 1 else (s,) for _ in range(num_samples)) for s in shots)

        res = qs.shape(dev)
        assert res == expected

        expected = qml.execute([qs], dev, gradient_fn=None)[0]
        expected_shape = tuple(tuple(e_.shape for e_ in e) for e in expected)

        assert res == expected_shape

    @pytest.mark.autograd
    def test_multi_measure_sample_wires_shot_vector(self):
        """Test that the expected output shape is obtained when using multiple
        qml.sample measurements with wires with a shot vector."""
        shots = (1, 1, 3, 3, 5, 1)
        dev = qml.device("default.qubit", wires=3, shots=shots)

        num_samples = 3
        ops = [qml.RY(0.3, 0), qml.RX(0.2, 0)]
        qs = QuantumScript(ops, [qml.sample()] * num_samples)

        expected = tuple(
            tuple((3,) if s == 1 else (s, 3) for _ in range(num_samples)) for s in shots
        )

        res = qs.shape(dev)
        assert res == expected

        expected = qml.execute([qs], dev, gradient_fn=None)[0]
        expected_shape = tuple(tuple(e_.shape for e_ in e) for e in expected)

        assert res == expected_shape


class TestNumericType:
    """Tests for determining the numeric type of the tape output."""

    @pytest.mark.parametrize(
        "ret", [qml.expval(qml.PauliZ(0)), qml.var(qml.PauliZ(0)), qml.probs(wires=[0])]
    )
    @pytest.mark.parametrize("shots", [None, 1, (1, 2, 3)])
    def test_float_measures(self, ret, shots):
        """Test that most measurements output floating point values and that
        the tape output domain correctly identifies this."""
        dev = qml.device("default.qubit", wires=3, shots=shots)

        a, b = 0.3, 0.2
        ops = [qml.RY(a, 0), qml.RZ(b, 0)]
        qs = QuantumScript(ops, [ret])

        result = qml.execute([qs], dev, gradient_fn=None)[0]
        if not isinstance(result, tuple):
            result = (result,)

        # Double-check the domain of the QNode output
        assert all(np.issubdtype(res.dtype, float) for res in result)
        assert qs.numeric_type is float

    @pytest.mark.parametrize(
        "ret", [qml.state(), qml.density_matrix(wires=[0, 1]), qml.density_matrix(wires=[2, 0])]
    )
    def test_complex_state(self, ret):
        """Test that a tape with qml.state correctly determines that the output
        domain will be complex."""
        dev = qml.device("default.qubit", wires=3)

        a, b = 0.3, 0.2
        ops = [qml.RY(a, 0), qml.RZ(b, 0)]
        qs = QuantumScript(ops, [ret])

        result = qml.execute([qs], dev, gradient_fn=None)[0]

        # Double-check the domain of the QNode output
        assert np.issubdtype(result.dtype, complex)
        assert qs.numeric_type is complex

    @pytest.mark.parametrize("ret", [qml.sample(), qml.sample(qml.PauliZ(wires=0))])
    def test_sample_int_eigvals(self, ret):
        """Test that the tape can correctly determine the output domain for a
        sampling measurement with a Hermitian observable with integer
        eigenvalues."""
        dev = qml.device("default.qubit", wires=3, shots=5)

        arr = np.array(
            [
                1.32,
                2.312,
            ]
        )
        herm = np.outer(arr, arr)
        qs = QuantumScript([qml.RY(0.4, 0)], [ret])

        result = qml.execute([qs], dev, gradient_fn=None)[0]

        # Double-check the domain of the QNode output
        assert np.issubdtype(result.dtype, np.int64)
        assert qs.numeric_type is int

    # TODO: add cases for each interface once qml.Hermitian supports other
    # interfaces
    def test_sample_real_eigvals(self):
        """Test that the tape can correctly determine the output domain when
        sampling a Hermitian observable with real eigenvalues."""
        dev = qml.device("default.qubit", wires=3, shots=5)

        arr = np.array(
            [
                1.32,
                2.312,
            ]
        )
        herm = np.outer(arr, arr)

        qs = QuantumScript([qml.RY(0.4, 0)], [qml.sample(qml.Hermitian(herm, wires=0))])

        result = qml.execute([qs], dev, gradient_fn=None)[0]

        # Double-check the domain of the QNode output
        assert np.issubdtype(result.dtype, float)
        assert qs.numeric_type is float

    @pytest.mark.autograd
    def test_sample_real_and_int_eigvals(self):
        """Test that the tape can correctly determine the output domain for
        multiple sampling measurements with a Hermitian observable with real
        eigenvalues and another one with integer eigenvalues."""
        dev = qml.device("default.qubit", wires=3, shots=5)

        arr = np.array(
            [
                1.32,
                2.312,
            ]
        )
        herm = np.outer(arr, arr)

        a, b = 0, 3
        ops = [qml.RY(a, 0), qml.RX(b, 0)]
        m = [qml.sample(qml.Hermitian(herm, wires=0)), qml.sample(qml.PauliZ(1))]
        qs = QuantumScript(ops, m)

        result = qml.execute([qs], dev, gradient_fn=None)[0]

        # Double-check the domain of the QNode output
        assert np.issubdtype(result[0].dtype, float)
        assert np.issubdtype(result[1].dtype, np.int64)
        assert qs.numeric_type == (float, int)
