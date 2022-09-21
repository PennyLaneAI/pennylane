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
"""Unit tests for the QuantumTape"""
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
from pennylane.tape import QuantumTape, TapeError


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

        with qml.tape.QuantumTape() as tape:
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            qml.apply(measurement)

        shot_dim = shots if not isinstance(shots, tuple) else len(shots)
        if expected_shape is None:
            expected_shape = shot_dim if shot_dim == 1 else (shot_dim,)

        if measurement.return_type is qml.measurements.Sample:
            if measurement.obs is not None:
                expected_shape = () if shots == 1 else (shots,)
            else:
                expected_shape = (num_wires,) if shots == 1 else (shots, num_wires)

        assert tape.shape(dev) == expected_shape

    @pytest.mark.parametrize("measurement, expected_shape", measures)
    @pytest.mark.parametrize("shots", [None, 1, 10, (1, 2, 5, 3)])
    def test_output_shapes_single_qnode_check(self, measurement, expected_shape, shots):
        """Test that the output shape produced by the tape matches the output
        shape of a QNode for a single measurement."""
        if shots is None and measurement.return_type is qml.measurements.Sample:
            pytest.skip("Sample doesn't support analytic computations.")

        if shots is not None and measurement.return_type is qml.measurements.State:
            pytest.skip("State and density matrix don't support finite shots and raise a warning.")

        # TODO: revisit when qml.sample without an observable has been updated
        # with shot vectors
        if (
            isinstance(shots, tuple)
            and measurement.return_type is qml.measurements.Sample
            and not measurement.obs
        ):
            pytest.skip("qml.sample with no observable is to be updated for shot vectors.")

        dev = qml.device("default.qubit", wires=3, shots=shots)

        a = np.array(0.1)
        b = np.array(0.2)

        with qml.tape.QuantumTape() as tape:
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            qml.apply(measurement)

        # TODO: test gradient_fn is not None when the interface `execute` functions are implemented
        res = qml.execute([tape], dev, gradient_fn=None)[0]

        if isinstance(shots, tuple):
            res_shape = tuple(r.shape for r in res)
        else:
            res_shape = res.shape

        assert tape.shape(dev) == res_shape

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

        a = np.array(0.1)
        b = np.array(0.2)

        with qml.tape.QuantumTape() as tape:
            qml.probs(wires=[0])

        @qml.qnode(dev)
        def circuit(a, b):
            return qml.probs(wires=[0])

        res_shape = qml.execute([tape], dev, gradient_fn=None)[0]
        assert tape.shape(dev) == res_shape.shape

    @pytest.mark.autograd
    @pytest.mark.parametrize("measurements, expected", multi_measurements)
    @pytest.mark.parametrize("shots", [None, 1, 10])
    def test_multi_measure(self, measurements, expected, shots):
        """Test that the expected output shape is obtained when using multiple
        expectation value, variance and probability measurements."""
        dev = qml.device("default.qubit", wires=3, shots=shots)

        a = np.array(0.1)
        b = np.array(0.2)

        with qml.tape.QuantumTape() as tape:
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            for m in measurements:
                qml.apply(m)

        if measurements[0].return_type is qml.measurements.Sample:
            expected[1] = shots
            expected = tuple(expected)

        res = tape.shape(dev)
        assert res == expected

        # TODO: test gradient_fn is not None when the interface `execute` functions are implemented
        res = qml.execute([tape], dev, gradient_fn=None)[0]
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

        with qml.tape.QuantumTape() as tape:
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            for m in measurements:
                qml.apply(m)

        # Update expected as we're using a shotvector
        expected = tuple(expected for _ in shots)
        res = tape.shape(dev)
        assert res == expected

        # TODO: test gradient_fn is not None when the interface `execute` functions are implemented
        res = qml.execute([tape], dev, gradient_fn=None)[0]
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
        with qml.tape.QuantumTape() as tape:
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            for i in range(num_samples):
                qml.sample(qml.PauliZ(i))

        expected = tuple(() if shots == 1 else (shots,) for _ in range(num_samples))

        res = tape.shape(dev)
        assert res == expected

        res = qml.execute([tape], dev, gradient_fn=None)[0]
        res_shape = tuple(r.shape for r in res)

        assert res_shape == expected

    @pytest.mark.autograd
    def test_multi_measure_sample_shot_vector(self):
        """Test that the expected output shape is obtained when using multiple
        qml.sample measurements with an observable with a shot vector."""
        shots = (1, 1, 3, 3, 5, 1)
        dev = qml.device("default.qubit", wires=3, shots=shots)

        a = np.array(0.1)
        b = np.array(0.2)

        num_samples = 3
        with qml.tape.QuantumTape() as tape:
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            for i in range(num_samples):
                qml.sample(qml.PauliZ(i))

        expected = tuple(tuple(() if s == 1 else (s,) for _ in range(num_samples)) for s in shots)

        res = tape.shape(dev)
        assert res == expected

        expected = qml.execute([tape], dev, gradient_fn=None)[0]
        expected_shape = tuple(tuple(e_.shape for e_ in e) for e in expected)

        assert res == expected_shape

    @pytest.mark.autograd
    def test_multi_measure_sample_shot_vector(self):
        """Test that the expected output shape is obtained when using multiple
        qml.sample measurements with wires with a shot vector."""
        shots = (1, 1, 3, 3, 5, 1)
        dev = qml.device("default.qubit", wires=3, shots=shots)

        num_samples = 3
        with qml.tape.QuantumTape() as tape:
            qml.RY(0.3, wires=0)
            qml.RX(0.2, wires=0)
            for i in range(num_samples):
                qml.sample()

        expected = tuple(
            tuple((3,) if s == 1 else (s, 3) for _ in range(num_samples)) for s in shots
        )

        res = tape.shape(dev)
        assert res == expected

        expected = qml.execute([tape], dev, gradient_fn=None)[0]
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

        with qml.tape.QuantumTape() as tape:
            qml.RY(a, wires=[0])
            qml.RZ(b, wires=[0])
            qml.apply(ret)

        result = qml.execute([tape], dev, gradient_fn=None)[0]
        if not isinstance(result, tuple):
            result = (result,)

        # Double-check the domain of the QNode output
        assert all(np.issubdtype(res.dtype, float) for res in result)
        assert tape.numeric_type is float

    @pytest.mark.parametrize(
        "ret", [qml.state(), qml.density_matrix(wires=[0, 1]), qml.density_matrix(wires=[2, 0])]
    )
    def test_complex_state(self, ret):
        """Test that a tape with qml.state correctly determines that the output
        domain will be complex."""
        dev = qml.device("default.qubit", wires=3)

        a, b = 0.3, 0.2

        with qml.tape.QuantumTape() as tape:
            qml.RY(a, wires=[0])
            qml.RZ(b, wires=[0])
            qml.apply(ret)

        result = qml.execute([tape], dev, gradient_fn=None)[0]

        # Double-check the domain of the QNode output
        assert np.issubdtype(result.dtype, complex)
        assert tape.numeric_type is complex

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

        with qml.tape.QuantumTape() as tape:
            qml.RY(0.4, wires=[0])
            qml.apply(ret)

        result = qml.execute([tape], dev, gradient_fn=None)[0]

        # Double-check the domain of the QNode output
        assert np.issubdtype(result.dtype, np.int64)
        assert tape.numeric_type is int

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

        with qml.tape.QuantumTape() as tape:
            qml.RY(0.4, wires=[0])
            qml.sample(qml.Hermitian(herm, wires=0))

        result = qml.execute([tape], dev, gradient_fn=None)[0]

        # Double-check the domain of the QNode output
        assert np.issubdtype(result.dtype, float)
        assert tape.numeric_type is float

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
        with qml.tape.QuantumTape() as tape:
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            qml.sample(qml.Hermitian(herm, wires=0)), qml.sample(qml.PauliZ(1))

        result = qml.execute([tape], dev, gradient_fn=None)[0]

        # Double-check the domain of the QNode output
        assert np.issubdtype(result[0].dtype, float)
        assert np.issubdtype(result[1].dtype, np.int64)
        assert tape.numeric_type == (float, int)
