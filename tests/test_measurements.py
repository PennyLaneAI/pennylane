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
"""Unit tests for the measure module"""
from _pytest.nodes import Node
import pytest
import numpy as np

import pennylane as qml
from pennylane import numpy as pnp
from pennylane.devices import DefaultQubit
from pennylane.operation import DecompositionUndefinedError

from pennylane.queuing import AnnotatedQueue
from pennylane.measurements import (
    expval,
    var,
    sample,
    probs,
    state,
    density_matrix,
    Expectation,
    Sample,
    State,
    Variance,
    Probability,
    MeasurementProcess,
    MeasurementValue,
    MeasurementValueError,
)


def test_no_measure(tol):
    """Test that failing to specify a measurement
    raises an exception"""
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def circuit(x):
        qml.RX(x, wires=0)
        return qml.PauliY(0)

    with pytest.raises(qml.QuantumFunctionError, match="must return either a single measurement"):
        res = circuit(0.65)


class TestExpval:
    """Tests for the expval function"""

    def test_value(self, tol):
        """Test that the expval interface works"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliY(0))

        x = 0.54
        res = circuit(x)
        expected = -np.sin(x)

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_not_an_observable(self):
        """Test that a qml.QuantumFunctionError is raised if the provided
        argument is not an observable"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.52, wires=0)
            return qml.expval(qml.CNOT(wires=[0, 1]))

        with pytest.raises(qml.QuantumFunctionError, match="CNOT is not an observable"):
            res = circuit()

    def test_observable_return_type_is_expectation(self):
        """Test that the return type of the observable is :attr:`ObservableReturnTypes.Expectation`"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            res = qml.expval(qml.PauliZ(0))
            assert res.return_type is Expectation
            return res

        circuit()


class TestVar:
    """Tests for the var function"""

    def test_value(self, tol):
        """Test that the var function works"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.var(qml.PauliZ(0))

        x = 0.54
        res = circuit(x)
        expected = np.sin(x) ** 2

        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_not_an_observable(self):
        """Test that a qml.QuantumFunctionError is raised if the provided
        argument is not an observable"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.52, wires=0)
            return qml.var(qml.CNOT(wires=[0, 1]))

        with pytest.raises(qml.QuantumFunctionError, match="CNOT is not an observable"):
            res = circuit()

    def test_observable_return_type_is_variance(self):
        """Test that the return type of the observable is :attr:`ObservableReturnTypes.Variance`"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            res = qml.var(qml.PauliZ(0))
            assert res.return_type is Variance
            return res

        circuit()


class TestSample:
    """Tests for the sample function"""

    def test_sample_dimension(self, tol):
        """Test that the sample function outputs samples of the right size"""
        n_sample = 10

        dev = qml.device("default.qubit", wires=2, shots=n_sample)

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.54, wires=0)
            return qml.sample(qml.PauliZ(0)), qml.sample(qml.PauliX(1))

        sample = circuit()

        assert np.array_equal(sample.shape, (2, n_sample))

    def test_sample_combination(self, tol):
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

    def test_single_wire_sample(self, tol):
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

    def test_multi_wire_sample_regular_shape(self, tol):
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

    def test_sample_output_type_in_combination(self, tol):
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
        """Test that a qml.QuantumFunctionError is raised if the provided
        argument is not an observable"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.52, wires=0)
            return qml.sample(qml.CNOT(wires=[0, 1]))

        with pytest.raises(qml.QuantumFunctionError, match="CNOT is not an observable"):
            sample = circuit()

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
            res = circuit()

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


class TestMeasure:
    """Tests for the measure function"""

    def test_many_wires_error(self):
        """Test that an error is raised if multiple wires are passed to
        measure."""
        with pytest.raises(
            qml.QuantumFunctionError,
            match="Only a single qubit can be measured in the middle of the circuit",
        ):
            qml.measure(wires=[0, 1])


class TestMeasurementValue:
    """Tests for the MeasurementValue class"""

    @pytest.mark.parametrize("val_pair", [(0, 1), (1, 0), (-1, 1)])
    @pytest.mark.parametrize("control_val_idx", [0, 1])
    def test_measurement_value_assertion(self, val_pair, control_val_idx):
        """Test that asserting the value of a measurement works well."""
        zero_case = val_pair[0]
        one_case = val_pair[1]
        mv = MeasurementValue(measurement_id="1234", zero_case=zero_case, one_case=one_case)
        mv == val_pair[control_val_idx]
        assert mv._control_value == val_pair[control_val_idx]

    @pytest.mark.parametrize("val_pair", [(0, 1), (1, 0), (-1, 1)])
    @pytest.mark.parametrize("num_inv, expected_idx", [(1, 0), (2, 1), (3, 0)])
    def test_measurement_value_inversion(self, val_pair, num_inv, expected_idx):
        """Test that inverting the value of a measurement works well even with
        multiple inversions.

        Double-inversion should leave the control value of the measurement
        value in place.
        """
        zero_case = val_pair[0]
        one_case = val_pair[1]
        mv = MeasurementValue(measurement_id="1234", zero_case=zero_case, one_case=one_case)
        for _ in range(num_inv):
            mv = mv.__invert__()

        assert mv._control_value == val_pair[expected_idx]

    def test_measurement_value_assertion_error(self):
        """Test that the return_type related info is updated for a
        measurement."""
        mv = MeasurementValue(measurement_id="1234")

        with pytest.raises(MeasurementValueError, match="Unknown measurement value asserted"):
            mv == -1


@pytest.mark.parametrize(
    "stat_func,return_type", [(expval, Expectation), (var, Variance), (sample, Sample)]
)
class TestStatisticsQueuing:
    """Tests for annotating the return types of the statistics functions"""

    @pytest.mark.parametrize(
        "op",
        [qml.PauliX, qml.PauliY, qml.PauliZ, qml.Hadamard, qml.Identity],
    )
    def test_annotating_obs_return_type(self, stat_func, return_type, op):
        """Test that the return_type related info is updated for a
        measurement"""
        with AnnotatedQueue() as q:
            A = op(0)
            stat_func(A)

        assert q.queue[:-1] == [A]
        meas_proc = q.queue[-1]
        assert isinstance(meas_proc, MeasurementProcess)
        assert meas_proc.return_type == return_type

        assert q._get_info(A) == {"owner": meas_proc}
        assert q._get_info(meas_proc) == {"owns": (A)}

    def test_annotating_tensor_hermitian(self, stat_func, return_type):
        """Test that the return_type related info is updated for a measurement
        when called for an Hermitian observable"""

        mx = np.array([[1, 0], [0, 1]])

        with AnnotatedQueue() as q:
            Herm = qml.Hermitian(mx, wires=[1])
            stat_func(Herm)

        assert q.queue[:-1] == [Herm]
        meas_proc = q.queue[-1]
        assert isinstance(meas_proc, MeasurementProcess)
        assert meas_proc.return_type == return_type

        assert q._get_info(Herm) == {"owner": meas_proc}
        assert q._get_info(meas_proc) == {"owns": (Herm)}

    @pytest.mark.parametrize(
        "op1,op2",
        [
            (qml.PauliY, qml.PauliX),
            (qml.Hadamard, qml.Hadamard),
            (qml.PauliY, qml.Identity),
            (qml.Identity, qml.Identity),
        ],
    )
    def test_annotating_tensor_return_type(self, op1, op2, stat_func, return_type):
        """Test that the return_type related info is updated for a measurement
        when called for an Tensor observable"""
        with AnnotatedQueue() as q:
            A = op1(0)
            B = op2(1)
            tensor_op = A @ B
            stat_func(tensor_op)

        assert q.queue[:-1] == [A, B, tensor_op]
        meas_proc = q.queue[-1]
        assert isinstance(meas_proc, MeasurementProcess)
        assert meas_proc.return_type == return_type

        assert q._get_info(A) == {"owner": tensor_op}
        assert q._get_info(B) == {"owner": tensor_op}
        assert q._get_info(tensor_op) == {"owns": (A, B), "owner": meas_proc}

    @pytest.mark.parametrize(
        "op1,op2",
        [
            (qml.PauliY, qml.PauliX),
            (qml.Hadamard, qml.Hadamard),
            (qml.PauliY, qml.Identity),
            (qml.Identity, qml.Identity),
        ],
    )
    def test_queueing_tensor_observable(self, op1, op2, stat_func, return_type):
        """Test that if the constituent components of a tensor operation are not
        found in the queue for annotation, that they are queued first and then annotated."""
        A = op1(0)
        B = op2(1)

        with AnnotatedQueue() as q:
            tensor_op = A @ B
            stat_func(tensor_op)

        assert q.queue[:-1] == [A, B, tensor_op]
        meas_proc = q.queue[-1]
        assert isinstance(meas_proc, MeasurementProcess)
        assert meas_proc.return_type == return_type

        assert q._get_info(A) == {"owner": tensor_op}
        assert q._get_info(B) == {"owner": tensor_op}
        assert q._get_info(tensor_op) == {"owns": (A, B), "owner": meas_proc}


@pytest.mark.parametrize("stat_func", [expval, var, sample])
class TestBetaStatisticsError:
    """Tests for errors arising for the beta statistics functions"""

    def test_not_an_observable(self, stat_func):
        """Test that a qml.QuantumFunctionError is raised if the provided
        argument is not an observable"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.RX(0.52, wires=0)
            return stat_func(qml.CNOT(wires=[0, 1]))

        with pytest.raises(qml.QuantumFunctionError, match="CNOT is not an observable"):
            res = circuit()


class TestBetaProbs:
    """Tests for annotating the return types of the probs function"""

    @pytest.mark.parametrize("wires", [[0], [0, 1], [1, 0, 2]])
    def test_annotating_probs(self, wires):
        with AnnotatedQueue() as q:
            probs(wires)

        assert len(q.queue) == 1

        meas_proc = q.queue[0]
        assert isinstance(meas_proc, MeasurementProcess)
        assert meas_proc.return_type == Probability


class TestProperties:
    """Test for the properties"""

    def test_wires_match_observable(self):
        """Test that the wires of the measurement process
        match an internal observable"""
        obs = qml.Hermitian(np.diag([1, 2, 3]), wires=["a", "b", "c"])
        m = MeasurementProcess(Expectation, obs=obs)

        assert np.all(m.wires == obs.wires)

    def test_eigvals_match_observable(self):
        """Test that the eigenvalues of the measurement process
        match an internal observable"""
        obs = qml.Hermitian(np.diag([1, 2, 3]), wires=[0, 1, 2])
        m = MeasurementProcess(Expectation, obs=obs)

        assert np.all(m.get_eigvals() == np.array([1, 2, 3]))

        # changing the observable data should be reflected
        obs.data = [np.diag([5, 6, 7])]
        assert np.all(m.get_eigvals() == np.array([5, 6, 7]))

    def test_error_obs_and_eigvals(self):
        """Test that providing both eigenvalues and an observable
        results in an error"""
        obs = qml.Hermitian(np.diag([1, 2, 3]), wires=[0, 1, 2])

        with pytest.raises(ValueError, match="Cannot set the eigenvalues"):
            MeasurementProcess(Expectation, obs=obs, eigvals=[0, 1])

    def test_error_obs_and_wires(self):
        """Test that providing both wires and an observable
        results in an error"""
        obs = qml.Hermitian(np.diag([1, 2, 3]), wires=[0, 1, 2])

        with pytest.raises(ValueError, match="Cannot set the wires"):
            MeasurementProcess(Expectation, obs=obs, wires=qml.wires.Wires([0, 1]))

    def test_observable_with_no_eigvals(self):
        """An observable with no eigenvalues defined should cause
        the eigvals method to return a NotImplementedError"""
        obs = qml.NumberOperator(wires=0)
        m = MeasurementProcess(Expectation, obs=obs)
        assert m.get_eigvals() is None

    def test_repr(self):
        """Test the string representation of a MeasurementProcess."""
        m = MeasurementProcess(Expectation, obs=qml.PauliZ(wires="a") @ qml.PauliZ(wires="b"))
        expected = "expval(PauliZ(wires=['a']) @ PauliZ(wires=['b']))"
        assert str(m) == expected

        m = MeasurementProcess(Probability, obs=qml.PauliZ(wires="a"))
        expected = "probs(PauliZ(wires=['a']))"
        assert str(m) == expected


class TestExpansion:
    """Test for measurement expansion"""

    def test_expand_pauli(self):
        """Test the expansion of a Pauli observable"""
        obs = qml.PauliX(0) @ qml.PauliY(1)
        m = MeasurementProcess(Expectation, obs=obs)
        tape = m.expand()

        assert len(tape.operations) == 4

        assert tape.operations[0].name == "Hadamard"
        assert tape.operations[0].wires.tolist() == [0]

        assert tape.operations[1].name == "PauliZ"
        assert tape.operations[1].wires.tolist() == [1]
        assert tape.operations[2].name == "S"
        assert tape.operations[2].wires.tolist() == [1]
        assert tape.operations[3].name == "Hadamard"
        assert tape.operations[3].wires.tolist() == [1]

        assert len(tape.measurements) == 1
        assert tape.measurements[0].return_type is Expectation
        assert tape.measurements[0].wires.tolist() == [0, 1]
        assert np.all(tape.measurements[0].get_eigvals() == np.array([1, -1, -1, 1]))

    def test_expand_hermitian(self, tol):
        """Test the expansion of an hermitian observable"""
        H = np.array([[1, 2], [2, 4]])
        obs = qml.Hermitian(H, wires=["a"])

        m = MeasurementProcess(Expectation, obs=obs)
        tape = m.expand()

        assert len(tape.operations) == 1

        assert tape.operations[0].name == "QubitUnitary"
        assert tape.operations[0].wires.tolist() == ["a"]
        assert np.allclose(
            tape.operations[0].parameters[0],
            np.array([[-2, 1], [1, 2]]) / np.sqrt(5),
            atol=tol,
            rtol=0,
        )

        assert len(tape.measurements) == 1
        assert tape.measurements[0].return_type is Expectation
        assert tape.measurements[0].wires.tolist() == ["a"]
        assert np.all(tape.measurements[0].get_eigvals() == np.array([0, 5]))

    def test_expand_no_observable(self):
        """Check that an exception is raised if the measurement to
        be expanded has no observable"""
        with pytest.raises(DecompositionUndefinedError):
            MeasurementProcess(Probability, wires=qml.wires.Wires([0, 1])).expand()


class TestDiagonalizingGates:
    def test_no_expansion(self):
        """Test a measurement that has no expansion"""
        m = qml.sample()

        assert m.diagonalizing_gates() == []

    def test_obs_diagonalizing_gates(self):
        """Test diagonalizing_gates method with and observable."""
        m = qml.expval(qml.PauliY(0))

        res = m.diagonalizing_gates()

        assert len(res) == 3

        expected_classes = [qml.PauliZ, qml.S, qml.Hadamard]
        for op, c in zip(res, expected_classes):
            assert isinstance(op, c)


class TestState:
    """Tests for the state function"""

    @pytest.mark.parametrize("wires", range(2, 5))
    def test_state_shape_and_dtype(self, wires):
        """Test that the state is of correct size and dtype for a trivial circuit"""

        dev = qml.device("default.qubit", wires=wires)

        @qml.qnode(dev)
        def func():
            return state()

        state_val = func()
        assert state_val.shape == (2**wires,)
        assert state_val.dtype == np.complex128

    def test_return_type_is_state(self):
        """Test that the return type of the observable is State"""

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def func():
            qml.Hadamard(0)
            return state()

        func()
        obs = func.qtape.observables
        assert len(obs) == 1
        assert obs[0].return_type is State

    @pytest.mark.parametrize("wires", range(2, 5))
    def test_state_correct_ghz(self, wires):
        """Test that the correct state is returned when the circuit prepares a GHZ state"""

        dev = qml.device("default.qubit", wires=wires)

        @qml.qnode(dev)
        def func():
            qml.Hadamard(wires=0)
            for i in range(wires - 1):
                qml.CNOT(wires=[i, i + 1])
            return state()

        state_val = func()
        assert np.allclose(np.sum(np.abs(state_val) ** 2), 1)
        assert np.allclose(state_val[0], 1 / np.sqrt(2))
        assert np.allclose(state_val[-1], 1 / np.sqrt(2))

    def test_return_with_other_types(self):
        """Test that an exception is raised when a state is returned along with another return
        type"""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def func():
            qml.Hadamard(wires=0)
            return state(), expval(qml.PauliZ(1))

        with pytest.raises(
            qml.QuantumFunctionError,
            match="The state or density matrix cannot be returned in combination with other return types",
        ):
            func()

    @pytest.mark.parametrize("wires", range(2, 5))
    def test_state_equal_to_dev_state(self, wires):
        """Test that the returned state is equal to the one stored in dev.state for a template
        circuit"""

        dev = qml.device("default.qubit", wires=wires)

        weights = np.random.random(
            qml.templates.StronglyEntanglingLayers.shape(n_layers=3, n_wires=wires)
        )

        @qml.qnode(dev)
        def func():
            qml.templates.StronglyEntanglingLayers(weights, wires=range(wires))
            return state()

        state_val = func()
        assert np.allclose(state_val, func.device.state)

    @pytest.mark.usefixtures("skip_if_no_tf_support")
    def test_interface_tf(self, skip_if_no_tf_support):
        """Test that the state correctly outputs in the tensorflow interface"""
        import tensorflow as tf

        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev, interface="tf")
        def func():
            for i in range(4):
                qml.Hadamard(i)
            return state()

        state_expected = 0.25 * tf.ones(16)
        state_val = func()

        assert isinstance(state_val, tf.Tensor)
        assert state_val.dtype == tf.complex128
        assert np.allclose(state_expected, state_val.numpy())
        assert state_val.shape == (16,)

    def test_interface_torch(self):
        """Test that the state correctly outputs in the torch interface"""
        torch = pytest.importorskip("torch", minversion="1.6")

        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev, interface="torch")
        def func():
            for i in range(4):
                qml.Hadamard(i)
            return state()

        state_expected = 0.25 * torch.ones(16, dtype=torch.complex128)
        state_val = func()

        assert isinstance(state_val, torch.Tensor)
        assert state_val.dtype == torch.complex128
        assert torch.allclose(state_expected, state_val)
        assert state_val.shape == (16,)

    def test_jacobian_not_supported(self):
        """Test if an error is raised if the jacobian method is called via qml.grad"""
        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev, diff_method="parameter-shift")
        def func(x):
            for i in range(4):
                qml.RX(x, wires=i)
            return state()

        d_func = qml.jacobian(func)

        with pytest.raises(
            ValueError,
            match="Computing the gradient of circuits that return the state is not supported",
        ):
            d_func(pnp.array(0.1, requires_grad=True))

    def test_no_state_capability(self, monkeypatch):
        """Test if an error is raised for devices that are not capable of returning the state.
        This is tested by changing the capability of default.qubit"""
        dev = qml.device("default.qubit", wires=1)
        capabilities = dev.capabilities().copy()
        capabilities["returns_state"] = False

        @qml.qnode(dev)
        def func():
            return state()

        with monkeypatch.context() as m:
            m.setattr(DefaultQubit, "capabilities", lambda *args, **kwargs: capabilities)
            with pytest.raises(qml.QuantumFunctionError, match="The current device is not capable"):
                func()

    def test_state_not_supported(self, monkeypatch):
        """Test if an error is raised for devices inheriting from the base Device class,
        which do not currently support returning the state"""
        dev = qml.device("default.gaussian", wires=1)

        @qml.qnode(dev)
        def func():
            return state()

        with pytest.raises(qml.QuantumFunctionError, match="Returning the state is not supported"):
            func()

    @pytest.mark.usefixtures("skip_if_no_tf_support")
    @pytest.mark.parametrize(
        "device", ["default.qubit", "default.qubit.tf", "default.qubit.autograd"]
    )
    @pytest.mark.parametrize("diff_method", ["best", "finite-diff", "parameter-shift"])
    def test_devices(self, device, diff_method, skip_if_no_tf_support):
        """Test that the returned state is equal to the expected returned state for all of
        PennyLane's built in statevector devices"""

        dev = qml.device(device, wires=4)

        @qml.qnode(dev, diff_method=diff_method)
        def func():
            for i in range(4):
                qml.Hadamard(i)
            return state()

        state_val = func()
        state_expected = 0.25 * np.ones(16)

        assert np.allclose(state_val, state_expected)
        assert np.allclose(state_val, dev.state)

    @pytest.mark.usefixtures("skip_if_no_tf_support")
    def test_gradient_with_passthru_tf(self, skip_if_no_tf_support):
        """Test that the gradient of the state is accessible when using default.qubit.tf with the
        backprop diff_method."""
        import tensorflow as tf

        dev = qml.device("default.qubit.tf", wires=1)

        @qml.qnode(dev, interface="tf", diff_method="backprop")
        def func(x):
            qml.RY(x, wires=0)
            return state()

        x = tf.Variable(0.1, dtype=tf.float64)

        with tf.GradientTape() as tape:
            result = func(x)

        grad = tape.jacobian(result, x)
        expected = tf.stack([-0.5 * tf.sin(x / 2), 0.5 * tf.cos(x / 2)])
        assert np.allclose(grad, expected)

    def test_gradient_with_passthru_autograd(self):
        """Test that the gradient of the state is accessible when using default.qubit.autograd
        with the backprop diff_method."""
        from pennylane import numpy as anp

        dev = qml.device("default.qubit.autograd", wires=1)

        @qml.qnode(dev, interface="autograd", diff_method="backprop")
        def func(x):
            qml.RY(x, wires=0)
            return state()

        x = anp.array(0.1, requires_grad=True)

        def loss_fn(x):
            res = func(x)
            return anp.real(res)  # This errors without the real. Likely an issue with complex
            # numbers in autograd

        d_loss_fn = qml.jacobian(loss_fn)

        grad = d_loss_fn(x)
        expected = np.array([-0.5 * np.sin(x / 2), 0.5 * np.cos(x / 2)])
        assert np.allclose(grad, expected)

    @pytest.mark.parametrize("wires", [[0, 2, 3, 1], ["a", -1, "b", 1000]])
    def test_custom_wire_labels(self, wires):
        """Test if an error is raised when custom wire labels are used"""
        dev = qml.device("default.qubit", wires=wires)

        @qml.qnode(dev, diff_method="parameter-shift")
        def func():
            qml.Hadamard(wires=wires[0])
            for i in range(3):
                qml.CNOT(wires=[wires[i], wires[i + 1]])
            return state()

        with pytest.raises(qml.QuantumFunctionError, match="custom wire labels"):
            func()


class TestDensityMatrix:
    """Tests for the density matrix function"""

    @pytest.mark.parametrize("wires", range(2, 5))
    @pytest.mark.parametrize("dev_name", ["default.qubit", "default.mixed"])
    def test_density_matrix_shape_and_dtype(self, dev_name, wires):
        """Test that the density matrix is of correct size and dtype for a
        trivial circuit"""

        dev = qml.device(dev_name, wires=wires)

        @qml.qnode(dev)
        def circuit():
            return density_matrix([0])

        state_val = circuit()

        assert state_val.shape == (2, 2)
        assert state_val.dtype == np.complex128

    @pytest.mark.parametrize("dev_name", ["default.qubit", "default.mixed"])
    def test_return_type_is_state(self, dev_name):
        """Test that the return type of the observable is State"""

        dev = qml.device(dev_name, wires=2)

        @qml.qnode(dev)
        def func():
            qml.Hadamard(0)
            return density_matrix(0)

        func()
        obs = func.qtape.observables
        assert len(obs) == 1
        assert obs[0].return_type is State

    @pytest.mark.parametrize("dev_name", ["default.qubit", "default.mixed"])
    def test_correct_density_matrix_product_state_first(self, dev_name):
        """Test that the correct density matrix is returned when
        tracing out a product state"""

        dev = qml.device(dev_name, wires=2)

        @qml.qnode(dev)
        def func():
            qml.Hadamard(wires=1)
            qml.PauliY(wires=0)
            return density_matrix(0)

        density_first = func()

        assert np.allclose(
            np.array([[0.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.0 + 0.0j]]), density_first
        )

    @pytest.mark.parametrize("dev_name", ["default.qubit", "default.mixed"])
    def test_correct_density_matrix_product_state_second(self, dev_name):
        """Test that the correct density matrix is returned when
        tracing out a product state"""

        dev = qml.device(dev_name, wires=2)

        @qml.qnode(dev)
        def func():
            qml.Hadamard(wires=1)
            qml.PauliY(wires=0)
            return density_matrix(1)

        density_second = func()
        assert np.allclose(
            np.array([[0.5 + 0.0j, 0.5 + 0.0j], [0.5 + 0.0j, 0.5 + 0.0j]]), density_second
        )

    @pytest.mark.parametrize("dev_name", ["default.qubit", "default.mixed"])
    def test_correct_density_matrix_three_wires_first(self, dev_name):
        """Test that the correct density matrix for an example with three wires"""

        dev = qml.device(dev_name, wires=3)

        @qml.qnode(dev)
        def func():
            qml.Hadamard(wires=1)
            qml.PauliY(wires=0)
            return density_matrix([0, 1])

        density_full = func()
        assert np.allclose(
            np.array(
                [
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.5 + 0.0j, 0.5 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.5 + 0.0j, 0.5 + 0.0j],
                ]
            ),
            density_full,
        )

    @pytest.mark.parametrize("dev_name", ["default.qubit", "default.mixed"])
    def test_correct_density_matrix_three_wires_second(self, dev_name):
        """Test that the correct density matrix for an example with three wires"""

        dev = qml.device(dev_name, wires=3)

        @qml.qnode(dev)
        def func():
            qml.Hadamard(0)
            qml.Hadamard(1)
            qml.CNOT(wires=[1, 2])
            return qml.density_matrix(wires=[1, 2])

        density = func()

        assert np.allclose(
            np.array(
                [
                    [
                        [0.5 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.5 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                        [0.5 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.5 + 0.0j],
                    ]
                ]
            ),
            density,
        )

    @pytest.mark.parametrize("dev_name", ["default.qubit", "default.mixed"])
    def test_correct_density_matrix_mixed_state(self, dev_name):
        """Test that the correct density matrix for an example with a mixed state"""

        dev = qml.device(dev_name, wires=2)

        @qml.qnode(dev)
        def func():
            qml.Hadamard(0)
            qml.CNOT(wires=[0, 1])
            return qml.density_matrix(wires=[1])

        density = func()

        assert np.allclose(np.array([[0.5 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.5 + 0.0j]]), density)

    @pytest.mark.parametrize("dev_name", ["default.qubit", "default.mixed"])
    def test_correct_density_matrix_all_wires(self, dev_name):
        """Test that the correct density matrix is returned when all wires are given"""

        dev = qml.device(dev_name, wires=2)

        @qml.qnode(dev)
        def func():
            qml.Hadamard(0)
            qml.CNOT(wires=[0, 1])
            return qml.density_matrix(wires=[0, 1])

        density = func()

        assert np.allclose(
            np.array(
                [
                    [0.5 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.5 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.5 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.5 + 0.0j],
                ]
            ),
            density,
        )

    @pytest.mark.parametrize("dev_name", ["default.qubit", "default.mixed"])
    def test_return_with_other_types(self, dev_name):
        """Test that an exception is raised when a state is returned along with another return
        type"""

        dev = qml.device(dev_name, wires=2)

        @qml.qnode(dev)
        def func():
            qml.Hadamard(wires=0)
            return density_matrix(0), expval(qml.PauliZ(1))

        with pytest.raises(
            qml.QuantumFunctionError,
            match="The state or density matrix"
            " cannot be returned in combination"
            " with other return types",
        ):
            func()

    def test_no_state_capability(self, monkeypatch):
        """Test if an error is raised for devices that are not capable of returning
        the density matrix. This is tested by changing the capability of default.qubit"""
        dev = qml.device("default.qubit", wires=2)
        capabilities = dev.capabilities().copy()
        capabilities["returns_state"] = False

        @qml.qnode(dev)
        def func():
            return density_matrix(0)

        with monkeypatch.context() as m:
            m.setattr(DefaultQubit, "capabilities", lambda *args, **kwargs: capabilities)
            with pytest.raises(
                qml.QuantumFunctionError,
                match="The current device is not capable" " of returning the state",
            ):
                func()

    def test_density_matrix_not_supported(self):
        """Test if an error is raised for devices inheriting from the base Device class,
        which do not currently support returning the state"""
        dev = qml.device("default.gaussian", wires=2)

        @qml.qnode(dev)
        def func():
            return density_matrix(0)

        with pytest.raises(qml.QuantumFunctionError, match="Returning the state is not supported"):
            func()

    @pytest.mark.parametrize("wires", [[0, 2, 3, 1], ["a", -1, "b", 1000]])
    @pytest.mark.parametrize("dev_name", ["default.qubit", "default.mixed"])
    def test_custom_wire_labels(self, wires, dev_name):
        """Test if an error is raised when custom wire labels are used"""
        dev = qml.device(dev_name, wires=wires)

        @qml.qnode(dev, diff_method="parameter-shift")
        def func():
            qml.Hadamard(wires=wires[0])
            for i in range(3):
                qml.CNOT(wires=[wires[i], wires[i + 1]])
            return density_matrix(0)

        with pytest.raises(qml.QuantumFunctionError, match="custom wire labels"):
            func()
