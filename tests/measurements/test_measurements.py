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
"""Unit tests for the measurements module"""
import numpy as np
import pytest

import pennylane as qml
from pennylane.exceptions import DecompositionUndefinedError, DeviceError, QuantumFunctionError
from pennylane.measurements import (
    ClassicalShadowMP,
    CountsMP,
    ExpectationMP,
    MeasurementProcess,
    MeasurementTransform,
    MeasurementValue,
    MidMeasureMP,
    MutualInfoMP,
    ProbabilityMP,
    PurityMP,
    SampleMeasurement,
    SampleMP,
    ShadowExpvalMP,
    Shots,
    StateMeasurement,
    StateMP,
    VarianceMP,
    VnEntropyMP,
    expval,
    sample,
    var,
)
from pennylane.queuing import AnnotatedQueue
from pennylane.wires import Wires

# pylint: disable=too-few-public-methods, unused-argument


class NotValidMeasurement(MeasurementProcess):
    _shortname = "NotValidReturnType"


def test_no_measure():
    """Test that failing to specify a measurement
    raises an exception"""
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def circuit(x):
        qml.RX(x, wires=0)
        return qml.PauliY(0)

    with pytest.raises(QuantumFunctionError, match="must return either a single measurement"):
        _ = circuit(0.65)


def test_numeric_type_unrecognized_error():
    """Test that querying the numeric type of a measurement process with an
    unrecognized return type raises an error."""

    mp = NotValidMeasurement()
    with pytest.raises(
        QuantumFunctionError,
        match="The numeric type of the measurement NotValidMeasurement is not defined",
    ):
        _ = mp.numeric_type


def test_shape_unrecognized_error():
    """Test that querying the shape of a measurement process with an
    unrecognized return type raises an error."""
    dev = qml.device("default.qubit", wires=2)
    mp = NotValidMeasurement()
    with pytest.raises(
        QuantumFunctionError,
        match="The shape of the measurement NotValidMeasurement is not defined",
    ):
        mp.shape(dev, Shots(None))


def test_eq_correctness():
    """Test that using `==` on measurement processes behaves the same as
    `qml.equal`."""

    class DummyMP(MeasurementProcess):
        """Dummy measurement process with no return type."""

    mp1 = DummyMP(wires=qml.wires.Wires(0))
    mp2 = DummyMP(wires=qml.wires.Wires(0))

    assert mp1 == mp1  # pylint: disable=comparison-with-itself
    assert mp1 == mp2


def test_hash_correctness():
    """Test that the hash of two equivalent measurement processes is the same."""

    class DummyMP(MeasurementProcess):
        """Dummy measurement process with no return type."""

    mp1 = DummyMP(wires=qml.wires.Wires(0))
    mp2 = DummyMP(wires=qml.wires.Wires(0))

    assert len({mp1, mp2}) == 1
    assert hash(mp1) == mp1.hash
    assert hash(mp2) == mp2.hash
    assert hash(mp1) == hash(mp2)


mv = qml.measure(0)

valid_meausurements = [
    ClassicalShadowMP(wires=Wires(0), seed=42),
    ShadowExpvalMP(qml.s_prod(3.0, qml.PauliX(0)), seed=97, k=2),
    ShadowExpvalMP([qml.PauliZ(0), 4.0 * qml.PauliX(0)], seed=86, k=4),
    CountsMP(obs=2.0 * qml.PauliX(0), all_outcomes=True),
    CountsMP(eigvals=[0.5, 0.6], wires=Wires(0), all_outcomes=False),
    CountsMP(obs=mv, all_outcomes=True),
    ExpectationMP(obs=qml.s_prod(2.0, qml.PauliX(0))),
    ExpectationMP(eigvals=[0.5, 0.6], wires=Wires("a")),
    ExpectationMP(obs=mv),
    MidMeasureMP(wires=Wires("a"), reset=True, id="abcd"),
    MutualInfoMP(wires=(Wires("a"), Wires("b")), log_base=3),
    ProbabilityMP(wires=Wires("a"), eigvals=[0.5, 0.6]),
    ProbabilityMP(obs=3.0 * qml.PauliX(0)),
    ProbabilityMP(obs=mv),
    PurityMP(wires=Wires("a")),
    SampleMP(obs=3.0 * qml.PauliY(0)),
    SampleMP(wires=Wires("a"), eigvals=[0.5, 0.6]),
    SampleMP(obs=mv),
    StateMP(),
    StateMP(wires=("a", "b")),
    VarianceMP(obs=qml.s_prod(0.5, qml.PauliX(0))),
    VarianceMP(eigvals=[0.6, 0.7], wires=Wires(0)),
    VarianceMP(obs=mv),
    VnEntropyMP(wires=Wires("a"), log_base=3),
]


# pylint: disable=protected-access
@pytest.mark.parametrize("mp", valid_meausurements)
def test_flatten_unflatten(mp):
    """Test flatten and unflatten methods."""

    data, metadata = mp._flatten()
    assert hash(metadata)

    new_mp = type(mp)._unflatten(data, metadata)
    qml.assert_equal(new_mp, mp)


@pytest.mark.jax
@pytest.mark.parametrize("mp", valid_meausurements)
def test_jax_pytree_integration(mp):
    """Test that measurement processes are jax pytrees."""
    import jax

    jax.tree_util.tree_flatten(mp)


@pytest.mark.parametrize(
    "stat_func,return_type", [(expval, ExpectationMP), (var, VarianceMP), (sample, SampleMP)]
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

        assert len(q.queue) == 1
        meas_proc = q.queue[0]
        assert isinstance(meas_proc, return_type)

    def test_annotating_tensor_hermitian(self, stat_func, return_type):
        """Test that the return_type related info is updated for a measurement
        when called for an Hermitian observable"""

        mx = np.array([[1, 0], [0, 1]])

        with AnnotatedQueue() as q:
            Herm = qml.Hermitian(mx, wires=[1])
            stat_func(Herm)

        assert len(q.queue) == 1
        meas_proc = q.queue[0]
        assert isinstance(meas_proc, return_type)

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

        assert len(q.queue) == 1
        meas_proc = q.queue[0]
        assert isinstance(meas_proc, return_type)

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
        found in the queue for annotation, they are not queued or annotated."""
        A = op1(0)
        B = op2(1)

        with AnnotatedQueue() as q:
            tensor_op = A @ B
            stat_func(tensor_op)

        assert len(q.queue) == 1

        meas_proc = q.queue[0]
        assert isinstance(meas_proc, return_type)


class TestProperties:
    """Test for the properties"""

    def test_wires_match_observable(self):
        """Test that the wires of the measurement process
        match an internal observable"""
        obs = qml.Hermitian(np.diag([1, 2, 3, 4]), wires=["a", "b"])
        m = qml.expval(op=obs)

        assert np.all(m.wires == obs.wires)

    def test_eigvals_match_observable(self):
        """Test that the eigenvalues of the measurement process
        match an internal observable"""
        obs = qml.Hermitian(np.diag([1, 2, 3, 4]), wires=[0, 1])
        m = qml.expval(op=obs)

        assert np.all(m.eigvals() == np.array([1, 2, 3, 4]))

        # changing the observable data should be reflected
        obs.data = [np.diag([5, 6, 7, 8])]
        assert np.all(m.eigvals() == np.array([5, 6, 7, 8]))

    def test_measurement_value_eigvals(self):
        """Test that eigenvalues of the measurement process
        are correct if the internal observable is a
        MeasurementValue."""
        m0 = qml.measure(0)
        m0.measurements[0]._id = "abc"  # pylint: disable=protected-access
        m1 = qml.measure(1)
        m1.measurements[0]._id = "def"  # pylint: disable=protected-access

        mp1 = qml.sample(op=[m0, m1])
        assert np.all(mp1.eigvals() == [0, 1, 2, 3])

        mp2 = qml.sample(op=3 * m0 - m1 / 2)
        assert np.all(mp2.eigvals() == [0, -0.5, 3, 2.5])

    def test_error_obs_and_eigvals(self):
        """Test that providing both eigenvalues and an observable
        results in an error"""
        obs = qml.Hermitian(np.diag([1, 2, 3, 4]), wires=[0, 1])

        with pytest.raises(ValueError, match="Cannot set the eigenvalues"):
            ExpectationMP(obs=obs, eigvals=[0, 1])

    def test_error_obs_and_wires(self):
        """Test that providing both wires and an observable
        results in an error"""
        obs = qml.Hermitian(np.diag([1, 2, 3, 4]), wires=[0, 1])

        with pytest.raises(ValueError, match="Cannot set the wires"):
            ExpectationMP(obs=obs, wires=qml.wires.Wires([0, 1]))

    def test_observable_with_no_eigvals(self):
        """An observable with no eigenvalues defined should cause
        the eigvals method to return a NotImplementedError"""
        obs = qml.NumberOperator(wires=0)
        m = qml.expval(op=obs)
        with pytest.raises(qml.operation.EigvalsUndefinedError):
            _ = m.eigvals()

    def test_repr(self):
        """Test the string representation of a MeasurementProcess."""
        m = qml.expval(op=qml.PauliZ(wires="a") @ qml.PauliZ(wires="b"))
        expected = "expval(Z('a') @ Z('b'))"
        assert str(m) == expected

        m = qml.probs(op=qml.PauliZ(wires="a"))
        expected = "probs(Z('a'))"
        assert str(m) == expected

        m = ProbabilityMP(eigvals=(1, 0), wires=qml.wires.Wires(0))
        assert repr(m) == "probs(eigvals=[1 0], wires=[0])"

        m0 = MeasurementValue([MidMeasureMP(Wires(0), id="0")], lambda v: v)
        m1 = MeasurementValue([MidMeasureMP(Wires(1), id="1")], lambda v: v)
        m = ProbabilityMP(obs=[m0, m1])
        expected = "probs([MeasurementValue(wires=[0]), MeasurementValue(wires=[1])])"
        assert repr(m) == expected

        m = ProbabilityMP(obs=m0 * m1)
        expected = "probs(MeasurementValue(wires=[0, 1]))"

    def test_measurement_value_map_wires(self):
        """Test that MeasurementProcess.map_wires works correctly when mp.mv
        is not None."""
        m0 = qml.measure("a")
        m1 = qml.measure("b")
        m2 = qml.measure(0)
        m3 = qml.measure(1)
        m2.measurements[0]._id = m0.measurements[0].id  # pylint: disable=protected-access
        m3.measurements[0]._id = m1.measurements[0].id  # pylint: disable=protected-access

        wire_map = {"a": 0, "b": 1}

        mp1 = qml.sample(op=[m0, m1])
        mapped_mp1 = mp1.map_wires(wire_map)
        qml.assert_equal(mapped_mp1, qml.sample(op=[m2, m3]))

        mp2 = qml.sample(op=m0 * m1)
        mapped_mp2 = mp2.map_wires(wire_map)
        qml.assert_equal(mapped_mp2, qml.sample(op=m2 * m3))


class TestExpansion:
    """Test for measurement expansion"""

    def test_expand_pauli(self):
        """Test the expansion of a Pauli observable"""
        obs = qml.PauliX(0) @ qml.PauliY(1)
        m = qml.expval(op=obs)
        with pytest.warns(qml.exceptions.PennyLaneDeprecationWarning):
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
        assert isinstance(tape.measurements[0], ExpectationMP)
        assert tape.measurements[0].wires.tolist() == [0, 1]
        assert np.all(tape.measurements[0].eigvals() == np.array([1, -1, -1, 1]))

    def test_expand_hermitian(self, tol):
        """Test the expansion of an hermitian observable"""
        H = np.array([[1, 2], [2, 4]])
        obs = qml.Hermitian(H, wires=["a"])

        m = qml.expval(op=obs)
        with pytest.warns(qml.exceptions.PennyLaneDeprecationWarning):
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
        assert isinstance(tape.measurements[0], ExpectationMP)
        assert tape.measurements[0].wires.tolist() == ["a"]
        assert np.all(tape.measurements[0].eigvals() == np.array([0, 5]))

    def test_expand_no_observable(self):
        """Check that an exception is raised if the measurement to
        be expanded has no observable"""
        with pytest.warns(qml.exceptions.PennyLaneDeprecationWarning):
            with pytest.raises(DecompositionUndefinedError):
                ProbabilityMP(wires=qml.wires.Wires([0, 1])).expand()

    @pytest.mark.parametrize(
        "m",
        [
            ExpectationMP(obs=qml.PauliX(0) @ qml.PauliY(1)),
            VarianceMP(obs=qml.PauliX(0) @ qml.PauliY(1)),
            ProbabilityMP(obs=qml.PauliX(0) @ qml.PauliY(1)),
            ExpectationMP(obs=qml.PauliX(5)),
            VarianceMP(obs=qml.PauliZ(0) @ qml.Identity(3)),
            ProbabilityMP(obs=qml.PauliZ(0) @ qml.Identity(3)),
        ],
    )
    def test_has_decomposition_true_pauli(self, m):
        """Test that measurements of Paulis report to have a decomposition."""
        assert m.has_decomposition is True

    def test_has_decomposition_true_hermitian(self):
        """Test that measurements of Hermitians report to have a decomposition."""
        H = np.array([[1, 2], [2, 4]])
        obs = qml.Hermitian(H, wires=["a"])
        m = qml.expval(op=obs)
        assert m.has_decomposition is True

    def test_has_decomposition_false_hermitian_wo_diaggates(self):
        """Test that measurements of Hermitians report to have a decomposition."""

        class HermitianNoDiagGates(qml.Hermitian):
            @property
            def has_diagonalizing_gates(
                self,
            ):  # pylint: disable=invalid-overridden-method, arguments-renamed
                return False

        H = np.array([[1, 2], [2, 4]])
        obs = HermitianNoDiagGates(H, wires=["a"])
        m = ExpectationMP(obs=obs)
        assert m.has_decomposition is False

    def test_has_decomposition_false_no_observable(self):
        """Check a MeasurementProcess without observable to report not having a decomposition"""
        m = ProbabilityMP(wires=qml.wires.Wires([0, 1]))
        assert m.has_decomposition is False

        m = ExpectationMP(wires=qml.wires.Wires([0, 1]), eigvals=np.ones(4))
        assert m.has_decomposition is False

    @pytest.mark.parametrize(
        "m",
        [
            SampleMP(),
            SampleMP(wires=["a", 1]),
            CountsMP(all_outcomes=True),
            CountsMP(wires=["a", 1], all_outcomes=True),
            CountsMP(),
            CountsMP(wires=["a", 1]),
            StateMP(),
            VnEntropyMP(wires=["a", 1]),
            MutualInfoMP(wires=[["a", 1], ["b", 2]]),
            ProbabilityMP(wires=["a", 1]),
        ],
    )
    def test_samples_computational_basis_true(self, m):
        """Test that measurements of Paulis report to have a decomposition."""
        assert m.samples_computational_basis is True

    @pytest.mark.parametrize(
        "m",
        [
            ExpectationMP(obs=qml.PauliX(2)),
            VarianceMP(obs=qml.PauliX("a")),
            ProbabilityMP(obs=qml.PauliX("b")),
            SampleMP(obs=qml.PauliX("a")),
            CountsMP(obs=qml.PauliX("a")),
            ShadowExpvalMP(H=qml.PauliX("a")),
            ClassicalShadowMP(wires=[["a", 1], ["b", 2]]),
        ],
    )
    def test_samples_computational_basis_false(self, m):
        """Test that measurements of Paulis report to have a decomposition."""
        assert m.samples_computational_basis is False


class TestDiagonalizingGates:
    def test_no_expansion(self):
        """Test a measurement that has no expansion"""
        m = qml.sample()

        assert len(m.diagonalizing_gates()) == 0

    def test_obs_diagonalizing_gates(self):
        """Test diagonalizing_gates method with and observable."""
        m = qml.expval(qml.PauliY(0))

        res = m.diagonalizing_gates()

        assert len(res) == 3

        expected_classes = [qml.PauliZ, qml.S, qml.Hadamard]
        for op, c in zip(res, expected_classes):
            assert isinstance(op, c)


class TestSampleMeasurement:
    """Tests for the SampleMeasurement class."""

    def test_custom_sample_measurement(self):
        """Test the execution of a custom sampled measurement."""

        class MyMeasurement(SampleMeasurement):
            # pylint: disable=signature-differs
            def process_samples(self, samples, wire_order, shot_range=None, bin_size=None):
                return qml.math.sum(samples[..., self.wires])

            def process_counts(self, counts: dict, wire_order: Wires):
                return counts

        dev = qml.device("default.qubit", wires=2)

        @qml.set_shots(1000)
        @qml.qnode(dev)
        def circuit():
            qml.PauliX(0)
            return MyMeasurement(wires=[0]), MyMeasurement(wires=[1])

        assert qml.math.allequal(circuit(), [1000, 0])

    def test_sample_measurement_without_shots(self):
        """Test that executing a sampled measurement with ``shots=None`` raises an error."""

        class MyMeasurement(SampleMeasurement):
            # pylint: disable=signature-differs
            def process_samples(self, samples, wire_order, shot_range, bin_size):
                return qml.math.sum(samples[..., self.wires])

            def process_counts(self, counts: dict, wire_order: Wires):
                return counts

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.PauliX(0)
            return MyMeasurement(wires=[0]), MyMeasurement(wires=[1])

        with pytest.raises(
            DeviceError,
            match="not accepted for analytic simulation on default.qubit",
        ):
            circuit()


class TestStateMeasurement:
    """Tests for the StateMeasurement class."""

    def test_custom_state_measurement(self):
        """Test the execution of a custom state measurement."""

        class MyMeasurement(StateMeasurement):
            def process_state(self, state, wire_order):
                return qml.math.sum(state)

            _shortname = "state"

            def shape(self):
                return ()

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            return MyMeasurement()

        assert circuit() == 1

    def test_state_measurement_with_shots(self):
        """Test that executing a state measurement with shots raises an error."""

        class MyMeasurement(StateMeasurement):
            def process_state(self, state, wire_order):
                return qml.math.sum(state)

            _shortname = "state"

            def shape(self):
                return ()

        dev = qml.device("default.qubit", wires=2)

        @qml.set_shots(1000)
        @qml.qnode(dev)
        def circuit():
            return MyMeasurement()

        with pytest.raises(DeviceError, match="not accepted with finite shots on default.qubit"):
            circuit()

    def test_state_measurement_process_density_matrix_not_implemented(self):
        """Test that the process_density_matrix method of StateMeasurement raises
        NotImplementedError."""

        class MyMeasurement(StateMeasurement):
            def process_state(self, state, wire_order):
                return qml.math.sum(state)

        with pytest.raises(NotImplementedError):
            MyMeasurement().process_density_matrix(
                density_matrix=qml.math.array([[1, 0], [0, 0]]), wire_order=Wires([0, 1])
            )


class TestMeasurementTransform:
    """Tests for the MeasurementTransform class."""

    def test_custom_measurement(self):
        """Test the execution of a custom measurement."""

        class CountTapesMP(MeasurementTransform, SampleMeasurement):
            def process(self, tape, device):
                program = device.preprocess_transforms()
                tapes, _ = program([tape])
                return len(tapes)

            def process_samples(self, samples, wire_order, shot_range=None, bin_size=None):
                return [True]

            def process_counts(self, counts: dict, wire_order: Wires):
                return counts

        dev = qml.device("default.qubit", wires=2)

        @qml.set_shots(1000)
        @qml.qnode(dev)
        def circuit():
            return CountTapesMP(wires=[0])

        assert circuit() == [True]


class TestMeasurementProcess:
    """Tests for the shape and numeric type of a measurement process"""

    measurements_no_shots = [
        (qml.expval(qml.PauliZ(0)), ()),
        (qml.var(qml.PauliZ(0)), ()),
        (qml.probs(wires=[0, 1]), (4,)),
        (qml.state(), (8,)),
        (qml.density_matrix(wires=[0, 1]), (4, 4)),
        (qml.mutual_info(wires0=[0], wires1=[1]), ()),
        (qml.vn_entropy(wires=[0, 1]), ()),
    ]

    measurements_finite_shots = [
        (qml.expval(qml.PauliZ(0)), ()),
        (qml.var(qml.PauliZ(0)), ()),
        (qml.probs(wires=[0, 1]), (4,)),
        (qml.state(), (8,)),
        (qml.density_matrix(wires=[0, 1]), (4, 4)),
        (qml.sample(qml.PauliZ(0)), (10,)),
        (qml.sample(), (10, 3)),
        (qml.mutual_info(wires0=0, wires1=1), ()),
        (qml.vn_entropy(wires=[0, 1]), ()),
    ]

    @pytest.mark.parametrize("measurement, expected_shape", measurements_no_shots)
    def test_output_shapes_no_shots(self, measurement, expected_shape):
        """Test that the output shape of the measurement process is expected
        when shots=None"""

        assert measurement.shape(shots=None, num_device_wires=3) == expected_shape

    @pytest.mark.parametrize("measurement, expected_shape", measurements_finite_shots)
    def test_output_shapes_finite_shots(self, measurement, expected_shape):
        """Test that the output shape of the measurement process is expected
        when shots is finite"""
        assert measurement.shape(shots=10, num_device_wires=3) == expected_shape

    def test_undefined_shape_error(self):
        """Test that an error is raised for a measurement with an undefined shape"""
        measurement = qml.counts(wires=[0, 1])
        msg = "The shape of the measurement CountsMP is not defined"

        with pytest.raises(QuantumFunctionError, match=msg):
            measurement.shape(shots=None, num_device_wires=2)
