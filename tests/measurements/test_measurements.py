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

import pennylane as qp
from pennylane.exceptions import DeviceError, QuantumFunctionError
from pennylane.measurements import (
    ClassicalShadowMP,
    CountsMP,
    ExpectationMP,
    MeasurementProcess,
    MeasurementTransform,
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


def test_measurements_module_getattr():
    """Test that the getattr raises an attribute error for things that dont exist."""
    with pytest.raises(AttributeError):
        qp.measurements.not_here  # pylint: disable=pointless-statement


def test_mid_measure_deprecations():

    assert qp.measurements.MidMeasureMP == qp.ops.MidMeasure
    assert qp.measurements.MeasurementValue == qp.ops.MeasurementValue
    assert qp.measurements.measure == qp.ops.measure
    assert qp.measurements.get_mcm_predicates == qp.ops.mid_measure.get_mcm_predicates
    from pennylane.devices.qubit.simulate import _find_post_processed_mcms

    assert qp.measurements.find_post_processed_mcms == _find_post_processed_mcms


class NotValidMeasurement(MeasurementProcess):
    _shortname = "NotValidReturnType"


def test_no_measure():
    """Test that failing to specify a measurement
    raises an exception"""
    dev = qp.device("default.qubit", wires=2)

    @qp.qnode(dev)
    def circuit(x):
        qp.RX(x, wires=0)
        return qp.PauliY(0)

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
    dev = qp.device("default.qubit", wires=2)
    mp = NotValidMeasurement()
    with pytest.raises(
        QuantumFunctionError,
        match="The shape of the measurement NotValidMeasurement is not defined",
    ):
        mp.shape(dev, Shots(None))


def test_eq_correctness():
    """Test that using `==` on measurement processes behaves the same as
    `qp.equal`."""

    class DummyMP(MeasurementProcess):
        """Dummy measurement process with no return type."""

    mp1 = DummyMP(wires=qp.wires.Wires(0))
    mp2 = DummyMP(wires=qp.wires.Wires(0))

    assert mp1 == mp1  # pylint: disable=comparison-with-itself
    assert mp1 == mp2


def test_hash_correctness():
    """Test that the hash of two equivalent measurement processes is the same."""

    class DummyMP(MeasurementProcess):
        """Dummy measurement process with no return type."""

    mp1 = DummyMP(wires=qp.wires.Wires(0))
    mp2 = DummyMP(wires=qp.wires.Wires(0))

    assert len({mp1, mp2}) == 1
    assert hash(mp1) == mp1.hash
    assert hash(mp2) == mp2.hash
    assert hash(mp1) == hash(mp2)


mv = qp.measure(0)

valid_meausurements = [
    ClassicalShadowMP(wires=Wires(0), seed=42),
    ShadowExpvalMP(qp.s_prod(3.0, qp.PauliX(0)), seed=97, k=2),
    ShadowExpvalMP([qp.PauliZ(0), 4.0 * qp.PauliX(0)], seed=86, k=4),
    CountsMP(obs=2.0 * qp.PauliX(0), all_outcomes=True),
    CountsMP(eigvals=[0.5, 0.6], wires=Wires(0), all_outcomes=False),
    CountsMP(obs=mv, all_outcomes=True),
    ExpectationMP(obs=qp.s_prod(2.0, qp.PauliX(0))),
    ExpectationMP(eigvals=[0.5, 0.6], wires=Wires("a")),
    ExpectationMP(obs=mv),
    MutualInfoMP(wires=(Wires("a"), Wires("b")), log_base=3),
    ProbabilityMP(wires=Wires("a"), eigvals=[0.5, 0.6]),
    ProbabilityMP(obs=3.0 * qp.PauliX(0)),
    ProbabilityMP(obs=mv),
    PurityMP(wires=Wires("a")),
    SampleMP(obs=3.0 * qp.PauliY(0)),
    SampleMP(wires=Wires("a"), eigvals=[0.5, 0.6]),
    SampleMP(obs=mv),
    StateMP(),
    StateMP(wires=("a", "b")),
    VarianceMP(obs=qp.s_prod(0.5, qp.PauliX(0))),
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
    qp.assert_equal(new_mp, mp)


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
        [qp.PauliX, qp.PauliY, qp.PauliZ, qp.Hadamard, qp.Identity],
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
            Herm = qp.Hermitian(mx, wires=[1])
            stat_func(Herm)

        assert len(q.queue) == 1
        meas_proc = q.queue[0]
        assert isinstance(meas_proc, return_type)

    @pytest.mark.parametrize(
        "op1,op2",
        [
            (qp.PauliY, qp.PauliX),
            (qp.Hadamard, qp.Hadamard),
            (qp.PauliY, qp.Identity),
            (qp.Identity, qp.Identity),
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
            (qp.PauliY, qp.PauliX),
            (qp.Hadamard, qp.Hadamard),
            (qp.PauliY, qp.Identity),
            (qp.Identity, qp.Identity),
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
        obs = qp.Hermitian(np.diag([1, 2, 3, 4]), wires=["a", "b"])
        m = qp.expval(op=obs)

        assert np.all(m.wires == obs.wires)

    def test_eigvals_match_observable(self):
        """Test that the eigenvalues of the measurement process
        match an internal observable"""
        obs = qp.Hermitian(np.diag([1, 2, 3, 4]), wires=[0, 1])
        m = qp.expval(op=obs)

        assert np.all(m.eigvals() == np.array([1, 2, 3, 4]))

        # changing the observable data should be reflected
        obs.data = [np.diag([5, 6, 7, 8])]
        assert np.all(m.eigvals() == np.array([5, 6, 7, 8]))

    def test_measurement_value_eigvals(self):
        """Test that eigenvalues of the measurement process
        are correct if the internal observable is a
        MeasurementValue."""
        m0 = qp.measure(0)
        m0.measurements[0]._id = "abc"  # pylint: disable=protected-access
        m1 = qp.measure(1)
        m1.measurements[0]._id = "def"  # pylint: disable=protected-access

        mp1 = qp.sample(op=[m0, m1])
        assert np.all(mp1.eigvals() == [0, 1, 2, 3])

        mp2 = qp.sample(op=3 * m0 - m1 / 2)
        assert np.all(mp2.eigvals() == [0, -0.5, 3, 2.5])

    def test_error_obs_and_eigvals(self):
        """Test that providing both eigenvalues and an observable
        results in an error"""
        obs = qp.Hermitian(np.diag([1, 2, 3, 4]), wires=[0, 1])

        with pytest.raises(ValueError, match="Cannot set the eigenvalues"):
            ExpectationMP(obs=obs, eigvals=[0, 1])

    def test_error_obs_and_wires(self):
        """Test that providing both wires and an observable
        results in an error"""
        obs = qp.Hermitian(np.diag([1, 2, 3, 4]), wires=[0, 1])

        with pytest.raises(ValueError, match="Cannot set the wires"):
            ExpectationMP(obs=obs, wires=qp.wires.Wires([0, 1]))

    def test_observable_with_no_eigvals(self):
        """An observable with no eigenvalues defined should cause
        the eigvals method to return a NotImplementedError"""
        obs = qp.NumberOperator(wires=0)
        m = qp.expval(op=obs)
        with pytest.raises(qp.operation.EigvalsUndefinedError):
            _ = m.eigvals()

    def test_repr(self):
        """Test the string representation of a MeasurementProcess."""
        m = qp.expval(op=qp.PauliZ(wires="a") @ qp.PauliZ(wires="b"))
        expected = "expval(Z('a') @ Z('b'))"
        assert str(m) == expected

        m = qp.probs(op=qp.PauliZ(wires="a"))
        expected = "probs(Z('a'))"
        assert str(m) == expected

        m = ProbabilityMP(eigvals=(1, 0), wires=qp.wires.Wires(0))
        assert repr(m) == "probs(eigvals=[1 0], wires=[0])"

        m0 = qp.ops.MeasurementValue([qp.ops.MidMeasure(Wires(0), id="0")], lambda v: v)
        m1 = qp.ops.MeasurementValue([qp.ops.MidMeasure(Wires(1), id="1")], lambda v: v)
        m = ProbabilityMP(obs=[m0, m1])
        expected = "probs([MeasurementValue(wires=[0]), MeasurementValue(wires=[1])])"
        assert repr(m) == expected

        m = ProbabilityMP(obs=m0 * m1)
        expected = "probs(MeasurementValue(wires=[0, 1]))"

    def test_measurement_value_map_wires(self):
        """Test that MeasurementProcess.map_wires works correctly when mp.mv
        is not None."""
        m0 = qp.measure("a")
        m1 = qp.measure("b")
        m2 = qp.measure(0)
        m3 = qp.measure(1)
        m2.measurements[0]._id = m0.measurements[0].id  # pylint: disable=protected-access
        m3.measurements[0]._id = m1.measurements[0].id  # pylint: disable=protected-access

        wire_map = {"a": 0, "b": 1}

        mp1 = qp.sample(op=[m0, m1])
        mapped_mp1 = mp1.map_wires(wire_map)
        qp.assert_equal(mapped_mp1, qp.sample(op=[m2, m3]))

        mp2 = qp.sample(op=m0 * m1)
        mapped_mp2 = mp2.map_wires(wire_map)
        qp.assert_equal(mapped_mp2, qp.sample(op=m2 * m3))


class TestDiagonalizingGates:
    def test_no_expansion(self):
        """Test a measurement that has no expansion"""
        m = qp.sample()

        assert len(m.diagonalizing_gates()) == 0

    def test_obs_diagonalizing_gates(self):
        """Test diagonalizing_gates method with and observable."""
        m = qp.expval(qp.PauliY(0))

        res = m.diagonalizing_gates()

        assert len(res) == 3

        expected_classes = [qp.PauliZ, qp.S, qp.Hadamard]
        for op, c in zip(res, expected_classes):
            assert isinstance(op, c)


class TestSampleMeasurement:
    """Tests for the SampleMeasurement class."""

    def test_custom_sample_measurement(self):
        """Test the execution of a custom sampled measurement."""

        class MyMeasurement(SampleMeasurement):
            # pylint: disable=signature-differs
            def process_samples(self, samples, wire_order, shot_range=None, bin_size=None):
                return qp.math.sum(samples[..., self.wires])

            def process_counts(self, counts: dict, wire_order: Wires):
                return counts

        dev = qp.device("default.qubit", wires=2)

        @qp.set_shots(1000)
        @qp.qnode(dev)
        def circuit():
            qp.PauliX(0)
            return MyMeasurement(wires=[0]), MyMeasurement(wires=[1])

        assert qp.math.allequal(circuit(), [1000, 0])

    def test_sample_measurement_without_shots(self):
        """Test that executing a sampled measurement with ``shots=None`` raises an error."""

        class MyMeasurement(SampleMeasurement):
            # pylint: disable=signature-differs
            def process_samples(self, samples, wire_order, shot_range, bin_size):
                return qp.math.sum(samples[..., self.wires])

            def process_counts(self, counts: dict, wire_order: Wires):
                return counts

        dev = qp.device("default.qubit", wires=2)

        @qp.qnode(dev)
        def circuit():
            qp.PauliX(0)
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
                return qp.math.sum(state)

            _shortname = "state"

            def shape(self):
                return ()

        dev = qp.device("default.qubit", wires=2)

        @qp.qnode(dev)
        def circuit():
            return MyMeasurement()

        assert circuit() == 1

    def test_state_measurement_with_shots(self):
        """Test that executing a state measurement with shots raises an error."""

        class MyMeasurement(StateMeasurement):
            def process_state(self, state, wire_order):
                return qp.math.sum(state)

            _shortname = "state"

            def shape(self):
                return ()

        dev = qp.device("default.qubit", wires=2)

        @qp.set_shots(1000)
        @qp.qnode(dev)
        def circuit():
            return MyMeasurement()

        with pytest.raises(DeviceError, match="not accepted with finite shots on default.qubit"):
            circuit()

    def test_state_measurement_process_density_matrix_not_implemented(self):
        """Test that the process_density_matrix method of StateMeasurement raises
        NotImplementedError."""

        class MyMeasurement(StateMeasurement):
            def process_state(self, state, wire_order):
                return qp.math.sum(state)

        with pytest.raises(NotImplementedError):
            MyMeasurement().process_density_matrix(
                density_matrix=qp.math.array([[1, 0], [0, 0]]), wire_order=Wires([0, 1])
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

        dev = qp.device("default.qubit", wires=2)

        @qp.set_shots(1000)
        @qp.qnode(dev)
        def circuit():
            return CountTapesMP(wires=[0])

        assert circuit() == [True]


class TestMeasurementProcess:
    """Tests for the shape and numeric type of a measurement process"""

    measurements_no_shots = [
        (qp.expval(qp.PauliZ(0)), ()),
        (qp.var(qp.PauliZ(0)), ()),
        (qp.probs(wires=[0, 1]), (4,)),
        (qp.state(), (8,)),
        (qp.density_matrix(wires=[0, 1]), (4, 4)),
        (qp.mutual_info(wires0=[0], wires1=[1]), ()),
        (qp.vn_entropy(wires=[0, 1]), ()),
    ]

    measurements_finite_shots = [
        (qp.expval(qp.PauliZ(0)), ()),
        (qp.var(qp.PauliZ(0)), ()),
        (qp.probs(wires=[0, 1]), (4,)),
        (qp.state(), (8,)),
        (qp.density_matrix(wires=[0, 1]), (4, 4)),
        (qp.sample(qp.PauliZ(0)), (10,)),
        (qp.sample(), (10, 3)),
        (qp.mutual_info(wires0=0, wires1=1), ()),
        (qp.vn_entropy(wires=[0, 1]), ()),
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
        measurement = qp.counts(wires=[0, 1])
        msg = "The shape of the measurement CountsMP is not defined"

        with pytest.raises(QuantumFunctionError, match=msg):
            measurement.shape(shots=None, num_device_wires=2)
