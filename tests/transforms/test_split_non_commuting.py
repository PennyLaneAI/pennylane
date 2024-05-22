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
""" Tests for the transform ``qml.transform.split_non_commuting()`` """
import functools

# pylint: disable=no-self-use, import-outside-toplevel, no-member, import-error
import itertools

import numpy as np
import pytest

import pennylane as qml
import pennylane.numpy as pnp
from pennylane.queuing import AnnotatedQueue
from pennylane.tape import QuantumScript
from pennylane.transforms import split_non_commuting

# list of observables with 2 commuting groups [[1, 3], [0, 2, 4]]
obs_list_2 = [
    qml.prod(qml.Z(0), qml.Z(1)),
    qml.prod(qml.PauliX(0), qml.PauliX(1)),
    qml.PauliZ(0),
    qml.PauliX(0),
    qml.PauliZ(1),
]

# list of observables with 3 commuting groups [[0,3], [1,4], [2,5]]
obs_list_3 = [
    qml.prod(qml.PauliZ(0), qml.PauliZ(1)),
    qml.prod(qml.PauliX(0), qml.PauliX(1)),
    qml.prod(qml.PauliY(0), qml.PauliY(1)),
    qml.PauliZ(0),
    qml.PauliX(0),
    qml.PauliY(0),
]

# measurements that accept observables as arguments
obs_meas_fn = [qml.expval, qml.var, qml.probs, qml.counts, qml.sample]

# measurements that accept wires as arguments
wire_meas_fn = [qml.probs, qml.counts, qml.sample]


class TestUnittestSplitNonCommuting:
    """Unit tests on ``qml.transforms.split_non_commuting()``"""

    @pytest.mark.parametrize("convert_to_opmath", (True, False))
    @pytest.mark.parametrize("meas_type", obs_meas_fn)
    def test_commuting_group_no_split(self, mocker, meas_type, convert_to_opmath):
        """Testing that commuting groups are not split for all supported measurement types"""
        with qml.queuing.AnnotatedQueue() as q:
            qml.PauliZ(0)
            qml.Hadamard(0)
            qml.CNOT((0, 1))
            meas_type(op=qml.PauliZ(0))
            meas_type(op=qml.PauliZ(0))
            meas_type(op=qml.PauliX(1))
            meas_type(op=qml.PauliZ(2))
            if convert_to_opmath:
                meas_type(op=qml.prod(qml.Z(0), qml.Z(3)))
            else:
                meas_type(op=qml.operation.Tensor(qml.Z(0), qml.Z(3)))

        # test transform on tape
        tape = qml.tape.QuantumScript.from_queue(q, shots=100)
        split, fn = split_non_commuting(tape)
        for t in split:
            assert t.shots == tape.shots

        spy = mocker.spy(qml.math, "concatenate")

        assert len(split) == 1
        assert all(isinstance(t, qml.tape.QuantumScript) for t in split)
        assert fn([[0.1, 0.2, 0.3, 0.4]]) == (0.1, 0.1, 0.2, 0.3, 0.4)

        # test transform on qscript
        qs = qml.tape.QuantumScript(tape.operations, tape.measurements, shots=50)
        split, fn = split_non_commuting(qs)
        for t in split:
            assert t.shots == qs.shots

        assert len(split) == 1
        assert all(isinstance(i_qs, qml.tape.QuantumScript) for i_qs in split)
        assert fn([[0.1, 0.2, 0.3, 0.4]]) == (0.1, 0.1, 0.2, 0.3, 0.4)

        spy.assert_not_called()

    @pytest.mark.parametrize("convert_to_opmath", (True, False))
    @pytest.mark.parametrize("meas_type", wire_meas_fn)
    def test_wire_commuting_group_no_split(self, mocker, meas_type, convert_to_opmath):
        """Testing that commuting MPs initialized using wires or observables are not split"""
        with qml.queuing.AnnotatedQueue() as q:
            qml.PauliZ(0)
            qml.Hadamard(0)
            qml.CNOT((0, 1))
            meas_type()  # Included to check splitting with all-wire measurements
            meas_type(wires=[0])
            meas_type(wires=[1])
            meas_type(wires=[0, 1])
            meas_type(op=qml.PauliZ(0))
            if convert_to_opmath:
                meas_type(op=qml.prod(qml.PauliZ(0), qml.PauliZ(2)))
            else:
                meas_type(op=qml.operation.Tensor(qml.PauliZ(0), qml.PauliZ(2)))

        # test transform on tape
        tape = qml.tape.QuantumScript.from_queue(q)
        split, fn = split_non_commuting(tape)

        spy = mocker.spy(qml.math, "concatenate")

        assert len(split) == 1
        assert all(isinstance(t, qml.tape.QuantumScript) for t in split)
        assert fn([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]) == (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)

        # test transform on qscript
        qs = qml.tape.QuantumScript(tape.operations, tape.measurements)
        split, fn = split_non_commuting(qs)

        assert len(split) == 1
        assert all(isinstance(i_qs, qml.tape.QuantumScript) for i_qs in split)
        assert fn([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]) == (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)

        spy.assert_not_called()

    @pytest.mark.parametrize("convert_to_opmath", (True, False))
    @pytest.mark.parametrize("meas_type", obs_meas_fn)
    @pytest.mark.parametrize("obs_list, expected", [(obs_list_2, 2), (obs_list_3, 3)])
    def test_non_commuting_group_right_number(
        self, meas_type, obs_list, expected, convert_to_opmath
    ):
        """Test that the no. of tapes after splitting into commuting groups is of the right size"""

        if not convert_to_opmath:
            obs_list = [
                qml.operation.Tensor(*o) if isinstance(o, qml.ops.Prod) else o for o in obs_list
            ]
        # create a queue with several measurements of same type but with differnent non-commuting
        # observables
        with qml.queuing.AnnotatedQueue() as q:
            qml.PauliZ(0)
            qml.Hadamard(0)
            qml.CNOT((0, 1))
            for obs in obs_list:
                meas_type(op=obs)

            # if MP type can accept wires, then add two extra measurements using wires and test no.
            # of tapes after splitting commuting groups
            if meas_type in wire_meas_fn:
                meas_type(wires=[0])
                meas_type(wires=[0, 1])

        tape = qml.tape.QuantumScript.from_queue(q)
        split, _ = split_non_commuting(tape)
        assert len(split) == expected

        qs = qml.tape.QuantumScript(tape.operations, tape.measurements)
        split, _ = split_non_commuting(qs)
        assert len(split) == expected

    @pytest.mark.parametrize("convert_to_opmath", (True, False))
    @pytest.mark.parametrize("meas_type", obs_meas_fn)
    @pytest.mark.parametrize(
        "obs_list, group_coeffs",
        [(obs_list_2, [[1, 3], [0, 2, 4]]), (obs_list_3, [[0, 3], [1, 4], [2, 5]])],
    )
    def test_non_commuting_group_right_reorder(
        self, meas_type, obs_list, convert_to_opmath, group_coeffs
    ):
        """Test that the output is of the correct order"""
        # create a queue with several measurements of same type but with differnent non-commuting
        # observables

        if not convert_to_opmath:
            obs_list = [
                qml.operation.Tensor(*o) if isinstance(o, qml.ops.Prod) else o for o in obs_list
            ]
        with qml.queuing.AnnotatedQueue() as q:
            qml.PauliZ(0)
            qml.Hadamard(0)
            qml.CNOT((0, 1))
            for obs in obs_list:
                meas_type(op=obs)

        tape = qml.tape.QuantumScript.from_queue(q)
        _, fn = split_non_commuting(tape)
        assert all(np.array(fn(group_coeffs)) == np.arange(len(tape.measurements)))

        qs = qml.tape.QuantumScript(tape.operations, tape.measurements)
        _, fn = split_non_commuting(qs)
        assert all(np.array(fn(group_coeffs)) == np.arange(len(tape.measurements)))

    @pytest.mark.parametrize("convert_to_opmath", (True, False))
    @pytest.mark.parametrize("meas_type", wire_meas_fn)
    @pytest.mark.parametrize(
        "obs_list, group_coeffs",
        [(obs_list_2, [[1, 3], [0, 2, 4, 5]]), (obs_list_3, [[1, 4], [2, 5], [0, 3, 6]])],
    )
    def test_wire_non_commuting_group_right_reorder(
        self, meas_type, obs_list, convert_to_opmath, group_coeffs
    ):
        """Test that the output is of the correct order with wire MPs initialized using a
        combination of wires and observables"""
        # create a queue with several measurements of same type but with differnent non-commuting
        # observables
        if not convert_to_opmath:
            obs_list = [
                qml.operation.Tensor(*o) if isinstance(o, qml.ops.Prod) else o for o in obs_list
            ]
        with qml.queuing.AnnotatedQueue() as q:
            qml.PauliZ(0)
            qml.Hadamard(0)
            qml.CNOT((0, 1))
            for obs in obs_list:
                meas_type(op=obs)

            # initialize measurements using wires
            meas_type(wires=[0])

        tape = qml.tape.QuantumScript.from_queue(q)
        _, fn = split_non_commuting(tape)
        assert all(np.array(fn(group_coeffs)) == np.arange(len(tape.measurements)))

        qs = qml.tape.QuantumScript(tape.operations, tape.measurements)
        _, fn = split_non_commuting(qs)
        assert all(np.array(fn(group_coeffs)) == np.arange(len(tape.measurements)))

    @pytest.mark.parametrize("convert_to_opmath", (True, False))
    @pytest.mark.parametrize("meas_type", obs_meas_fn)
    def test_different_measurement_types(self, meas_type, convert_to_opmath):
        """Test that the measurements types of the split tapes are correct"""

        prod_type = qml.prod if convert_to_opmath else qml.operation.Tensor
        with qml.queuing.AnnotatedQueue() as q:
            qml.PauliZ(0)
            qml.Hadamard(0)
            qml.CNOT((0, 1))
            meas_type(op=prod_type(qml.Z(0), qml.Z(1)))
            meas_type(op=prod_type(qml.X(0), qml.X(1)))
            meas_type(op=qml.PauliZ(0))
            meas_type(op=qml.PauliX(0))

            # if the MP can also accept wires as arguments, add extra measurements to test
            if meas_type in wire_meas_fn:
                meas_type(wires=[0])
                meas_type(wires=[0, 1])

        tape = qml.tape.QuantumScript.from_queue(q)
        the_return_type = tape.measurements[0].return_type
        split, _ = split_non_commuting(tape)
        for new_tape in split:
            for meas in new_tape.measurements:
                assert meas.return_type == the_return_type

        qs = qml.tape.QuantumScript(tape.operations, tape.measurements)
        split, _ = split_non_commuting(qs)
        for new_tape in split:
            for meas in new_tape.measurements:
                assert meas.return_type == the_return_type

    @pytest.mark.parametrize("meas_type_1, meas_type_2", itertools.combinations(obs_meas_fn, 2))
    def test_mixed_measurement_types(self, meas_type_1, meas_type_2):
        """Test that mixing different combintations of MPs initialized using obs works correctly."""

        with qml.queuing.AnnotatedQueue() as q:
            qml.Hadamard(0)
            qml.Hadamard(1)
            meas_type_1(op=qml.PauliX(0))
            meas_type_1(op=qml.PauliZ(1))
            meas_type_2(op=qml.PauliZ(0))

        tape = qml.tape.QuantumScript.from_queue(q)
        split, _ = split_non_commuting(tape)

        assert len(split) == 2
        assert qml.equal(split[0].measurements[0], meas_type_1(op=qml.PauliX(0)))
        assert qml.equal(split[0].measurements[1], meas_type_1(op=qml.PauliZ(1)))
        assert qml.equal(split[1].measurements[0], meas_type_2(op=qml.PauliZ(0)))

    @pytest.mark.parametrize("meas_type_1, meas_type_2", itertools.combinations(wire_meas_fn, 2))
    def test_mixed_wires_measurement_types(self, meas_type_1, meas_type_2):
        """Test that mixing different combinations of MPs initialized using wires works correctly"""

        with qml.queuing.AnnotatedQueue() as q:
            qml.Hadamard(0)
            qml.Hadamard(1)
            meas_type_1(op=qml.PauliX(0))
            meas_type_1(wires=[1])
            meas_type_2(wires=[0])

        tape = qml.tape.QuantumScript.from_queue(q)
        split, _ = split_non_commuting(tape)

        assert len(split) == 2
        assert qml.equal(split[0].measurements[0], meas_type_1(op=qml.PauliX(0)))
        assert qml.equal(split[0].measurements[1], meas_type_1(wires=[1]))
        assert qml.equal(split[1].measurements[0], meas_type_2(wires=[0]))

    @pytest.mark.parametrize(
        "meas_type_1, meas_type_2", itertools.product(obs_meas_fn, wire_meas_fn)
    )
    def test_mixed_wires_obs_measurement_types(self, meas_type_1, meas_type_2):
        """Test that mixing different combinations of MPs initialized using wires and obs works
        correctly"""

        with qml.queuing.AnnotatedQueue() as q:
            qml.Hadamard(0)
            qml.Hadamard(1)
            meas_type_1(op=qml.PauliX(0))
            meas_type_2()
            meas_type_2(wires=[1])
            meas_type_2(wires=[0, 1])

        tape = qml.tape.QuantumScript.from_queue(q)
        split, _ = split_non_commuting(tape)

        assert len(split) == 2
        assert qml.equal(split[0].measurements[0], meas_type_1(op=qml.PauliX(0)))
        assert qml.equal(split[0].measurements[1], meas_type_2(wires=[1]))
        assert qml.equal(split[1].measurements[0], meas_type_2())
        assert qml.equal(split[1].measurements[1], meas_type_2(wires=[0, 1]))

    @pytest.mark.parametrize("batch_type", (tuple, list))
    def test_batch_of_tapes(self, batch_type):
        """Test that `split_non_commuting` can transform a batch of tapes"""

        # create a batch with two simple tapes
        tape1 = qml.tape.QuantumScript(
            [qml.RX(1.2, 0)], [qml.expval(qml.X(0)), qml.expval(qml.Y(0)), qml.expval(qml.X(1))]
        )
        tape2 = qml.tape.QuantumScript(
            [qml.RY(0.5, 0)], [qml.expval(qml.Z(0)), qml.expval(qml.Y(0))]
        )
        batch = batch_type([tape1, tape2])

        # test transform on the batch
        new_batch, post_proc_fn = split_non_commuting(batch)

        # test that transform has been applied correctly on the batch by explicitly comparing with splitted tapes
        tp1 = qml.tape.QuantumScript([qml.RX(1.2, 0)], [qml.expval(qml.X(0)), qml.expval(qml.X(1))])
        tp2 = qml.tape.QuantumScript([qml.RX(1.2, 0)], [qml.expval(qml.Y(0))])
        tp3 = qml.tape.QuantumScript([qml.RY(0.5, 0)], [qml.expval(qml.Z(0))])
        tp4 = qml.tape.QuantumScript([qml.RY(0.5, 0)], [qml.expval(qml.Y(0))])

        assert all(qml.equal(tapeA, tapeB) for tapeA, tapeB in zip(new_batch, [tp1, tp2, tp3, tp4]))

        # final (double) check: test postprocessing function on a fictitious results
        result = ([0.1, 0.2], 0.2, 0.3, 0.4)
        assert post_proc_fn(result) == ((0.1, 0.2, 0.2), (0.3, 0.4))

    def test_sprod_support(self):
        """Test that split_non_commuting works with scalar product pauli words."""

        ob1 = 2.0 * qml.prod(qml.X(0), qml.X(1))
        ob2 = 3.0 * qml.prod(qml.Y(0), qml.Y(1))
        ob3 = qml.s_prod(4.0, qml.X(1))

        tape = qml.tape.QuantumScript([], [qml.expval(o) for o in [ob1, ob2, ob3]])
        batch, fn = qml.transforms.split_non_commuting(tape)

        tape0 = qml.tape.QuantumScript([], [qml.expval(qml.prod(qml.Y(0), qml.Y(1)))])
        assert qml.equal(tape0, batch[0])
        tape1 = qml.tape.QuantumScript(
            [], [qml.expval(qml.prod(qml.X(0), qml.X(1))), qml.expval(qml.X(1))]
        )
        assert qml.equal(tape1, batch[1])

        in_res = (1.0, (2.0, 3.0))
        assert fn(in_res) == (4.0, 3.0, 12.0)


# measurements that require shots=True
required_shot_meas_fn = [qml.sample, qml.counts]

# measurements that can optionally have shots=True
optional_shot_meas_fn = [qml.probs, qml.expval, qml.var]


class TestIntegration:
    """Integration tests for ``qml.transforms.split_non_commuting()``"""

    @pytest.mark.parametrize("convert_to_opmath", (True, False))
    def test_expval_non_commuting_observables(self, convert_to_opmath):
        """Test expval with multiple non-commuting operators"""
        dev = qml.device("default.qubit", wires=6)

        prod_type = qml.prod if convert_to_opmath else qml.operation.Tensor

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(1)
            qml.Hadamard(0)
            qml.PauliZ(0)
            qml.Hadamard(3)
            qml.Hadamard(5)
            qml.T(5)
            return (
                qml.expval(prod_type(qml.Z(0), qml.Z(1))),
                qml.expval(qml.PauliX(0)),
                qml.expval(qml.PauliZ(1)),
                qml.expval(prod_type(qml.X(1), qml.X(4))),
                qml.expval(qml.PauliX(3)),
                qml.expval(qml.PauliY(5)),
            )

        res = circuit()
        assert isinstance(res, tuple)
        assert len(res) == 6
        assert all(isinstance(r, np.ndarray) for r in res)
        assert all(r.shape == () for r in res)

        res = qml.math.stack(res)

        assert all(np.isclose(res, np.array([0.0, -1.0, 0.0, 0.0, 1.0, 1 / np.sqrt(2)])))

    @pytest.mark.parametrize("convert_to_opmath", (True, False))
    def test_expval_non_commuting_observables_qnode(self, convert_to_opmath):
        """Test expval with multiple non-commuting operators as a transform program on the qnode."""
        dev = qml.device("default.qubit", wires=6)

        prod_type = qml.prod if convert_to_opmath else qml.operation.Tensor

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(1)
            qml.Hadamard(0)
            qml.PauliZ(0)
            qml.Hadamard(3)
            qml.Hadamard(5)
            qml.T(5)
            return (
                qml.expval(prod_type(qml.Z(0), qml.Z(1))),
                qml.expval(qml.PauliX(0)),
                qml.expval(qml.PauliZ(1)),
                qml.expval(prod_type(qml.X(1), qml.X(4))),
                qml.expval(qml.PauliX(3)),
                qml.expval(qml.PauliY(5)),
            )

        res = split_non_commuting(circuit)()

        assert isinstance(res, tuple)
        assert len(res) == 6
        assert all(isinstance(r, np.ndarray) for r in res)
        assert all(r.shape == () for r in res)

        res = qml.math.stack(res)

        assert all(np.isclose(res, np.array([0.0, -1.0, 0.0, 0.0, 1.0, 1 / np.sqrt(2)])))

    @pytest.mark.parametrize("convert_to_opmath", (True, False))
    def test_expval_probs_non_commuting_observables_qnode(self, convert_to_opmath):
        """Test expval with multiple non-commuting operators and probs with non-commuting wires as a
        transform program on the qnode."""
        dev = qml.device("default.qubit", wires=6)

        prod_type = qml.prod if convert_to_opmath else qml.operation.Tensor

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(1)
            qml.Hadamard(0)
            qml.PauliZ(0)
            qml.Hadamard(3)
            qml.Hadamard(5)
            qml.T(5)
            return (
                qml.probs(wires=[0, 1]),
                qml.probs(wires=[1]),
                qml.expval(qml.PauliZ(0)),
                qml.expval(prod_type(qml.X(1), qml.X(4))),
                qml.expval(qml.PauliX(3)),
                qml.expval(qml.PauliY(5)),
            )

        res = split_non_commuting(circuit)()

        assert isinstance(res, tuple)
        assert len(res) == 6
        assert all(isinstance(r, np.ndarray) for r in res)

        res_probs = qml.math.concatenate(res[:2], axis=0)
        res_expval = qml.math.stack(res[2:])

        assert all(np.isclose(res_probs, np.array([0.25, 0.25, 0.25, 0.25, 0.5, 0.5])))

        assert all(np.isclose(res_expval, np.array([0.0, 0.0, 1.0, 1 / np.sqrt(2)])))

    @pytest.mark.parametrize("convert_to_opmath", (True, False))
    def test_shot_vector_support(self, convert_to_opmath):
        """Test output is correct when using shot vectors"""

        dev = qml.device("default.qubit", wires=6, shots=(10000, (20000, 2), 30000))

        prod_type = qml.prod if convert_to_opmath else qml.operation.Tensor

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(1)
            qml.Hadamard(0)
            qml.PauliZ(0)
            qml.Hadamard(3)
            qml.Hadamard(5)
            qml.T(5)
            return (
                qml.expval(prod_type(qml.Z(0), qml.Z(1))),
                qml.expval(qml.PauliX(0)),
                qml.expval(qml.PauliZ(1)),
                qml.expval(prod_type(qml.Y(0), qml.Y(1), qml.Z(3), qml.Y(4), qml.X(5))),
                qml.expval(prod_type(qml.X(1), qml.X(4))),
                qml.expval(qml.PauliX(3)),
                qml.expval(qml.PauliY(5)),
            )

        res = circuit()
        assert isinstance(res, tuple)
        assert len(res) == 4
        assert all(isinstance(shot_res, tuple) for shot_res in res)
        assert all(len(shot_res) == 7 for shot_res in res)
        # pylint:disable=not-an-iterable
        assert all(
            all(list(isinstance(r, np.ndarray) and r.shape == () for r in shot_res))
            for shot_res in res
        )

        res = qml.math.stack([qml.math.stack(r) for r in res])

        assert np.allclose(
            res, np.array([0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1 / np.sqrt(2)]), atol=0.05
        )

    def test_shot_vector_support_sample(self):
        """Test output is correct when using shots and sample and expval measurements"""

        dev = qml.device("default.qubit", wires=2, shots=(10, 20))

        @qml.qnode(dev)
        def circuit():
            qml.PauliZ(0)
            return (qml.sample(wires=[0, 1]), qml.expval(qml.PauliZ(0)))

        res = split_non_commuting(circuit)()
        assert isinstance(res, tuple)
        assert len(res) == 2
        assert all(isinstance(shot_res, tuple) for shot_res in res)
        assert all(len(shot_res) == 2 for shot_res in res)
        # pylint:disable=not-an-iterable
        assert all(all(list(isinstance(r, np.ndarray) for r in shot_res)) for shot_res in res)

        assert all(
            shot_res[0].shape in [(10, 2), (20, 2)] and shot_res[1].shape == () for shot_res in res
        )

        # check all the wire samples are as expected
        sample_res = qml.math.concatenate(
            [qml.math.concatenate(shot_res[0], axis=0) for shot_res in res], axis=0
        )
        assert np.allclose(sample_res, 0.0, atol=0.05)

        expval_res = qml.math.stack([shot_res[1] for shot_res in res])
        assert np.allclose(expval_res, np.array([1.0, 1.0]), atol=0.05)

    def test_shot_vector_support_counts(self):
        """Test output is correct when using shots, counts and expval measurements"""

        dev = qml.device("default.qubit", wires=2, shots=(10, 20))

        @qml.qnode(dev)
        def circuit():
            qml.PauliZ(0)
            return (qml.counts(wires=[0, 1]), qml.expval(qml.PauliZ(0)))

        res = split_non_commuting(circuit)()
        assert isinstance(res, tuple)
        assert len(res) == 2
        assert all(isinstance(shot_res, tuple) for shot_res in res)
        assert all(len(shot_res) == 2 for shot_res in res)
        # pylint:disable=not-an-iterable
        assert all(
            isinstance(shot_res[0], dict) and isinstance(shot_res[1], np.ndarray)
            for shot_res in res
        )

        assert all(shot_res[1].shape == () for shot_res in res)

        # check all the wire counts are as expected
        assert all(shot_res[0]["00"] in [10, 20] for shot_res in res)

        expval_res = qml.math.stack([shot_res[1] for shot_res in res])
        assert np.allclose(expval_res, np.array([1.0, 1.0]), atol=0.05)


# Autodiff tests
exp_res = np.array([0.77015115, -0.47942554, 0.87758256])
exp_grad = np.array(
    [[-4.20735492e-01, -4.20735492e-01], [-8.77582562e-01, 0.0], [-4.79425539e-01, 0.0]]
)

exp_res_probs = np.array([0.88132907, 0.05746221, 0.05746221, 0.00374651, 0.0])
exp_grad_probs = np.array(
    [
        [-0.22504026, -0.22504026],
        [-0.01467251, 0.22504026],
        [0.22504026, -0.01467251],
        [0.01467251, 0.01467251],
        [0.0, 0.0],
    ]
)


class TestAutodiffSplitNonCommuting:
    """Autodiff tests for all frameworks"""

    @pytest.mark.parametrize("prod", (qml.prod, qml.operation.Tensor))
    @pytest.mark.autograd
    def test_split_with_autograd(self, prod):
        """Test that results after splitting are still differentiable with autograd"""
        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def circuit(params):
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=1)
            return (
                qml.expval(prod(qml.PauliZ(0), qml.PauliZ(1))),
                qml.expval(qml.PauliY(0)),
                qml.expval(qml.PauliZ(0)),
            )

        def cost(params):
            res = circuit(params)
            return qml.math.stack(res)

        params = pnp.array([0.5, 0.5])
        res = cost(params)
        grad = qml.jacobian(cost)(params)
        assert all(np.isclose(res, exp_res))
        assert all(np.isclose(grad, exp_grad).flatten())

    @pytest.mark.autograd
    @pytest.mark.parametrize("prod", (qml.prod, qml.operation.Tensor))
    def test_split_with_autograd_probs(self, prod):
        """Test resulting after splitting non-commuting tapes with expval and probs measurements
        are still differentiable with autograd"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(params):
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=1)
            return qml.probs(wires=[0, 1]), qml.expval(prod(qml.PauliX(0), qml.PauliX(1)))

        def cost(params):
            res = split_non_commuting(circuit)(params)
            return qml.math.concatenate([res[0]] + [qml.math.stack(res[1:])], axis=0)

        params = pnp.array([0.5, 0.5])
        res = cost(params)
        grad = qml.jacobian(cost)(params)
        assert all(np.isclose(res, exp_res_probs))
        assert all(np.isclose(grad, exp_grad_probs).flatten())

    @pytest.mark.jax
    @pytest.mark.parametrize("prod", (qml.prod, qml.operation.Tensor))
    def test_split_with_jax(self, prod):
        """Test that results after splitting are still differentiable with jax"""

        import jax
        import jax.numpy as jnp

        dev = qml.device("default.qubit.jax", wires=3)

        @qml.qnode(dev)
        def circuit(params):
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=1)
            return (
                qml.expval(prod(qml.PauliZ(0), qml.PauliZ(1))),
                qml.expval(qml.PauliY(0)),
                qml.expval(qml.PauliZ(0)),
            )

        params = jnp.array([0.5, 0.5])
        res = split_non_commuting(circuit)(params)
        grad = jax.jacobian(split_non_commuting(circuit))(params)
        assert all(np.isclose(res, exp_res, atol=0.05))
        assert all(np.isclose(grad, exp_grad, atol=0.05).flatten())

    @pytest.mark.jax
    @pytest.mark.parametrize("prod", (qml.prod, qml.operation.Tensor))
    def test_split_with_jax_probs(self, prod):
        """Test resulting after splitting non-commuting tapes with expval and probs measurements
        are still differentiable with jax"""
        import jax
        import jax.numpy as jnp

        dev = qml.device("default.qubit.jax", wires=2)

        @qml.qnode(dev)
        def circuit(params):
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=1)
            return (qml.probs(wires=[0, 1]), qml.expval(prod(qml.X(0), qml.X(1))))

        params = jnp.array([0.5, 0.5])
        res = split_non_commuting(circuit)(params)
        res = jnp.concatenate([res[0]] + [jnp.stack(res[1:])], axis=0)

        grad = jax.jacobian(split_non_commuting(circuit))(params)
        grad = jnp.concatenate([grad[0]] + [jnp.stack(grad[1:])], axis=0)

        assert all(np.isclose(res, exp_res_probs, atol=0.05))
        assert all(np.isclose(grad, exp_grad_probs, atol=0.05).flatten())

    @pytest.mark.jax
    @pytest.mark.parametrize("prod", (qml.prod, qml.operation.Tensor))
    def test_split_with_jax_multi_params(self, prod):
        """Test that results after splitting are still differentiable with jax
        with multiple parameters"""

        import jax
        import jax.numpy as jnp

        dev = qml.device("default.qubit.jax", wires=3)

        @qml.qnode(dev)
        def circuit(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            return (
                qml.expval(prod(qml.Z(0), qml.Z(1))),
                qml.expval(qml.PauliY(0)),
                qml.expval(qml.PauliZ(0)),
            )

        x = jnp.array(0.5)
        y = jnp.array(0.5)

        res = split_non_commuting(circuit)(x, y)
        grad = jax.jacobian(split_non_commuting(circuit), argnums=[0, 1])(x, y)

        assert all(np.isclose(res, exp_res))

        assert isinstance(grad, tuple)
        assert len(grad) == 3

        for i, meas_grad in enumerate(grad):
            assert isinstance(meas_grad, tuple)
            assert len(meas_grad) == 2
            assert all(isinstance(g, jnp.ndarray) and g.shape == () for g in meas_grad)

            assert np.allclose(meas_grad, exp_grad[i], atol=1e-5)

    @pytest.mark.jax
    @pytest.mark.parametrize("prod", (qml.prod, qml.operation.Tensor))
    def test_split_with_jax_multi_params_probs(self, prod):
        """Test that results after splitting are still differentiable with jax
        with multiple parameters"""

        import jax
        import jax.numpy as jnp

        dev = qml.device("default.qubit.jax", wires=2)

        @qml.qnode(dev)
        def circuit(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            return (qml.probs(wires=[0, 1]), qml.expval(prod(qml.X(0), qml.X(1))))

        x = jnp.array(0.5)
        y = jnp.array(0.5)

        res = split_non_commuting(circuit)(x, y)
        res = jnp.concatenate([res[0]] + [jnp.stack(res[1:])], axis=0)
        assert all(np.isclose(res, exp_res_probs))

        grad = jax.jacobian(split_non_commuting(circuit), argnums=[0, 1])(x, y)

        assert isinstance(grad, tuple)
        assert len(grad) == 2

        for meas_grad in grad:
            assert isinstance(meas_grad, tuple)
            assert len(meas_grad) == 2
            assert all(isinstance(g, jnp.ndarray) for g in meas_grad)

        # reshape the returned gradient to the right shape
        grad = jnp.concatenate(
            [
                jnp.concatenate([grad[0][0].reshape(-1, 1), grad[0][1].reshape(-1, 1)], axis=1),
                jnp.concatenate([grad[1][0].reshape(-1, 1), grad[1][1].reshape(-1, 1)], axis=1),
            ],
            axis=0,
        )
        assert all(np.isclose(grad, exp_grad_probs, atol=0.05).flatten())

    @pytest.mark.jax
    @pytest.mark.parametrize("prod", (qml.prod, qml.operation.Tensor))
    def test_split_with_jax_jit(self, prod):
        """Test that results after splitting are still differentiable with jax-jit"""

        import jax
        import jax.numpy as jnp

        dev = qml.device("default.qubit", wires=3)

        @jax.jit
        @qml.qnode(dev)
        def circuit(params):
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=1)
            return (
                qml.expval(prod(qml.Z(0), qml.PauliZ(1))),
                qml.expval(qml.PauliY(0)),
                qml.expval(qml.PauliZ(0)),
            )

        params = jnp.array([0.5, 0.5])
        res = circuit(params)
        grad = jax.jacobian(circuit)(params)
        assert all(np.isclose(res, exp_res))
        assert all(np.isclose(grad, exp_grad, atol=1e-5).flatten())

    @pytest.mark.jax
    @pytest.mark.parametrize("prod", (qml.prod, qml.operation.Tensor))
    def test_split_with_jax_jit_probs(self, prod):
        """Test resulting after splitting non-commuting tapes with expval and probs measurements
        are still differentiable with jax"""

        import jax
        import jax.numpy as jnp

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(params):
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=1)
            return (qml.probs(wires=[0, 1]), qml.expval(prod(qml.X(0), qml.PauliX(1))))

        params = jnp.array([0.5, 0.5])
        res = split_non_commuting(circuit)(params)
        res = jnp.concatenate([res[0]] + [jnp.stack(res[1:])], axis=0)

        grad = jax.jacobian(split_non_commuting(circuit))(params)
        grad = jnp.concatenate([grad[0]] + [jnp.stack(grad[1:])], axis=0)

        assert all(np.isclose(res, exp_res_probs, atol=0.05))
        assert all(np.isclose(grad, exp_grad_probs, atol=0.05).flatten())

    @pytest.mark.torch
    @pytest.mark.parametrize("prod", (qml.prod, qml.operation.Tensor))
    def test_split_with_torch(self, prod):
        """Test that results after splitting are still differentiable with torch"""

        import torch
        from torch.autograd.functional import jacobian

        dev = qml.device("default.qubit.torch", wires=3)

        @qml.qnode(dev)
        def circuit(params):
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=1)
            return (
                qml.expval(prod(qml.Z(0), qml.Z(1))),
                qml.expval(qml.PauliY(0)),
                qml.expval(qml.PauliZ(0)),
            )

        def cost(params):
            res = split_non_commuting(circuit)(params)
            return qml.math.stack(res)

        params = torch.tensor([0.5, 0.5], requires_grad=True)
        res = cost(params)
        grad = jacobian(cost, (params))
        assert all(np.isclose(res.detach().numpy(), exp_res))
        assert all(np.isclose(grad.detach().numpy(), exp_grad, atol=1e-5).flatten())

    @pytest.mark.torch
    @pytest.mark.parametrize("prod", (qml.prod, qml.operation.Tensor))
    def test_split_with_torch_probs(self, prod):
        """Test resulting after splitting non-commuting tapes with expval and probs measurements
        are still differentiable with torch"""

        import torch
        from torch.autograd.functional import jacobian

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(params):
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=1)
            return (qml.probs(wires=[0, 1]), qml.expval(prod(qml.X(0), qml.X(1))))

        def cost(params):
            res = split_non_commuting(circuit)(params)
            return qml.math.concatenate([res[0]] + [qml.math.stack(res[1:])], axis=0)

        params = torch.tensor([0.5, 0.5], requires_grad=True)
        res = cost(params)
        grad = jacobian(cost, (params))
        assert all(np.isclose(res.detach().numpy(), exp_res_probs))
        assert all(np.isclose(grad.detach().numpy(), exp_grad_probs, atol=1e-5).flatten())

    @pytest.mark.tf
    @pytest.mark.parametrize("prod", (qml.prod, qml.operation.Tensor))
    def test_split_with_tf(self, prod):
        """Test that results after splitting are still differentiable with tf"""

        import tensorflow as tf

        dev = qml.device("default.qubit.tf", wires=3)

        @qml.qnode(dev)
        def circuit(params):
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=1)
            return (
                qml.expval(prod(qml.Z(0), qml.Z(1))),
                qml.expval(qml.PauliY(0)),
                qml.expval(qml.PauliZ(0)),
            )

        params = tf.Variable([0.5, 0.5])
        res = circuit(params)
        with tf.GradientTape() as tape:
            loss = split_non_commuting(circuit)(params)
            loss = tf.stack(loss)

        grad = tape.jacobian(loss, params)
        assert all(np.isclose(res, exp_res))
        assert all(np.isclose(grad, exp_grad, atol=1e-5).flatten())

    @pytest.mark.parametrize("prod", (qml.prod, qml.operation.Tensor))
    @pytest.mark.tf
    def test_split_with_tf_probs(self, prod):
        """Test that results after splitting are still differentiable with tf"""

        import tensorflow as tf

        dev = qml.device("default.qubit.tf", wires=2)

        @qml.qnode(dev)
        def circuit(params):
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=1)
            return (qml.probs(wires=[0, 1]), qml.expval(prod(qml.X(0), qml.X(1))))

        params = tf.Variable([0.5, 0.5])
        with tf.GradientTape() as tape:
            res = split_non_commuting(circuit)(params)
            res = tf.concat([res[0]] + [tf.stack(res[1:])], axis=0)

        grad = tape.jacobian(res, params)
        assert all(np.isclose(res, exp_res_probs))
        assert all(np.isclose(grad, exp_grad_probs, atol=1e-5).flatten())


# Defines the device used for all tests
dev = qml.device("default.qubit", wires=4)

# Defines circuits to be used in queueing/output tests
with AnnotatedQueue() as q_tape1:
    qml.PauliX(0)
    H1 = qml.Hamiltonian([1.5], [qml.PauliZ(0) @ qml.PauliZ(1)])
    qml.expval(H1)
tape1 = QuantumScript.from_queue(q_tape1)

with AnnotatedQueue() as q_tape2:
    qml.Hadamard(0)
    qml.Hadamard(1)
    qml.PauliZ(1)
    qml.PauliX(2)
    H2 = qml.Hamiltonian(
        [1, 3, -2, 1, 1],
        [
            qml.PauliX(0) @ qml.PauliZ(2),
            qml.PauliZ(2),
            qml.PauliX(0),
            qml.PauliX(2),
            qml.PauliZ(0) @ qml.PauliX(1),
        ],
    )
    qml.expval(H2)
tape2 = QuantumScript.from_queue(q_tape2)

H3 = qml.Hamiltonian([1.5, 0.3], [qml.Z(0) @ qml.Z(1), qml.X(1)])

with AnnotatedQueue() as q3:
    qml.PauliX(0)
    qml.expval(H3)

tape3 = QuantumScript.from_queue(q3)

H4 = qml.Hamiltonian(
    [1, 3, -2, 1, 1, 1],
    [
        qml.PauliX(0) @ qml.PauliZ(2),
        qml.PauliZ(2),
        qml.PauliX(0),
        qml.PauliZ(2),
        qml.PauliZ(2),
        qml.PauliZ(0) @ qml.PauliX(1) @ qml.PauliY(2),
    ],
).simplify()

with AnnotatedQueue() as q4:
    qml.Hadamard(0)
    qml.Hadamard(1)
    qml.PauliZ(1)
    qml.PauliX(2)

    qml.expval(H4)

tape4 = QuantumScript.from_queue(q4)
TAPES = [tape1, tape2, tape3, tape4]
OUTPUTS = [-1.5, -6, -1.5, -8]


class TestSingleHamiltonian:
    """Tests that split_non_commuting works with a single Hamiltonian"""

    @pytest.mark.parametrize(("tape", "output"), zip(TAPES, OUTPUTS))
    def test_hamiltonians(self, tape, output):
        """Tests that the split_non_commuting transform returns the correct value"""

        tapes, fn = split_non_commuting(tape)
        results = dev.execute(tapes)
        expval = fn(results)

        assert np.isclose(output, expval)

        qs = QuantumScript(tape.operations, tape.measurements)
        tapes, fn = split_non_commuting(qs)
        results = dev.execute(tapes)
        expval = fn(results)
        assert np.isclose(output, expval)

    @pytest.mark.parametrize(("tape", "output"), zip(TAPES, OUTPUTS))
    def test_hamiltonians_no_grouping(self, tape, output):
        """Tests that the split_non_commuting transform returns the correct value
        if we switch grouping off"""

        tapes, fn = split_non_commuting(tape, group=False)
        results = dev.execute(tapes)
        expval = fn(results)

        assert np.isclose(output, expval)

        qs = QuantumScript(tape.operations, tape.measurements)
        tapes, fn = split_non_commuting(qs, group=False)
        results = dev.execute(tapes)
        expval = fn(results)

        assert np.isclose(output, expval)

    def test_grouping_is_used(self):
        """Test that the grouping in a Hamiltonian is used"""

        H = qml.Hamiltonian(
            [1.0, 2.0, 3.0], [qml.PauliZ(0), qml.PauliX(1), qml.PauliX(0)], grouping_type="qwc"
        )
        assert H.grouping_indices is not None

        with AnnotatedQueue() as q:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PauliX(wires=2)
            qml.expval(H)

        tape = QuantumScript.from_queue(q)
        tapes, _ = split_non_commuting(tape, group=False)
        assert len(tapes) == 2

        qs = QuantumScript(tape.operations, tape.measurements)
        tapes, _ = split_non_commuting(qs, group=False)
        assert len(tapes) == 2

    def test_number_of_tapes(self):
        """Tests that the the correct number of tapes is produced"""

        H = qml.Hamiltonian([1.0, 2.0, 3.0], [qml.PauliZ(0), qml.PauliX(1), qml.PauliX(0)])

        with AnnotatedQueue() as q:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PauliX(wires=2)
            qml.expval(H)

        tape = QuantumScript.from_queue(q)
        tapes, _ = split_non_commuting(tape, group=False)
        assert len(tapes) == 3

        tapes, _ = split_non_commuting(tape, group=True)
        assert len(tapes) == 2

    def test_number_of_qscripts(self):
        """Tests the correct number of quantum scripts are produced."""

        H = qml.Hamiltonian([1.0, 2.0, 3.0], [qml.PauliZ(0), qml.PauliX(1), qml.PauliX(0)])
        qs = QuantumScript(measurements=[qml.expval(H)])

        tapes, _ = split_non_commuting(qs, group=False)
        assert len(tapes) == 3

        tapes, _ = split_non_commuting(qs, group=True)
        assert len(tapes) == 2

    @pytest.mark.parametrize("shots", [None, 100])
    @pytest.mark.parametrize("group", [True, False])
    def test_shots_attribute(self, shots, group):
        """Tests that the shots attribute is copied to the new tapes"""
        H = qml.Hamiltonian([1.0, 2.0, 3.0], [qml.PauliZ(0), qml.PauliX(1), qml.PauliX(0)])

        with AnnotatedQueue() as q:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PauliX(wires=2)
            qml.expval(H)

        tape = QuantumScript.from_queue(q, shots=shots)
        new_tapes, _ = split_non_commuting(tape, group=group)

        assert all(new_tape.shots == tape.shots for new_tape in new_tapes)

    @pytest.mark.autograd
    def test_hamiltonian_dif_autograd(self, tol):
        """Tests that the split_non_commuting tape transform is differentiable with the Autograd interface"""

        H = qml.Hamiltonian(
            [-0.2, 0.5, 1], [qml.PauliX(1), qml.PauliZ(1) @ qml.PauliY(2), qml.PauliZ(0)]
        )

        var = pnp.array([0.1, 0.67, 0.3, 0.4, -0.5, 0.7, -0.2, 0.5, 1.0], requires_grad=True)
        output = 0.42294409781940356
        output2 = [
            9.68883500e-02,
            -2.90832724e-01,
            -1.04448033e-01,
            -1.94289029e-09,
            3.50307411e-01,
            -3.41123470e-01,
            0.0,
            -0.43657,
            0.64123,
        ]

        with AnnotatedQueue() as q:
            for _ in range(2):
                qml.RX(np.array(0), wires=0)
                qml.RX(np.array(0), wires=1)
                qml.RX(np.array(0), wires=2)
                qml.CNOT(wires=[0, 1])
                qml.CNOT(wires=[1, 2])
                qml.CNOT(wires=[2, 0])

            qml.expval(H)

        tape = QuantumScript.from_queue(q)

        def cost(x):
            new_tape = tape.bind_new_parameters(x, list(range(9)))
            tapes, fn = split_non_commuting(new_tape)
            res = qml.execute(tapes, dev, qml.gradients.param_shift)
            return fn(res)

        assert np.isclose(cost(var), output)

        grad = qml.grad(cost)(var)
        assert len(grad) == len(output2)
        for g, o in zip(grad, output2):
            assert np.allclose(g, o, atol=tol)

    @pytest.mark.tf
    def test_hamiltonian_dif_tensorflow(self):
        """Tests that the split_non_commuting tape transform is differentiable with the Tensorflow interface"""

        import tensorflow as tf

        inner_dev = qml.device("default.qubit")

        H = qml.Hamiltonian(
            [-0.2, 0.5, 1], [qml.PauliX(1), qml.PauliZ(1) @ qml.PauliY(2), qml.PauliZ(0)]
        )
        var = tf.Variable([[0.1, 0.67, 0.3], [0.4, -0.5, 0.7]], dtype=tf.float64)
        output = 0.42294409781940356
        output2 = [
            9.68883500e-02,
            -2.90832724e-01,
            -1.04448033e-01,
            -1.94289029e-09,
            3.50307411e-01,
            -3.41123470e-01,
        ]

        with tf.GradientTape() as gtape:
            with AnnotatedQueue() as q:
                for _i in range(2):
                    qml.RX(var[_i, 0], wires=0)
                    qml.RX(var[_i, 1], wires=1)
                    qml.RX(var[_i, 2], wires=2)
                    qml.CNOT(wires=[0, 1])
                    qml.CNOT(wires=[1, 2])
                    qml.CNOT(wires=[2, 0])
                qml.expval(H)

            tape = QuantumScript.from_queue(q)
            tapes, fn = split_non_commuting(tape)
            res = fn(qml.execute(tapes, inner_dev, qml.gradients.param_shift))

            assert np.isclose(res, output)

            g = gtape.gradient(res, var)
            assert np.allclose(list(g[0]) + list(g[1]), output2)

    @pytest.mark.parametrize(
        "H, expected",
        [
            # Contains only groups with single coefficients
            (qml.Hamiltonian([1, 2.0], [qml.PauliZ(0), qml.PauliX(0)]), -1),
            # Contains groups with multiple coefficients
            (qml.Hamiltonian([1.0, 2.0, 3.0], [qml.X(0), qml.X(0) @ qml.X(1), qml.Z(0)]), -3),
        ],
    )
    @pytest.mark.parametrize("grouping", [True, False])
    def test_processing_function_shot_vectors(self, H, expected, grouping):
        """Tests that the processing function works with shot vectors
        and grouping with different number of coefficients in each group"""

        dev_with_shot_vector = qml.device("default.qubit", shots=[(20000, 4)])
        if grouping:
            H.compute_grouping()

        @functools.partial(qml.transforms.split_non_commuting, group=grouping)
        @qml.qnode(dev_with_shot_vector)
        def circuit(inputs):
            qml.RX(inputs, wires=0)
            return qml.expval(H)

        res = circuit(np.pi)
        assert qml.math.shape(res) == (4,)
        assert qml.math.allclose(res, np.ones((4,)) * expected, atol=0.1)

    @pytest.mark.parametrize(
        "H, expected",
        [
            # Contains only groups with single coefficients
            (qml.Hamiltonian([1, 2.0], [qml.PauliZ(0), qml.PauliX(0)]), [1, 0, -1]),
            # Contains groups with multiple coefficients
            (
                qml.Hamiltonian([1.0, 2.0, 3.0], [qml.X(0), qml.X(0) @ qml.X(1), qml.Z(0)]),
                [3, 0, -3],
            ),
        ],
    )
    @pytest.mark.parametrize("grouping", [True, False])
    def test_processing_function_shot_vectors_broadcasting(self, H, expected, grouping):
        """Tests that the processing function works with shot vectors, parameter broadcasting,
        and grouping with different number of coefficients in each group"""

        dev_with_shot_vector = qml.device("default.qubit", shots=[(10000, 4)])
        if grouping:
            H.compute_grouping()

        @functools.partial(qml.transforms.split_non_commuting, group=grouping)
        @qml.qnode(dev_with_shot_vector)
        def circuit(inputs):
            qml.RX(inputs, wires=0)
            return qml.expval(H)

        res = circuit([0, np.pi / 2, np.pi])
        assert qml.math.shape(res) == (4, 3)
        assert qml.math.allclose(res, qml.math.stack([expected] * 4), atol=0.1)

    def test_constant_offset_grouping(self):
        """Test that split_non_commuting can handle a multi-term observable with a constant offset and grouping."""

        H = 2.0 * qml.I() + 3 * qml.X(0) + 4 * qml.X(0) @ qml.Y(1) + qml.Z(0)
        tape = qml.tape.QuantumScript([], [qml.expval(H)], shots=50)
        batch, fn = qml.transforms.split_non_commuting(tape, group=True)

        assert len(batch) == 2

        tape_0 = qml.tape.QuantumScript([], [qml.expval(qml.Z(0))], shots=50)
        tape_1 = qml.tape.QuantumScript(
            [], [qml.expval(qml.X(0)), qml.expval(qml.X(0) @ qml.Y(1))], shots=50
        )

        assert qml.equal(batch[0], tape_0)
        assert qml.equal(batch[1], tape_1)

        dummy_res = (1.0, (1.0, 1.0))
        processed_res = fn(dummy_res)
        assert qml.math.allclose(processed_res, 10.0)

    def test_constant_offset_no_grouping(self):
        """Test that split_non_commuting can handle a multi-term observable with a constant offset and no grouping.."""

        H = 2.0 * qml.I() + 3 * qml.X(0) + 4 * qml.X(0) @ qml.Y(1) + qml.Z(0)
        tape = qml.tape.QuantumScript([], [qml.expval(H)], shots=50)
        batch, fn = qml.transforms.split_non_commuting(tape, group=False)

        assert len(batch) == 3

        tape_0 = qml.tape.QuantumScript([], [qml.expval(qml.X(0))], shots=50)
        tape_1 = qml.tape.QuantumScript([], [qml.expval(qml.X(0) @ qml.Y(1))], shots=50)
        tape_2 = qml.tape.QuantumScript([], [qml.expval(qml.Z(0))], shots=50)

        assert qml.equal(batch[0], tape_0)
        assert qml.equal(batch[1], tape_1)
        assert qml.equal(batch[2], tape_2)

        dummy_res = (1.0, 1.0, 1.0)
        processed_res = fn(dummy_res)
        assert qml.math.allclose(processed_res, 10.0)

    def test_only_constant_offset(self):
        """Tests that split_non_commuting can handle a single Identity observable"""

        H = qml.Hamiltonian([1.5, 2.5], [qml.I(), qml.I()])

        @functools.partial(qml.transforms.split_non_commuting, group=False)
        @qml.qnode(dev)
        def circuit():
            return qml.expval(H)

        with dev.tracker:
            res = circuit()
        assert dev.tracker.totals == {}
        assert qml.math.allclose(res, 4.0)


with AnnotatedQueue() as s_tape1:
    qml.PauliX(0)
    S1 = qml.s_prod(1.5, qml.sum(qml.prod(qml.PauliZ(0), qml.PauliZ(1)), qml.Identity()))
    qml.expval(S1)
    qml.state()
    qml.expval(S1)

with AnnotatedQueue() as s_tape2:
    qml.Hadamard(0)
    qml.Hadamard(1)
    qml.PauliZ(1)
    qml.PauliX(2)
    S2 = qml.sum(
        qml.prod(qml.PauliX(0), qml.PauliZ(2)),
        qml.s_prod(3, qml.PauliZ(2)),
        qml.s_prod(-2, qml.PauliX(0)),
        qml.Identity(),
        qml.PauliX(2),
        qml.prod(qml.PauliZ(0), qml.PauliX(1)),
    )
    qml.expval(S2)
    qml.probs(op=qml.PauliZ(0))
    qml.expval(S2)

S3 = qml.sum(
    qml.s_prod(1.5, qml.prod(qml.PauliZ(0), qml.PauliZ(1))),
    qml.s_prod(0.3, qml.PauliX(1)),
    qml.Identity(),
)

with AnnotatedQueue() as s_tape3:
    qml.PauliX(0)
    qml.expval(S3)
    qml.probs(wires=[1, 3])
    qml.expval(qml.PauliX(1))
    qml.expval(S3)
    qml.probs(op=qml.PauliY(0))


S4 = qml.sum(
    qml.prod(qml.PauliX(0), qml.PauliZ(2), qml.Identity()),
    qml.s_prod(3, qml.PauliZ(2)),
    qml.s_prod(-2, qml.PauliX(0)),
    qml.s_prod(1.5, qml.Identity()),
    qml.PauliZ(2),
    qml.PauliZ(2),
    qml.prod(qml.PauliZ(0), qml.PauliX(1), qml.PauliY(2)),
)

with AnnotatedQueue() as s_tape4:
    qml.Hadamard(0)
    qml.Hadamard(1)
    qml.PauliZ(1)
    qml.PauliX(2)
    qml.expval(S4)
    qml.expval(qml.PauliX(2))
    qml.expval(S4)
    qml.expval(qml.PauliX(2))

s_qscript1 = QuantumScript.from_queue(s_tape1)
s_qscript2 = QuantumScript.from_queue(s_tape2)
s_qscript3 = QuantumScript.from_queue(s_tape3)
s_qscript4 = QuantumScript.from_queue(s_tape4)

SUM_QSCRIPTS = [s_qscript1, s_qscript2, s_qscript3, s_qscript4]
SUM_OUTPUTS = [
    [
        0,
        np.array(
            [
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                1.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
            ]
        ),
        0,
    ],
    [-5, np.array([0.5, 0.5]), -5],
    [-0.5, np.array([1.0, 0.0, 0.0, 0.0]), 0.0, -0.5, np.array([0.5, 0.5])],
    [-6.5, 0, -6.5, 0],
]


class TestSums:
    """Tests for the split_non_commuting with Sums"""

    def test_observables_on_same_wires(self):
        """Test that even if the observables are on the same wires, if they are different operations, they are separated.
        This is testing for a case that gave rise to a bug that occured due to a problem in MeasurementProcess.hash.
        """
        obs1 = qml.prod(qml.PauliX(0), qml.PauliX(1))
        obs2 = qml.prod(qml.PauliX(0), qml.PauliY(1))

        circuit = QuantumScript(measurements=[qml.expval(obs1), qml.expval(obs2)])
        batch, _ = split_non_commuting(circuit)
        assert len(batch) == 2
        assert qml.equal(batch[0][0], qml.expval(obs1))
        assert qml.equal(batch[1][0], qml.expval(obs2))

    @pytest.mark.parametrize(("qscript", "output"), zip(SUM_QSCRIPTS, SUM_OUTPUTS))
    def test_sums(self, qscript, output):
        """Tests that the split_non_commuting transform returns the correct value"""
        processed, _ = dev.preprocess()[0]([qscript])
        assert len(processed) == 1
        qscript = processed[0]
        tapes, fn = split_non_commuting(qscript)
        results = dev.execute(tapes)
        expval = fn(results)

        assert all(qml.math.allclose(o, e) for o, e in zip(output, expval))

    @pytest.mark.parametrize(("qscript", "output"), zip(SUM_QSCRIPTS, SUM_OUTPUTS))
    def test_sums_legacy_device(self, qscript, output):
        """Tests that the split_non_commuting transform returns the correct value"""
        dev_old = qml.device("default.qubit.legacy", wires=4)
        tapes, fn = qml.transforms.split_non_commuting(qscript)
        results = dev_old.batch_execute(tapes)
        expval = fn(results)

        assert all(qml.math.allclose(o, e) for o, e in zip(output, expval))

    @pytest.mark.parametrize(("qscript", "output"), zip(SUM_QSCRIPTS, SUM_OUTPUTS))
    def test_sums_no_grouping(self, qscript, output):
        """Tests that the split_non_commuting transform returns the correct value
        if we switch grouping off"""
        processed, _ = dev.preprocess()[0]([qscript])
        assert len(processed) == 1
        qscript = processed[0]
        tapes, fn = split_non_commuting(qscript, group=False)
        results = dev.execute(tapes)
        expval = fn(results)

        assert all(qml.math.allclose(o, e) for o, e in zip(output, expval))

    def test_grouping(self):
        """Test the grouping functionality"""
        S = qml.sum(qml.PauliZ(0), qml.s_prod(2, qml.PauliX(1)), qml.s_prod(3, qml.PauliX(0)))

        with AnnotatedQueue() as q:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PauliX(wires=2)
            qml.expval(S)

        qscript = QuantumScript.from_queue(q)

        tapes, _ = split_non_commuting(qscript, group=True)
        assert len(tapes) == 2

    def test_number_of_qscripts(self):
        """Tests the correct number of quantum scripts are produced."""

        S = qml.sum(qml.PauliZ(0), qml.s_prod(2, qml.PauliX(1)), qml.s_prod(3, qml.PauliX(0)))
        qs = QuantumScript(measurements=[qml.expval(S)])

        tapes, _ = split_non_commuting(qs, group=False)
        assert len(tapes) == 3

        tapes, _ = split_non_commuting(qs, group=True)
        assert len(tapes) == 2

    @pytest.mark.parametrize("shots", [None, 100])
    @pytest.mark.parametrize("group", [True, False])
    def test_shots_attribute(self, shots, group):
        """Tests that the shots attribute is copied to the new tapes"""
        H = qml.Hamiltonian([1.0, 2.0, 3.0], [qml.PauliZ(0), qml.PauliX(1), qml.PauliX(0)])

        with AnnotatedQueue() as q:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PauliX(wires=2)
            qml.expval(H)

        tape = QuantumScript.from_queue(q, shots=shots)
        new_tapes, _ = split_non_commuting(tape, group=group)

        assert all(new_tape.shots == tape.shots for new_tape in new_tapes)

    def test_non_sum_tape(self):
        """Test that the ``split_non_commuting`` function returns the input tape if it does not
        contain a single measurement with the expectation value of a Sum."""

        with AnnotatedQueue() as q:
            qml.expval(qml.PauliZ(0))

        tape = QuantumScript.from_queue(q)

        tapes, fn = split_non_commuting(tape)

        assert len(tapes) == 1
        assert isinstance(list(tapes[0])[0].obs, qml.PauliZ)
        # Old return types return a list for a single value:
        # e.g. qml.expval(qml.PauliX(0)) = [1.23]
        res = [1.23]
        assert fn(res) == 1.23

    @pytest.mark.parametrize("grouping", [True, False])
    def test_prod_tape(self, grouping):
        """Tests that ``split_non_commuting`` works with a single Prod measurement"""

        _dev = qml.device("default.qubit", wires=1)

        @functools.partial(qml.transforms.split_non_commuting, group=grouping)
        @qml.qnode(_dev)
        def circuit():
            return qml.expval(qml.prod(qml.PauliZ(0), qml.I()))

        assert circuit() == 1.0

    @pytest.mark.parametrize("grouping", [True, False])
    def test_sprod_tape(self, grouping):
        """Tests that ``split_non_commuting`` works with a single SProd measurement"""

        _dev = qml.device("default.qubit", wires=1)

        @functools.partial(qml.transforms.split_non_commuting, group=grouping)
        @qml.qnode(_dev)
        def circuit():
            return qml.expval(qml.s_prod(1.5, qml.Z(0)))

        assert circuit() == 1.5

    @pytest.mark.parametrize("grouping", [True, False])
    def test_no_obs_tape(self, grouping):
        """Tests tapes with only constant offsets (only measurements on Identity)"""

        _dev = qml.device("default.qubit", wires=1)

        @functools.partial(qml.transforms.split_non_commuting, group=grouping)
        @qml.qnode(_dev)
        def circuit():
            return qml.expval(qml.s_prod(1.5, qml.I(0)))

        with _dev.tracker:
            res = circuit()
        assert _dev.tracker.totals == {}
        assert qml.math.allclose(res, 1.5)

    @pytest.mark.parametrize("grouping", [True, False])
    def test_no_obs_tape_multi_measurement(self, grouping):
        """Tests tapes with only constant offsets (only measurements on Identity)"""

        _dev = qml.device("default.qubit", wires=1)

        @functools.partial(qml.transforms.split_non_commuting, group=grouping)
        @qml.qnode(_dev)
        def circuit():
            return qml.expval(qml.s_prod(1.5, qml.I())), qml.expval(qml.s_prod(2.5, qml.I()))

        with _dev.tracker:
            res = circuit()
        assert _dev.tracker.totals == {}
        assert qml.math.allclose(res, [1.5, 2.5])

    @pytest.mark.parametrize("grouping", [True, False])
    def test_split_non_commuting_broadcasting(self, grouping):
        """Tests that the split_non_commuting transform works with broadcasting"""

        _dev = qml.device("default.qubit", wires=3)

        @functools.partial(qml.transforms.split_non_commuting, group=grouping)
        @qml.qnode(_dev)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.RY(x, wires=1)
            qml.RX(x, wires=2)
            return (
                qml.expval(qml.PauliZ(0)),
                qml.expval(qml.prod(qml.PauliZ(1), qml.sum(qml.PauliY(2), qml.PauliX(2)))),
                qml.expval(qml.sum(qml.PauliZ(0), qml.s_prod(1.5, qml.PauliX(1)))),
            )

        res = circuit([0, np.pi / 3, np.pi / 2, np.pi])

        def _expected(theta):
            return [
                np.cos(theta / 2) ** 2 - np.sin(theta / 2) ** 2,
                -(np.cos(theta / 2) ** 2 - np.sin(theta / 2) ** 2) * np.sin(theta),
                np.cos(theta / 2) ** 2 - np.sin(theta / 2) ** 2 + 1.5 * np.sin(theta),
            ]

        expected = np.array([_expected(t) for t in [0, np.pi / 3, np.pi / 2, np.pi]]).T
        assert qml.math.allclose(res, expected)

    @pytest.mark.parametrize(
        "theta", [0, np.pi / 3, np.pi / 2, np.pi, [0, np.pi / 3, np.pi / 2, np.pi]]
    )
    @pytest.mark.parametrize("grouping", [True, False])
    def test_split_non_commuting_shot_vector(self, grouping, theta):
        """Tests that the split_non_commuting transform works with shot vectors"""

        _dev = qml.device("default.qubit", wires=3, shots=[(20000, 5)])

        @functools.partial(qml.transforms.split_non_commuting, group=grouping)
        @qml.qnode(_dev)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.RY(x, wires=1)
            qml.RX(x, wires=2)
            return (
                qml.expval(qml.PauliZ(0)),
                qml.expval(qml.prod(qml.PauliZ(1), qml.sum(qml.PauliY(2), qml.PauliX(2)))),
                qml.expval(qml.sum(qml.PauliZ(0), qml.s_prod(1.5, qml.PauliX(1)))),
            )

        if isinstance(theta, list):
            theta = np.array(theta)

        expected = [
            np.cos(theta / 2) ** 2 - np.sin(theta / 2) ** 2,
            -(np.cos(theta / 2) ** 2 - np.sin(theta / 2) ** 2) * np.sin(theta),
            np.cos(theta / 2) ** 2 - np.sin(theta / 2) ** 2 + 1.5 * np.sin(theta),
        ]

        res = circuit(theta)

        if isinstance(theta, np.ndarray):
            assert qml.math.shape(res) == (5, 3, 4)
        else:
            assert qml.math.shape(res) == (5, 3)

        for r in res:
            assert qml.math.allclose(r, expected, atol=0.05)

    @pytest.mark.autograd
    def test_sum_dif_autograd(self, tol):
        """Tests that the split_non_commuting tape transform is differentiable with the Autograd interface"""
        S = qml.sum(
            qml.s_prod(-0.2, qml.PauliX(1)),
            qml.s_prod(0.5, qml.prod(qml.PauliZ(1), qml.PauliY(2))),
            qml.s_prod(1, qml.PauliZ(0)),
        )

        var = pnp.array([0.1, 0.67, 0.3, 0.4, -0.5, 0.7, -0.2, 0.5, 1], requires_grad=True)
        output = 0.42294409781940356
        output2 = [
            9.68883500e-02,
            -2.90832724e-01,
            -1.04448033e-01,
            -1.94289029e-09,
            3.50307411e-01,
            -3.41123470e-01,
            0.0,
            -4.36578753e-01,
            6.41233474e-01,
        ]

        with AnnotatedQueue() as q:
            for _ in range(2):
                qml.RX(np.array(0), wires=0)
                qml.RX(np.array(0), wires=1)
                qml.RX(np.array(0), wires=2)
                qml.CNOT(wires=[0, 1])
                qml.CNOT(wires=[1, 2])
                qml.CNOT(wires=[2, 0])

            qml.expval(S)

        qscript = QuantumScript.from_queue(q)

        def cost(x):
            new_qscript = qscript.bind_new_parameters(x, list(range(9)))
            tapes, fn = split_non_commuting(new_qscript)
            res = qml.execute(tapes, dev, qml.gradients.param_shift)
            return fn(res)

        assert np.isclose(cost(var), output)

        grad = qml.grad(cost)(var)
        assert len(grad) == len(output2)
        for g, o in zip(grad, output2):
            assert np.allclose(g, o, atol=tol)

    @pytest.mark.tf
    def test_sum_dif_tensorflow(self):
        """Tests that the split_non_commuting tape transform is differentiable with the Tensorflow interface"""

        import tensorflow as tf

        S = qml.sum(
            qml.s_prod(-0.2, qml.PauliX(1)),
            qml.s_prod(0.5, qml.prod(qml.PauliZ(1), qml.PauliY(2))),
            qml.s_prod(1, qml.PauliZ(0)),
        )
        var = tf.Variable([[0.1, 0.67, 0.3], [0.4, -0.5, 0.7]], dtype=tf.float64)
        output = 0.42294409781940356
        output2 = [
            9.68883500e-02,
            -2.90832724e-01,
            -1.04448033e-01,
            -1.94289029e-09,
            3.50307411e-01,
            -3.41123470e-01,
        ]

        with tf.GradientTape() as gtape:
            with AnnotatedQueue() as q:
                for _i in range(2):
                    qml.RX(var[_i, 0], wires=0)
                    qml.RX(var[_i, 1], wires=1)
                    qml.RX(var[_i, 2], wires=2)
                    qml.CNOT(wires=[0, 1])
                    qml.CNOT(wires=[1, 2])
                    qml.CNOT(wires=[2, 0])
                qml.expval(S)

            qscript = QuantumScript.from_queue(q)
            tapes, fn = split_non_commuting(qscript)
            res = fn(qml.execute(tapes, dev, qml.gradients.param_shift))

            assert np.isclose(res, output)

            g = gtape.gradient(res, var)
            assert np.allclose(list(g[0]) + list(g[1]), output2)

    @pytest.mark.jax
    def test_sum_dif_jax(self, tol):
        """Tests that the split_non_commuting tape transform is differentiable with the Jax interface"""
        import jax
        from jax import numpy as jnp

        S = qml.sum(
            qml.s_prod(-0.2, qml.PauliX(1)),
            qml.s_prod(0.5, qml.prod(qml.PauliZ(1), qml.PauliY(2))),
            qml.s_prod(1, qml.PauliZ(0)),
        )

        var = jnp.array([0.1, 0.67, 0.3, 0.4, -0.5, 0.7, -0.2, 0.5, 1])
        output = 0.42294409781940356
        output2 = [
            9.68883500e-02,
            -2.90832724e-01,
            -1.04448033e-01,
            -1.94289029e-09,
            3.50307411e-01,
            -3.41123470e-01,
            0.0,
            -4.36578753e-01,
            6.41233474e-01,
        ]

        with AnnotatedQueue() as q:
            for _ in range(2):
                qml.RX(np.array(0), wires=0)
                qml.RX(np.array(0), wires=1)
                qml.RX(np.array(0), wires=2)
                qml.CNOT(wires=[0, 1])
                qml.CNOT(wires=[1, 2])
                qml.CNOT(wires=[2, 0])

            qml.expval(S)

        qscript = QuantumScript.from_queue(q)

        def cost(x):
            new_qscript = qscript.bind_new_parameters(x, list(range(9)))
            tapes, fn = split_non_commuting(new_qscript)
            res = qml.execute(tapes, dev, qml.gradients.param_shift)
            return fn(res)

        assert np.isclose(cost(var), output)

        grad = jax.grad(cost)(var)
        assert len(grad) == len(output2)
        for g, o in zip(grad, output2):
            assert np.allclose(g, o, atol=tol)
