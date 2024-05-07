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
# pylint: disable=no-self-use, import-outside-toplevel, no-member, import-error
import itertools

import numpy as np
import pytest

import pennylane as qml
import pennylane.numpy as pnp
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
        assert fn([0.5]) == 0.5

        # test transform on qscript
        qs = qml.tape.QuantumScript(tape.operations, tape.measurements, shots=50)
        split, fn = split_non_commuting(qs)
        for t in split:
            assert t.shots == qs.shots

        assert len(split) == 1
        assert all(isinstance(i_qs, qml.tape.QuantumScript) for i_qs in split)
        assert fn([0.5]) == 0.5

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
        assert fn([0.5]) == 0.5

        # test transform on qscript
        qs = qml.tape.QuantumScript(tape.operations, tape.measurements)
        split, fn = split_non_commuting(qs)

        assert len(split) == 1
        assert all(isinstance(i_qs, qml.tape.QuantumScript) for i_qs in split)
        assert fn([0.5]) == 0.5

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

        # test postprocessing function applied to the transformed batch
        assert all(
            qml.equal(tapeA, tapeB)
            for sublist1, sublist2 in zip(post_proc_fn(new_batch), ((tp1, tp2), (tp3, tp4)))
            for tapeA, tapeB in zip(sublist1, sublist2)
        )

        # final (double) check: test postprocessing function on a fictitious results
        result = ("tp1", "tp2", "tp3", "tp4")
        assert post_proc_fn(result) == (("tp1", "tp2"), ("tp3", "tp4"))

    def test_sprod_support(self):
        """Test that split_non_commuting works with scalar product pauli words."""

        ob1 = 2.0 * qml.prod(qml.X(0), qml.X(1))
        ob2 = 3.0 * qml.prod(qml.Y(0), qml.Y(1))
        ob3 = qml.s_prod(4.0, qml.X(1))

        tape = qml.tape.QuantumScript([], [qml.expval(o) for o in [ob1, ob2, ob3]])
        batch, fn = qml.transforms.split_non_commuting(tape)

        tape0 = qml.tape.QuantumScript([], [qml.expval(ob2)])
        assert qml.equal(tape0, batch[0])
        tape1 = qml.tape.QuantumScript([], [qml.expval(ob1), qml.expval(ob3)])
        assert qml.equal(tape1, batch[1])

        in_res = (1.0, (2.0, 3.0))
        assert fn(in_res) == (2.0, 1.0, 3.0)


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
