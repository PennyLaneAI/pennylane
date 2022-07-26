# Copyright 2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Unit tests for the new return types.
"""
import pytest

import numpy as np
import pennylane as qml

herm = np.diag([1, 2, 3, 4])
probs_data = [
    (None, [0]),
    (None, [0, 1]),
    (qml.PauliZ(0), None),
    (qml.Hermitian(herm, wires=[1, 0]), None),
]

wires = [2, 3, 4]


class TestSingleReturnExecute:
    """Test that single measurements return behavior does not change."""

    @pytest.mark.parametrize("wires", wires)
    def test_state_default(self, wires):
        """Return state with default.qubit."""
        dev = qml.device("default.qubit", wires=wires)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.state()

        qnode = qml.QNode(circuit, dev)
        qnode.construct([0.5], {})
        res = qml.execute_new(tapes=[qnode.tape], device=dev, gradient_fn=None)

        assert res[0].shape == (2**wires,)
        assert isinstance(res[0], np.ndarray)

    @pytest.mark.parametrize("wires", wires)
    def test_state_mixed(self, wires):
        """Return state with default.mixed."""
        dev = qml.device("default.mixed", wires=wires)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.state()

        qnode = qml.QNode(circuit, dev)
        qnode.construct([0.5], {})

        res = qml.execute_new(tapes=[qnode.tape], device=dev, gradient_fn=None)

        assert res[0].shape == (2**wires, 2**wires)
        assert isinstance(res[0], np.ndarray)

    @pytest.mark.parametrize("d_wires", wires)
    def test_density_matrix_default(self, d_wires):
        """Return density matrix with default.qubit."""
        dev = qml.device("default.qubit", wires=4)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.density_matrix(wires=range(0, d_wires))

        qnode = qml.QNode(circuit, dev)
        qnode.construct([0.5], {})

        res = qml.execute_new(tapes=[qnode.tape], device=dev, gradient_fn=None)

        assert res[0].shape == (2**d_wires, 2**d_wires)
        assert isinstance(res[0], np.ndarray)

    @pytest.mark.parametrize("d_wires", wires)
    def test_density_matrix_mixed(self, d_wires):
        """Return density matrix with default.mixed."""
        dev = qml.device("default.mixed", wires=4)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.density_matrix(wires=range(0, d_wires))

        qnode = qml.QNode(circuit, dev)
        qnode.construct([0.5], {})
        res = qml.execute_new(tapes=[qnode.tape], device=dev, gradient_fn=None)
        assert res[0].shape == (2**d_wires, 2**d_wires)
        assert isinstance(res[0], np.ndarray)

    def test_expval(self):
        """Return a single expval."""
        dev = qml.device("default.qubit", wires=2)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.expval(qml.PauliZ(wires=1))

        qnode = qml.QNode(circuit, dev)
        qnode.construct([0.5], {})

        res = qml.execute_new(tapes=[qnode.tape], device=dev, gradient_fn=None)

        assert res[0].shape == ()
        assert isinstance(res[0], np.ndarray)

    def test_var(self):
        """Return a single var."""
        dev = qml.device("default.qubit", wires=2)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.var(qml.PauliZ(wires=1))

        qnode = qml.QNode(circuit, dev)
        qnode.construct([0.5], {})

        res = qml.execute_new(tapes=[qnode.tape], device=dev, gradient_fn=None)

        assert res[0].shape == ()
        assert isinstance(res[0], np.ndarray)

    def test_vn_entropy(self):
        """Return a single vn entropy."""
        dev = qml.device("default.qubit", wires=2)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.vn_entropy(wires=0)

        qnode = qml.QNode(circuit, dev)
        qnode.construct([0.5], {})

        res = qml.execute_new(tapes=[qnode.tape], device=dev, gradient_fn=None)

        assert res[0].shape == ()
        assert isinstance(res[0], np.ndarray)

    def test_mutual_info(self):
        """Return a single mutual information."""
        dev = qml.device("default.qubit", wires=2)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.mutual_info(wires0=[0], wires1=[1])

        qnode = qml.QNode(circuit, dev)
        qnode.construct([0.5], {})

        res = qml.execute_new(tapes=[qnode.tape], device=dev, gradient_fn=None)

        assert res[0].shape == ()
        assert isinstance(res[0], np.ndarray)

    @pytest.mark.parametrize("op,wires", probs_data)
    def test_probs(self, op, wires):
        """Return a single prob."""
        dev = qml.device("default.qubit", wires=3)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.probs(op=op, wires=wires)

        qnode = qml.QNode(circuit, dev)
        qnode.construct([0.5], {})

        res = qml.execute_new(tapes=[qnode.tape], device=dev, gradient_fn=None)

        if wires is None:
            wires = op.wires

        assert res[0].shape == (2 ** len(wires),)
        assert isinstance(res[0], np.ndarray)

    # Samples and counts


# op1, wires1, op2, wires2
multi_probs_data = [
    (None, [0], None, [0]),
    (None, [0], None, [0, 1]),
    (None, [0, 1], None, [0]),
    (None, [0, 1], None, [0, 1]),
    (qml.PauliZ(0), None, qml.PauliZ(1), None),
    (None, [0], qml.PauliZ(1), None),
    (qml.PauliZ(0), None, None, [0]),
    (qml.PauliZ(1), None, qml.PauliZ(0), None),
]

wires = [([0], [1]), ([1], [0]), ([0], [0]), ([1], [1])]


class TestMultipleReturns:
    """Test the new return types for multiple measurements, it should always return a tuple containing the single
    measurements.
    """

    def test_multiple_expval(self):
        """Return multiple expvals."""
        dev = qml.device("default.qubit", wires=2)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.expval(qml.PauliZ(wires=0)), qml.expval(qml.PauliZ(wires=1))

        qnode = qml.QNode(circuit, dev)
        qnode.construct([0.5], {})

        res = qml.execute_new(tapes=[qnode.tape], device=dev, gradient_fn=None)

        assert isinstance(res[0], tuple)
        assert len(res[0]) == 2

        assert isinstance(res[0][0], np.ndarray)
        assert res[0][0].shape == ()

        assert isinstance(res[0][1], np.ndarray)
        assert res[0][1].shape == ()

    def test_multiple_var(self):
        """Return multiple vars."""
        dev = qml.device("default.qubit", wires=2)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.var(qml.PauliZ(wires=0)), qml.var(qml.PauliZ(wires=1))

        qnode = qml.QNode(circuit, dev)
        qnode.construct([0.5], {})

        res = qml.execute_new(tapes=[qnode.tape], device=dev, gradient_fn=None)

        assert isinstance(res[0], tuple)
        assert len(res[0]) == 2

        assert isinstance(res[0][0], np.ndarray)
        assert res[0][0].shape == ()

        assert isinstance(res[0][1], np.ndarray)
        assert res[0][1].shape == ()

    @pytest.mark.parametrize("op1,wires1,op2,wires2", multi_probs_data)
    def test_multiple_prob(self, op1, op2, wires1, wires2):
        """Return multiple probs."""
        dev = qml.device("default.qubit", wires=2)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.probs(op=op1, wires=wires1), qml.probs(op=op2, wires=wires2)

        qnode = qml.QNode(circuit, dev)
        qnode.construct([0.5], {})

        res = qml.execute_new(tapes=[qnode.tape], device=dev, gradient_fn=None)

        assert isinstance(res[0], tuple)
        assert len(res[0]) == 2

        if wires1 is None:
            wires1 = op1.wires

        if wires2 is None:
            wires2 = op2.wires

        assert isinstance(res[0][0], np.ndarray)
        assert res[0][0].shape == (2 ** len(wires1),)

        assert isinstance(res[0][1], np.ndarray)
        assert res[0][1].shape == (2 ** len(wires2),)

    @pytest.mark.parametrize("op1,wires1,op2,wires2", multi_probs_data)
    @pytest.mark.parametrize("wires3, wires4", wires)
    def test_mix_meas(self, op1, wires1, op2, wires2, wires3, wires4):
        """Return multiple different measurements."""
        dev = qml.device("default.qubit", wires=2)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return (
                qml.probs(op=op1, wires=wires1),
                qml.vn_entropy(wires=wires3),
                qml.probs(op=op2, wires=wires2),
                qml.expval(qml.PauliZ(wires=wires4)),
            )

        qnode = qml.QNode(circuit, dev)
        qnode.construct([0.5], {})

        res = qml.execute_new(tapes=[qnode.tape], device=dev, gradient_fn=None)

        if wires1 is None:
            wires1 = op1.wires

        if wires2 is None:
            wires2 = op2.wires

        assert isinstance(res[0], tuple)
        assert len(res[0]) == 4

        assert isinstance(res[0][0], np.ndarray)
        assert res[0][0].shape == (2 ** len(wires1),)

        assert isinstance(res[0][1], np.ndarray)
        assert res[0][1].shape == ()

        assert isinstance(res[0][2], np.ndarray)
        assert res[0][2].shape == (2 ** len(wires2),)

        assert isinstance(res[0][3], np.ndarray)
        assert res[0][3].shape == ()

    wires = [2, 3, 4, 5]

    @pytest.mark.parametrize("wires", wires)
    def test_list_multiple_expval(self, wires):
        """Return a comprehension list of multiple expvals."""
        dev = qml.device("default.qubit", wires=wires)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(0, wires)]

        qnode = qml.QNode(circuit, dev)
        qnode.construct([0.5], {})

        res = qml.execute_new(tapes=[qnode.tape], device=dev, gradient_fn=None)

        assert isinstance(res[0], tuple)
        assert len(res[0]) == wires

        for i in range(0, wires):
            assert isinstance(res[0][i], np.ndarray)
            assert res[0][i].shape == ()


pauliz = qml.PauliZ(wires=1)
proj = qml.Projector([1], wires=1)
hermitian = qml.Hermitian(np.diag([1, 2]), wires=0)

# Note: mutual info and vn_entropy do not support some shot vectors
# qml.mutual_info(wires0=[0], wires1=[1]), qml.vn_entropy(wires=[0])]
single_scalar_output_measurements = [
    qml.expval(pauliz),
    qml.var(pauliz),
    qml.expval(proj),
    qml.var(proj),
    qml.expval(hermitian),
    qml.var(hermitian),
]

herm = np.diag([1, 2, 3, 4])
probs_data = [
    (None, [0]),
    (None, [0, 1]),
    (qml.PauliZ(0), None),
    (qml.Hermitian(herm, wires=[1, 0]), None),
]


@pytest.mark.parametrize("shot_vector", [[10, 1000], [1, 10, 10, 1000], [1, (10, 2), 1000]])
class TestShotVectorsAutograd:
    """Test the support for executing tapes with single measurements using a
    device with shot vectors."""

    @pytest.mark.parametrize("measurement", single_scalar_output_measurements)
    def test_single_scalar(self, shot_vector, measurement):
        """Test a single scalar-valued measurement."""
        dev = qml.device("default.qubit", wires=2, shots=shot_vector)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.apply(measurement)

        qnode = qml.QNode(circuit, dev)
        qnode.construct([0.5], {})
        qnode.tape.is_sampled = True

        res = qml.execute_new(tapes=[qnode.tape], device=dev, gradient_fn=None)

        all_shots = sum([shot_tuple.copies for shot_tuple in dev.shot_vector])

        assert isinstance(res[0], tuple)
        assert len(res[0]) == all_shots
        assert all(r.shape == () for r in res[0])

    @pytest.mark.parametrize("op,wires", probs_data)
    def test_probs(self, shot_vector, op, wires):
        """Test a single probability measurement."""
        dev = qml.device("default.qubit", wires=2, shots=shot_vector)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.probs(op=op, wires=wires)

        qnode = qml.QNode(circuit, dev)
        qnode.construct([0.5], {})
        qnode.tape.is_sampled = True

        res = qml.execute_new(tapes=[qnode.tape], device=dev, gradient_fn=None)

        all_shots = sum([shot_tuple.copies for shot_tuple in dev.shot_vector])

        assert isinstance(res[0], tuple)
        assert len(res[0]) == all_shots
        wires_to_use = wires if wires else op.wires
        assert all(r.shape == (2 ** len(wires_to_use),) for r in res[0])

    @pytest.mark.parametrize("wires", [[0], [2, 0], [1, 0], [2, 0, 1]])
    @pytest.mark.xfail
    def test_density_matrix(self, shot_vector, wires):
        """Test a density matrix measurement."""
        dev = qml.device("default.qubit", wires=3, shots=shot_vector)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.density_matrix(wires=wires)

        qnode = qml.QNode(circuit, dev)
        qnode.construct([0.5], {})
        qnode.tape.is_sampled = True

        res = qml.execute_new(tapes=[qnode.tape], device=dev, gradient_fn=None)

        all_shots = sum([shot_tuple.copies for shot_tuple in dev.shot_vector])

        assert isinstance(res[0], tuple)
        assert len(res[0]) == all_shots
        dim = 2 ** len(wires)
        assert all(r.shape == (dim, dim) for r in res[0])

    @pytest.mark.parametrize("measurement", [qml.sample(qml.PauliZ(0)), qml.sample(wires=[0])])
    def test_samples(self, shot_vector, measurement):
        """Test the sample measurement."""
        dev = qml.device("default.qubit", wires=2, shots=shot_vector)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.apply(measurement)

        qnode = qml.QNode(circuit, dev)
        qnode.construct([0.5], {})
        qnode.tape.is_sampled = True

        res = qml.execute_new(tapes=[qnode.tape], device=dev, gradient_fn=None)

        all_shot_copies = [
            shot_tuple.shots for shot_tuple in dev.shot_vector for _ in range(shot_tuple.copies)
        ]

        assert len(res[0]) == len(all_shot_copies)
        for r, shots in zip(res[0], all_shot_copies):

            if shots == 1:
                # Scalar tensors
                assert r.shape == ()
            else:
                assert r.shape == (shots,)

    @pytest.mark.parametrize(
        "measurement", [qml.sample(qml.PauliZ(0), counts=True), qml.sample(wires=[0], counts=True)]
    )
    def test_counts(self, shot_vector, measurement):
        """Test the counts measurement."""
        dev = qml.device("default.qubit", wires=2, shots=shot_vector)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.apply(measurement)

        qnode = qml.QNode(circuit, dev)
        qnode.construct([0.5], {})
        qnode.tape.is_sampled = True

        res = qml.execute_new(tapes=[qnode.tape], device=dev, gradient_fn=None)

        all_shots = sum([shot_tuple.copies for shot_tuple in dev.shot_vector])

        assert isinstance(res[0], tuple)
        assert len(res[0]) == all_shots
        assert all(isinstance(r, dict) for r in res[0])


# 1. Expval/var with another measurement
#
# What else do we have:
# a) Probs
# b) i) Sample
#    ii) Counts
# c) Sample & density: no tests

pauliz_w2 = qml.PauliZ(wires=2)
proj_w2 = qml.Projector([1], wires=2)
hermitian = qml.Hermitian(np.diag([1, 2]), wires=0)
tensor_product = qml.PauliY(wires=2) @ qml.PauliX(wires=1)

single_scalar_probs_multi = [
    # Expval
    (qml.expval(pauliz_w2), qml.probs(wires=[2, 0])),
    (qml.expval(proj_w2), qml.probs(wires=[2, 0])),
    (qml.expval(tensor_product), qml.probs(wires=[2, 0])),
    # Var
    (qml.var(qml.PauliZ(wires=1)), qml.probs(wires=[0, 1])),
    (qml.var(proj_w2), qml.probs(wires=[2, 0])),
    (qml.var(tensor_product), qml.probs(wires=[2, 0])),
]

single_scalar_sample_multi = [
    # Expval
    (qml.expval(pauliz_w2), qml.sample(op=qml.PauliZ(1) @ qml.PauliZ(0))),
    (qml.expval(proj_w2), qml.sample(op=qml.PauliZ(1) @ qml.PauliZ(0))),
    (qml.expval(tensor_product), qml.sample(op=qml.PauliZ(0))),
    # Var
    (qml.var(proj_w2), qml.sample(op=qml.PauliZ(1) @ qml.PauliZ(0))),
    (qml.var(pauliz_w2), qml.sample(op=qml.PauliZ(1) @ qml.PauliZ(0))),
    (qml.var(tensor_product), qml.sample(op=qml.PauliZ(0))),
]

single_scalar_sample_no_obs_multi = [
    # TODO: for copy=1, the wires syntax has a bug
    # -----
    (qml.expval(qml.PauliZ(wires=1)), qml.sample(wires=[0, 1])),
    (qml.var(qml.PauliZ(wires=1)), qml.sample(wires=[0, 1])),
]

single_scalar_counts_multi = [
    # Expval
    (qml.expval(pauliz_w2), qml.sample(op=qml.PauliZ(1) @ qml.PauliZ(0), counts=True)),
    (qml.expval(proj_w2), qml.sample(op=qml.PauliZ(1) @ qml.PauliZ(0), counts=True)),
    (qml.expval(tensor_product), qml.sample(op=qml.PauliZ(0), counts=True)),
    # Var
    (qml.var(proj_w2), qml.sample(op=qml.PauliZ(1) @ qml.PauliZ(0), counts=True)),
    (qml.var(pauliz_w2), qml.sample(op=qml.PauliZ(1) @ qml.PauliZ(0), counts=True)),
    (qml.var(tensor_product), qml.sample(op=qml.PauliZ(0), counts=True)),
]

single_scalar_counts_no_obs_multi = [
    # TODO: for copy=1, the wires syntax has a bug
    # -----
    (qml.expval(qml.PauliZ(wires=1)), qml.sample(wires=[0, 1], counts=True)),
    (qml.var(qml.PauliZ(wires=1)), qml.sample(wires=[0, 1], counts=True)),
]


@pytest.mark.parametrize("shot_vector", [[10, 1000], [1, 10, 10, 1000], [1, (10, 2), 1000]])
class TestShotVectorsAutogradMultiMeasure:
    """Test the support for executing tapes with multiple measurements using a
    device with shot vectors"""

    @pytest.mark.parametrize("meas1,meas2", single_scalar_probs_multi)
    def test_single_scalar_probs(self, shot_vector, meas1, meas2):
        """Test scalar-valued and probability measurements"""
        dev = qml.device("default.qubit", wires=3, shots=shot_vector)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.apply(meas1), qml.apply(meas2)

        qnode = qml.QNode(circuit, dev)
        qnode.construct([0.5], {})
        qnode.tape.is_sampled = True

        res = qml.execute_new(tapes=[qnode.tape], device=dev, gradient_fn=None)

        all_shots = sum([shot_tuple.copies for shot_tuple in dev.shot_vector])

        assert isinstance(res[0], tuple)
        assert len(res[0]) == all_shots
        assert all(isinstance(r, tuple) for r in res[0])
        assert all(isinstance(m, np.ndarray) for measurement_res in res[0] for m in measurement_res)
        for meas_res in res[0]:
            for i, r in enumerate(meas_res):
                if i % 2 == 0:
                    assert r.shape == ()
                else:
                    assert r.shape == (2**2,)

    @pytest.mark.parametrize("meas1,meas2", single_scalar_sample_multi)
    def test_single_scalar_sample_with_obs(self, shot_vector, meas1, meas2):
        """Test scalar-valued and sample measurements where sample takes an
        observable."""
        dev = qml.device("default.qubit", wires=3, shots=shot_vector)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.apply(meas1), qml.apply(meas2)

        qnode = qml.QNode(circuit, dev)
        qnode.construct([0.5], {})
        qnode.tape.is_sampled = True

        res = qml.execute_new(tapes=[qnode.tape], device=dev, gradient_fn=None)

        all_shots = sum([shot_tuple.copies for shot_tuple in dev.shot_vector])

        assert isinstance(res[0], tuple)
        assert len(res[0]) == all_shots
        assert all(isinstance(r, tuple) for r in res[0])
        assert all(isinstance(m, np.ndarray) for measurement_res in res[0] for m in measurement_res)

        idx = 0
        for shot_tuple in dev.shot_vector:
            for _ in range(shot_tuple.copies):
                for i, r in enumerate(res[0][idx]):
                    if i % 2 == 0 or shot_tuple.shots == 1:
                        obs_provided = meas2.obs is not None
                        expected_shape = ()
                        assert r.shape == expected_shape
                    else:
                        assert r.shape == (shot_tuple.shots,)
                idx += 1

    @pytest.mark.parametrize("meas1,meas2", single_scalar_sample_no_obs_multi)
    @pytest.mark.xfail
    def test_single_scalar_sample_no_obs(self, shot_vector, meas1, meas2):
        """Test scalar-valued and computational basis sample measurements."""
        dev = qml.device("default.qubit", wires=3, shots=shot_vector)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.apply(meas1), qml.apply(meas2)

        qnode = qml.QNode(circuit, dev)
        qnode.construct([0.5], {})
        qnode.tape.is_sampled = True

        res = qml.execute_new(tapes=[qnode.tape], device=dev, gradient_fn=None)

        all_shots = sum([shot_tuple.copies for shot_tuple in dev.shot_vector])

        assert isinstance(res[0], tuple)
        assert len(res[0]) == all_shots
        assert all(isinstance(r, tuple) for r in res[0])
        assert all(isinstance(m, np.ndarray) for measurement_res in res[0] for m in measurement_res)

        for shot_tuple in dev.shot_vector:
            for idx in range(shot_tuple.copies):
                for i, r in enumerate(res[0][idx]):
                    expected_sample_shape_item = len(meas2.wires)
                    if i % 2 == 0 or shot_tuple.shots == 1:
                        obs_provided = meas2.obs is not None
                        expected_shape = ()
                        assert r.shape == expected_shape
                    else:
                        assert r.shape == (shot_tuple.shots,)

    @pytest.mark.parametrize("meas1,meas2", single_scalar_counts_multi)
    def test_single_scalar_counts_with_obs(self, shot_vector, meas1, meas2):
        """Test scalar-valued and counts measurements where counts takes an
        observable."""
        dev = qml.device("default.qubit", wires=3, shots=shot_vector)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.apply(meas1), qml.apply(meas2)

        qnode = qml.QNode(circuit, dev)
        qnode.construct([0.5], {})
        qnode.tape.is_sampled = True

        res = qml.execute_new(tapes=[qnode.tape], device=dev, gradient_fn=None)

        all_shots = sum([shot_tuple.copies for shot_tuple in dev.shot_vector])

        assert isinstance(res[0], tuple)
        assert len(res[0]) == all_shots
        assert all(isinstance(r, tuple) for r in res[0])

        for r in res[0]:
            assert isinstance(r[0], np.ndarray)
            assert isinstance(r[1], dict)

        idx = 0
        expected_outcomes = {-1, 1}
        for shot_tuple in dev.shot_vector:
            for _ in range(shot_tuple.copies):
                for i, r in enumerate(res[0][idx]):
                    if i % 2 == 0:
                        obs_provided = meas2.obs is not None
                        expected_shape = ()
                        assert r.shape == expected_shape
                    else:
                        # Samples are either -1 or 1
                        assert set(r.keys()).issubset(expected_outcomes)
                        assert sum(r.values()) == shot_tuple.shots
                idx += 1

    @pytest.mark.parametrize("meas1,meas2", single_scalar_counts_no_obs_multi)
    @pytest.mark.xfail
    def test_single_scalar_counts_no_obs(self, shot_vector, meas1, meas2):
        """Test scalar-valued and computational basis counts measurements."""
        dev = qml.device("default.qubit", wires=3, shots=shot_vector)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.apply(meas1), qml.apply(meas2)

        qnode = qml.QNode(circuit, dev)
        qnode.construct([0.5], {})
        qnode.tape.is_sampled = True

        res = qml.execute_new(tapes=[qnode.tape], device=dev, gradient_fn=None)

        all_shots = sum([shot_tuple.copies for shot_tuple in dev.shot_vector])

        assert isinstance(res[0], tuple)
        assert len(res[0]) == all_shots
        assert all(isinstance(r, tuple) for r in res[0])
        assert all(isinstance(m, np.ndarray) for measurement_res in res[0] for m in measurement_res)

        idx = 0
        for shot_tuple in dev.shot_vector:
            for _ in range(shot_tuple.copies):
                for i, r in enumerate(res[0][idx]):
                    # First component of the shot vector is 1
                    shots_is_one = idx == 0

                    expected_sample_shape_item = len(meas2.wires)
                    if i % 2 == 0 or shots_is_one:
                        obs_provided = meas2.obs is not None
                        expected_shape = ()
                        assert r.shape == expected_shape
                    else:
                        assert r.shape == (shot_tuple.shots,)
                idx += 1
