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
"""
Unit tests for the :mod:`pennylane.utils` module.
"""
# pylint: disable=no-self-use,too-many-arguments,protected-access
import functools
import itertools
import pytest

import numpy as np

import pennylane as qml
import pennylane.utils as pu
import scipy.sparse

from pennylane import Identity, PauliX, PauliY, PauliZ
from pennylane.operation import Tensor


single_scalar_output_measurements = [qml.expval(qml.PauliZ(wires=1)), qml.var(qml.PauliZ(wires=1))]

# Note: mutual info and vn_entropy do not support some shot vectors
# qml.mutual_info(wires0=[0], wires1=[1]), qml.vn_entropy(wires=[0])]

herm = np.diag([1, 2, 3, 4])
probs_data = [
    (None, [0]),
    (None, [0, 1]),
    (qml.PauliZ(0), None),
    (qml.Hermitian(herm, wires=[1, 0]), None),
]


@pytest.mark.parametrize("shot_vector", [[1, 10, 10, 1000], [1, (10, 2), 1000]])
class TestShotVectorsAutograd:
    """TODO"""

    @pytest.mark.parametrize("measurement", single_scalar_output_measurements)
    def test_expval(self, shot_vector, measurement):
        """TODO"""
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
        """TODO"""
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
        """TODO"""
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
        """TODO"""
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
        """TODO"""
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


expval_probs_multi = [
    (qml.expval(qml.PauliZ(wires=2)), qml.probs(wires=[2, 0])),
    (qml.expval(qml.PauliZ(wires=2)), qml.probs(op=qml.PauliZ(1) @ qml.PauliZ(0))),
    (qml.var(qml.PauliZ(wires=1)), qml.probs(wires=[0, 1])),
]

expval_sample_multi = [
    # TODO:
    # For copy=1, the wires syntax has a bug
    # (qml.expval(qml.PauliZ(wires=2)), qml.sample(wires=[2,0])),
    # (qml.var(qml.PauliZ(wires=1)), qml.sample(wires=[0, 1])),
    # -----
    (qml.expval(qml.PauliZ(wires=2)), qml.sample(op=qml.PauliZ(1) @ qml.PauliZ(0))),
    (qml.var(qml.PauliZ(wires=2)), qml.sample(op=qml.PauliZ(1) @ qml.PauliZ(0))),
]

# TODO: test Projector expval/var!


@pytest.mark.parametrize("shot_vector", [[1, 10, 10, 1000], [1, (10, 2), 1000]])
class TestShotVectorsAutogradMultiMeasure:
    """TODO"""

    @pytest.mark.parametrize("meas1,meas2", expval_probs_multi)
    def test_expval_probs(self, shot_vector, meas1, meas2):
        """TODO"""
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

    @pytest.mark.parametrize("meas1,meas2", expval_sample_multi)
    def test_expval_sample(self, shot_vector, meas1, meas2):
        """TODO"""
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
                    if i % 2 == 0 or idx == 0:
                        assert r.shape == ()
                    else:
                        assert r.shape == (shot_tuple.shots,)
                idx += 1
