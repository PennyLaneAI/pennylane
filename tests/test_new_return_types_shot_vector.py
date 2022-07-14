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

# Note: mutual info
# qml.mutual_info(wires0=[0], wires1=[1]), qml.vn_entropy(wires=[0])]


@pytest.mark.parametrize("shot_vector", [[1, 10, 10, 1000], [1, (10, 2), 1000]])
class TestShotVectorsAutograd:
    """TODO"""

    @pytest.mark.parametrize("measurement", single_scalar_output_measurements)
    def test_expval(self, shot_vector, measurement):
        """TODO"""
        dev = qml.device("default.qubit", wires=2, shots=shot_vector)

        @qml.qnode(device=dev)
        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.apply(measurement)

        res = circuit(0.5)

        all_shots = sum([shot_tuple.copies for shot_tuple in dev.shot_vector])

        assert isinstance(res, tuple)
        assert len(res) == all_shots
        assert all(r.shape == () for r in res)

    @pytest.mark.parametrize("wires", [[0], [2, 0], [1, 0], [2, 0, 1]])
    @pytest.mark.xfail
    def test_density_matrix(self, shot_vector, wires):
        """TODO"""
        dev = qml.device("default.qubit", wires=3, shots=shot_vector)

        @qml.qnode(device=dev)
        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.density_matrix(wires=wires)

        res = circuit(0.5)

        all_shots = sum([shot_tuple.copies for shot_tuple in dev.shot_vector])

        assert isinstance(res, tuple)
        assert len(res) == all_shots
        dim = 2 ** len(wires)
        assert all(r.shape == (dim, dim) for r in res)

    @pytest.mark.parametrize("measurement", [qml.sample(qml.PauliZ(0)), qml.sample(wires=[0])])
    def test_samples(self, shot_vector, measurement):
        """TODO"""
        dev = qml.device("default.qubit", wires=2, shots=shot_vector)

        @qml.qnode(device=dev)
        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.apply(measurement)

        res = circuit(0.5)

        all_shot_copies = [
            shot_tuple.shots for shot_tuple in dev.shot_vector for _ in range(shot_tuple.copies)
        ]

        assert len(res) == len(all_shot_copies)
        for r, shots in zip(res, all_shot_copies):

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

        @qml.qnode(device=dev)
        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.apply(measurement)

        res = circuit(0.5)

        all_shots = sum([shot_tuple.copies for shot_tuple in dev.shot_vector])

        assert isinstance(res, tuple)
        assert len(res) == all_shots
        assert all(isinstance(r, dict) for r in res)
