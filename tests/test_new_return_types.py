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
Unit tests for the :mod:`pennylane.utils` module.
"""
# pylint: disable=no-self-use,too-many-arguments,protected-access
import functools
import itertools

import numpy as np

import pennylane.numpy as pnp
import pytest

import pennylane as qml


class TestSingleReturnExecute:
    def test_state_default(self):
        dev = qml.device("default.qubit", wires=2)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.state()

        qnode = qml.QNode(circuit, dev)
        qnode.construct([0.5], {})
        res = qml.execute(tapes=[qnode.tape], device=dev, gradient_fn=None)

        assert res[0].shape == (1, 4)
        assert isinstance(res[0], np.ndarray)

    def test_state_mixed(self):
        dev = qml.device("default.mixed", wires=2)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.state()

        qnode = qml.QNode(circuit, dev)
        qnode.construct([0.5], {})
        res = qml.execute_new(tapes=[qnode.tape], device=dev, gradient_fn=None)
        assert res[0].shape == (4, 4)
        assert isinstance(res[0], np.ndarray)

    def test_density_matrix_default(self):
        dev = qml.device("default.qubit", wires=2)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.density_matrix(wires=0)

        qnode = qml.QNode(circuit, dev)
        qnode.construct([0.5], {})
        res = qml.execute_new(tapes=[qnode.tape], device=dev, gradient_fn=None)
        assert res[0].shape == (2, 2)
        assert isinstance(res[0], np.ndarray)

    def test_density_matrix_mixed(self):
        dev = qml.device("default.mixed", wires=2)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.density_matrix(wires=0)

        qnode = qml.QNode(circuit, dev)
        qnode.construct([0.5], {})
        res = qml.execute_new(tapes=[qnode.tape], device=dev, gradient_fn=None)
        assert res[0].shape == (2, 2)
        assert isinstance(res[0], np.ndarray)

    def test_expval(self):
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

    def test_mutual_info(self):
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

    def test_probs(self):
        dev = qml.device("default.qubit", wires=2)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.probs(wires=[0, 1])

        qnode = qml.QNode(circuit, dev)
        qnode.construct([0.5], {})

        res = qml.execute_new(tapes=[qnode.tape], device=dev, gradient_fn=None)

        assert res[0].shape == (4,)
        assert isinstance(res[0], np.ndarray)

    # Samples and counts


class TestMultipleReturns:
    def test_multiple_expval(self):
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

    def test_multiple_prob(self):
        dev = qml.device("default.qubit", wires=2)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.probs(wires=0), qml.probs(wires=[0, 1])

        qnode = qml.QNode(circuit, dev)
        qnode.construct([0.5], {})

        res = qml.execute_new(tapes=[qnode.tape], device=dev, gradient_fn=None)

        assert isinstance(res[0], tuple)
        assert len(res[0]) == 2

        assert isinstance(res[0][0], np.ndarray)
        assert res[0][0].shape == (2,)

        assert isinstance(res[0][1], np.ndarray)
        assert res[0][1].shape == (4,)

    def test_mix_probs_vn(self):
        dev = qml.device("default.qubit", wires=2)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return qml.probs(wires=[0, 1]), qml.vn_entropy(wires=0), qml.probs(wires=[1])

        qnode = qml.QNode(circuit, dev)
        qnode.construct([0.5], {})

        res = qml.execute_new(tapes=[qnode.tape], device=dev, gradient_fn=None)

        assert isinstance(res[0], tuple)
        assert len(res[0]) == 3

        assert isinstance(res[0][0], np.ndarray)
        assert res[0][0].shape == (4,)

        assert isinstance(res[0][1], np.ndarray)
        assert res[0][1].shape == ()

        assert isinstance(res[0][2], np.ndarray)
        assert res[0][2].shape == (2,)

    def test_list_multiple_expval(self):
        dev = qml.device("default.qubit", wires=3)

        def circuit(x):
            qml.Hadamard(wires=[0])
            qml.CRX(x, wires=[0, 1])
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(0, 3)]

        qnode = qml.QNode(circuit, dev)
        qnode.construct([0.5], {})

        res = qml.execute_new(tapes=[qnode.tape], device=dev, gradient_fn=None)

        assert isinstance(res[0], tuple)
        assert len(res[0]) == 3

        assert isinstance(res[0][0], np.ndarray)
        assert res[0][0].shape == ()

        assert isinstance(res[0][1], np.ndarray)
        assert res[0][1].shape == ()

        assert isinstance(res[0][2], np.ndarray)
        assert res[0][1].shape == ()
