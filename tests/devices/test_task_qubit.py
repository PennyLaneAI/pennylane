# Copyright 2021 Xanadu Quantum Technologies Inc.

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
Tests for the accessibility of the Task-Qubit device
"""
import pennylane as qml
import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", minversion="2.4")
dist = pytest.importorskip("dask.distributed")


def test_integration(tol=1e-5):
    """Test that the execution of task.qubit is possible and agrees with general use of default.qubit.tf"""
    tol = 1e-5
    wires = 3
    dev_tf = qml.device("default.qubit", wires=wires)
    dev_task = qml.device("task.qubit", wires=wires, backend="default.qubit")
    p_tf = tf.Variable(np.random.rand(4))

    cluster = dist.LocalCluster(n_workers=1, threads_per_worker=1)
    client = dist.Client(cluster)

    @qml.qnode(dev_tf, cache=False, interface="tf", diff_method="backprop")
    def circuit_tf_bp(x):
        for i in range(0, 4, 2):
            qml.RX(x[i], wires=0)
            qml.RY(x[i + 1], wires=1)
        for i in range(2):
            qml.CNOT(wires=[i, i + 1])
        return qml.expval(qml.PauliZ(0))

    @qml.qnode(dev_tf, cache=False, interface="tf", diff_method="parameter-shift")
    def circuit_tf_ps(x):
        for i in range(0, 4, 2):
            qml.RX(x[i], wires=0)
            qml.RY(x[i + 1], wires=1)
        for i in range(2):
            qml.CNOT(wires=[i, i + 1])
        return qml.expval(qml.PauliZ(0))

    @qml.qnode(dev_task, cache=False, interface="tf", diff_method="backprop")
    def circuit_task_bp(x):
        for i in range(0, 4, 2):
            qml.RX(x[i], wires=0)
            qml.RY(x[i + 1], wires=1)
        for i in range(2):
            qml.CNOT(wires=[i, i + 1])
        return qml.expval(qml.PauliZ(0))

    @qml.qnode(dev_task, cache=False, interface="tf", diff_method="parameter-shift")
    def circuit_task_ps(x):
        for i in range(0, 4, 2):
            qml.RX(x[i], wires=0)
            qml.RY(x[i + 1], wires=1)
        for i in range(2):
            qml.CNOT(wires=[i, i + 1])
        return qml.expval(qml.PauliZ(0))

    with tf.GradientTape() as tape_tf_bp:
        res_tf_bp = circuit_tf_bp(p_tf)

    with tf.GradientTape() as tape_tf_ps:
        res_tf_ps = circuit_tf_ps(p_tf)

    # Tasks must be submitted through the client
    res_task_bp = client.submit(circuit_task_bp, p_tf).result()
    res_task_ps = client.submit(circuit_task_ps, p_tf).result()

    assert np.allclose(res_tf_bp, res_task_bp, atol=tol, rtol=0)
    assert np.allclose(res_tf_ps, res_task_ps, atol=tol, rtol=0)
    assert np.allclose(res_tf_bp, res_task_ps, atol=tol, rtol=0)

    gres_tf_bp = tape_tf_bp.gradient(res_tf_bp, [p_tf])
    gres_tf_ps = tape_tf_ps.gradient(res_tf_ps, [p_tf])

    def grad_task_bp(params):
        with tf.GradientTape() as tape_task_bp:
            res_task_bp = circuit_task_bp(params)
        return tape_task_bp.gradient(res_task_bp, [params])

    def grad_task_ps(params):
        with tf.GradientTape() as tape_task_ps:
            res_task_ps = circuit_task_ps(params)
        return tape_task_ps.gradient(res_task_ps, [params])

    gres_task_bp = client.submit(grad_task_bp, p_tf).result()
    gres_task_ps = client.submit(grad_task_ps, p_tf).result()

    assert np.allclose(gres_tf_bp, gres_task_bp, atol=tol, rtol=0)
    assert np.allclose(gres_tf_ps, gres_task_ps, atol=tol, rtol=0)
    assert np.allclose(gres_tf_bp, gres_task_ps, atol=tol, rtol=0)
