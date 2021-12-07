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
import os

# from _pytest import monkeypatch

# Ensure GPU devices disabled if available
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

tf = pytest.importorskip("tensorflow", minversion="2.4")
dist = pytest.importorskip("dask.distributed")


@pytest.mark.parametrize(
    "BACKEND",
    [
        ("default.qubit", "parameter-shift"),
        ("default.qubit", "backprop"),
        ("lightning.qubit", "parameter-shift"),
    ],
)
def test_integration_tf(dask_setup_teardown, BACKEND, tol=1e-5):
    """Test that the execution of task.qubit is possible and agrees with general use of default.qubit.tf"""
    tol = 1e-5
    wires = 3
    dev_tf = qml.device("default.qubit.tf", wires=wires)
    dev_task = qml.device("task.qubit", wires=wires, backend=BACKEND[0])
    p_tf = tf.Variable(np.random.rand(4))

    client = dist.Client(address=dask_setup_teardown)

    @qml.qnode(dev_tf, cache=False, interface="tf", diff_method=BACKEND[1])
    def circuit_tf(x):
        for i in range(0, 4, 2):
            qml.RX(x[i], wires=0)
            qml.RY(x[i + 1], wires=1)
        for i in range(2):
            qml.CNOT(wires=[i, i + 1])
        return qml.expval(qml.PauliZ(0))

    @qml.qnode(dev_task, cache=False, interface="tf", diff_method=BACKEND[1])
    def circuit_task(x):
        for i in range(0, 4, 2):
            qml.RX(x[i], wires=0)
            qml.RY(x[i + 1], wires=1)
        for i in range(2):
            qml.CNOT(wires=[i, i + 1])
        return qml.expval(qml.PauliZ(0))

    with tf.GradientTape() as tape_tf:
        res_tf = circuit_tf(p_tf)

    # Tasks must be submitted through the client
    res_task = client.submit(circuit_task, p_tf).result()

    gres_tf = tape_tf.gradient(res_tf, [p_tf])

    def grad_task(params):
        with tf.GradientTape() as tape_task:
            res_task = circuit_task(params)
        return tape_task.gradient(res_task, [params])

    gres_task = client.submit(grad_task, p_tf).result()
    client.close()

    assert np.allclose(res_tf, res_task, atol=tol, rtol=0)
    assert np.allclose(gres_tf, gres_task, atol=tol, rtol=0)
