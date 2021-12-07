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
Tests for the accessibility of the Task-Qubit device with Pyjax interface
"""
import pennylane as qml
import numpy as np
import pytest
import os

# Ensure GPU devices disabled if available
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

jax = pytest.importorskip("jax", minversion="0.2")
dist = pytest.importorskip("dask.distributed")


@pytest.mark.parametrize(
    "BACKEND",
    [
        ("default.qubit", "parameter-shift"),
        ("default.qubit", "backprop"),
        ("lightning.qubit", "parameter-shift"),
    ],
)
def test_integration_jax(dask_setup_teardown, BACKEND, tol=1e-5):
    """Test that the execution of task.qubit is possible and agrees with general use of default.qubit.jax"""
    tol = 1e-5
    wires = 3
    dev_jax = qml.device("default.qubit.jax", wires=wires)
    dev_task = qml.device("task.qubit", wires=wires, backend=BACKEND[0])
    p_jax = jax.numpy.array(np.random.rand(4))

    # Pull address from fixture
    client = dist.Client(address=dask_setup_teardown)

    @qml.qnode(dev_jax, cache=False, interface="jax", diff_method=BACKEND[1])
    def circuit_jax(x):
        for i in range(0, 4, 2):
            qml.RX(x[i], wires=0)
            qml.RY(x[i + 1], wires=1)
        for i in range(2):
            qml.CNOT(wires=[i, i + 1])
        return qml.expval(qml.PauliZ(0))

    @qml.qnode(dev_task, cache=False, interface="jax", diff_method=BACKEND[1])
    def circuit_task(x):
        for i in range(0, 4, 2):
            qml.RX(x[i], wires=0)
            qml.RY(x[i + 1], wires=1)
        for i in range(2):
            qml.CNOT(wires=[i, i + 1])
        return qml.expval(qml.PauliZ(0))

    res_jax = circuit_jax(p_jax)
    res_task = client.submit(circuit_task, p_jax).result()

    grads = jax.grad(circuit_jax, argnums=(0))
    gres_jax = grads(p_jax)

    def grad_task(params):
        grads = jax.grad(circuit_task, argnums=(0))
        gres_jax = grads(params)
        return gres_jax

    client.scatter([1, 2, 3], broadcast=True)
    gres_task = client.submit(
        grad_task,
        p_jax,
    ).result()
    client.close()

    assert jax.numpy.allclose(res_jax, res_task, atol=tol, rtol=0)
    assert jax.numpy.allclose(gres_jax, gres_task, atol=tol, rtol=0)
