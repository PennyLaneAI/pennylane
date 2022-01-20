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
Tests for the accessibility of the Task-Qubit device with autograd interface
"""
import pennylane as qml
import numpy as np
import pytest
import os
from typing import List, Tuple, Dict

# Ensure GPU devices disabled if available
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
dist = pytest.importorskip("dask.distributed")


@pytest.mark.parametrize(
    "BACKEND",
    [
        ("default.qubit.autograd", "parameter-shift"),
        ("default.qubit.autograd", "backprop"),
        ("lightning.qubit", "parameter-shift"),
    ],
)
def test_integration_autograd(dask_setup_teardown, BACKEND, tol=1e-5):
    """Test that the execution of task.qubit is possible and agrees with general use of default.qubit.autograd"""
    tol = 1e-5
    wires = 3
    dev = qml.device("default.qubit.autograd", wires=wires)
    dev_task = qml.device("task.qubit", wires=wires, backend=BACKEND[0])
    p = qml.numpy.array(np.random.rand(4), requires_grad=True)

    # Pull address from fixture
    client = dist.Client(address=dask_setup_teardown)

    @qml.qnode(dev, cache=False, interface="autograd", diff_method=BACKEND[1])
    def circuit(x):
        for i in range(0, 4, 2):
            qml.RX(x[i], wires=0)
            qml.RY(x[i + 1], wires=1)
        for i in range(2):
            qml.CNOT(wires=[i, i + 1])
        return [qml.expval(qml.PauliZ(i)) for i in range(wires)]

    @qml.qnode(dev_task, cache=False, interface="autograd", diff_method=BACKEND[1])
    def circuit_task(x):
        for i in range(0, 4, 2):
            qml.RX(x[i], wires=0)
            qml.RY(x[i + 1], wires=1)
        for i in range(2):
            qml.CNOT(wires=[i, i + 1])
        return [qml.expval(qml.PauliZ(i)) for i in range(wires)]

    res = circuit(p)
    res_task = client.submit(circuit_task, p).result()

    g_res = qml.jacobian(circuit)(p)

    def grad_task(params):
        gres_task = qml.jacobian(circuit_task)(params)
        return gres_task

    client.scatter([1, 2, 3], broadcast=True)
    gres_task = client.submit(
        grad_task,
        p,
    ).result()
    client.close()

    assert qml.numpy.allclose(res, res_task, atol=tol, rtol=0)
    assert qml.numpy.allclose(g_res, gres_task, atol=tol, rtol=0)


def test_autograd_serialization(dask_setup_teardown):
    "Test the serialization of qml.numpy.tensor datatypes"
    arr = qml.numpy.array(np.random.rand(4), requires_grad=True)
    client = dist.Client(address=dask_setup_teardown)

    # Serialize as part of client submission
    c_arr_future = client.scatter(arr, broadcast=True)
    c_arr = c_arr_future.result()

    assert qml.numpy.allclose(arr, c_arr)
