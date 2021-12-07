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
Tests for the accessibility of the Task-Qubit device with PyTorch interface
"""
import pennylane as qml
import numpy as np
import pytest
import os

# Ensure GPU devices disabled if available
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

torch = pytest.importorskip("torch", minversion="1.8")
dist = pytest.importorskip("dask.distributed")


@pytest.mark.parametrize(
    "BACKEND",
    [
        ("default.qubit", "parameter-shift"),
        ("default.qubit", "backprop"),
        ("lightning.qubit", "parameter-shift"),
    ],
)
def test_integration_torch(dask_setup_teardown, BACKEND, tol=1e-5):
    """Test that the execution of task.qubit is possible and agrees with general use of default.qubit.torch"""
    tol = 1e-5
    wires = 3
    dev_torch = qml.device("default.qubit.torch", wires=wires)
    dev_task = qml.device("task.qubit", wires=wires, backend=BACKEND[0])
    p_torch = torch.tensor(np.random.rand(4), requires_grad=True)

    # Pull address from fixture
    client = dist.Client(address=dask_setup_teardown)

    @qml.qnode(dev_torch, cache=False, interface="torch", diff_method=BACKEND[1])
    def circuit_torch(x):
        for i in range(0, 4, 2):
            qml.RX(x[i], wires=0)
            qml.RY(x[i + 1], wires=1)
        for i in range(2):
            qml.CNOT(wires=[i, i + 1])
        return qml.expval(qml.PauliZ(0))

    @qml.qnode(dev_task, cache=False, interface="torch", diff_method=BACKEND[1])
    def circuit_task(x):
        for i in range(0, 4, 2):
            qml.RX(x[i], wires=0)
            qml.RY(x[i + 1], wires=1)
        for i in range(2):
            qml.CNOT(wires=[i, i + 1])
        return qml.expval(qml.PauliZ(0))

    res_torch = circuit_torch(p_torch)
    res_task = client.submit(circuit_task, p_torch).result()

    res_torch.backward()
    gres_torch = circuit_torch(p_torch.grad)

    def grad_task(params):
        res_task = circuit_task(params)
        res_task.backward()
        return circuit_task(params.grad)

    gres_task = client.submit(grad_task, p_torch).result()
    client.close()

    assert torch.allclose(res_torch, res_task, atol=tol, rtol=0)
    assert torch.allclose(gres_torch, gres_task, atol=tol, rtol=0)
