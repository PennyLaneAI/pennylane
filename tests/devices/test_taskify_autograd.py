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
Tests for the taskify interface with autograd as INTERFACE.
"""
import pennylane as qml
import numpy as np
import pytest
import os

# Ensure GPU devices disabled if available
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

dist = pytest.importorskip("dask.distributed")


@pytest.mark.parametrize(
    "BACKEND",
    [
        ("default.qubit", "parameter-shift"),
        ("default.qubit", "backprop"),
        ("lightning.qubit", "parameter-shift"),
    ],
)
@pytest.mark.parametrize(
    "INTERFACE",
    [("autograd", lambda x: qml.numpy.array(x, requires_grad=True))],
)
def test_taskify_func(dask_setup_teardown, BACKEND, INTERFACE, tol=1e-5):
    """Test that the execution of task-based submission of circuit evaluations"""
    tol = 1e-5
    wires = 3
    dev = qml.device(BACKEND[0], wires=wires)
    dev_task = qml.device("task.qubit", wires=wires, backend=BACKEND[0])
    p = INTERFACE[1](np.random.rand(4))

    # Pull address from fixture
    client = dist.Client(address=dask_setup_teardown)

    @qml.qnode(dev, cache=False, interface=INTERFACE[0], diff_method=BACKEND[1])
    def circuit(x):
        for i in range(0, 4, 2):
            qml.RX(x[i], wires=0)
            qml.RY(x[i + 1], wires=1)
        for i in range(2):
            qml.CNOT(wires=[i, i + 1])
        return qml.expval(qml.PauliZ(0))

    @qml.qnode(dev_task, cache=False, interface=INTERFACE[0], diff_method=BACKEND[1])
    def circuit_task(x):
        for i in range(0, 4, 2):
            qml.RX(x[i], wires=0)
            qml.RY(x[i + 1], wires=1)
        for i in range(2):
            qml.CNOT(wires=[i, i + 1])
        return qml.expval(qml.PauliZ(0))

    res = circuit(p)
    res_task = qml.taskify(circuit_task)(p)
    res_task_f = qml.taskify(circuit_task, futures=True)(p)

    assert np.allclose(res, res_task, atol=tol, rtol=0)
    assert np.allclose(res, res_task_f.result(), atol=tol, rtol=0)
    client.close()


@pytest.mark.parametrize(
    "BACKEND",
    [
        ("default.qubit", "parameter-shift"),
        ("default.qubit", "backprop"),
        ("lightning.qubit", "parameter-shift"),
    ],
)
@pytest.mark.parametrize(
    "INTERFACE",
    [("autograd", lambda x: qml.numpy.array(x, requires_grad=True))],
)
def test_taskify_device(dask_setup_teardown, BACKEND, INTERFACE, tol=1e-5):
    """Test that conversion of a device to `task.qubit` equivalent"""
    tol = 1e-5
    wires = 3
    dev = qml.device(BACKEND[0], wires=wires)
    dev_task = qml.taskify_dev(dev)
    p = INTERFACE[1](np.random.rand(4))

    # Pull address from fixture
    client = dist.Client(address=dask_setup_teardown)

    @qml.qnode(dev, cache=False, interface=INTERFACE[0], diff_method=BACKEND[1])
    def circuit(x):
        for i in range(0, 4, 2):
            qml.RX(x[i], wires=0)
            qml.RY(x[i + 1], wires=1)
        for i in range(2):
            qml.CNOT(wires=[i, i + 1])
        return qml.expval(qml.PauliZ(0))

    @qml.qnode(dev_task, cache=False, interface=INTERFACE[0], diff_method=BACKEND[1])
    def circuit_task(x):
        for i in range(0, 4, 2):
            qml.RX(x[i], wires=0)
            qml.RY(x[i + 1], wires=1)
        for i in range(2):
            qml.CNOT(wires=[i, i + 1])
        return qml.expval(qml.PauliZ(0))

    res = circuit(p)
    res_task = qml.taskify(circuit_task)(p)
    res_task_f = qml.taskify(circuit_task, futures=True)(p)

    assert np.allclose(res, res_task, atol=tol, rtol=0)
    assert np.allclose(res, res_task_f.result(), atol=tol, rtol=0)
    client.close()


@pytest.mark.parametrize(
    "BACKEND",
    [
        ("default.qubit", "parameter-shift"),
        ("default.qubit", "backprop"),
        ("lightning.qubit", "parameter-shift"),
    ],
)
@pytest.mark.parametrize(
    "INTERFACE",
    [("autograd", lambda x: qml.numpy.array(x, requires_grad=True))],
)
def test_untaskify_result(dask_setup_teardown, BACKEND, INTERFACE, tol=1e-5):
    """Test that the sync of results to host"""
    tol = 1e-5
    wires = 3
    dev = qml.device(BACKEND[0], wires=wires)
    dev_task = qml.device("task.qubit", wires=wires, backend=BACKEND[0])
    p = INTERFACE[1](np.random.rand(4))

    # Pull address from fixture
    client = dist.Client(address=dask_setup_teardown)

    @qml.qnode(dev, cache=False, interface=INTERFACE[0], diff_method=BACKEND[1])
    def circuit(x):
        for i in range(0, 4, 2):
            qml.RX(x[i], wires=0)
            qml.RY(x[i + 1], wires=1)
        for i in range(2):
            qml.CNOT(wires=[i, i + 1])
        return qml.expval(qml.PauliZ(0))

    @qml.qnode(dev_task, cache=False, interface=INTERFACE[0], diff_method=BACKEND[1])
    def circuit_task(x):
        for i in range(0, 4, 2):
            qml.RX(x[i], wires=0)
            qml.RY(x[i + 1], wires=1)
        for i in range(2):
            qml.CNOT(wires=[i, i + 1])
        return qml.expval(qml.PauliZ(0))

    res = circuit(p)
    res_task_f = qml.taskify(circuit_task, futures=True)(p)

    assert np.allclose(res, qml.untaskify(res_task_f)(), atol=tol, rtol=0)
    client.close()


@pytest.mark.parametrize(
    "METHOD",
    [qml.taskify, qml.untaskify],
)
def test_taskify_result_noclient(METHOD):
    """Test untaskify exception throwing"""
    with pytest.raises(RuntimeError) as ex:
        METHOD(lambda _: "This will fail")([])

    assert "No running Dask client detected." in str(ex.value)
