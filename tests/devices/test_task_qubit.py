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
import pytest
from pennylane.devices.task_qubit import ProxyHybridMethod
from pathlib import Path

dist = pytest.importorskip("dask.distributed")


@pytest.mark.parametrize(
    "BACKEND",
    [
        "my.qubit",
        "your.qubit",
    ],
)
def test_unsupported_backend(dask_setup_teardown, BACKEND, tol=1e-5):
    """Test that the execution of task.qubit is possible and agrees with general use of default.qubit.tf"""
    wires = 3
    client = dist.Client(address=dask_setup_teardown)

    with pytest.raises(qml.DeviceError) as excinfo:
        qml.device("task.qubit", wires=wires, backend=BACKEND)

    client.close()
    assert "Unsupported device backend" in str(excinfo.value)


def test_defines_correct_capabilities():
    """Test that the device defines the right capabilities"""

    cap = qml.devices.task_qubit.TaskQubit.capabilities()
    capabilities = {
        "model": "qubit",
        "supports_finite_shots": False,
        "supports_tensor_observables": True,
        "returns_probs": False,
        "returns_state": False,
        "supports_reversible_diff": False,
        "supports_inverse_operations": False,
        "supports_analytic_computation": False,
        "provides_adjoint_method": True,
        "passthru_devices": {},
        "is_proxy": True,
    }
    assert cap == capabilities


@pytest.mark.parametrize(
    "PERF_REPORT",
    [
        "myreport.html",
        True,
    ],
)
def test_perf_report(dask_setup_teardown, PERF_REPORT):
    """Test for the creation of a performance report from the tasks"""
    wires = 1
    client = dist.Client(address=dask_setup_teardown)
    dev_task = qml.device(
        "task.qubit", wires=wires, backend="default.qubit", gen_report=PERF_REPORT
    )

    @qml.qnode(dev_task, cache=False, diff_method="parameter-shift")
    def circuit(x):
        qml.RX(x[0], wires=0)
        qml.RY(x[1], wires=0)
        return qml.expval(qml.PauliZ(0))

    arr = qml.numpy.random.rand(2, requires_grad=True)
    client.submit(qml.grad(circuit), arr).result()
    rep = Path("dask-report.html") if isinstance(PERF_REPORT, bool) else Path(f"{PERF_REPORT}")
    rep_exists = rep.is_file()
    if rep_exists:
        rep.unlink()
    assert rep_exists


def test_return_future(dask_setup_teardown):
    """Test return of futures from device"""
    wires = 2
    client = dist.Client(address=dask_setup_teardown)
    dev_task = qml.device("task.qubit", wires=wires, backend="default.qubit", future=True)

    @qml.qnode(
        dev_task, cache=False, interface="autograd", diff_method="backprop"
    )  # caching must be disabled due to proxy interface
    def circuit(x):
        qml.RX(x[0], wires=0)
        qml.RY(x[1], wires=0)
        qml.RZ(x[2], wires=0)
        qml.RZ(x[0], wires=1)
        qml.RX(x[1], wires=1)
        qml.RY(x[2], wires=1)
        return [qml.expval(qml.PauliZ(i)) for i in range(2)]

    arr = qml.numpy.random.rand(3, requires_grad=True)
    res = circuit(arr)
    future = dev_task.batch_execute([circuit.tape])

    assert isinstance(future[0], dist.client.Future)
    assert qml.numpy.allclose(future[0].result(), res)
    client.close()


def test_execute_return_future(dask_setup_teardown):
    """Test return of futures from device"""
    wires = 2
    client = dist.Client(address=dask_setup_teardown)
    dev_task = qml.device("task.qubit", wires=wires, backend="default.qubit", future=True)

    @qml.qnode(
        dev_task, cache=False, interface="autograd", diff_method="backprop"
    )  # caching must be disabled due to proxy interface
    def circuit(x):
        qml.RX(x[0], wires=0)
        qml.RY(x[1], wires=0)
        qml.RZ(x[2], wires=0)
        qml.RZ(x[0], wires=1)
        qml.RX(x[1], wires=1)
        qml.RY(x[2], wires=1)
        return [qml.expval(qml.PauliZ(i)) for i in range(2)]

    arr = qml.numpy.random.rand(3, requires_grad=True)
    res = circuit(arr)
    future = dev_task.execute(circuit.tape)

    assert isinstance(future, dist.client.Future)
    assert qml.numpy.allclose(future.result(), res)
    client.close()


def test_str_repr_output():
    "Test device string representation"
    d_path = qml.devices.task_qubit
    wires = 1
    dev_task = qml.device(
        "task.qubit",
        wires=wires,
        backend="default.qubit",
    )
    dev = qml.device("default.qubit", wires=wires)

    assert "Backend: default.qubit" in str(dev_task)
    assert "Backend: default.qubit" in d_path.TaskQubit._str_dynamic(dev, Backend="default.qubit")
    assert "Backend: default.qubit" in repr(dev_task)
    assert "Backend: default.qubit" in d_path.TaskQubit._repr_dynamic(dev, Backend="default.qubit")


def test_instance_vs_class_method():
    "Test the ability to have different return results for class and instance methods with the same name"

    class DummyClass:
        def __init__(self):
            pass

        @ProxyHybridMethod
        def instance_or_class(cls):
            return True

        @instance_or_class.instancemethod
        def instance_or_class(self):
            return False

    d = DummyClass()
    assert d.instance_or_class() == False
    assert DummyClass.instance_or_class() == True
