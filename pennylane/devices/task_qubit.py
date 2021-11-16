# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module contains a proxy qubit object for spawning multiple instances of 
a given qubit type for run on a background scheduler.
"""
from typing import List, Union, Tuple, Dict
from contextlib import nullcontext
import pennylane as qml
from pennylane import QubitDevice, DeviceError, QubitStateVector, BasisState
from .default_qubit import DefaultQubit
from .._version import __version__

try:
    import dask
    import dask.distributed as dist
    from distributed import get_client, secede, rejoin, worker_client
    from distributed.protocol import (
        dask_serialize,
        dask_deserialize,
        serialize,
        deserialize,
    )

    @dask_serialize.register(qml.numpy.tensor)
    def serialize(tensor: qml.numpy.tensor) -> Tuple[Dict, List[bytes]]:
        header, frames = dist.protocol.numpy.serialize_numpy_ndarray(tensor)
        header["type"] = "qml.numpy.tensor"
        header["requires_grad"] = tensor.requires_grad
        frames = [tensor.data]
        return header, frames

    @dask_deserialize.register(qml.numpy.tensor)
    def deserialize(header: Dict, frames: List[bytes]) -> qml.numpy.tensor:
        return qml.numpy.tensor(frames[0], requires_grad=header["requires_grad"])


except ImportError as e:  # pragma: no cover
    raise ImportError("task.qubit requires installing dask and dask.distributed") from e


class ProxyInstanceCLS(classmethod):
    """
    This utility class allows the use of both an instance
    as well as class method types. For situations where
    the explicit class method is needed, we can use this as a 
    decorator, and select the appropriate call dynamically.
    This is an essential support for accessing backend 
    functionality supports with the proxy `task.qubit`.

    The implementation is based on the Python descriptor guide
    as mentioned https://stackoverflow.com/questions/28237955/same-name-for-classmethod-and-instancemethod/28238047#28238047
    """
    def __get__(self, instance, type_):
        descr_get = super().__get__ if instance is None else self.__func__.__get__
        return descr_get(instance, type_)

class TaskQubit(QubitDevice):
    """Proxy simulator plugin written using Dask.Distributed as a task-distribution scheduling backend.

    **Short name:** ``task.qubit``

    This device provides a pure-state qubit simulator wrapping both ``"default.qubit"`` and ``"lightning.qubit"``,
    and written to allow batched offloading to a Dask scheduler.

    To use this device, you will need to install dask and dask.distributed:

    .. code-block:: console

        pip install dask distributed

    **Example**

    >>> import pennylane as qml
    >>> import tensorflow as tf
    >>> import dask.distributed as dist

    >>> if __name__ == '__main__': # Required for LocalCluster
    ...     dask_cluster = dist.LocalCluster(n_workers=4, threads_per_worker=1)
    ...     dask_client = dist.Client(dask_cluster)
    ...     dask_backend = "default.qubit.tf"
    ...     dev = qml.device("task.qubit", wires=6, backend=dask_backend)
    ...     @qml.beta.qnode(dev, cache=False, interface="tf") # caching must be disabled due to proxy interface
    ...     def circuit(x):
    ...         qml.RX(x[0], wires=0)
    ...         qml.RY(x[1], wires=0)
    ...         qml.RZ(x[2], wires=0)
    ...         qml.RZ(x[0], wires=1)
    ...         qml.RX(x[1], wires=1)
    ...         qml.RY(x[2], wires=1)
    ...         return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
    ...     weights = tf.Variable(qml.numpy.random.rand(3))

    ...     with tf.GradientTape() as tape:
    ...         # Use the circuit to calculate the loss value
    ...         loss = tf.abs(circuit(weights)-0.5)**2

    ...     print(tape.gradient(loss, weights))
    tf.Tensor([-0.39259075 -0.30759767 -0.49561258], shape=(3,), dtype=float64)

    Args:
        wires (int): The number of wires to initialize the device with.
        backend (None, str): Indicates the PennyLane device type to use for offloading
            computation tasks. The TensorFlow and PyTorch interfaces are the preferred types
            for gradient computations. This is due to existing support in Dask for
            TF and Torch datatypes.
        future (None, bool): Indicates whether the internal circuit evaluation returns a future
            to a result. This allows building of dependent workflows, but currently only works with
            explicit calls to `device.batch_execute` with a PennyLane native device type such as
            (`default.qubit`, `lightning.qubit`).
        gen_report: Indicates whether the backend task-scheduler will generate a performance report based on the tasks that were run.
    """

    operations = DefaultQubit.operations
    observables = DefaultQubit.observables

    name = "Task-based proxy PennyLane plugin"
    short_name = "task.qubit"
    pennylane_requires = __version__
    version = __version__
    author = "Xanadu Inc."

    supported_devices = {
        "default.qubit",
        "default.qubit.tf",
        "default.qubit.jax",
        "default.qubit.torch",
        "lightning.qubit",
    }

    def __init__(
        self,
        wires,
        *,
        shots=None,
        analytic=None,
        backend="default.qubit",
        gen_report: Union[bool, str] = False,
        future=False
    ):
        super().__init__(0, shots, cache=False)

        self._backend = backend
        self._backend_cls = qml.plugin_devices[backend].load()
        self._backend_dev = self._backend_cls(wires=0)
        self._wires = wires
        self._gen_report = gen_report
        self.num_wires = wires if isinstance(wires, int) else len(wires)
        self._shots = None
        self._cache = None
        self._future = future
        self.expand_fn = self.default_expand_fn
        self._shot_vector = None
        self._num_executions = 0

        if backend not in TaskQubit.supported_devices:
            raise ("Unsupported device backend.")

    def __get__(self, obj, objtype=None):
        if obj not in self.__dict__:
            return self._backend_cls.obj
        return self.__dict__[obj]

    def batch_execute(self, circuits: Union[List[qml.tape.QuantumTape], qml.beta.QNode], **kwargs):
        if self._gen_report:
            filename = self._gen_report if isinstance(self._gen_report, str) else "dask-report.html"
            from dask.distributed import performance_report

            cm = performance_report(filename=filename)
        else:
            cm = nullcontext()

        with cm:
            results = []
            if isinstance(circuits, qml.beta.QNode):
                with worker_client() as client:
                    results.append(
                        client.submit(
                            TaskQubit._execute_wrapper,
                            self._backend,
                            self._wires,
                            circuit,
                        )
                    )
            elif isinstance(circuits, dask.distributed.client.Future):
                with worker_client() as client:
                    results = client.submit(
                        lambda backend, wires, tapes: [
                            TaskQubit._execute_wrapper(backend, wires, i) for i in tapes
                        ],
                        self._backend,
                        self._wires,
                        circuits,
                    )
            else:
                with worker_client() as client:
                    for idx, circuit in enumerate(circuits):
                        results.append(
                            client.submit(
                                TaskQubit._execute_wrapper,
                                self._backend,
                                self._wires,
                                circuit,
                            )
                        )

            if self._future:
                return results
            else:
                with worker_client() as client:
                    res = client.gather(results)
                return res

    def apply():
        pass

    # Since we are using a proxy device, capabilities are handled by chosen backend
    @ProxyInstanceCLS
    def capabilities(self_cls):
        if not isinstance(self_cls, type):
            return self_cls._backend_cls.capabilities()
        else:
            capabilities = super().capabilities().copy()
            capabilities.update(
                model="qubit",
                supports_finite_shots=False,
                supports_reversible_diff=False,
                supports_inverse_operations=False,
                supports_analytic_computation=False,
                returns_state=False,
                returns_probs=False,
                passthru_devices={},
            )
            return capabilities

    @staticmethod
    def _execute_wrapper(
        backend: str,
        wires: int,
        circuit: Union[qml.QNode, qml.tape.QuantumTape],
    ):
        dev = qml.device(backend, wires=wires)
        return dev.execute(circuit)
