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
from typing import List, Union, Tuple, Dict, Iterable
from contextlib import nullcontext
import pennylane as qml

from .default_qubit import DefaultQubit
from .._version import __version__

try:
    import dask
    import dask.distributed as dist
    from dask.distributed import worker_client, performance_report
    import dask

    # Ensure tasks are scheduled on processes rather than threads
    dask.config.set(scheduler="processes")

    from distributed.protocol import (
        dask_serialize,
        dask_deserialize,
    )

except ImportError as e:  # pragma: no cover
    raise ImportError("task.qubit requires installing dask and dask.distributed") from e


@dask_serialize.register(qml.numpy.tensor)
def serialize(tensor: qml.numpy.tensor) -> Tuple[Dict, List[bytes]]:
    "Defines a serializer for the Dask backend"
    header, frames = dist.protocol.numpy.serialize_numpy_ndarray(tensor)
    header["type"] = "qml.numpy.tensor"
    header["requires_grad"] = tensor.requires_grad
    frames = [tensor.data]
    return header, frames


@dask_deserialize.register(qml.numpy.tensor)
def deserialize(header: Dict, frames: List[bytes]) -> qml.numpy.tensor:
    "Defines a deserializer for the Dask backend"
    return qml.numpy.tensor(frames[0], requires_grad=header["requires_grad"])


class ProxyHybridMethod:
    """
    This utility class allows the use of both an instance
    as well as class method types. For situations where
    the explicit class method is needed, we can use this as a
    decorator, and select the appropriate call dynamically.
    This is an essential support for accessing backend
    functionality supports with the proxy `task.qubit`.

    The implementation is based on the Python descriptor guide
    as mentioned at https://stackoverflow.com/questions/28237955/same-name-for-classmethod-and-instancemethod/28238047#28238047
    """

    def __init__(self, fclass, finstance=None, doc=None):
        self.fclass = fclass
        self.finstance = finstance
        self.__doc__ = doc or fclass.__doc__
        self.__isabstractmethod__ = bool(getattr(fclass, "__isabstractmethod__", False))

    def classmethod(self, fclass):
        """
        Defines the decorated method as a class-method.
        """
        return type(self)(fclass, self.finstance, None)

    def instancemethod(self, finstance):
        """
        Defines the decorated method as an instance-method.
        """
        return type(self)(self.fclass, finstance, self.__doc__)

    def __get__(self, instance, cls):
        if instance is None or self.finstance is None:
            return self.fclass.__get__(cls, None)
        return self.finstance.__get__(instance, cls)


class TaskQubit(DefaultQubit):
    """Proxy simulator plugin written using Dask.Distributed as a task-distribution scheduling backend.

    **Short name:** ``task.qubit``

    This device provides a pure-state qubit simulator wrapping both ``"default.qubit"`` and ``"lightning.qubit"``,
    and written to allow batched offloading to a Dask scheduler. The ``task.qubit`` device works with the Autograd,
    TensorFlow, PyTorch and (non-JIT) JAX interfaces. Currently, support exists for both parameter-shift and
    backpropagation differentation methods; adjoint support is not currently enabled.

    To use this device, you will need to install dask and dask.distributed:

    .. code-block:: console

        pip install dask distributed

    **Example**

    >>> import pennylane as qml
    >>> import pennylane.numpy as np
    >>> import tensorflow as tf
    >>> import dask.distributed as dist

    >>> if __name__ == '__main__': # Required for LocalCluster
    ...     cluster = dist.LocalCluster(n_workers=4, threads_per_worker=1)
    ...     client = dist.Client(cluster)
    ...     backend = "default.qubit"
    ...     dev = qml.device("task.qubit", wires=6, backend=backend)
    ...     @qml.qnode(dev, cache=False, interface="tf", diff_method="parameter-shift") # caching must be disabled due to proxy interface
    ...     def circuit(x):
    ...         qml.RX(x[0], wires=0)
    ...         qml.RY(x[1], wires=0)
    ...         qml.RZ(x[2], wires=0)
    ...         qml.RZ(x[0], wires=1)
    ...         qml.RX(x[1], wires=1)
    ...         qml.RY(x[2], wires=1)
    ...         return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
    ...     weights = tf.Variable(np.random.rand(3))
    ...     def f_submit(weights):
    ...         with tf.GradientTape() as tape:
    ...             # Use the circuit to calculate the loss value
    ...             loss = tf.abs(circuit(weights)-0.5)**2
    ...         return tape.gradient(loss, weights)
    ...     print(client.submit(f_submit, weights).result())
    tf.Tensor([0.01776833 0.05199685 0.03689981], shape=(3,), dtype=float64)

    For self-encapsulated workflows, the task-based ``qml.taskify`` command can also be employed, which will allow automatic offloading using the
    batch execute support in PennyLane devices. As an example:

    >>> def my_workflow(params, backend, interface, diff_method):
    ...     qpu = qml.device(
    ...         "task.qubit",
    ...         wires=3,
    ...         backend=backend
    ...     )
    ...     # Caching must be disabled, as we are using a proxy device
    ...     @qml.qnode(qpu, cache=False, interface=interface, diff_method=diff_method)
    ...     def circuit(x):
    ...         qml.RX(x[0], wires=0)
    ...         qml.RY(x[0], wires=0)
    ...         qml.RZ(x[1], wires=0)
    ...         qml.RX(x[1], wires=1)
    ...         qml.RY(x[2], wires=1)
    ...         qml.RZ(x[2], wires=1)
    ...         return [qml.expval(qml.PauliZ(i) @ qml.PauliZ((i+1)%3)) for i in range(3)]
    ...
    ...     # Need a local copy of data on device
    ...     if interface == "tf":
    ...         weights = tf.Variable(params)
    ...         with tf.GradientTape() as tape:
    ...             # Use the circuit to calculate the loss value
    ...             loss = circuit(weights)
    ...         w_grad = tape.jacobian(loss, [weights])
    ...     elif interface == "torch":
    ...         weights = torch.tensor(params, requires_grad=True)
    ...         w_grad = torch.autograd.functional.jacobian(circuit, weights)
    ...     else:
    ...         weights = qml.numpy.array(params, requires_grad=True)
    ...         w_grad = qml.jacobian(circuit)(weights)
    ...     return w_grad

    >>> qml.taskify(my_workflow)(np.array([1.0,2.0,3.0]), "default.qubit", "autograd", "backprop")
    array([[-3.74614396e-01,  2.62791617e-01,  1.71438687e-02],
       [ 1.11022302e-16,  9.00197630e-01,  5.87266449e-02],
       [-9.09297427e-01, -1.38777878e-16, -3.46944695e-17]])

    For large batches of workloads, one can use the futures interface, and submit the tasks to be run asynchronously.
    As an example, we can modify the above example to adjust the supplied weights for circuit execution, and gather the
    results altogether when complete.

    >>> func = qml.taskify(my_workflow, futures=True)
    >>> futures = [func(i*np.array([1.0,2.0,3.0]), "default.qubit", "autograd", "backprop") for i in range(5)]
    >>> futures
    [<Future: finished, type: numpy.ndarray, key: my_workflow-238c01c1b1b88ef140d2a0404a226f9e>,
     <Future: finished, type: numpy.ndarray, key: my_workflow-276bf61f8de5488ec4e05ae93e3d0b85>,
     <Future: finished, type: numpy.ndarray, key: my_workflow-ccbf9977e25ec480b726b93c88cfc9e3>,
     <Future: finished, type: numpy.ndarray, key: my_workflow-e644cd634c006efc6fc7dd14837ed639>,
     <Future: finished, type: numpy.ndarray, key: my_workflow-ce39fdf431490ef17c836800b4a7318b>]
    >>> qml.untaskify(futures)()
    [array([[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]]),
     array([[-3.74614396e-01,  2.62791617e-01,  1.71438687e-02],
            [ 1.11022302e-16,  9.00197630e-01,  5.87266449e-02],
            [-9.09297427e-01, -1.38777878e-16, -3.46944695e-17]]),
     array([[-4.74976196e-01,  1.25841537e-01, -3.16289455e-02],
            [ 1.66533454e-16,  7.26659269e-01, -1.82638158e-01],
            [ 7.56802495e-01,  9.71445147e-17,  6.93889390e-18]]),
     array([[-2.44443912e-01, -2.49513914e-01, -3.87823537e-01],
            [ 2.77555756e-17, -2.54583916e-01, -3.95703924e-01],
            [ 2.79415498e-01,  0.00000000e+00,  0.00000000e+00]]),
     array([[ 1.21474177e-01, -3.56699848e-01, -3.33559948e-02],
            [ 2.77555756e-17, -8.34873873e-01, -7.80713777e-02],
            [-9.89358247e-01, -2.22044605e-16,  1.38777878e-17]])]

    Args:
        wires (int): The number of wires to initialize the device with.
        backend (None, str): Indicates the PennyLane device type to use for offloading
            computation tasks. This can be `default.qubit`, one of its subclasses or
            `lightning.qubit`. The TensorFlow, PyTorch or JAX interfaces are the preferred types
            for gradient computations. This is due to existing support in Dask for
            TF and Torch datatypes, and natural support for JAX via numpy.
        future (None, bool): Indicates whether the internal circuit evaluation returns a future
            to a result. This allows building of dependent workflows, but currently only works with
            explicit calls to `device.batch_execute` with a PennyLane native device type such as
            (`default.qubit`, `lightning.qubit`).
        gen_report (bool, str): Indicates whether the backend task-scheduler will generate a performance report based on the tasks that were run.
    """

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=no-self-argument

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
        backend="default.qubit",
        gen_report: Union[bool, str] = False,
        future=False,
    ):

        self._backend = backend
        self._backend_cls = qml.plugin_devices[backend].load()
        self._backend_dev = self._backend_cls(wires=0)

        # pylint: disable=invalid-class-object
        self.__class__ = type(
            "TaskQubit",
            (self._backend_cls,),
            {
                "execute": self.execute,
                "batch_execute": self.batch_execute,
                "_apply_ops": self._backend_dev._apply_ops,
            },
        )
        super(self._backend_cls, self).__init__(wires)

        if not isinstance(wires, Iterable):
            # interpret wires as the number of consecutive wires
            self._wires = range(wires)

        self.num_wires = wires if isinstance(wires, int) else len(self._wires)
        self._gen_report = gen_report
        self._future = future

        if backend not in TaskQubit.supported_devices:
            raise Exception("Unsupported device backend.")

    def __str__(self):
        return super().__str__() + (
            "\n"
            f"Backend: {self._backend}\n"
            f"Futures: {self._future}\n"
            f"Report: {self._gen_report}"
        )

    def __get__(self, obj, objtype=None):
        if obj not in self.__dict__:
            return self._backend_cls.obj
        return self.__dict__[obj]

    def batch_execute(self, circuits: List[qml.tape.QuantumTape]):
        # This overloads the batch_execute functionality of QubitDevice to offload
        # computations to a user-chosen backend device. This allows scaling of the
        # available workers to instantiate a given number of backends for concurrent
        # circuit evaluations.
        if self._gen_report:
            filename = self._gen_report if isinstance(self._gen_report, str) else "dask-report.html"
            cm = performance_report(filename=filename)
        else:
            cm = nullcontext()

        with cm:
            results = []
            if isinstance(circuits, dask.distributed.client.Future):
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
                    for circuit in circuits:
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
            with worker_client() as client:
                res = client.gather(results)
            return res

    # pylint: disable=arguments-differ
    def apply(self, operations, **kwargs):
        "The apply method of task.qubit should not be explicitly used."
        pass

    @ProxyHybridMethod
    def capabilities(cls):
        # Since we are using a proxy device, capabilities are handled by chosen backend
        # upon instantiation. If accessing class attributes, a limited set is provided.
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
            is_proxy=True,
        )
        return capabilities

    @capabilities.instancemethod
    def capabilities(self):
        return self._backend_cls.capabilities()

    @staticmethod
    def _execute_wrapper(
        backend: str,
        wires: int,
        circuit: qml.tape.QuantumTape,
    ):
        "This function provides a carrier function for instantiating and evaluating a tape on a given backend device."
        dev = qml.device(backend, wires=wires)
        return dev.execute(circuit)

    def execute(self, circuit, **kwargs):
        self.batch_execute([circuit])
