# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
This module contains the next generation successor to default qubit
"""

from functools import partial
from numbers import Number
from typing import Union, Callable, Tuple, Optional, Sequence
import concurrent.futures
import os
import warnings
import numpy as np

from pennylane.tape import QuantumTape, QuantumScript
from pennylane.typing import Result, ResultBatch
from pennylane.transforms import convert_to_numpy_parameters
from pennylane._version import __version__
from pennylane import DeviceError, Snapshot

from . import Device
from .execution_config import ExecutionConfig, DefaultExecutionConfig
from ..qubit.simulate import simulate, get_final_state, measure_final_state
from ..qubit.preprocess import preprocess, validate_and_expand_adjoint
from ..qubit.adjoint_jacobian import adjoint_jacobian, adjoint_vjp, adjoint_jvp

Result_or_ResultBatch = Union[Result, ResultBatch]
QuantumTapeBatch = Sequence[QuantumTape]
QuantumTape_or_Batch = Union[QuantumTape, QuantumTapeBatch]
# always a function from a resultbatch to either a result or a result batch
PostprocessingFn = Callable[[ResultBatch], Result_or_ResultBatch]


class DefaultQubit2(Device):
    """A PennyLane device written in Python and capable of backpropagation derivatives.

    Args:
        shots (int, Sequence[int], Sequence[Union[int, Sequence[int]]]): The default number of shots to use in executions involving
            this device.
        seed="global" (Union[str, None, int, array_like[int], SeedSequence, BitGenerator, Generator]): A
            seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng`` or
            a request to seed from numpy's global random number generator.
            The default, ``seed="global"`` pulls a seed from NumPy's global generator. ``seed=None``
            will pull a seed from the OS entropy.
        max_workers=None (int): A ``ProcessPoolExecutor`` executes tapes asynchronously
            using a pool of at most ``max_workers`` processes. If ``max_workers`` is ``None``,
            only the current process executes tapes. If you experience any
            issue, say using JAX, TensorFlow, Torch, try setting ``max_workers`` to ``None``.

    **Example:**

    .. code-block:: python

        n_layers = 5
        n_wires = 10
        num_qscripts = 5

        shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_wires)
        rng = qml.numpy.random.default_rng(seed=42)

        qscripts = []
        for i in range(num_qscripts):
            params = rng.random(shape)
            op = qml.StronglyEntanglingLayers(params, wires=range(n_wires))
            qs = qml.tape.QuantumScript([op], [qml.expval(qml.PauliZ(0))])
            qscripts.append(qs)

    >>> dev = DefaultQubit2()
    >>> new_batch, post_processing_fn, execution_config = dev.preprocess(qscripts)
    >>> results = dev.execute(new_batch, execution_config=execution_config)
    >>> post_processing_fn(results)
    [-0.0006888975950537501,
    0.025576307134457577,
    -0.0038567269892757494,
    0.1339705146860149,
    -0.03780669772690448]

    Suppose one has a processor with 5 cores or more, these scripts can be executed in
    parallel as follows

    >>> dev = DefaultQubit2(max_workers=5)
    >>> new_batch, post_processing_fn, execution_config = dev.preprocess(qscripts)
    >>> results = dev.execute(new_batch, execution_config=execution_config)
    >>> post_processing_fn(results)

    If you monitor your CPU usage, you should see 5 new Python processes pop up to
    crunch through those ``QuantumScript``'s. Beware not oversubscribing your machine.
    This may happen if a single device already uses many cores, if NumPy uses a multi-
    threaded BLAS library like MKL or OpenBLAS for example. The number of threads per
    process times the number of processes should not exceed the number of cores on your
    machine. You can control the number of threads per process with the environment
    variables:

    * OMP_NUM_THREADS
    * MKL_NUM_THREADS
    * OPENBLAS_NUM_THREADS

    where the last two are specific to the MKL and OpenBLAS libraries specifically.

    This device currently supports backpropagation derivatives:

    >>> from pennylane.devices import ExecutionConfig
    >>> dev.supports_derivatives(ExecutionConfig(gradient_method="backprop"))
    True

    For example, we can use jax to jit computing the derivative:

    .. code-block:: python

        import jax

        @jax.jit
        def f(x):
            qs = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.PauliZ(0))])
            new_batch, post_processing_fn, execution_config = dev.preprocess(qs)
            results = dev.execute(new_batch, execution_config=execution_config)
            return post_processing_fn(results)

    >>> f(jax.numpy.array(1.2))
    DeviceArray(0.36235774, dtype=float32)
    >>> jax.grad(f)(jax.numpy.array(1.2))
    DeviceArray(-0.93203914, dtype=float32, weak_type=True)

    """

    pennylane_requires = __version__

    @property
    def name(self):
        """The name of the device."""
        return "default.qubit.2"

    def __init__(self, wires=None, shots=None, seed="global", max_workers=None) -> None:
        super().__init__(wires=wires, shots=shots)
        self._max_workers = max_workers
        seed = np.random.randint(0, high=10000000) if seed == "global" else seed
        self._rng = np.random.default_rng(seed)
        self._debugger = None

    def supports_derivatives(
        self,
        execution_config: Optional[ExecutionConfig] = None,
        circuit: Optional[QuantumTape] = None,
    ) -> bool:
        """Check whether or not derivatives are available for a given configuration and circuit.

        ``DefaultQubit2`` supports backpropagation derivatives with analytic results, as well as
        adjoint differentiation.

        Args:
            execution_config (ExecutionConfig): The configuration of the desired derivative calculation
            circuit (QuantumTape): An optional circuit to check derivatives support for.

        Returns:
            Bool: Whether or not a derivative can be calculated provided the given information

        """
        if execution_config is None:
            return True
        # backpropagation currently supported for all supported circuits
        # will later need to add logic if backprop requested with finite shots
        # do once device accepts finite shots
        if (
            execution_config.gradient_method == "backprop"
            and self._get_max_workers(execution_config) is None
        ):
            return True

        if execution_config.gradient_method == "adjoint" and execution_config.use_device_gradient:
            if circuit is None:
                return True

            return isinstance(validate_and_expand_adjoint(circuit), QuantumScript)

        return False

    def preprocess(
        self,
        circuits: QuantumTape_or_Batch,
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ) -> Tuple[QuantumTapeBatch, PostprocessingFn, ExecutionConfig]:
        """Converts an arbitrary circuit or batch of circuits into a batch natively executable by the :meth:`~.execute` method.

        Args:
            circuits (Union[QuantumTape, Sequence[QuantumTape]]): The circuit or a batch of circuits to preprocess
                before execution on the device
            execution_config (Union[ExecutionConfig, Sequence[ExecutionConfig]]): A data structure describing the parameters needed to fully describe
                the execution. Includes such information as shots.

        Returns:
            Tuple[QuantumTape], Callable, ExecutionConfig: QuantumTapes that the device can natively execute,
            a postprocessing function to be called after execution, and a configuration with unset specifications filled in.

        This device:

        * Supports any qubit operations that provide a matrix
        * Currently does not support finite shots
        * Currently does not intrinsically support parameter broadcasting

        """
        is_single_circuit = False
        if isinstance(circuits, QuantumScript):
            circuits = [circuits]
            is_single_circuit = True

        batch, post_processing_fn, config = preprocess(circuits, execution_config=execution_config)

        if is_single_circuit:

            def convert_batch_to_single_output(results: ResultBatch) -> Result:
                """Unwraps a dimension so that executing the batch of circuits looks like executing a single circuit."""
                return post_processing_fn(results)[0]

            return batch, convert_batch_to_single_output, config

        return batch, post_processing_fn, config

    def execute(
        self,
        circuits: QuantumTape_or_Batch,
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ) -> Result_or_ResultBatch:
        is_single_circuit = False
        if isinstance(circuits, QuantumScript):
            is_single_circuit = True
            circuits = [circuits]

        if self.tracker.active:
            for c in circuits:
                self.tracker.update(resources=c.specs["resources"])
            self.tracker.update(batches=1, executions=len(circuits))
            self.tracker.record()

        max_workers = self._get_max_workers(execution_config)
        if max_workers is None:
            results = tuple(simulate(c, rng=self._rng, debugger=self._debugger) for c in circuits)
        else:
            self._validate_multiprocessing_circuits(circuits)
            vanilla_circuits = [convert_to_numpy_parameters(c) for c in circuits]
            seeds = self._rng.integers(2**31 - 1, size=len(vanilla_circuits))
            _wrap_simulate = partial(simulate, debugger=None)
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                exec_map = executor.map(_wrap_simulate, vanilla_circuits, seeds)
                results = tuple(circuit for circuit in exec_map)

            # reset _rng to mimic serial behavior
            self._rng = np.random.default_rng(self._rng.integers(2**31 - 1))

        return results[0] if is_single_circuit else results

    def compute_derivatives(
        self,
        circuits: QuantumTape_or_Batch,
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ):
        is_single_circuit = False
        if isinstance(circuits, QuantumScript):
            is_single_circuit = True
            circuits = [circuits]

        if self.tracker.active:
            self.tracker.update(derivative_batches=1, derivatives=len(circuits))
            self.tracker.record()

        max_workers = self._get_max_workers(execution_config)
        if max_workers is None:
            res = tuple(adjoint_jacobian(circuit) for circuit in circuits)
        else:
            vanilla_circuits = [convert_to_numpy_parameters(c) for c in circuits]
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                exec_map = executor.map(adjoint_jacobian, vanilla_circuits)
                res = tuple(circuit for circuit in exec_map)

            # reset _rng to mimic serial behavior
            self._rng = np.random.default_rng(self._rng.integers(2**31 - 1))

        return res[0] if is_single_circuit else res

    def execute_and_compute_derivatives(
        self,
        circuits: QuantumTape_or_Batch,
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ):
        is_single_circuit = False
        if isinstance(circuits, QuantumScript):
            is_single_circuit = True
            circuits = [circuits]

        if self.tracker.active:
            for c in circuits:
                self.tracker.update(resources=c.specs["resources"])
            self.tracker.update(
                execute_and_derivative_batches=1,
                executions=len(circuits),
                derivatives=len(circuits),
            )
            self.tracker.record()

        max_workers = self._get_max_workers(execution_config)
        if max_workers is None:
            results = tuple(
                _adjoint_jac_wrapper(c, rng=self._rng, debugger=self._debugger) for c in circuits
            )
            results, jacs = tuple(zip(*results))
        else:
            self._validate_multiprocessing_circuits(circuits)

            vanilla_circuits = [convert_to_numpy_parameters(c) for c in circuits]
            seeds = self._rng.integers(2**31 - 1, size=len(vanilla_circuits))

            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                results = tuple(executor.map(_adjoint_jac_wrapper, vanilla_circuits, seeds))

            results, jacs = tuple(zip(*results))

            # reset _rng to mimic serial behavior
            self._rng = np.random.default_rng(self._rng.integers(2**31 - 1))

        return (results[0], jacs[0]) if is_single_circuit else (results, jacs)

    def supports_jvp(
        self,
        execution_config: Optional[ExecutionConfig] = None,
        circuit: Optional[QuantumTape] = None,
    ) -> bool:
        """Whether or not this device defines a custom jacobian vector product.

        ``DefaultQubit2`` supports backpropagation derivatives with analytic results, as well as
        adjoint differentiation.

        Args:
            execution_config (ExecutionConfig): The configuration of the desired derivative calculation
            circuit (QuantumTape): An optional circuit to check derivatives support for.

        Returns:
            bool: Whether or not a derivative can be calculated provided the given information
        """
        return self.supports_derivatives(execution_config, circuit)

    def compute_jvp(
        self,
        circuits: QuantumTape_or_Batch,
        tangents: Tuple[Number],
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ):
        is_single_circuit = False
        if isinstance(circuits, QuantumScript):
            is_single_circuit = True
            circuits = [circuits]
            tangents = [tangents]

        if self.tracker.active:
            self.tracker.update(jvp_batches=1, jvps=len(circuits))
            self.tracker.record()

        max_workers = self._get_max_workers(execution_config)
        if max_workers is None:
            res = tuple(adjoint_jvp(circuit, tans) for circuit, tans in zip(circuits, tangents))
        else:
            vanilla_circuits = [convert_to_numpy_parameters(c) for c in circuits]
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                res = tuple(executor.map(adjoint_jvp, vanilla_circuits, tangents))

            # reset _rng to mimic serial behavior
            self._rng = np.random.default_rng(self._rng.integers(2**31 - 1))

        return res[0] if is_single_circuit else res

    def execute_and_compute_jvp(
        self,
        circuits: QuantumTape_or_Batch,
        tangents: Tuple[Number],
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ):
        is_single_circuit = False
        if isinstance(circuits, QuantumScript):
            is_single_circuit = True
            circuits = [circuits]
            tangents = [tangents]

        if self.tracker.active:
            for c in circuits:
                self.tracker.update(resources=c.specs["resources"])
            self.tracker.update(
                execute_and_jvp_batches=1, executions=len(circuits), jvps=len(circuits)
            )
            self.tracker.record()

        max_workers = self._get_max_workers(execution_config)
        if max_workers is None:
            results = tuple(
                _adjoint_jvp_wrapper(c, t, rng=self._rng, debugger=self._debugger)
                for c, t in zip(circuits, tangents)
            )
            results, jvps = tuple(zip(*results))
        else:
            self._validate_multiprocessing_circuits(circuits)

            vanilla_circuits = [convert_to_numpy_parameters(c) for c in circuits]
            seeds = self._rng.integers(2**31 - 1, size=len(vanilla_circuits))

            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                results = tuple(
                    executor.map(_adjoint_jvp_wrapper, vanilla_circuits, tangents, seeds)
                )

            results, jvps = tuple(zip(*results))

            # reset _rng to mimic serial behavior
            self._rng = np.random.default_rng(self._rng.integers(2**31 - 1))

        return (results[0], jvps[0]) if is_single_circuit else (results, jvps)

    def supports_vjp(
        self,
        execution_config: Optional[ExecutionConfig] = None,
        circuit: Optional[QuantumTape] = None,
    ) -> bool:
        """Whether or not this device defines a custom vector jacobian product.

        ``DefaultQubit2`` supports backpropagation derivatives with analytic results, as well as
        adjoint differentiation.

        Args:
            execution_config (ExecutionConfig): A description of the hyperparameters for the desired computation.
            circuit (None, QuantumTape): A specific circuit to check differentation for.

        Returns:
            bool: Whether or not a derivative can be calculated provided the given information
        """
        return self.supports_derivatives(execution_config, circuit)

    def compute_vjp(
        self,
        circuits: QuantumTape_or_Batch,
        cotangents: Tuple[Number],
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ):
        is_single_circuit = False
        if isinstance(circuits, QuantumScript):
            is_single_circuit = True
            circuits = [circuits]
            cotangents = [cotangents]

        if self.tracker.active:
            self.tracker.update(vjp_batches=1, vjps=len(circuits))
            self.tracker.record()

        max_workers = self._get_max_workers(execution_config)
        if max_workers is None:
            res = tuple(adjoint_vjp(circuit, cots) for circuit, cots in zip(circuits, cotangents))
        else:
            vanilla_circuits = [convert_to_numpy_parameters(c) for c in circuits]
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                res = tuple(executor.map(adjoint_vjp, vanilla_circuits, cotangents))

            # reset _rng to mimic serial behavior
            self._rng = np.random.default_rng(self._rng.integers(2**31 - 1))

        return res[0] if is_single_circuit else res

    def execute_and_compute_vjp(
        self,
        circuits: QuantumTape_or_Batch,
        cotangents: Tuple[Number],
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ):
        is_single_circuit = False
        if isinstance(circuits, QuantumScript):
            is_single_circuit = True
            circuits = [circuits]
            cotangents = [cotangents]

        if self.tracker.active:
            for c in circuits:
                self.tracker.update(resources=c.specs["resources"])
            self.tracker.update(
                execute_and_vjp_batches=1, executions=len(circuits), vjps=len(circuits)
            )
            self.tracker.record()

        max_workers = self._get_max_workers(execution_config)
        if max_workers is None:
            results = tuple(
                _adjoint_vjp_wrapper(c, t, rng=self._rng, debugger=self._debugger)
                for c, t in zip(circuits, cotangents)
            )
            results, vjps = tuple(zip(*results))
        else:
            self._validate_multiprocessing_circuits(circuits)

            vanilla_circuits = [convert_to_numpy_parameters(c) for c in circuits]
            seeds = self._rng.integers(2**31 - 1, size=len(vanilla_circuits))

            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                results = tuple(
                    executor.map(_adjoint_vjp_wrapper, vanilla_circuits, cotangents, seeds)
                )

            results, vjps = tuple(zip(*results))

            # reset _rng to mimic serial behavior
            self._rng = np.random.default_rng(self._rng.integers(2**31 - 1))

        return (results[0], vjps[0]) if is_single_circuit else (results, vjps)

    # pylint: disable=missing-function-docstring
    def _get_max_workers(self, execution_config=None):
        max_workers = None
        if (
            execution_config
            and execution_config.device_options
            and "max_workers" in execution_config.device_options
        ):
            max_workers = execution_config.device_options["max_workers"]
        else:
            max_workers = self._max_workers
        _validate_multiprocessing_workers(max_workers)
        return max_workers

    def _validate_multiprocessing_circuits(self, circuits):
        """Make sure the tapes can be processed by a ProcessPoolExecutor instance.

        Args:
            circuits (QuantumTape_or_Batch): Quantum tapes
        """
        if self._debugger and self._debugger.active:
            raise DeviceError("Debugging with ``Snapshots`` is not available with multiprocessing.")

        def _has_snapshot(circuit):
            return any(isinstance(c, Snapshot) for c in circuit)

        if any(_has_snapshot(c) for c in circuits):
            raise RuntimeError(
                """ProcessPoolExecutor cannot execute a QuantumScript with
                a ``Snapshot`` operation. Change the value of ``max_workers``
                to ``None`` or execute the QuantumScript separately."""
            )


def _validate_multiprocessing_workers(max_workers):
    """Validates the number of workers for multiprocessing.

    Checks that the CPU is not oversubscribed and warns user if it is,
    making suggestions for the number of workers and/or the number of
    threads per worker.

    Args:
        max_workers (int): Maximal number of multiprocessing workers
    """
    if max_workers is None:
        return
    threads_per_proc = os.cpu_count()  # all threads by default
    varname = "OMP_NUM_THREADS"
    varnames = ["MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"]
    for var in varnames:
        if os.getenv(var):  # pragma: no cover
            varname = var
            threads_per_proc = int(os.getenv(var))
            break
    num_threads = threads_per_proc * max_workers
    num_cpu = os.cpu_count()
    num_threads_suggest = max(1, os.cpu_count() // max_workers)
    num_workers_suggest = max(1, os.cpu_count() // threads_per_proc)
    if num_threads > num_cpu:
        warnings.warn(
            f"""The device requested {num_threads} threads ({max_workers} processes
            times {threads_per_proc} threads per process), but the processor only has
            {num_cpu} logical cores. The processor is likely oversubscribed, which may
            lead to performance deterioration. Consider decreasing the number of processes,
            setting the device or execution config argument `max_workers={num_workers_suggest}`
            for example, or decreasing the number of threads per process by setting the
            environment variable `{varname}={num_threads_suggest}`.""",
            UserWarning,
        )


def _adjoint_jac_wrapper(c, rng=None, debugger=None):
    state, is_state_batched = get_final_state(c, debugger=debugger)
    jac = adjoint_jacobian(c, state=state)
    res = measure_final_state(c, state, is_state_batched, rng=rng)
    return res, jac


def _adjoint_jvp_wrapper(c, t, rng=None, debugger=None):
    state, is_state_batched = get_final_state(c, debugger=debugger)
    jvp = adjoint_jvp(c, t, state=state)
    res = measure_final_state(c, state, is_state_batched, rng=rng)
    return res, jvp


def _adjoint_vjp_wrapper(c, t, rng=None, debugger=None):
    state, is_state_batched = get_final_state(c, debugger=debugger)
    vjp = adjoint_vjp(c, t, state=state)
    res = measure_final_state(c, state, is_state_batched, rng=rng)
    return res, vjp
