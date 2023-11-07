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
The default.qubit device is PennyLane's standard qubit-based device.
"""

from dataclasses import replace
from functools import partial
from numbers import Number
from typing import Union, Callable, Tuple, Optional, Sequence
import concurrent.futures
import inspect
import logging
import numpy as np

import pennylane as qml
from pennylane.tape import QuantumTape, QuantumScript
from pennylane.typing import Result, ResultBatch
from pennylane.transforms import convert_to_numpy_parameters
from pennylane.transforms.core import TransformProgram

from . import Device
from .preprocess import (
    decompose,
    validate_observables,
    validate_measurements,
    validate_multiprocessing_workers,
    validate_device_wires,
    warn_about_trainable_observables,
    no_sampling,
)
from .execution_config import ExecutionConfig, DefaultExecutionConfig
from .qubit.simulate import simulate, get_final_state, measure_final_state
from .qubit.sampling import get_num_shots_and_executions
from .qubit.adjoint_jacobian import adjoint_jacobian, adjoint_vjp, adjoint_jvp

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

Result_or_ResultBatch = Union[Result, ResultBatch]
QuantumTapeBatch = Sequence[QuantumTape]
QuantumTape_or_Batch = Union[QuantumTape, QuantumTapeBatch]
# always a function from a resultbatch to either a result or a result batch
PostprocessingFn = Callable[[ResultBatch], Result_or_ResultBatch]


observables = {
    "PauliX",
    "PauliY",
    "PauliZ",
    "Hadamard",
    "Hermitian",
    "Identity",
    "Projector",
    "SparseHamiltonian",
    "Hamiltonian",
    "Sum",
    "SProd",
    "Prod",
    "Exp",
    "Evolution",
}


def observable_stopping_condition(obs: qml.operation.Operator) -> bool:
    """Specifies whether or not an observable is accepted by DefaultQubit."""
    return obs.name in observables


def stopping_condition(op: qml.operation.Operator) -> bool:
    """Specify whether or not an Operator object is supported by the device."""
    if op.name == "QFT" and len(op.wires) >= 6:
        return False
    if op.name == "GroverOperator" and len(op.wires) >= 13:
        return False
    if op.name == "Snapshot":
        return True
    if op.__class__.__name__ == "Pow" and qml.operation.is_trainable(op):
        return False

    return op.has_matrix


def accepted_sample_measurement(m: qml.measurements.MeasurementProcess) -> bool:
    """Specifies whether or not a measurement is accepted when sampling."""
    return isinstance(
        m,
        (
            qml.measurements.SampleMeasurement,
            qml.measurements.ClassicalShadowMP,
            qml.measurements.ShadowExpvalMP,
        ),
    )


def _add_adjoint_transforms(program: TransformProgram) -> None:
    """Private helper function for ``preprocess`` that adds the transforms specific
    for adjoint differentiation.

    Args:
        program (TransformProgram): where we will add the adjoint differentiation transforms

    Side Effects:
        Adds transforms to the input program.

    """

    def adjoint_ops(op: qml.operation.Operator) -> bool:
        """Specify whether or not an Operator is supported by adjoint differentiation."""
        return op.num_params == 0 or op.num_params == 1 and op.has_generator

    def adjoint_observables(obs: qml.operation.Operator) -> bool:
        """Specifies whether or not an observable is compatible with adjoint differentiation on DefaultQubit."""
        return obs.has_matrix

    def accepted_adjoint_measurement(m: qml.measurements.MeasurementProcess) -> bool:
        return isinstance(m, qml.measurements.ExpectationMP)

    name = "adjoint + default.qubit"
    program.add_transform(no_sampling, name=name)
    program.add_transform(
        decompose,
        stopping_condition=adjoint_ops,
        name=name,
    )
    program.add_transform(validate_observables, adjoint_observables, name=name)
    program.add_transform(
        validate_measurements,
        analytic_measurements=accepted_adjoint_measurement,
        name=name,
    )
    program.add_transform(qml.transforms.broadcast_expand)
    program.add_transform(warn_about_trainable_observables)


class DefaultQubit(Device):
    """A PennyLane device written in Python and capable of backpropagation derivatives.

    Args:
        wires (int, Iterable[Number, str]): Number of wires present on the device, or iterable that
            contains unique labels for the wires as numbers (i.e., ``[-1, 0, 2]``) or strings
            (``['ancilla', 'q1', 'q2']``). Default ``None`` if not specified.
        shots (int, Sequence[int], Sequence[Union[int, Sequence[int]]]): The default number of shots
            to use in executions involving this device.
        seed (Union[str, None, int, array_like[int], SeedSequence, BitGenerator, Generator, jax.random.PRNGKey]): A
            seed-like parameter matching that of ``seed`` for ``numpy.random.default_rng``, or
            a request to seed from numpy's global random number generator.
            The default, ``seed="global"`` pulls a seed from NumPy's global generator. ``seed=None``
            will pull a seed from the OS entropy.
            If a ``jax.random.PRNGKey`` is passed as the seed, a JAX-specific sampling function using
            ``jax.random.choice`` and the ``PRNGKey`` will be used for sampling rather than
            ``numpy.random.default_rng``.
        max_workers (int): A ``ProcessPoolExecutor`` executes tapes asynchronously
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

    >>> dev = DefaultQubit()
    >>> program, execution_config = dev.preprocess()
    >>> new_batch, post_processing_fn = program(qscripts)
    >>> results = dev.execute(new_batch, execution_config=execution_config)
    >>> post_processing_fn(results)
    [-0.0006888975950537501,
    0.025576307134457577,
    -0.0038567269892757494,
    0.1339705146860149,
    -0.03780669772690448]

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
            program, execution_config = dev.preprocess()
            new_batch, post_processing_fn = program([qs])
            results = dev.execute(new_batch, execution_config=execution_config)
            return post_processing_fn(results)

    >>> f(jax.numpy.array(1.2))
    DeviceArray(0.36235774, dtype=float32)
    >>> jax.grad(f)(jax.numpy.array(1.2))
    DeviceArray(-0.93203914, dtype=float32, weak_type=True)

    .. details::
        :title: Tracking

        ``DefaultQubit`` tracks:

        * ``executions``: the number of unique circuits that would be required on quantum hardware
        * ``shots``: the number of shots
        * ``resources``: the :class:`~.resource.Resources` for the executed circuit.
        * ``simulations``: the number of simulations performed. One simulation can cover multiple QPU executions, such as for non-commuting measurements and batched parameters.
        * ``batches``: The number of times :meth:`~.execute` is called.
        * ``results``: The results of each call of :meth:`~.execute`
        * ``derivative_batches``: How many times :meth:`~.compute_derivatives` is called.
        * ``execute_and_derivative_batches``: How many times :meth:`~.execute_and_compute_derivatives` is called
        * ``vjp_batches``: How many times :meth:`~.compute_vjp` is called
        * ``execute_and_vjp_batches``: How many times :meth:`~.execute_and_compute_vjp` is called
        * ``jvp_batches``: How many times :meth:`~.compute_jvp` is called
        * ``execute_and_jvp_batches``: How many times :meth:`~.execute_and_compute_jvp` is called
        * ``derivatives``: How many circuits are submitted to :meth:`~.compute_derivatives` or :meth:`~.execute_and_compute_derivatives`.
        * ``vjps``: How many circuits are submitted to :meth:`~.compute_vjp` or :meth:`~.execute_and_compute_vjp`
        * ``jvps``: How many circuits are submitted to :meth:`~.compute_jvp` or :meth:`~.execute_and_compute_jvp`


    .. details::
        :title: Accelerate calculations with multiprocessing

        Suppose one has a processor with 5 cores or more, these scripts can be executed in
        parallel as follows

        >>> dev = DefaultQubit(max_workers=5)
        >>> program, execution_config = dev.preprocess()
        >>> new_batch, post_processing_fn = program(qscripts)
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

        .. warning::

            Multiprocessing may fail depending on your platform and environment (Python shell,
            script with a protected entry point, Jupyter notebook, etc.) This may be solved
            changing the so-called start method. The supported start methods are the following:

            * Windows (win32): spawn (default).
            * macOS (darwin): spawn (default), fork, forkserver.
            * Linux (unix): spawn, fork (default), forkserver.

            which can be changed with ``multiprocessing.set_start_method()``. For example,
            if multiprocessing fails on macOS in your Jupyter notebook environment, try
            restarting the session and adding the following at the beginning of the file:

            .. code-block:: python

                import multiprocessing
                multiprocessing.set_start_method("fork")

            Additional information can be found in the
            `multiprocessing doc <https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods>`_.


    """

    @property
    def name(self):
        """The name of the device."""
        return "default.qubit"

    # pylint:disable = too-many-arguments
    def __init__(
        self,
        wires=None,
        shots=None,
        seed="global",
        max_workers=None,
    ) -> None:
        super().__init__(wires=wires, shots=shots)
        self._max_workers = max_workers
        seed = np.random.randint(0, high=10000000) if seed == "global" else seed
        if qml.math.get_interface(seed) == "jax":
            self._prng_key = seed
            self._rng = np.random.default_rng(None)
        else:
            self._prng_key = None
            self._rng = np.random.default_rng(seed)
        self._debugger = None

    def supports_derivatives(
        self,
        execution_config: Optional[ExecutionConfig] = None,
        circuit: Optional[QuantumTape] = None,
    ) -> bool:
        """Check whether or not derivatives are available for a given configuration and circuit.

        ``DefaultQubit`` supports backpropagation derivatives with analytic results, as well as
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
            and execution_config.device_options.get("max_workers", self._max_workers) is None
            and execution_config.interface is not None
        ):
            return True

        if (
            execution_config.gradient_method == "adjoint"
            and execution_config.use_device_gradient is not False
        ):
            if circuit is None:
                return True

            prog = TransformProgram()
            _add_adjoint_transforms(prog)

            try:
                prog((circuit,))
            except (qml.operation.DecompositionUndefinedError, qml.DeviceError):
                return False
            return True

        return False

    def preprocess(
        self,
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ) -> Tuple[TransformProgram, ExecutionConfig]:
        """This function defines the device transform program to be applied and an updated device configuration.

        Args:
            execution_config (Union[ExecutionConfig, Sequence[ExecutionConfig]]): A data structure describing the
                parameters needed to fully describe the execution.

        Returns:
            TransformProgram, ExecutionConfig: A transform program that when called returns QuantumTapes that the device
            can natively execute as well as a postprocessing function to be called after execution, and a configuration with
            unset specifications filled in.

        This device:

        * Supports any qubit operations that provide a matrix
        * Currently does not support finite shots
        * Currently does not intrinsically support parameter broadcasting

        """
        config = self._setup_execution_config(execution_config)
        transform_program = TransformProgram()

        transform_program.add_transform(validate_device_wires, self.wires, name=self.name)
        transform_program.add_transform(qml.defer_measurements, device=self)
        transform_program.add_transform(
            decompose, stopping_condition=stopping_condition, name=self.name
        )
        transform_program.add_transform(
            validate_measurements, sample_measurements=accepted_sample_measurement, name=self.name
        )
        transform_program.add_transform(
            validate_observables, stopping_condition=observable_stopping_condition, name=self.name
        )

        # Validate multi processing
        max_workers = config.device_options.get("max_workers", self._max_workers)
        if max_workers:
            transform_program.add_transform(validate_multiprocessing_workers, max_workers, self)

        if config.gradient_method == "backprop":
            transform_program.add_transform(no_sampling, name="backprop + default.qubit")

        if config.gradient_method == "adjoint":
            _add_adjoint_transforms(transform_program)

        return transform_program, config

    def _setup_execution_config(self, execution_config: ExecutionConfig) -> ExecutionConfig:
        """This is a private helper for ``preprocess`` that sets up the execution config.

        Args:
            execution_config (ExecutionConfig)

        Returns:
            ExecutionConfig: a preprocessed execution config

        """
        updated_values = {}
        if execution_config.gradient_method == "best":
            updated_values["gradient_method"] = "backprop"
        if execution_config.use_device_gradient is None:
            updated_values["use_device_gradient"] = execution_config.gradient_method in {
                "best",
                "adjoint",
                "backprop",
            }
        if execution_config.use_device_jacobian_product is None:
            updated_values["use_device_jacobian_product"] = (
                execution_config.gradient_method == "adjoint"
            )
        if execution_config.grad_on_execution is None:
            updated_values["grad_on_execution"] = execution_config.gradient_method == "adjoint"
        updated_values["device_options"] = dict(execution_config.device_options)  # copy
        if "max_workers" not in updated_values["device_options"]:
            updated_values["device_options"]["max_workers"] = self._max_workers
        if "rng" not in updated_values["device_options"]:
            updated_values["device_options"]["rng"] = self._rng
        if "prng_key" not in updated_values["device_options"]:
            updated_values["device_options"]["prng_key"] = self._prng_key
        return replace(execution_config, **updated_values)

    def execute(
        self,
        circuits: QuantumTape_or_Batch,
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ) -> Result_or_ResultBatch:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                """Entry with args=(circuits=%s) called by=%s""",
                circuits,
                "::L".join(
                    str(i) for i in inspect.getouterframes(inspect.currentframe(), 2)[1][1:3]
                ),
            )

        is_single_circuit = False
        if isinstance(circuits, QuantumScript):
            is_single_circuit = True
            circuits = [circuits]

        max_workers = execution_config.device_options.get("max_workers", self._max_workers)
        interface = (
            execution_config.interface
            if execution_config.gradient_method in {"backprop", None}
            else None
        )
        if max_workers is None:
            results = tuple(
                simulate(
                    c,
                    rng=self._rng,
                    prng_key=self._prng_key,
                    debugger=self._debugger,
                    interface=interface,
                )
                for c in circuits
            )
        else:
            vanilla_circuits = [convert_to_numpy_parameters(c) for c in circuits]
            seeds = self._rng.integers(2**31 - 1, size=len(vanilla_circuits))
            _wrap_simulate = partial(simulate, debugger=None, interface=interface)
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                exec_map = executor.map(
                    _wrap_simulate,
                    vanilla_circuits,
                    seeds,
                    [self._prng_key] * len(vanilla_circuits),
                )
                results = tuple(exec_map)

            # reset _rng to mimic serial behavior
            self._rng = np.random.default_rng(self._rng.integers(2**31 - 1))

        if self.tracker.active:
            self.tracker.update(batches=1)
            self.tracker.record()
            for i, c in enumerate(circuits):
                qpu_executions, shots = get_num_shots_and_executions(c)
                res = np.array(results[i]) if isinstance(results[i], Number) else results[i]
                if c.shots:
                    self.tracker.update(
                        simulations=1,
                        executions=qpu_executions,
                        results=res,
                        shots=shots,
                        resources=c.specs["resources"],
                    )
                else:
                    self.tracker.update(
                        simulations=1,
                        executions=qpu_executions,
                        results=res,
                        resources=c.specs["resources"],
                    )
                self.tracker.record()

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

        max_workers = execution_config.device_options.get("max_workers", self._max_workers)
        if max_workers is None:
            res = tuple(adjoint_jacobian(circuit) for circuit in circuits)
        else:
            vanilla_circuits = [convert_to_numpy_parameters(c) for c in circuits]
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                exec_map = executor.map(adjoint_jacobian, vanilla_circuits)
                res = tuple(exec_map)

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

        max_workers = execution_config.device_options.get("max_workers", self._max_workers)
        if max_workers is None:
            results = tuple(
                _adjoint_jac_wrapper(
                    c, rng=self._rng, debugger=self._debugger, prng_key=self._prng_key
                )
                for c in circuits
            )
        else:
            vanilla_circuits = [convert_to_numpy_parameters(c) for c in circuits]
            seeds = self._rng.integers(2**31 - 1, size=len(vanilla_circuits))

            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                results = tuple(
                    executor.map(
                        _adjoint_jac_wrapper,
                        vanilla_circuits,
                        seeds,
                        [self._prng_key] * len(vanilla_circuits),
                    )
                )

            # reset _rng to mimic serial behavior
            self._rng = np.random.default_rng(self._rng.integers(2**31 - 1))

        results, jacs = tuple(zip(*results))
        return (results[0], jacs[0]) if is_single_circuit else (results, jacs)

    def supports_jvp(
        self,
        execution_config: Optional[ExecutionConfig] = None,
        circuit: Optional[QuantumTape] = None,
    ) -> bool:
        """Whether or not this device defines a custom jacobian vector product.

        ``DefaultQubit`` supports backpropagation derivatives with analytic results, as well as
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

        max_workers = execution_config.device_options.get("max_workers", self._max_workers)
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

        max_workers = execution_config.device_options.get("max_workers", self._max_workers)
        if max_workers is None:
            results = tuple(
                _adjoint_jvp_wrapper(
                    c, t, rng=self._rng, debugger=self._debugger, prng_key=self._prng_key
                )
                for c, t in zip(circuits, tangents)
            )
        else:
            vanilla_circuits = [convert_to_numpy_parameters(c) for c in circuits]
            seeds = self._rng.integers(2**31 - 1, size=len(vanilla_circuits))

            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                results = tuple(
                    executor.map(
                        _adjoint_jvp_wrapper,
                        vanilla_circuits,
                        tangents,
                        seeds,
                        [self._prng_key] * len(vanilla_circuits),
                    )
                )

            # reset _rng to mimic serial behavior
            self._rng = np.random.default_rng(self._rng.integers(2**31 - 1))

        results, jvps = tuple(zip(*results))
        return (results[0], jvps[0]) if is_single_circuit else (results, jvps)

    def supports_vjp(
        self,
        execution_config: Optional[ExecutionConfig] = None,
        circuit: Optional[QuantumTape] = None,
    ) -> bool:
        """Whether or not this device defines a custom vector jacobian product.

        ``DefaultQubit`` supports backpropagation derivatives with analytic results, as well as
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

        max_workers = execution_config.device_options.get("max_workers", self._max_workers)
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

        max_workers = execution_config.device_options.get("max_workers", self._max_workers)
        if max_workers is None:
            results = tuple(
                _adjoint_vjp_wrapper(
                    c, t, rng=self._rng, prng_key=self._prng_key, debugger=self._debugger
                )
                for c, t in zip(circuits, cotangents)
            )
        else:
            vanilla_circuits = [convert_to_numpy_parameters(c) for c in circuits]
            seeds = self._rng.integers(2**31 - 1, size=len(vanilla_circuits))

            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                results = tuple(
                    executor.map(
                        _adjoint_vjp_wrapper,
                        vanilla_circuits,
                        cotangents,
                        seeds,
                        [self._prng_key] * len(vanilla_circuits),
                    )
                )

            # reset _rng to mimic serial behavior
            self._rng = np.random.default_rng(self._rng.integers(2**31 - 1))

        results, vjps = tuple(zip(*results))
        return (results[0], vjps[0]) if is_single_circuit else (results, vjps)


def _adjoint_jac_wrapper(c, rng=None, prng_key=None, debugger=None):
    state, is_state_batched = get_final_state(c, debugger=debugger)
    jac = adjoint_jacobian(c, state=state)
    res = measure_final_state(c, state, is_state_batched, rng=rng, prng_key=prng_key)
    return res, jac


def _adjoint_jvp_wrapper(c, t, rng=None, prng_key=None, debugger=None):
    state, is_state_batched = get_final_state(c, debugger=debugger)
    jvp = adjoint_jvp(c, t, state=state)
    res = measure_final_state(c, state, is_state_batched, rng=rng, prng_key=prng_key)
    return res, jvp


def _adjoint_vjp_wrapper(c, t, rng=None, prng_key=None, debugger=None):
    state, is_state_batched = get_final_state(c, debugger=debugger)
    vjp = adjoint_vjp(c, t, state=state)
    res = measure_final_state(c, state, is_state_batched, rng=rng, prng_key=prng_key)
    return res, vjp
