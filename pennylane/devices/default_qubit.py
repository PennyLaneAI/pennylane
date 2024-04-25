# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
from pennylane.ops.op_math.condition import Conditional
from pennylane.measurements.mid_measure import MidMeasureMP
from pennylane.tape import QuantumTape
from pennylane.typing import Result, ResultBatch
from pennylane.transforms import convert_to_numpy_parameters
from pennylane.transforms.core import TransformProgram

from . import Device
from .modifiers import single_tape_support, simulator_tracking
from .preprocess import (
    decompose,
    mid_circuit_measurements,
    validate_observables,
    validate_measurements,
    validate_multiprocessing_workers,
    validate_device_wires,
    validate_adjoint_trainable_params,
    no_sampling,
)
from .execution_config import ExecutionConfig, DefaultExecutionConfig
from .qubit.sampling import jax_random_split
from .qubit.simulate import simulate, get_final_state, measure_final_state
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
    "LinearCombination",
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
    if op.__class__.__name__[:3] == "Pow" and qml.operation.is_trainable(op):
        return False

    return op.has_matrix


def stopping_condition_shots(op: qml.operation.Operator) -> bool:
    """Specify whether or not an Operator object is supported by the device with shots."""
    return isinstance(op, (Conditional, MidMeasureMP)) or stopping_condition(op)


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


def null_postprocessing(results):
    """An empty post-processing function."""
    return results[0]


def all_state_postprocessing(results, measurements, wire_order):
    """Process a state measurement back into the original measurements."""
    result = tuple(m.process_state(results[0], wire_order=wire_order) for m in measurements)
    return result[0] if len(measurements) == 1 else result


@qml.transform
def adjoint_state_measurements(
    tape: QuantumTape, device_vjp=False
) -> (Tuple[QuantumTape], Callable):
    """Perform adjoint measurement preprocessing.

    * Allows a tape with only expectation values through unmodified
    * Raises an error if non-expectation value measurements exist and any have diagonalizing gates
    * Turns the circuit into a state measurement + classical postprocesssing for arbitrary measurements

    Args:
        tape (QuantumTape): the input circuit

    """
    if all(isinstance(m, qml.measurements.ExpectationMP) for m in tape.measurements):
        return (tape,), null_postprocessing

    if any(len(m.diagonalizing_gates()) > 0 for m in tape.measurements):
        raise qml.DeviceError(
            "adjoint diff supports either all expectation values or only measurements without observables."
        )

    params = tape.get_parameters()

    if device_vjp:
        for p in params:
            if (
                qml.math.requires_grad(p)
                and qml.math.get_interface(p) == "tensorflow"
                and qml.math.get_dtype_name(p) in {"float32", "complex64"}
            ):
                raise ValueError(
                    "tensorflow with adjoint differentiation of the state requires float64 or complex128 parameters."
                )

    complex_data = [qml.math.cast(p, complex) for p in params]
    tape = tape.bind_new_parameters(complex_data, list(range(len(params))))
    new_mp = qml.measurements.StateMP(wires=tape.wires)
    state_tape = qml.tape.QuantumScript(tape.operations, [new_mp])
    return (state_tape,), partial(
        all_state_postprocessing, measurements=tape.measurements, wire_order=tape.wires
    )


def adjoint_ops(op: qml.operation.Operator) -> bool:
    """Specify whether or not an Operator is supported by adjoint differentiation."""
    return not isinstance(op, MidMeasureMP) and (
        op.num_params == 0
        or not qml.operation.is_trainable(op)
        or (op.num_params == 1 and op.has_generator)
    )


def adjoint_observables(obs: qml.operation.Operator) -> bool:
    """Specifies whether or not an observable is compatible with adjoint differentiation on DefaultQubit."""
    return obs.has_matrix


def _supports_adjoint(circuit):
    if circuit is None:
        return True

    prog = TransformProgram()
    _add_adjoint_transforms(prog)

    try:
        prog((circuit,))
    except (qml.operation.DecompositionUndefinedError, qml.DeviceError, AttributeError):
        return False
    return True


def _add_adjoint_transforms(program: TransformProgram, device_vjp=False) -> None:
    """Private helper function for ``preprocess`` that adds the transforms specific
    for adjoint differentiation.

    Args:
        program (TransformProgram): where we will add the adjoint differentiation transforms

    Side Effects:
        Adds transforms to the input program.

    """

    name = "adjoint + default.qubit"
    program.add_transform(no_sampling, name=name)
    program.add_transform(
        decompose, stopping_condition=adjoint_ops, name=name, skip_initial_state_prep=False
    )
    program.add_transform(validate_observables, adjoint_observables, name=name)
    program.add_transform(
        validate_measurements,
        name=name,
    )
    program.add_transform(adjoint_state_measurements, device_vjp=device_vjp)
    program.add_transform(qml.transforms.broadcast_expand)
    program.add_transform(validate_adjoint_trainable_params)


@simulator_tracking
@single_tape_support
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
            qs = qml.tape.QuantumScript([op], [qml.expval(qml.Z(0))])
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
            qs = qml.tape.QuantumScript([qml.RX(x, 0)], [qml.expval(qml.Z(0))])
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

        * ``OMP_NUM_THREADS``
        * ``MKL_NUM_THREADS``
        * ``OPENBLAS_NUM_THREADS``

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

    def get_prng_keys(self, num: int = 1):
        """Get ``num`` new keys with ``jax.random.split``.

        A user may provide a ``jax.random.PRNGKey`` as a random seed.
        It will be used by the device when executing circuits with finite shots.
        The JAX RNG is notably different than the NumPy RNG as highlighted in the
        `JAX documentation <https://jax.readthedocs.io/en/latest/jax-101/05-random-numbers.html>`_.
        JAX does not keep track of a global seed or key, but needs one anytime it draws from a random number distribution.
        Generating randomness therefore requires changing the key every time, which is done by "splitting" the key.
        For example, when executing ``n`` circuits, the ``PRNGkey`` is split ``n`` times into 2 new keys
        using ``jax.random.split`` to simulate a non-deterministic behaviour.
        The device seed is modified in-place using the first key, and the second key is fed to the
        circuit, and hence can be discarded after returning the results.
        This same key may be split further down the stack if necessary so that no one key is ever
        reused.
        """
        if num < 1:
            raise ValueError("Argument num must be a positive integer.")
        if num > 1:
            return [self.get_prng_keys()[0] for _ in range(num)]
        self._prng_key, *keys = jax_random_split(self._prng_key)
        return keys

    def reset_prng_key(self):
        """Reset the RNG key to its initial value."""
        self._prng_key = self._prng_seed

    _state_cache: Optional[dict] = None
    """
    A cache to store the "pre-rotated state" for reuse between the forward pass call to ``execute`` and
    subsequent calls to ``compute_vjp``. ``None`` indicates that no caching is required.
    """

    _device_options = ("max_workers", "rng", "prng_key")
    """
    tuple of string names for all the device options.
    """

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
            self._prng_seed = seed
            self._prng_key = seed
            self._rng = np.random.default_rng(None)
        else:
            self._prng_seed = None
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

        no_max_workers = (
            execution_config.device_options.get("max_workers", self._max_workers) is None
        )

        if execution_config.gradient_method in {"backprop", "best"} and no_max_workers:
            if circuit is None:
                return True
            return not circuit.shots and not any(
                isinstance(m.obs, qml.SparseHamiltonian) for m in circuit.measurements
            )

        if execution_config.gradient_method in {"adjoint", "best"}:
            return _supports_adjoint(circuit=circuit)
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

        This device supports any qubit operations that provide a matrix

        """
        config = self._setup_execution_config(execution_config)
        transform_program = TransformProgram()

        transform_program.add_transform(validate_device_wires, self.wires, name=self.name)
        transform_program.add_transform(mid_circuit_measurements, device=self)
        transform_program.add_transform(
            decompose,
            stopping_condition=stopping_condition,
            stopping_condition_shots=stopping_condition_shots,
            name=self.name,
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
            _add_adjoint_transforms(
                transform_program, device_vjp=config.use_device_jacobian_product
            )

        return transform_program, config

    def _setup_execution_config(self, execution_config: ExecutionConfig) -> ExecutionConfig:
        """This is a private helper for ``preprocess`` that sets up the execution config.

        Args:
            execution_config (ExecutionConfig)

        Returns:
            ExecutionConfig: a preprocessed execution config

        """
        updated_values = {}

        for option in execution_config.device_options:
            if option not in self._device_options:
                raise qml.DeviceError(f"device option {option} not present on {self}")

        gradient_method = execution_config.gradient_method
        if execution_config.gradient_method == "best":
            no_max_workers = (
                execution_config.device_options.get("max_workers", self._max_workers) is None
            )
            gradient_method = "backprop" if no_max_workers else "adjoint"
            updated_values["gradient_method"] = gradient_method

        if execution_config.use_device_gradient is None:
            updated_values["use_device_gradient"] = gradient_method in {
                "adjoint",
                "backprop",
            }
        if execution_config.use_device_jacobian_product is None:
            updated_values["use_device_jacobian_product"] = gradient_method == "adjoint"
        if execution_config.grad_on_execution is None:
            updated_values["grad_on_execution"] = gradient_method == "adjoint"

        updated_values["device_options"] = dict(execution_config.device_options)  # copy
        for option in self._device_options:
            if option not in updated_values["device_options"]:
                updated_values["device_options"][option] = getattr(self, f"_{option}")
        return replace(execution_config, **updated_values)

    def execute(
        self,
        circuits: QuantumTape_or_Batch,
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ) -> Result_or_ResultBatch:
        self.reset_prng_key()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                """Entry with args=(circuits=%s) called by=%s""",
                circuits,
                "::L".join(
                    str(i) for i in inspect.getouterframes(inspect.currentframe(), 2)[1][1:3]
                ),
            )

        max_workers = execution_config.device_options.get("max_workers", self._max_workers)
        self._state_cache = {} if execution_config.use_device_jacobian_product else None
        interface = (
            execution_config.interface
            if execution_config.gradient_method in {"backprop", None}
            else None
        )
        prng_keys = [self.get_prng_keys()[0] for _ in range(len(circuits))]

        if max_workers is None:
            return tuple(
                _simulate_wrapper(
                    c,
                    {
                        "rng": self._rng,
                        "debugger": self._debugger,
                        "interface": interface,
                        "state_cache": self._state_cache,
                        "prng_key": _key,
                    },
                )
                for c, _key in zip(circuits, prng_keys)
            )

        vanilla_circuits = [convert_to_numpy_parameters(c) for c in circuits]
        seeds = self._rng.integers(2**31 - 1, size=len(vanilla_circuits))
        simulate_kwargs = [{"rng": _rng, "prng_key": _key} for _rng, _key in zip(seeds, prng_keys)]

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            exec_map = executor.map(_simulate_wrapper, vanilla_circuits, simulate_kwargs)
            results = tuple(exec_map)

        # reset _rng to mimic serial behaviour
        self._rng = np.random.default_rng(self._rng.integers(2**31 - 1))

        return results

    def compute_derivatives(
        self,
        circuits: QuantumTape_or_Batch,
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ):
        max_workers = execution_config.device_options.get("max_workers", self._max_workers)
        if max_workers is None:
            return tuple(adjoint_jacobian(circuit) for circuit in circuits)

        vanilla_circuits = [convert_to_numpy_parameters(c) for c in circuits]
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            exec_map = executor.map(adjoint_jacobian, vanilla_circuits)
            res = tuple(exec_map)

        # reset _rng to mimic serial behaviour
        self._rng = np.random.default_rng(self._rng.integers(2**31 - 1))

        return res

    def execute_and_compute_derivatives(
        self,
        circuits: QuantumTape_or_Batch,
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ):
        self.reset_prng_key()
        max_workers = execution_config.device_options.get("max_workers", self._max_workers)
        if max_workers is None:
            results = tuple(_adjoint_jac_wrapper(c, debugger=self._debugger) for c in circuits)
        else:
            vanilla_circuits = [convert_to_numpy_parameters(c) for c in circuits]

            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                results = tuple(
                    executor.map(
                        _adjoint_jac_wrapper,
                        vanilla_circuits,
                    )
                )

        return tuple(zip(*results))

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
        max_workers = execution_config.device_options.get("max_workers", self._max_workers)
        if max_workers is None:
            return tuple(adjoint_jvp(circuit, tans) for circuit, tans in zip(circuits, tangents))

        vanilla_circuits = [convert_to_numpy_parameters(c) for c in circuits]
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            res = tuple(executor.map(adjoint_jvp, vanilla_circuits, tangents))

        # reset _rng to mimic serial behaviour
        self._rng = np.random.default_rng(self._rng.integers(2**31 - 1))

        return res

    def execute_and_compute_jvp(
        self,
        circuits: QuantumTape_or_Batch,
        tangents: Tuple[Number],
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ):
        self.reset_prng_key()
        max_workers = execution_config.device_options.get("max_workers", self._max_workers)
        if max_workers is None:
            results = tuple(
                _adjoint_jvp_wrapper(c, t, debugger=self._debugger)
                for c, t in zip(circuits, tangents)
            )
        else:
            vanilla_circuits = [convert_to_numpy_parameters(c) for c in circuits]

            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                results = tuple(
                    executor.map(
                        _adjoint_jvp_wrapper,
                        vanilla_circuits,
                        tangents,
                    )
                )

        return tuple(zip(*results))

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
        r"""The vector jacobian product used in reverse-mode differentiation. ``DefaultQubit`` uses the
        adjoint differentiation method to compute the VJP.

        Args:
            circuits (Union[QuantumTape, Sequence[QuantumTape]]): the circuit or batch of circuits
            cotangents (Tuple[Number, Tuple[Number]]): Gradient-output vector. Must have shape matching the output shape of the
                corresponding circuit. If the circuit has a single output, `cotangents` may be a single number, not an iterable
                of numbers.
            execution_config (ExecutionConfig): a datastructure with all additional information required for execution

        Returns:
            tensor-like: A numeric result of computing the vector jacobian product

        **Definition of vjp:**

        If we have a function with jacobian:

        .. math::

            \vec{y} = f(\vec{x}) \qquad J_{i,j} = \frac{\partial y_i}{\partial x_j}

        The vector jacobian product is the inner product of the derivatives of the output ``y`` with the
        Jacobian matrix. The derivatives of the output vector are sometimes called the **cotangents**.

        .. math::

            \text{d}x_i = \Sigma_{i} \text{d}y_i J_{i,j}

        **Shape of cotangents:**

        The value provided to ``cotangents`` should match the output of :meth:`~.execute`. For computing the full Jacobian,
        the cotangents can be batched to vectorize the computation. In this case, the cotangents can have the following
        shapes. ``batch_size`` below refers to the number of entries in the Jacobian:

        * For a state measurement, the cotangents must have shape ``(batch_size, 2 ** n_wires)``
        * For ``n`` expectation values, the cotangents must have shape ``(n, batch_size)``. If ``n = 1``,
          then the shape must be ``(batch_size,)``.

        """
        max_workers = execution_config.device_options.get("max_workers", self._max_workers)
        if max_workers is None:

            def _state(circuit):
                return (
                    None if self._state_cache is None else self._state_cache.get(circuit.hash, None)
                )

            return tuple(
                adjoint_vjp(circuit, cots, state=_state(circuit))
                for circuit, cots in zip(circuits, cotangents)
            )

        vanilla_circuits = [convert_to_numpy_parameters(c) for c in circuits]
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            res = tuple(executor.map(adjoint_vjp, vanilla_circuits, cotangents))

        # reset _rng to mimic serial behaviour
        self._rng = np.random.default_rng(self._rng.integers(2**31 - 1))

        return res

    def execute_and_compute_vjp(
        self,
        circuits: QuantumTape_or_Batch,
        cotangents: Tuple[Number],
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ):
        self.reset_prng_key()
        max_workers = execution_config.device_options.get("max_workers", self._max_workers)
        if max_workers is None:
            results = tuple(
                _adjoint_vjp_wrapper(c, t, debugger=self._debugger)
                for c, t in zip(circuits, cotangents)
            )
        else:
            vanilla_circuits = [convert_to_numpy_parameters(c) for c in circuits]

            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                results = tuple(
                    executor.map(
                        _adjoint_vjp_wrapper,
                        vanilla_circuits,
                        cotangents,
                    )
                )

        return tuple(zip(*results))


def _simulate_wrapper(circuit, kwargs):
    return simulate(circuit, **kwargs)


def _adjoint_jac_wrapper(c, debugger=None):
    state, is_state_batched = get_final_state(c, debugger=debugger)
    jac = adjoint_jacobian(c, state=state)
    res = measure_final_state(c, state, is_state_batched)
    return res, jac


def _adjoint_jvp_wrapper(c, t, debugger=None):
    state, is_state_batched = get_final_state(c, debugger=debugger)
    jvp = adjoint_jvp(c, t, state=state)
    res = measure_final_state(c, state, is_state_batched)
    return res, jvp


def _adjoint_vjp_wrapper(c, t, debugger=None):
    state, is_state_batched = get_final_state(c, debugger=debugger)
    vjp = adjoint_vjp(c, t, state=state)
    res = measure_final_state(c, state, is_state_batched)
    return res, vjp
