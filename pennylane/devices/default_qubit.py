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

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING

import numpy as np

from pennylane import capture, math, ops
from pennylane.exceptions import DeviceError
from pennylane.logging import debug_logger, debug_logger_init
from pennylane.measurements import (
    ClassicalShadowMP,
    CountsMP,
    ExpectationMP,
    MeasurementProcess,
    MidMeasureMP,
    SampleMeasurement,
    ShadowExpvalMP,
    Shots,
    StateMeasurement,
    StateMP,
)
from pennylane.operation import DecompositionUndefinedError
from pennylane.ops.op_math import Conditional
from pennylane.tape import QuantumScript, QuantumScriptBatch, QuantumScriptOrBatch
from pennylane.transforms import (
    broadcast_expand,
    convert_to_numpy_parameters,
)
from pennylane.transforms import decompose as transforms_decompose
from pennylane.transforms import (
    defer_measurements,
    dynamic_one_shot,
)
from pennylane.transforms.core import TransformProgram, transform
from pennylane.typing import PostprocessingFn, Result, ResultBatch, TensorLike

from .device_api import Device
from .execution_config import ExecutionConfig, MCMConfig
from .modifiers import simulator_tracking, single_tape_support
from .preprocess import (
    decompose,
    device_resolve_dynamic_wires,
    no_sampling,
    validate_adjoint_trainable_params,
    validate_device_wires,
    validate_measurements,
    validate_multiprocessing_workers,
    validate_observables,
)
from .qubit.adjoint_jacobian import adjoint_jacobian, adjoint_jvp, adjoint_vjp
from .qubit.sampling import jax_random_split
from .qubit.simulate import get_final_state, measure_final_state, simulate

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

if TYPE_CHECKING:
    from numbers import Number

    from jax.extend.core import Jaxpr

    from pennylane.operation import Operator


# Base gate set for DefaultQubit
_BASE_DQ_GATE_SET = {
    "CNOT",
    "CRX",
    "CRY",
    "CRZ",
    "CRot",
    "CSWAP",
    "CY",
    "CZ",
    "ControlledPhaseShift",
    "GlobalPhase",
    "Hadamard",
    "ISWAP",
    "Identity",
    "IsingXX",
    "IsingXY",
    "IsingYY",
    "IsingZZ",
    "MultiControlledX",
    "MultiRZ",
    "PSWAP",
    "PauliX",
    "PauliY",
    "PauliZ",
    "PhaseShift",
    "RX",
    "RY",
    "RZ",
    "S",
    "SWAP",
    "SX",
    "T",
    "Toffoli",
}


# Complete gate set including controlled and adjoint variants
ALL_DQ_GATE_SET = (
    _BASE_DQ_GATE_SET
    | {f"C({gate})" for gate in _BASE_DQ_GATE_SET}
    | {f"Adjoint({gate})" for gate in _BASE_DQ_GATE_SET}
)


_special_operator_support = {
    "QFT": lambda op: len(op.wires) < 6,
    "GroverOperator": lambda op: len(op.wires) < 13,
    "FromBloq": lambda op: len(op.wires) < 4 and op.has_matrix,
    "Snapshot": lambda _: True,
    "Allocate": lambda _: True,
    "Deallocate": lambda _: True,
}
"""Map from gates with a special support condition."""


def stopping_condition(op: Operator, allow_mcms=True) -> bool:
    """Specify whether or not an Operator object is supported by the device."""
    if constraint := _special_operator_support.get(op.name):
        return constraint(op)
    if op.__class__.__name__[:3] == "Pow" and any(math.requires_grad(d) for d in op.data):
        return False
    if isinstance(op, MidMeasureMP):
        return allow_mcms
    return op.has_matrix or op.has_sparse_matrix


# need to create these once so we can compare in tests
allow_mcms_stopping_condition = partial(stopping_condition, allow_mcms=True)
no_mcms_stopping_condition = partial(stopping_condition, allow_mcms=False)


def observable_accepts_sampling(obs: Operator) -> bool:
    """Verifies whether an observable supports sample measurement"""

    if isinstance(obs, ops.CompositeOp):
        return all(observable_accepts_sampling(o) for o in obs.operands)

    if isinstance(obs, ops.SymbolicOp):
        return observable_accepts_sampling(obs.base)

    return obs.has_diagonalizing_gates


def observable_accepts_analytic(obs: Operator, is_expval=False) -> bool:
    """Verifies whether an observable supports analytic measurement"""

    if isinstance(obs, ops.CompositeOp):
        return all(observable_accepts_analytic(o, is_expval) for o in obs.operands)

    if isinstance(obs, ops.SymbolicOp):
        return observable_accepts_analytic(obs.base, is_expval)

    if is_expval and isinstance(obs, (ops.SparseHamiltonian, ops.Hermitian)):
        return True

    return obs.has_diagonalizing_gates


def accepted_sample_measurement(m: MeasurementProcess) -> bool:
    """Specifies whether a measurement is accepted when sampling."""

    if not isinstance(
        m,
        (
            SampleMeasurement,
            ClassicalShadowMP,
            ShadowExpvalMP,
        ),
    ):
        return False

    if m.obs is not None:
        return observable_accepts_sampling(m.obs)

    return True


def accepted_analytic_measurement(m: MeasurementProcess) -> bool:
    """Specifies whether a measurement is accepted when analytic."""

    if not isinstance(m, StateMeasurement):
        return False

    if m.obs is not None:
        return observable_accepts_analytic(m.obs, isinstance(m, ExpectationMP))

    return True


def null_postprocessing(results):
    """An empty post-processing function."""
    return results[0]


def all_state_postprocessing(results, measurements, wire_order):
    """Process a state measurement back into the original measurements."""
    result = tuple(m.process_state(results[0], wire_order=wire_order) for m in measurements)
    return result[0] if len(measurements) == 1 else result


@transform
def _conditional_broastcast_expand(tape):
    """Apply conditional broadcast expansion to the tape if needed."""
    # Currently, default.qubit does not support native parameter broadcasting with
    # shadow operations. We need to expand the tape to include the broadcasted parameters.
    if any(isinstance(mp, (ShadowExpvalMP, ClassicalShadowMP)) for mp in tape.measurements):
        return broadcast_expand(tape)
    return (tape,), null_postprocessing


@transform
def no_counts(tape):
    """Throws an error on counts measurements."""
    if any(isinstance(mp, CountsMP) for mp in tape.measurements):
        raise NotImplementedError("The JAX-JIT interface doesn't support qml.counts.")
    return (tape,), null_postprocessing


@transform
def adjoint_state_measurements(
    tape: QuantumScript, device_vjp=False
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """Perform adjoint measurement preprocessing.

    * Allows a tape with only expectation values through unmodified
    * Raises an error if non-expectation value measurements exist and any have diagonalizing gates
    * Turns the circuit into a state measurement + classical postprocesssing for arbitrary measurements

    Args:
        tape (QuantumTape): the input circuit

    """
    if all(isinstance(m, ExpectationMP) for m in tape.measurements):
        return (tape,), null_postprocessing

    if any(len(m.diagonalizing_gates()) > 0 for m in tape.measurements):
        raise DeviceError(
            "adjoint diff supports either all expectation values or only measurements without observables."
        )

    params = tape.get_parameters()

    if device_vjp:
        for p in params:
            if (
                math.requires_grad(p)
                and math.get_interface(p) == "tensorflow"
                and math.get_dtype_name(p) in {"float32", "complex64"}
            ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
                raise ValueError(
                    "tensorflow with adjoint differentiation of the state requires float64 or complex128 parameters."
                )

    complex_data = [math.cast(p, complex) for p in params]
    tape = tape.bind_new_parameters(complex_data, list(range(len(params))))
    new_mp = StateMP(wires=tape.wires)
    state_tape = tape.copy(measurements=[new_mp])
    return (state_tape,), partial(
        all_state_postprocessing, measurements=tape.measurements, wire_order=tape.wires
    )


def adjoint_ops(op: Operator) -> bool:
    """Specify whether or not an Operator is supported by adjoint differentiation."""
    return not isinstance(op, (Conditional, MidMeasureMP)) and (
        op.num_params == 0
        or not any(math.requires_grad(d) for d in op.data)
        or (op.num_params == 1 and op.has_generator)
    )


def adjoint_observables(obs: Operator) -> bool:
    """Specifies whether or not an observable is compatible with adjoint differentiation on DefaultQubit."""
    return obs.has_matrix


def _supports_adjoint(circuit, device_wires, device_name):
    if circuit is None:
        return True

    program = TransformProgram()
    program.add_transform(validate_device_wires, device_wires, name=device_name)
    _add_adjoint_transforms(program, device_wires=device_wires)

    try:
        program((circuit,))
    except (
        DecompositionUndefinedError,
        DeviceError,
        AttributeError,
    ):
        return False
    return True


def _add_adjoint_transforms(program: TransformProgram, device_vjp=False, device_wires=None) -> None:
    """Private helper function for ``preprocess`` that adds the transforms specific
    for adjoint differentiation.

    Args:
        program (TransformProgram): where we will add the adjoint differentiation transforms
        device_vjp (bool): whether or not to use the device-provided Vector Jacobian Product (VJP).
        device_wires (Wires): the device wires, used to calculate available work wires

    Side Effects:
        Adds transforms to the input program.

    """

    name = "adjoint + default.qubit"
    program.add_transform(no_sampling, name=name)
    program.add_transform(
        decompose,
        stopping_condition=adjoint_ops,
        device_wires=device_wires,
        target_gates=ALL_DQ_GATE_SET,
        name=name,
        skip_initial_state_prep=False,
    )
    program.add_transform(validate_observables, adjoint_observables, name=name)
    program.add_transform(
        validate_measurements,
        name=name,
    )
    program.add_transform(adjoint_state_measurements, device_vjp=device_vjp)
    program.add_transform(broadcast_expand)
    program.add_transform(validate_adjoint_trainable_params)


@simulator_tracking
@single_tape_support
class DefaultQubit(Device):
    """A PennyLane device written in Python and capable of backpropagation derivatives.

    Args:
        wires (int, Iterable[Number, str]): Number of wires present on the device, or iterable that
            contains unique labels for the wires as numbers (i.e., ``[-1, 0, 2]``) or strings
            (``['auxiliary', 'q1', 'q2']``). Default ``None`` if not specified.
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
        max_workers (int): A :class:`~pennylane.concurrency.executors.base.RemoteExec` executes tapes asynchronously
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
        :title: Accelerate calculations with concurrent executors

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

            Concurrent executors using the multiprocessing backend (default) may fail depending on your platform and environment (Python shell,
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

    _state_cache: dict | None = None
    """
    A cache to store the "pre-rotated state" for reuse between the forward pass call to ``execute`` and
    subsequent calls to ``compute_vjp``. ``None`` indicates that no caching is required.
    """

    _device_options = ("max_workers", "rng", "prng_key")
    """
    tuple of string names for all the device options.
    """

    @debug_logger_init
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
        if math.get_interface(seed) == "jax":
            self._prng_seed = seed
            self._prng_key = seed
            self._rng = np.random.default_rng(None)
        else:
            self._prng_seed = None
            self._prng_key = None
            self._rng = np.random.default_rng(seed)
        self._debugger = None

    @debug_logger
    def supports_derivatives(
        self,
        execution_config: ExecutionConfig | None = None,
        circuit: QuantumScript | None = None,
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
                isinstance(m.obs, ops.SparseHamiltonian) for m in circuit.measurements
            )

        if execution_config.gradient_method in {"adjoint", "best"}:
            return _supports_adjoint(circuit, device_wires=self.wires, device_name=self.name)
        return False

    def _capture_preprocess_transforms(self, config: ExecutionConfig) -> TransformProgram:
        transform_program = TransformProgram()
        if config.mcm_config.mcm_method == "deferred":
            transform_program.add_transform(defer_measurements, num_wires=len(self.wires))
        transform_program.add_transform(transforms_decompose, gate_set=stopping_condition)

        return transform_program

    @debug_logger
    def preprocess_transforms(
        self, execution_config: ExecutionConfig | None = None
    ) -> TransformProgram:
        """This function defines the device transform program to be applied and an updated device configuration.

        Args:
            execution_config (ExecutionConfig | None): A data structure describing the
                parameters needed to fully describe the execution.

        Returns:
            TransformProgram:

        """
        config = execution_config or ExecutionConfig()

        if capture.enabled():
            return self._capture_preprocess_transforms(config)

        transform_program = TransformProgram()

        if config.interface == math.Interface.JAX_JIT:
            transform_program.add_transform(no_counts)

        if config.mcm_config.mcm_method == "deferred":
            transform_program.add_transform(defer_measurements, allow_postselect=True)
            _stopping_condition = no_mcms_stopping_condition
        else:
            _stopping_condition = allow_mcms_stopping_condition
        transform_program.add_transform(
            decompose,
            stopping_condition=_stopping_condition,
            device_wires=self.wires,
            target_gates=ALL_DQ_GATE_SET,
            name=self.name,
        )
        _allow_resets = config.mcm_config.mcm_method != "deferred"
        transform_program.add_transform(
            device_resolve_dynamic_wires, wires=self.wires, allow_resets=_allow_resets
        )
        transform_program.add_transform(validate_device_wires, self.wires, name=self.name)
        transform_program.add_transform(
            validate_measurements,
            analytic_measurements=accepted_analytic_measurement,
            sample_measurements=accepted_sample_measurement,
            name=self.name,
        )
        transform_program.add_transform(_conditional_broastcast_expand)
        if config.mcm_config.mcm_method == "tree-traversal":
            transform_program.add_transform(broadcast_expand)

        if config.mcm_config.mcm_method == "one-shot":
            transform_program.add_transform(
                dynamic_one_shot, postselect_mode=config.mcm_config.postselect_mode
            )
        # Validate multi processing
        max_workers = config.device_options.get("max_workers", self._max_workers)
        if max_workers:
            transform_program.add_transform(validate_multiprocessing_workers, max_workers, self)

        if config.gradient_method == "backprop":
            transform_program.add_transform(no_sampling, name="backprop + default.qubit")

        if config.gradient_method == "adjoint":
            _add_adjoint_transforms(
                transform_program,
                device_vjp=config.use_device_jacobian_product,
                device_wires=self.wires,
            )
        return transform_program

    @debug_logger
    def setup_execution_config(
        self, config: ExecutionConfig | None = None, circuit: QuantumScript | None = None
    ) -> ExecutionConfig:
        config = config or ExecutionConfig()
        updated_values = {}

        # uncomment once compilation overhead with jitting improved
        # TODO: [sc-82874]
        # jax_interfaces = {math.Interface.JAX, math.Interface.JAX_JIT}
        # updated_values["convert_to_numpy"] = (
        #    config.interface not in jax_interfaces
        #    or config.gradient_method == "adjoint"
        #    # need numpy to use caching, and need caching higher order derivatives
        #    or config.derivative_order > 1
        # )

        # If PRNGKey is present, we can't use a pure_callback, because that would cause leaked tracers
        # we assume that if someone provides a PRNGkey, they want to jit end-to-end
        if not capture.enabled():
            jax_interfaces = {math.Interface.JAX, math.Interface.JAX_JIT}
            updated_values["convert_to_numpy"] = not (
                self._prng_key is not None
                and config.interface in jax_interfaces
                and config.gradient_method != "adjoint"
                # need numpy to use caching, and need caching higher order derivatives
                and config.derivative_order == 1
            )

        for option, value in config.device_options.items():
            if option not in self._device_options:
                raise DeviceError(f"device option {option} not present on {self}")

            if capture.enabled():
                if option == "max_workers" and value is not None:
                    raise DeviceError("Cannot set 'max_workers' if program capture is enabled.")

        gradient_method = config.gradient_method
        if config.gradient_method == "best":
            no_max_workers = (
                config.device_options.get("max_workers", self._max_workers) is None
            ) or capture.enabled()
            gradient_method = "backprop" if no_max_workers else "adjoint"
            updated_values["gradient_method"] = gradient_method

        if config.use_device_gradient is None:
            updated_values["use_device_gradient"] = gradient_method in {
                "adjoint",
                "backprop",
            }
        if config.use_device_jacobian_product is None:
            updated_values["use_device_jacobian_product"] = gradient_method == "adjoint"
        if config.grad_on_execution is None:
            updated_values["grad_on_execution"] = gradient_method == "adjoint"

        updated_values["device_options"] = dict(config.device_options)  # copy
        for option in self._device_options:
            if option not in updated_values["device_options"]:
                updated_values["device_options"][option] = getattr(self, f"_{option}")

        mcm_config = self._setup_mcm_config(config.mcm_config, circuit)

        updated_values["mcm_config"] = mcm_config
        return replace(config, **updated_values)

    def _setup_mcm_config(self, mcm_config: MCMConfig, tape: QuantumScript) -> MCMConfig:
        if capture.enabled():
            return self._capture_setup_mcm_config(mcm_config)

        final_mcm_method = mcm_config.mcm_method
        if mcm_config.mcm_method is None:
            final_mcm_method = "one-shot" if getattr(tape, "shots", None) else "deferred"
        elif mcm_config.mcm_method == "device":
            final_mcm_method = "tree-traversal"

        supported_methods = {"one-shot", "deferred", "tree-traversal"}
        if final_mcm_method not in supported_methods:
            raise DeviceError(
                f"mcm_method {final_mcm_method} not supported on default.qubit. "
                f"Supported methods are {supported_methods}"
            )

        if mcm_config.postselect_mode == "fill-shots" and final_mcm_method != "deferred":
            raise DeviceError(
                "Using postselect_mode='fill-shots' is only supported with mcm_method='deferred'."
            )

        return replace(mcm_config, mcm_method=final_mcm_method)

    def _capture_setup_mcm_config(self, mcm_config):
        mcm_updated_values = {}
        mcm_method = mcm_config.mcm_method

        if mcm_method == "single-branch-statistics" and mcm_config.postselect_mode is not None:
            warnings.warn(
                "Setting 'postselect_mode' is not supported with mcm_method='single-branch-"
                "statistics'. 'postselect_mode' will be ignored.",
                UserWarning,
            )
            mcm_updated_values["postselect_mode"] = None
        if mcm_method is None:
            mcm_updated_values["mcm_method"] = "deferred"
        return replace(mcm_config, **mcm_updated_values)

    @debug_logger
    def execute(
        self,
        circuits: QuantumScriptOrBatch,
        execution_config: ExecutionConfig | None = None,
    ) -> Result | ResultBatch:
        if execution_config is None:
            execution_config = ExecutionConfig()
        self.reset_prng_key()
        max_workers = execution_config.device_options.get("max_workers", self._max_workers)
        self._state_cache = {} if execution_config.use_device_jacobian_product else None
        interface = (
            execution_config.interface
            if execution_config.gradient_method in {"backprop", None}
            else None
        )
        prng_keys = [self.get_prng_keys()[0] for _ in range(len(circuits))]

        if (
            not execution_config.convert_to_numpy
            and execution_config.interface == math.Interface.JAX_JIT
            and len(circuits) > 10
        ):
            warnings.warn(
                (
                    "Jitting executions with many circuits may have substantial classical overhead."
                    " To disable end-to-end jitting, please specify a integer seed instead of a PRNGKey."
                ),
                UserWarning,
            )

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
                        "mcm_method": execution_config.mcm_config.mcm_method,
                        "postselect_mode": execution_config.mcm_config.postselect_mode,
                    },
                )
                for c, _key in zip(circuits, prng_keys)
            )

        vanilla_circuits = convert_to_numpy_parameters(circuits)[0]
        seeds = self._rng.integers(2**31 - 1, size=len(vanilla_circuits))
        simulate_kwargs = [
            {
                "rng": _rng,
                "prng_key": _key,
                "mcm_method": execution_config.mcm_config.mcm_method,
                "postselect_mode": execution_config.mcm_config.postselect_mode,
            }
            for _rng, _key in zip(seeds, prng_keys)
        ]

        with execution_config.executor_backend(max_workers=max_workers) as executor:
            exec_map = executor.map(_simulate_wrapper, vanilla_circuits, simulate_kwargs)
            results = tuple(exec_map)

        # reset _rng to mimic serial behaviour
        self._rng = np.random.default_rng(self._rng.integers(2**31 - 1))

        return results

    @debug_logger
    def compute_derivatives(
        self,
        circuits: QuantumScriptOrBatch,
        execution_config: ExecutionConfig | None = None,
    ):
        if execution_config is None:
            execution_config = ExecutionConfig()
        max_workers = execution_config.device_options.get("max_workers", self._max_workers)
        if max_workers is None:
            return tuple(adjoint_jacobian(circuit) for circuit in circuits)

        vanilla_circuits = convert_to_numpy_parameters(circuits)[0]

        with execution_config.executor_backend(max_workers=max_workers) as executor:
            exec_map = executor.map(adjoint_jacobian, vanilla_circuits)
            res = tuple(exec_map)

        # reset _rng to mimic serial behaviour
        self._rng = np.random.default_rng(self._rng.integers(2**31 - 1))

        return res

    @debug_logger
    def execute_and_compute_derivatives(
        self,
        circuits: QuantumScriptOrBatch,
        execution_config: ExecutionConfig | None = None,
    ):
        if execution_config is None:
            execution_config = ExecutionConfig()
        self.reset_prng_key()
        max_workers = execution_config.device_options.get("max_workers", self._max_workers)
        if max_workers is None:
            results = tuple(_adjoint_jac_wrapper(c, debugger=self._debugger) for c in circuits)
        else:
            vanilla_circuits = convert_to_numpy_parameters(circuits)[0]

            with execution_config.executor_backend(max_workers=max_workers) as executor:
                results = tuple(
                    executor.map(
                        _adjoint_jac_wrapper,
                        vanilla_circuits,
                    )
                )

        return tuple(zip(*results))

    @debug_logger
    def supports_jvp(
        self,
        execution_config: ExecutionConfig | None = None,
        circuit: QuantumScript | None = None,
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

    @debug_logger
    def compute_jvp(
        self,
        circuits: QuantumScriptOrBatch,
        tangents: tuple[Number, ...],
        execution_config: ExecutionConfig | None = None,
    ):
        if execution_config is None:
            execution_config = ExecutionConfig()
        max_workers = execution_config.device_options.get("max_workers", self._max_workers)
        if max_workers is None:
            return tuple(adjoint_jvp(circuit, tans) for circuit, tans in zip(circuits, tangents))

        vanilla_circuits = convert_to_numpy_parameters(circuits)[0]
        with execution_config.executor_backend(max_workers=max_workers) as executor:
            res = tuple(executor.map(adjoint_jvp, vanilla_circuits, tangents))

        # reset _rng to mimic serial behaviour
        self._rng = np.random.default_rng(self._rng.integers(2**31 - 1))

        return res

    @debug_logger
    def execute_and_compute_jvp(
        self,
        circuits: QuantumScriptOrBatch,
        tangents: tuple[Number, ...],
        execution_config: ExecutionConfig | None = None,
    ):
        if execution_config is None:
            execution_config = ExecutionConfig()
        self.reset_prng_key()
        max_workers = execution_config.device_options.get("max_workers", self._max_workers)
        if max_workers is None:
            results = tuple(
                _adjoint_jvp_wrapper(c, t, debugger=self._debugger)
                for c, t in zip(circuits, tangents)
            )
        else:
            vanilla_circuits = convert_to_numpy_parameters(circuits)[0]

            with execution_config.executor_backend(max_workers=max_workers) as executor:
                results = tuple(
                    executor.map(
                        _adjoint_jvp_wrapper,
                        vanilla_circuits,
                        tangents,
                    )
                )

        return tuple(zip(*results))

    @debug_logger
    def supports_vjp(
        self,
        execution_config: ExecutionConfig | None = None,
        circuit: QuantumScript | None = None,
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

    @debug_logger
    def compute_vjp(
        self,
        circuits: QuantumScriptOrBatch,
        cotangents: tuple[Number, ...],
        execution_config: ExecutionConfig | None = None,
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
        if execution_config is None:
            execution_config = ExecutionConfig()
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

        vanilla_circuits = convert_to_numpy_parameters(circuits)[0]
        with execution_config.executor_backend(max_workers=max_workers) as executor:
            res = tuple(executor.map(adjoint_vjp, vanilla_circuits, cotangents))

        # reset _rng to mimic serial behaviour
        self._rng = np.random.default_rng(self._rng.integers(2**31 - 1))

        return res

    @debug_logger
    def execute_and_compute_vjp(
        self,
        circuits: QuantumScriptOrBatch,
        cotangents: tuple[Number, ...],
        execution_config: ExecutionConfig | None = None,
    ):
        if execution_config is None:
            execution_config = ExecutionConfig()
        self.reset_prng_key()
        max_workers = execution_config.device_options.get("max_workers", self._max_workers)
        if max_workers is None:
            results = tuple(
                _adjoint_vjp_wrapper(c, t, debugger=self._debugger)
                for c, t in zip(circuits, cotangents)
            )
        else:
            vanilla_circuits = convert_to_numpy_parameters(circuits)[0]

            with execution_config.executor_backend(max_workers=max_workers) as executor:
                results = tuple(
                    executor.map(
                        _adjoint_vjp_wrapper,
                        vanilla_circuits,
                        cotangents,
                    )
                )

        return tuple(zip(*results))

    # pylint: disable=import-outside-toplevel
    @debug_logger
    def eval_jaxpr(
        self,
        jaxpr: Jaxpr,
        consts: list[TensorLike],
        *args,
        execution_config=None,
        shots=Shots(None),
    ) -> list[TensorLike]:
        from .qubit.dq_interpreter import DefaultQubitInterpreter

        execution_config = execution_config or ExecutionConfig()
        if (mcm_method := execution_config.mcm_config.mcm_method) not in (
            "deferred",
            "single-branch-statistics",
            None,
        ):
            raise DeviceError(
                f"mcm_method='{mcm_method}' is not supported with default.qubit "
                "when program capture is enabled."
            )

        if self.wires is None:
            raise DeviceError("Device wires are required for jaxpr execution.")
        shots = Shots(shots)
        if shots.has_partitioned_shots:
            raise DeviceError("Shot vectors are unsupported with jaxpr execution.")
        if self._prng_key is not None:
            key = self.get_prng_keys()[0]
        else:
            import jax

            key = jax.random.PRNGKey(self._rng.integers(100000))

        interpreter = DefaultQubitInterpreter(
            num_wires=len(self.wires),
            shots=shots.total_shots,
            key=key,
            execution_config=execution_config,
        )
        return interpreter.eval(jaxpr, consts, *args)

    def _backprop_jvp(self, jaxpr, args, tangents, execution_config=None):
        import jax

        def _make_zero(tan, arg):
            return (
                jax.lax.zeros_like_array(arg).astype(tan.aval.dtype)
                if isinstance(tan, jax.interpreters.ad.Zero)
                else tan
            )

        def eval_wrapper(*inner_args):
            n_consts = len(jaxpr.constvars)
            consts = inner_args[:n_consts]
            non_const_args = inner_args[n_consts:]
            return self.eval_jaxpr(
                jaxpr, consts, *non_const_args, execution_config=execution_config
            )

        tangents = tuple(map(_make_zero, tangents, args))

        return jax.jvp(eval_wrapper, args, tangents)

    # pylint :disable=import-outside-toplevel, unused-argument
    @debug_logger
    def jaxpr_jvp(
        self,
        jaxpr,
        args: Sequence[TensorLike],
        tangents: Sequence[TensorLike],
        execution_config=None,
    ) -> tuple[Sequence[TensorLike], Sequence[TensorLike]]:
        gradient_method = getattr(execution_config, "gradient_method", "backprop")
        if gradient_method == "backprop":
            return self._backprop_jvp(jaxpr, args, tangents, execution_config=execution_config)

        if gradient_method == "adjoint":
            from .qubit.jaxpr_adjoint import execute_and_jvp

            return execute_and_jvp(jaxpr, args, tangents, num_wires=len(self.wires))

        raise NotImplementedError(
            f"DefaultQubit does not support gradient_method={gradient_method}"
        )


def _simulate_wrapper(circuit, kwargs):
    return simulate(circuit, **kwargs)


def _adjoint_jac_wrapper(c, debugger=None):
    c = c.map_to_standard_wires()
    state, is_state_batched = get_final_state(c, debugger=debugger)
    jac = adjoint_jacobian(c, state=state)
    res = measure_final_state(c, state, is_state_batched)
    return res, jac


def _adjoint_jvp_wrapper(c, t, debugger=None):
    c = c.map_to_standard_wires()
    state, is_state_batched = get_final_state(c, debugger=debugger)
    jvp = adjoint_jvp(c, t, state=state)
    res = measure_final_state(c, state, is_state_batched)
    return res, jvp


def _adjoint_vjp_wrapper(c, t, debugger=None):
    c = c.map_to_standard_wires()
    state, is_state_batched = get_final_state(c, debugger=debugger)
    vjp = adjoint_vjp(c, t, state=state)
    res = measure_final_state(c, state, is_state_batched)
    return res, vjp
