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
"""The default.qutrit.mixed device is PennyLane's standard qutrit simulator for mixed-state
computations."""
import logging
import warnings
from collections.abc import Callable, Sequence
from dataclasses import replace
from functools import partial

import numpy as np

import pennylane as qml
from pennylane.exceptions import DeviceError
from pennylane.logging import debug_logger, debug_logger_init
from pennylane.ops import _qutrit__channel__ops__ as channels
from pennylane.tape import QuantumScript, QuantumScriptOrBatch
from pennylane.transforms.core import TransformProgram
from pennylane.typing import Result, ResultBatch

from . import Device
from .default_qutrit import DefaultQutrit
from .execution_config import ExecutionConfig
from .modifiers import simulator_tracking, single_tape_support
from .preprocess import (
    decompose,
    no_sampling,
    null_postprocessing,
    validate_device_wires,
    validate_measurements,
    validate_observables,
)
from .qutrit_mixed.simulate import simulate

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


observables = {
    "THermitian",
    "GellMann",
}


def observable_stopping_condition(obs: qml.operation.Operator) -> bool:
    """Specifies whether an observable is accepted by DefaultQutritMixed."""
    if obs.name in {"Prod", "Sum"}:
        return all(observable_stopping_condition(observable) for observable in obs.operands)
    if obs.name == "LinearCombination":
        return all(observable_stopping_condition(observable) for observable in obs.terms()[1])
    if obs.name == "SProd":
        return observable_stopping_condition(obs.base)

    return obs.name in observables


def stopping_condition(op: qml.operation.Operator) -> bool:
    """Specify whether an Operator object is supported by the device."""
    expected_set = DefaultQutrit.operations | {"Snapshot"} | channels
    return op.name in expected_set


def accepted_sample_measurement(m: qml.measurements.MeasurementProcess) -> bool:
    """Specifies whether a measurement is accepted when sampling."""
    return isinstance(m, qml.measurements.SampleMeasurement)


@qml.transform
def warn_readout_error_state(
    tape: qml.tape.QuantumTape,
) -> tuple[Sequence[qml.tape.QuantumTape], Callable]:
    """If a measurement in the QNode is an analytic state or density_matrix, and a readout error
    parameter is defined, warn that readout error will not be applied.

    Args:
        tape (QuantumTape, .QNode, Callable): a quantum circuit.

    Returns:
        qnode (pennylane.QNode) or quantum function (callable) or tuple[List[.QuantumTape], function]:
        The unaltered input circuit.
    """
    if not tape.shots:
        for m in tape.measurements:
            if isinstance(m, qml.measurements.StateMP):
                warnings.warn(f"Measurement {m} is not affected by readout error.")

    return (tape,), null_postprocessing


def get_readout_errors(readout_relaxation_probs, readout_misclassification_probs):
    r"""Get the list of readout errors that should be applied to each measured wire.

    Args:
        readout_relaxation_probs (List[float]): Inputs for :class:`~.QutritAmplitudeDamping` channel
            of the form :math:`[\gamma_{10}, \gamma_{20}, \gamma_{21}]`. This error models
            amplitude damping associated with longer readout and varying relaxation times of
            transmon-based qudits.
        readout_misclassification_probs (List[float]): Inputs for :class:`~.TritFlip` channel
            of the form :math:`[p_{01}, p_{02}, p_{12}]`. This error models misclassification events
            in readout.

    Returns:
        readout_errors (List[Callable]): List of readout error channels that should be
        applied to each measured wire.
    """
    measure_funcs = []
    if readout_relaxation_probs is not None:
        try:
            with qml.queuing.QueuingManager.stop_recording():
                qml.QutritAmplitudeDamping(*readout_relaxation_probs, wires=0)
        except Exception as e:
            raise DeviceError("Applying damping readout error results in error.") from e
        measure_funcs.append(partial(qml.QutritAmplitudeDamping, *readout_relaxation_probs))
    if readout_misclassification_probs is not None:
        try:
            with qml.queuing.QueuingManager.stop_recording():
                qml.TritFlip(*readout_misclassification_probs, wires=0)
        except Exception as e:
            raise DeviceError("Applying trit flip readout error results in error.") from e
        measure_funcs.append(partial(qml.TritFlip, *readout_misclassification_probs))

    return None if len(measure_funcs) == 0 else measure_funcs


@simulator_tracking
@single_tape_support
class DefaultQutritMixed(Device):
    r"""A PennyLane Python-based device for mixed-state qutrit simulation.

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
        readout_relaxation_probs (List[float]): Input probabilities for relaxation errors implemented
            with the :class:`~.QutritAmplitudeDamping` channel. The input defines the
            channel's parameters :math:`[\gamma_{10}, \gamma_{20}, \gamma_{21}]`.
        readout_misclassification_probs (List[float]):  Input probabilities for state readout
            misclassification events implemented with the :class:`~.TritFlip` channel. The input defines the
            channel's parameters :math:`[p_{01}, p_{02}, p_{12}]`.

    **Example:**

    .. code-block:: python

        n_wires = 5
        num_qscripts = 5
        qscripts = []
        for i in range(num_qscripts):
            unitary = scipy.stats.unitary_group(dim=3**n_wires, seed=(42 + i)).rvs()
            op = qml.QutritUnitary(unitary, wires=range(n_wires))
            qs = qml.tape.QuantumScript([op], [qml.expval(qml.GellMann(0, 3))])
            qscripts.append(qs)

    >>> dev = DefaultQutritMixed()
    >>> program, execution_config = dev.preprocess()
    >>> new_batch, post_processing_fn = program(qscripts)
    >>> results = dev.execute(new_batch, execution_config=execution_config)
    >>> post_processing_fn(results)
    [0.08015701503959313,
    0.04521414211599359,
    -0.0215232130089687,
    0.062120285032425865,
    -0.0635052317625]

    This device currently supports backpropagation derivatives:

    >>> from pennylane.devices import ExecutionConfig
    >>> dev.supports_derivatives(ExecutionConfig(gradient_method="backprop"))
    True

    For example, we can use jax to jit computing the derivative:

    .. code-block:: python

        import jax

        @jax.jit
        def f(x):
            qs = qml.tape.QuantumScript([qml.TRX(x, 0)], [qml.expval(qml.GellMann(0, 3))])
            program, execution_config = dev.preprocess()
            new_batch, post_processing_fn = program([qs])
            results = dev.execute(new_batch, execution_config=execution_config)
            return post_processing_fn(results)[0]

    >>> f(jax.numpy.array(1.2))
    DeviceArray(0.36235774, dtype=float32)
    >>> jax.grad(f)(jax.numpy.array(1.2))
    DeviceArray(-0.93203914, dtype=float32, weak_type=True)

    .. details::
        :title: Readout Error

        ``DefaultQutritMixed`` includes readout error support. Two input arguments control
        the parameters of error channels applied to each measured wire of the state after
        it has been diagonalized for measurement:

        * ``readout_relaxation_probs``:  Input parameters of a :class:`~.QutritAmplitudeDamping` channel.
          This error models state relaxation error that occurs during readout of transmon-based qutrits.
          The motivation for this readout error is described in [`1 <https://arxiv.org/abs/2003.03307>`_] (Sec II.A).
        * ``readout_misclassification_probs``: Input parameters of a :class:`~.TritFlip` channel.
          This error models misclassification events in readout. An example of this readout error
          can be seen in [`2 <https://arxiv.org/abs/2309.11303>`_] (Fig 1a).

        In the case that both parameters are defined, relaxation error is applied first then
        misclassification error is applied.

        .. note::
            The readout errors will be applied to the state after it has been diagonalized for each
            measurement. This may give different results depending on how the observable is defined.
            This is because diagonalizing gates for the same observable may return eigenvalues in
            different orders. For example, measuring :class:`~.THermitian` with a non-diagonal
            GellMann matrix will result in a different measurement result then measuring the
            equivalent :class:`~.GellMann` observable, as the THermitian eigenvalues are returned
            in increasing order when explicitly diagonalized (i.e., ``[-1, 0, 1]``), while non-diagonal GellManns provided
            in PennyLane have their eigenvalues hardcoded (i.e., ``[1, -1, 0]``).

    .. details::
        :title: Tracking

        ``DefaultQutritMixed`` tracks:

        * ``executions``: the number of unique circuits that would be required on quantum hardware
        * ``shots``: the number of shots
        * ``resources``: the :class:`~.resource.Resources` for the executed circuit.
        * ``simulations``: the number of simulations performed. One simulation can cover multiple QPU executions, such as for non-commuting measurements and batched parameters.
        * ``batches``: The number of times :meth:`~.execute` is called.
        * ``results``: The results of each call of :meth:`~.execute`

    """

    _device_options = ("rng", "prng_key")  # tuple of string names for all the device options.

    @property
    def name(self):
        """The name of the device."""
        return "default.qutrit.mixed"

    @debug_logger_init
    def __init__(  # pylint: disable=too-many-arguments
        self,
        wires=None,
        shots=None,
        seed="global",
        readout_relaxation_probs=None,
        readout_misclassification_probs=None,
    ) -> None:
        super().__init__(wires=wires, shots=shots)
        seed = np.random.randint(0, high=10000000) if seed == "global" else seed
        if qml.math.get_interface(seed) == "jax":
            self._prng_key = seed
            self._rng = np.random.default_rng(None)
        else:
            self._prng_key = None
            self._rng = np.random.default_rng(seed)
        self._debugger = None

        self.readout_errors = get_readout_errors(
            readout_relaxation_probs, readout_misclassification_probs
        )

    @debug_logger
    def supports_derivatives(
        self,
        execution_config: ExecutionConfig | None = None,
        circuit: QuantumScript | None = None,
    ) -> bool:
        """Check whether or not derivatives are available for a given configuration and circuit.

        ``DefaultQutritMixed`` supports backpropagation derivatives with analytic results.

        Args:
            execution_config (ExecutionConfig): The configuration of the desired derivative calculation.
            circuit (QuantumTape): An optional circuit to check derivatives support for.

        Returns:
            bool: Whether or not a derivative can be calculated provided the given information.

        """
        if execution_config is None or execution_config.gradient_method in {"backprop", "best"}:
            return circuit is None or not circuit.shots
        return False

    def _setup_execution_config(self, execution_config: ExecutionConfig) -> ExecutionConfig:
        """This is a private helper for ``preprocess`` that sets up the execution config.

        Args:
            execution_config (ExecutionConfig): an unprocessed execution config.

        Returns:
            ExecutionConfig: a preprocessed execution config.
        """
        updated_values = {}
        for option in execution_config.device_options:
            if option not in self._device_options:
                raise DeviceError(f"device option {option} not present on {self}")

        if execution_config.gradient_method == "best":
            updated_values["gradient_method"] = "backprop"
        updated_values["use_device_gradient"] = False
        updated_values["grad_on_execution"] = False
        updated_values["device_options"] = dict(execution_config.device_options)  # copy

        for option in self._device_options:
            if option not in updated_values["device_options"]:
                updated_values["device_options"][option] = getattr(self, f"_{option}")
        return replace(execution_config, **updated_values)

    @debug_logger
    def preprocess(
        self,
        execution_config: ExecutionConfig | None = None,
    ) -> tuple[TransformProgram, ExecutionConfig]:
        """This function defines the device transform program to be applied and an updated device
        configuration.

        Args:
            execution_config (Union[ExecutionConfig, Sequence[ExecutionConfig]]): A data structure
                describing the parameters needed to fully describe the execution.

        Returns:
            TransformProgram, ExecutionConfig: A transform program that when called returns
            ``QuantumTape`` objects that the device can natively execute, as well as a postprocessing
            function to be called after execution, and a configuration with unset
            specifications filled in.

        This device:

        * Supports any qutrit operations that provide a matrix
        * Supports any qutrit channel that provides Kraus matrices

        """
        if execution_config is None:
            execution_config = ExecutionConfig()
        config = self._setup_execution_config(execution_config)
        transform_program = TransformProgram()

        transform_program.add_transform(validate_device_wires, self.wires, name=self.name)
        transform_program.add_transform(
            decompose,
            stopping_condition=stopping_condition,
            name=self.name,
        )
        transform_program.add_transform(
            validate_measurements, sample_measurements=accepted_sample_measurement, name=self.name
        )
        transform_program.add_transform(
            validate_observables, stopping_condition=observable_stopping_condition, name=self.name
        )

        if config.gradient_method == "backprop":
            transform_program.add_transform(no_sampling, name="backprop + default.qutrit")

        if self.readout_errors is not None:
            transform_program.add_transform(warn_readout_error_state)

        return transform_program, config

    @debug_logger
    def execute(
        self,
        circuits: QuantumScriptOrBatch,
        execution_config: ExecutionConfig | None = None,
    ) -> Result | ResultBatch:
        if execution_config is None:
            execution_config = ExecutionConfig()
        interface = (
            execution_config.interface
            if execution_config.gradient_method in {"best", "backprop", None}
            else None
        )

        return tuple(
            simulate(
                c,
                rng=self._rng,
                prng_key=self._prng_key,
                debugger=self._debugger,
                interface=interface,
                readout_errors=self.readout_errors,
            )
            for c in circuits
        )
