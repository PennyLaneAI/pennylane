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

from dataclasses import replace
from typing import Union, Callable, Tuple, Sequence, Optional
import inspect
import logging
import numpy as np

import pennylane as qml
from pennylane.transforms.core import TransformProgram
from pennylane.tape import QuantumTape
from pennylane.typing import Result, ResultBatch

from . import Device
from .modifiers import single_tape_support, simulator_tracking

from .preprocess import (
    decompose,
    validate_observables,
    validate_measurements,
    validate_device_wires,
    no_sampling,
)
from .execution_config import ExecutionConfig, DefaultExecutionConfig
from .qutrit_mixed.simulate import simulate
from .default_qutrit import DefaultQutrit

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

Result_or_ResultBatch = Union[Result, ResultBatch]
QuantumTapeBatch = Sequence[QuantumTape]
QuantumTape_or_Batch = Union[QuantumTape, QuantumTapeBatch]

# always a function from a resultbatch to either a result or a result batch
PostprocessingFn = Callable[[ResultBatch], Result_or_ResultBatch]

channels = set()
observables = {
    "THermitian",
    "GellMann",
}


def observable_stopping_condition(obs: qml.operation.Operator) -> bool:
    """Specifies whether an observable is accepted by DefaultQutritMixed."""
    if isinstance(obs, qml.operation.Tensor):
        return all(observable_stopping_condition(observable) for observable in obs.obs)
    if obs.name in {"Prod", "Sum"}:
        return all(observable_stopping_condition(observable) for observable in obs.operands)
    if obs.name in {"LinearCombination", "Hamiltonian"}:
        return all(observable_stopping_condition(observable) for observable in obs.terms()[1])
    if obs.name == "SProd":
        return observable_stopping_condition(obs.base)

    return obs.name in observables


def stopping_condition(op: qml.operation.Operator) -> bool:
    """Specify whether an Operator object is supported by the device."""
    expected_set = DefaultQutrit.operations | {"Snapshot"} | channels
    return op.name in expected_set


def stopping_condition_shots(op: qml.operation.Operator) -> bool:
    """Specify whether an Operator object is supported by the device with shots."""
    return stopping_condition(op)


def accepted_sample_measurement(m: qml.measurements.MeasurementProcess) -> bool:
    """Specifies whether a measurement is accepted when sampling."""
    return isinstance(m, qml.measurements.SampleMeasurement)


@simulator_tracking
@single_tape_support
class DefaultQutritMixed(Device):
    """A PennyLane Python-based device for mixed-state qutrit simulation.

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

    def __init__(
        self,
        wires=None,
        shots=None,
        seed="global",
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

    def supports_derivatives(
        self,
        execution_config: Optional[ExecutionConfig] = None,
        circuit: Optional[QuantumTape] = None,
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
                raise qml.DeviceError(f"device option {option} not present on {self}")

        if execution_config.gradient_method == "best":
            updated_values["gradient_method"] = "backprop"
        updated_values["use_device_gradient"] = False
        updated_values["grad_on_execution"] = False
        updated_values["device_options"] = dict(execution_config.device_options)  # copy

        for option in self._device_options:
            if option not in updated_values["device_options"]:
                updated_values["device_options"][option] = getattr(self, f"_{option}")
        return replace(execution_config, **updated_values)

    def preprocess(
        self,
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ) -> Tuple[TransformProgram, ExecutionConfig]:
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
        config = self._setup_execution_config(execution_config)
        transform_program = TransformProgram()

        transform_program.add_transform(validate_device_wires, self.wires, name=self.name)
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

        if config.gradient_method == "backprop":
            transform_program.add_transform(no_sampling, name="backprop + default.qutrit")

        return transform_program, config

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
            )
            for c in circuits
        )
