# Copyright 2022 Xanadu Quantum Technologies Inc.

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
The null.qubit device is a no-op device, useful for resource estimation, and for
benchmarking PennyLane's auxiliary functionality outside direct circuit evaluations.
"""
# pylint:disable=unused-argument

from dataclasses import replace
from functools import singledispatch
from numbers import Number
from typing import Union, Callable, Tuple, Sequence
import inspect
import logging
import numpy as np
from pennylane import math
from pennylane.devices.execution_config import ExecutionConfig
from pennylane.devices.modifiers import simulator_tracking, single_tape_support
from pennylane.devices.qubit.simulate import INTERFACE_TO_LIKE

from pennylane.tape import QuantumTape
from pennylane.transforms.core import TransformProgram
from pennylane.typing import Result, ResultBatch
from pennylane.measurements import (
    MeasurementProcess,
    CountsMP,
    StateMP,
    ProbabilityMP,
    Shots,
    MeasurementValue,
    ClassicalShadowMP,
    DensityMatrixMP,
)

from . import Device, DefaultQubit
from .execution_config import ExecutionConfig, DefaultExecutionConfig
from .preprocess import decompose

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

Result_or_ResultBatch = Union[Result, ResultBatch]
QuantumTapeBatch = Sequence[QuantumTape]
QuantumTape_or_Batch = Union[QuantumTape, QuantumTapeBatch]
# always a function from a resultbatch to either a result or a result batch
PostprocessingFn = Callable[[ResultBatch], Result_or_ResultBatch]


@singledispatch
def zero_measurement(
    mp: MeasurementProcess, obj_with_wires, shots: Shots, batch_size: int, interface: str
):
    """Create all-zero results for various measurement processes."""
    return _zero_measurement(mp, obj_with_wires, shots, batch_size, interface)


def _zero_measurement(mp, obj_with_wires, shots, batch_size, interface):
    shape = mp.shape(obj_with_wires, shots)
    if all(isinstance(s, int) for s in shape):
        if batch_size is not None:
            shape = (batch_size,) + shape
        return math.zeros(shape, like=interface, dtype=mp.numeric_type)
    if batch_size is not None:
        shape = ((batch_size,) + s for s in shape)
    return tuple(math.zeros(s, like=interface, dtype=mp.numeric_type) for s in shape)


@zero_measurement.register
def _(mp: ClassicalShadowMP, obj_with_wires, shots, batch_size, interface):
    shapes = [mp.shape(obj_with_wires, Shots(s)) for s in shots]
    if batch_size is not None:
        # shapes = [(batch_size,) + shape for shape in shapes]
        raise ValueError(
            "Parameter broadcasting is not supported with null.qubit and qml.classical_shadow"
        )
    results = tuple(math.zeros(shape, like=interface, dtype=np.int8) for shape in shapes)
    return results if shots.has_partitioned_shots else results[0]


@zero_measurement.register
def _(mp: CountsMP, obj_with_wires, shots, batch_size, interface):
    outcomes = []
    if mp.obs is None and not isinstance(mp.mv, MeasurementValue):
        num_wires = len(obj_with_wires.wires)
        state = "0" * num_wires
        results = tuple({state: math.asarray(s, like=interface)} for s in shots)
        if mp.all_outcomes:
            outcomes = [f"{x:0{num_wires}b}" for x in range(1, 2**num_wires)]
    else:
        outcomes = sorted(mp.eigvals())  # always assign shots to the smallest
        results = tuple({outcomes[0]: math.asarray(s, like=interface)} for s in shots)
        outcomes = outcomes[1:] if mp.all_outcomes else []

    if outcomes:
        zero = math.asarray(0, like=interface)
        for res in results:
            for val in outcomes:
                res[val] = zero
    if batch_size is not None:
        results = tuple([r] * batch_size for r in results)
    return results[0] if len(results) == 1 else results


zero_measurement.register(DensityMatrixMP)(_zero_measurement)


@zero_measurement.register(StateMP)
@zero_measurement.register(ProbabilityMP)
def _(mp: Union[StateMP, ProbabilityMP], obj_with_wires, shots, batch_size, interface):
    wires = mp.wires or obj_with_wires.wires
    state = [1.0] + [0.0] * (2 ** len(wires) - 1)
    if batch_size is not None:
        state = [state] * batch_size
    result = math.asarray(state, like=interface)
    return (result,) * shots.num_copies if shots.has_partitioned_shots else result


@simulator_tracking
@single_tape_support
class NullQubit(Device):
    """Null qubit device for PennyLane. This device performs no operations involved in numerical calculations.
    Instead the time spent in execution is dominated by support (or setting up) operations, like tape creation etc.

    Args:
        wires (int, Iterable[Number, str]): Number of wires present on the device, or iterable that
            contains unique labels for the wires as numbers (i.e., ``[-1, 0, 2]``) or strings
            (``['aux_wire', 'q1', 'q2']``). Default ``None`` if not specified.
        shots (int, Sequence[int], Sequence[Union[int, Sequence[int]]]): The default number of shots
            to use in executions involving this device.

    **Example:**

    .. code-block:: python

        qs = qml.tape.QuantumScript(
            [qml.Hadamard(0), qml.CNOT([0, 1])],
            [qml.expval(qml.PauliZ(0)), qml.probs()],
        )
        qscripts = [qs, qs, qs]

    >>> dev = NullQubit()
    >>> program, execution_config = dev.preprocess()
    >>> new_batch, post_processing_fn = program(qscripts)
    >>> results = dev.execute(new_batch, execution_config=execution_config)
    >>> post_processing_fn(results)
    ((array(0.), array([1., 0., 0., 0.])),
     (array(0.), array([1., 0., 0., 0.])),
     (array(0.), array([1., 0., 0., 0.])))


    This device currently supports (trivial) derivatives:

    >>> from pennylane.devices import ExecutionConfig
    >>> dev.supports_derivatives(ExecutionConfig(gradient_method="device"))
    True

    This device can be used to track resource usage:

    .. code-block:: python

        n_layers = 50
        n_wires = 100
        shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_wires)

        @qml.qnode(dev)
        def circuit(params):
            qml.StronglyEntanglingLayers(params, wires=range(n_wires))
            return [qml.expval(qml.Z(i)) for i in range(n_wires)]

        params = np.random.random(shape)

        with qml.Tracker(dev) as tracker:
            circuit(params)

    >>> tracker.history["resources"][0]
    wires: 100
    gates: 10000
    depth: 502
    shots: Shots(total=None)
    gate_types:
    {'Rot': 5000, 'CNOT': 5000}
    gate_sizes:
    {1: 5000, 2: 5000}


    .. details::
        :title: Tracking

        ``NullQubit`` tracks:

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

    """

    @property
    def name(self):
        """The name of the device."""
        return "null.qubit"

    def __init__(self, wires=None, shots=None) -> None:
        super().__init__(wires=wires, shots=shots)
        self._debugger = None

    def _simulate(self, circuit, interface):
        shots = circuit.shots
        obj_with_wires = self if self.wires else circuit
        results = tuple(
            zero_measurement(mp, obj_with_wires, shots, circuit.batch_size, interface)
            for mp in circuit.measurements
        )
        if len(results) == 1:
            return results[0]
        if shots.has_partitioned_shots:
            return tuple(zip(*results))
        return results

    def _derivatives(self, circuit, interface):
        shots = circuit.shots
        obj_with_wires = self if self.wires else circuit
        n = len(circuit.trainable_params)
        derivatives = tuple(
            (
                math.zeros_like(
                    zero_measurement(mp, obj_with_wires, shots, circuit.batch_size, interface)
                ),
            )
            * n
            for mp in circuit.measurements
        )
        if n == 1:
            derivatives = tuple(d[0] for d in derivatives)
        return derivatives[0] if len(derivatives) == 1 else derivatives

    @staticmethod
    def _vjp(circuit, interface):
        batch_size = circuit.batch_size
        n = len(circuit.trainable_params)
        res_shape = (n,) if batch_size is None else (n, batch_size)
        return math.zeros(res_shape, like=interface)

    @staticmethod
    def _jvp(circuit, interface):
        jvps = (math.asarray(0.0, like=interface),) * len(circuit.measurements)
        return jvps[0] if len(jvps) == 1 else jvps

    @staticmethod
    def _setup_execution_config(execution_config: ExecutionConfig) -> ExecutionConfig:
        """No-op function to allow for borrowing DefaultQubit.preprocess without AttributeErrors"""
        return execution_config

    @property
    def _max_workers(self):
        """No-op property to allow for borrowing DefaultQubit.preprocess without AttributeErrors"""
        return None

    # pylint: disable=cell-var-from-loop
    def preprocess(
        self, execution_config=DefaultExecutionConfig
    ) -> Tuple[TransformProgram, ExecutionConfig]:
        program, _ = DefaultQubit.preprocess(self, execution_config)
        for t in program:
            if t.transform == decompose.transform:
                original_stopping_condition = t.kwargs["stopping_condition"]

                def new_stopping_condition(op):
                    return (not op.has_decomposition) or original_stopping_condition(op)

                t.kwargs["stopping_condition"] = new_stopping_condition

                original_shots_stopping_condition = t.kwargs.get("stopping_condition_shots", None)
                if original_shots_stopping_condition:

                    def new_shots_stopping_condition(op):
                        return (not op.has_decomposition) or original_shots_stopping_condition(op)

                    t.kwargs["stopping_condition_shots"] = new_shots_stopping_condition

        updated_values = {}
        if execution_config.gradient_method in ["best", "adjoint"]:
            updated_values["gradient_method"] = "device"
        if execution_config.use_device_gradient is None:
            updated_values["use_device_gradient"] = execution_config.gradient_method in {
                "best",
                "device",
                "adjoint",
                "backprop",
            }
        if execution_config.use_device_jacobian_product is None:
            updated_values["use_device_jacobian_product"] = (
                execution_config.gradient_method == "device"
            )
        if execution_config.grad_on_execution is None:
            updated_values["grad_on_execution"] = execution_config.gradient_method == "device"
        return program, replace(execution_config, **updated_values)

    def execute(
        self,
        circuits: QuantumTape_or_Batch,
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ) -> Result_or_ResultBatch:
        if logger.isEnabledFor(logging.DEBUG):  # pragma: no cover
            logger.debug(
                """Entry with args=(circuits=%s) called by=%s""",
                circuits,
                "::L".join(
                    str(i) for i in inspect.getouterframes(inspect.currentframe(), 2)[1][1:3]
                ),
            )

        return tuple(
            self._simulate(c, INTERFACE_TO_LIKE[execution_config.interface]) for c in circuits
        )

    def supports_derivatives(self, execution_config=None, circuit=None):
        return execution_config is None or execution_config.gradient_method in (
            "device",
            "backprop",
            "adjoint",
        )

    def supports_vjp(self, execution_config=None, circuit=None):
        return execution_config is None or execution_config.gradient_method in (
            "device",
            "backprop",
            "adjoint",
        )

    def supports_jvp(self, execution_config=None, circuit=None):
        return execution_config is None or execution_config.gradient_method in (
            "device",
            "backprop",
            "adjoint",
        )

    def compute_derivatives(
        self,
        circuits: QuantumTape_or_Batch,
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ):
        return tuple(
            self._derivatives(c, INTERFACE_TO_LIKE[execution_config.interface]) for c in circuits
        )

    def execute_and_compute_derivatives(
        self,
        circuits: QuantumTape_or_Batch,
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ):
        results = tuple(
            self._simulate(c, INTERFACE_TO_LIKE[execution_config.interface]) for c in circuits
        )
        jacs = tuple(
            self._derivatives(c, INTERFACE_TO_LIKE[execution_config.interface]) for c in circuits
        )

        return results, jacs

    def compute_jvp(
        self,
        circuits: QuantumTape_or_Batch,
        tangents: Tuple[Number],
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ):
        return tuple(self._jvp(c, INTERFACE_TO_LIKE[execution_config.interface]) for c in circuits)

    def execute_and_compute_jvp(
        self,
        circuits: QuantumTape_or_Batch,
        tangents: Tuple[Number],
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ):
        results = tuple(
            self._simulate(c, INTERFACE_TO_LIKE[execution_config.interface]) for c in circuits
        )
        jvps = tuple(self._jvp(c, INTERFACE_TO_LIKE[execution_config.interface]) for c in circuits)

        return results, jvps

    def compute_vjp(
        self,
        circuits: QuantumTape_or_Batch,
        cotangents: Tuple[Number],
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ):
        return tuple(self._vjp(c, INTERFACE_TO_LIKE[execution_config.interface]) for c in circuits)

    def execute_and_compute_vjp(
        self,
        circuits: QuantumTape_or_Batch,
        cotangents: Tuple[Number],
        execution_config: ExecutionConfig = DefaultExecutionConfig,
    ):
        results = tuple(
            self._simulate(c, INTERFACE_TO_LIKE[execution_config.interface]) for c in circuits
        )
        vjps = tuple(self._vjp(c, INTERFACE_TO_LIKE[execution_config.interface]) for c in circuits)
        return results, vjps
