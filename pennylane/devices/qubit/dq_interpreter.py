# Copyright 2024 Xanadu Quantum Technologies Inc.

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
This module contains a class for executing plxpr using default qubit tools.
"""

import jax
import numpy as np

from pennylane.capture import pause
from pennylane.capture.base_interpreter import FlattenedInterpreter
from pennylane.capture.primitives import adjoint_transform_prim, ctrl_transform_prim, measure_prim
from pennylane.devices import ExecutionConfig
from pennylane.measurements import MidMeasureMP, Shots
from pennylane.ops import adjoint, ctrl
from pennylane.ops.qubit import Projector
from pennylane.tape.plxpr_conversion import CollectOpsandMeas

from .apply_operation import apply_operation
from .initialize_state import create_initial_state
from .measure import measure
from .sampling import measure_with_samples
from .simulate import _postselection_postprocess


# pylint: disable=too-many-instance-attributes
class DefaultQubitInterpreter(FlattenedInterpreter):
    """Implements a class for interpreting plxpr using python simulation tools.

    Args:
        num_wires (int): the number of wires to initialize the state with
        shots (int | None): the number of shots to use for the execution. Shot vectors are not supported yet.
        key (None, jax.numpy.ndarray): the ``PRNGKey`` to use for random number generation.


    >>> from pennylane.devices.qubit.dq_interpreter import DefaultQubitInterpreter
    >>> qml.capture.enable()
    >>> import jax
    >>> key = jax.random.PRNGKey(1234)
    >>> dq = DefaultQubitInterpreter(num_wires=2, shots=None, key=key)
    >>> @qml.for_loop(2)
    ... def g(i,y):
    ...     qml.RX(y,0)
    ...     return y
    >>> def f(x):
    ...     g(x)
    ...     return qml.expval(qml.Z(0))
    >>> dq(f)(0.5)
    Array(0.54030231, dtype=float64)
    >>> jaxpr = jax.make_jaxpr(f)(0.5)
    >>> dq.eval(jaxpr.jaxpr, jaxpr.consts, 0.5)
    Array(0.54030231, dtype=float64)

    This execution can be differentiated via backprop and jitted as normal. Note that finite shot executions
    still cannot be differentiated with backprop.

    >>> jax.grad(dq(f))(jax.numpy.array(0.5))
    Array(-1.68294197, dtype=float64, weak_type=True)
    >>> jax.jit(dq(f))(jax.numpy.array(0.5))
    Array(0.54030231, dtype=float64)
    """

    def __copy__(self):
        inst = DefaultQubitInterpreter.__new__(DefaultQubitInterpreter)
        inst.stateref = self.stateref
        inst.shots = self.shots
        inst.execution_config = self.execution_config
        return inst

    def __init__(
        self,
        num_wires: int,
        shots: int | None = None,
        key: None | jax.numpy.ndarray = None,
        execution_config: None | ExecutionConfig = None,
    ):
        self.num_wires = num_wires
        self.original_shots = Shots(shots)
        if self.original_shots.has_partitioned_shots:
            raise NotImplementedError(
                "DefaultQubitInterpreter does not yet support partitioned shots."
            )
        if key is None:
            key = jax.random.PRNGKey(np.random.randint(100000))

        self.initial_key = key
        self.stateref = None
        self.execution_config = execution_config or ExecutionConfig()

        super().__init__()

    @property
    def state(self):
        """The statevector"""
        try:
            return self.stateref["state"]
        except TypeError as e:
            raise AttributeError("execution not yet initialized.") from e

    @state.setter
    def state(self, new_val):
        try:
            self.stateref["state"] = new_val
        except TypeError as e:
            raise AttributeError("execution not yet initialized.") from e

    @property
    def shots(self):
        """The shots"""
        try:
            return self.stateref["shots"]
        except TypeError as e:
            raise AttributeError("execution not yet initialized.") from e

    @shots.setter
    def shots(self, new_val):
        try:
            self.stateref["shots"] = new_val
        except TypeError as e:
            raise AttributeError("execution not yet initialized.") from e

    @property
    def key(self):
        """A jax PRNGKey for random number generation."""
        try:
            return self.stateref["key"]
        except TypeError as e:
            raise AttributeError("execution not yet initialized.") from e

    @key.setter
    def key(self, new_val):
        try:
            self.stateref["key"] = new_val
        except TypeError as e:
            raise AttributeError("execution not yet initialized.") from e

    @property
    def is_state_batched(self) -> bool:
        """Whether or not the state vector is batched."""
        try:
            return self.stateref["is_state_batched"]
        except TypeError as e:
            raise AttributeError("execution not yet initialized.") from e

    @is_state_batched.setter
    def is_state_batched(self, new_val):
        try:
            self.stateref["is_state_batched"] = new_val
        except TypeError as e:
            raise AttributeError("execution not yet initialized.") from e

    def setup(self) -> None:
        if self.stateref is None:
            self.stateref = {
                "state": create_initial_state(range(self.num_wires), like="jax"),
                "shots": self.original_shots,
                "key": self.initial_key,
                "is_state_batched": False,
            }
        # else set by copying a parent interpreter and we need to modify same stateref

    def cleanup(self) -> None:
        self.initial_key = self.key  # be cautious of leaked tracers, but we should be fine.
        self.stateref = None

    def interpret_operation(self, op):
        self.state = apply_operation(op, self.state, is_state_batched=self.is_state_batched)
        if op.batch_size:
            self.is_state_batched = True

        if isinstance(op, Projector):
            self.key, new_key = jax.random.split(self.key, 2)
            self.state, self.shots = _postselection_postprocess(
                self.state,
                self.is_state_batched,
                self.shots,
                prng_key=new_key,
                postselect_mode=self.execution_config.mcm_config.postselect_mode,
            )

        return op

    def interpret_measurement_eqn(self, eqn: "jax.extend.core.JaxprEqn"):
        if "mcm" in eqn.primitive.name:
            raise NotImplementedError(
                "DefaultQubitInterpreter does not yet support postprocessing mcms"
            )
        return super().interpret_measurement_eqn(eqn)

    def interpret_measurement(self, measurement):
        # measurements can sometimes create intermediary mps,
        # but those intermediaries will not work with capture enabled when jitting
        with pause():
            if self.shots:
                self.key, new_key = jax.random.split(self.key, 2)
                # note that this does *not* group commuting measurements
                # further work could figure out how to perform multiple measurements at the same time
                output = measure_with_samples(
                    [measurement],
                    self.state,
                    shots=self.shots,
                    prng_key=new_key,
                    is_state_batched=self.is_state_batched,
                )[0]
            else:
                output = measure(measurement, self.state, is_state_batched=self.is_state_batched)
        return output


@DefaultQubitInterpreter.register_primitive(measure_prim)
def _(self, *invals, reset, postselect):
    mp = MidMeasureMP(invals, reset=reset, postselect=postselect)
    self.key, new_key = jax.random.split(self.key, 2)
    mcms = {}
    self.state = apply_operation(mp, self.state, mid_measurements=mcms, prng_key=new_key)
    if mp.postselect is not None:
        # Divide by zero to create NaNs for MCM values that differ from the postselection value
        self.state = self.state / (1 - jax.numpy.abs(mp.postselect - mcms[mp]))
    return mcms[mp]


@DefaultQubitInterpreter.register_primitive(adjoint_transform_prim)
def _(self, *invals, jaxpr, n_consts, lazy=True):
    consts = invals[:n_consts]
    args = invals[n_consts:]
    recorder = CollectOpsandMeas()
    recorder.eval(jaxpr, consts, *args)

    ops = recorder.state["ops"]
    with pause():
        for op in ops[::-1]:
            self.state = apply_operation(
                adjoint(op, lazy=lazy), self.state, is_state_batched=self.is_state_batched
            )
            if op.batch_size:
                self.is_state_batched = True

    return []


# pylint: disable=too-many-arguments
@DefaultQubitInterpreter.register_primitive(ctrl_transform_prim)
def _(self, *invals, n_control, jaxpr, control_values, work_wires, n_consts):
    consts = invals[:n_consts]
    control_wires = invals[-n_control:]
    args = invals[n_consts:-n_control]
    recorder = CollectOpsandMeas()
    recorder.eval(jaxpr, consts, *args)

    ops = recorder.state["ops"]
    with pause():
        for op in ops:
            self.state = apply_operation(
                ctrl(op, control_wires, control_values=control_values, work_wires=work_wires),
                self.state,
                is_state_batched=self.is_state_batched,
            )
            if op.batch_size:
                self.is_state_batched = True

    return []
