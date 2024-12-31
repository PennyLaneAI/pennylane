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

from pennylane.capture import disable, enable
from pennylane.capture.base_interpreter import FlattenedHigherOrderPrimitives, PlxprInterpreter
from pennylane.capture.primitives import adjoint_transform_prim, ctrl_transform_prim, measure_prim
from pennylane.measurements import MidMeasureMP, Shots

from .apply_operation import apply_operation
from .initialize_state import create_initial_state
from .measure import measure
from .sampling import measure_with_samples


# pylint: disable=attribute-defined-outside-init, access-member-before-definition
class DefaultQubitInterpreter(PlxprInterpreter):
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

    def __init__(
        self, num_wires: int, shots: int | None = None, key: None | jax.numpy.ndarray = None
    ):
        self.num_wires = num_wires
        self.shots = Shots(shots)
        if self.shots.has_partitioned_shots:
            raise NotImplementedError(
                "DefaultQubitInterpreter does not yet support partitioned shots."
            )
        if key is None:
            key = jax.random.PRNGKey(np.random.randint(100000))

        self.initial_key = key
        self.stateref = None
        super().__init__()

    def __getattr__(self, key):
        if key in {"state", "key", "is_state_batched"}:
            if self.stateref is None:
                raise AttributeError("execution not yet initialized.")
            return self.stateref[key]
        raise AttributeError(f"No attribute {key}")

    def __setattr__(self, __name: str, __value) -> None:
        if __name in {"state", "key", "is_state_batched"}:
            if self.stateref is None:
                raise AttributeError("execution not yet initialized")
            self.stateref[__name] = __value
        else:
            super().__setattr__(__name, __value)

    def setup(self) -> None:
        if self.stateref is None:
            self.stateref = {
                "state": create_initial_state(range(self.num_wires), like="jax"),
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
        return op

    def interpret_measurement_eqn(self, eqn: "jax.core.JaxprEqn"):
        if "mcm" in eqn.primitive.name:
            raise NotImplementedError(
                "DefaultQubitInterpreter does not yet support postprocessing mcms"
            )
        return super().interpret_measurement_eqn(eqn)

    def interpret_measurement(self, measurement):
        # measurements can sometimes create intermediary mps, but those intermediaries will not work with capture enabled
        disable()
        try:
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
        finally:
            enable()
        return output


# pylint: disable=protected-access
DefaultQubitInterpreter._primitive_registrations.update(FlattenedHigherOrderPrimitives)


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


# pylint: disable=unused-argument
@DefaultQubitInterpreter.register_primitive(adjoint_transform_prim)
def _(self, *invals, jaxpr, n_consts, lazy=True):
    # TODO: requires jaxpr -> list of ops first
    raise NotImplementedError


# pylint: disable=too-many-arguments
@DefaultQubitInterpreter.register_primitive(ctrl_transform_prim)
def _(self, *invals, n_control, jaxpr, control_values, work_wires, n_consts):
    # TODO: requires jaxpr -> list of ops first
    raise NotImplementedError
