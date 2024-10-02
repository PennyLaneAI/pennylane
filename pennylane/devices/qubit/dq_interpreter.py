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

from pennylane.capture import PlxprInterpreter
from pennylane.capture.primitives import (
    adjoint_transform_prim,
    cond_prim,
    ctrl_transform_prim,
    for_loop_prim,
    measure_prim,
    while_loop_prim,
)
from pennylane.measurements import MidMeasureMP

from .apply_operation import apply_operation
from .initialize_state import create_initial_state
from .measure import measure
from .sampling import measure_with_samples


class DefaultQubitInterpreter(PlxprInterpreter):
    """Implements a class for interpreting plxpr using default qubit.

    >>> key = jax.random.PRNGKey(1234)
    >>> dq = DefaultQubitInterpreter(num_wires=2, shots=qml.measurements.Shots(50), key=key)
    >>> @qml.for_loop(2)
    ... def g(i,y):
    ...     qml.RX(y,0)
    ...     return y
    >>> def f(x):
    ...     g(x)
    ...     return qml.expval(qml.Z(0))
    >>> dq(f)(0.5)
    Array(-0.79999995, dtype=float32)


    """

    def __init__(self, num_wires, shots, key=None, stateref=None):
        self.num_wires = num_wires
        self.shots = shots
        self.stateref = stateref or {"state": None}
        self.key = key

    @property
    def state(self):
        return self.stateref["state"]

    @state.setter
    def state(self, value):
        self.stateref["state"] = value

    def child(self) -> "DefaultQubitInterpreter":
        return type(self)(
            num_wires=self.num_wires, shots=self.shots, key=self.key, stateref=self.stateref
        )

    def setup(self):
        if self.state is None:
            self.state = create_initial_state(range(self.num_wires))

    def interpret_operation(self, op):
        self.state = apply_operation(op, self.state)

    def interpret_measurement_eqn(self, primitive, *invals, **params):
        mp = primitive.impl(*invals, **params)
        if self.shots:
            self.key, new_key = jax.random.split(self.key, 2)
            # note that this does *not* group commuting measurements
            # further work could figure out how to perform multiple measurements at the same time
            return measure_with_samples([mp], self.state, shots=self.shots, prng_key=new_key)[0]
        return measure(mp, self.state)


# pylint: disable=unused-argument
@DefaultQubitInterpreter.register_primitive(adjoint_transform_prim)
def _(self, *invals, jaxpr, n_consts, lazy=True):
    raise NotImplementedError("TODO?")


@DefaultQubitInterpreter.register_primitive(ctrl_transform_prim)
def _(self, *invals, n_control, jaxpr, control_values, work_wires, n_consts):
    raise NotImplementedError("TODO?")


@DefaultQubitInterpreter.register_primitive(for_loop_prim)
def _(self, *invals, jaxpr_body_fn, n_consts):
    start, stop, step = invals[0], invals[1], invals[2]
    consts = invals[3 : 3 + n_consts]
    init_state = invals[3 + n_consts :]

    res = None
    for i in range(start, stop, step):
        res = self.child().eval(jaxpr_body_fn, consts, i, *init_state)

    return res


@DefaultQubitInterpreter.register_primitive(while_loop_prim)
def _(self, *invals, jaxpr_body_fn, jaxpr_cond_fn, n_consts_body, n_consts_cond):
    consts_body = invals[:n_consts_body]
    consts_cond = invals[n_consts_body : n_consts_body + n_consts_cond]
    init_state = invals[n_consts_body + n_consts_cond :]

    fn_res = init_state
    while self.child().eval(jaxpr_cond_fn, consts_cond, *fn_res)[0]:
        fn_res = self.child().eval(jaxpr_body_fn, consts_body, *fn_res)

    return fn_res


@DefaultQubitInterpreter.register_primitive(measure_prim)
def _(self, *invals, reset, postselect):
    mp = MidMeasureMP(invals, reset=reset, postselect=postselect)
    mid_measurements = {}
    self.key, new_key = jax.random.split(self.key, 2)
    self.state = apply_operation(
        mp, self.state, mid_measurements=mid_measurements, prng_key=new_key
    )
    return mid_measurements[mp]


@DefaultQubitInterpreter.register_primitive(cond_prim)
def _(self, *invals, jaxpr_branches, n_consts_per_branch, n_args):
    n_branches = len(jaxpr_branches)
    conditions = invals[:n_branches]
    consts_flat = invals[n_branches + n_args :]
    args = invals[n_branches : n_branches + n_args]

    start = 0
    for pred, jaxpr, n_consts in zip(conditions, jaxpr_branches, n_consts_per_branch):
        consts = consts_flat[start : start + n_consts]
        start += n_consts
        if pred and jaxpr is not None:
            return self.child().eval_jaxpr(jaxpr, consts, *args)
    return ()
