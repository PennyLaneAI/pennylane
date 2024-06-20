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
This submodule defines a utility for converting plxpr into catalyst jaxpr.
"""
from dataclasses import dataclass, field
from functools import partial
from typing import Callable

import catalyst
import jax
from catalyst import jax_primitives as c_prims
from jax.extend.linear_util import wrap_init

from .capture_qnode import _get_qnode_prim, _get_shapes_for
from .primitives import _get_abstract_measurement, _get_abstract_operator

qnode_prim = _get_qnode_prim()
AbstractOperator = _get_abstract_operator()
AbstractMeasurement = _get_abstract_measurement()

measurement_map = {
    "sample_wires": c_prims.sample_p,
    "expval_obs": c_prims.expval_p,
    "var_obs": c_prims.var_p,
    "probs_wires": c_prims.probs_p,
    "state_wires": c_prims.state_p,
}


# pylint: disable=unidiomatic-typecheck
def _read(var, env: dict):
    return var.val if type(var) is jax.core.Literal else env[var]


def _get_device_kwargs(device: "pennylane.devices.Device") -> dict:
    """Calulcate the params for a device equation."""
    features = catalyst.utils.toml.ProgramFeatures(device.shots is not None)
    capabilities = catalyst.device.get_device_capabilities(device, features)
    info = catalyst.device.extract_backend_info(device, capabilities)
    # Note that the value of rtd_kwargs is a string version of the info kwargs, not the info kwargs itself!
    return {
        "rtd_kwargs": str(info.kwargs),
        "rtd_lib": info.lpath,
        "rtd_name": info.c_interface_name,
    }


def to_catalyst(plxpr: jax.core.Jaxpr) -> Callable[..., jax.core.Jaxpr]:
    """Convert pennylane variant jaxpr to catalyst variant jaxpr.

    Args:
        jaxpr (jax.core.Jaxpr): pennylane variant jaxpr

    Returns:
        Callable: a function that accepts the same arguments as the plxpr and returns catalyst
        variant jaxpr.

    Note that the input jaxpr should be workflow level and contain qnode primitives, rather than
    qfunc level with individual operators.

    .. code-block:: python

        from pennylane.capture.to_catalyst import  to_catalyst

        qml.capture.enable()

        @qml.qnode(qml.device('lightning.qubit', wires=2))
        def circuit(x):
            qml.RX(x, 0)
            return qml.probs(wires=(0, 1))

        def f(x):
            return circuit(2 * x) ** 2

        jaxpr = jax.make_jaxpr(circuit)(0.5)

        print(qml.capture.to_catalyst(jaxpr)(0.5))

    .. code-block:: none

        { lambda ; a:f64[]. let
            b:f64[4] = func[
            call_jaxpr={ lambda ; c:f64[]. let
                qdevice[
                    rtd_kwargs={'shots': 0, 'mcmc': False, 'num_burnin': 0, 'kernel_name': None}
                    rtd_lib=/Users/christina/Prog/catalyst/frontend/catalyst/utils/../../../runtime/build/lib/librtd_lightning.dylib
                    rtd_name=LightningSimulator
                ]
                d:AbstractQreg() = qalloc 2
                e:AbstractQbit() = qextract d 0
                f:AbstractQbit() = qinst[
                    adjoint=False
                    ctrl_len=0
                    op=RX
                    params_len=1
                    qubits_len=1
                ] e c
                g:AbstractQbit() = qextract d 1
                h:AbstractObs(num_qubits=2,primitive=compbasis) = compbasis f g
                i:f64[4] = probs[shape=(4,) shots=None] h
                j:AbstractQreg() = qinsert d 0 f
                qdealloc j
                in (i,) }
            fn=<QNode: device='<lightning.qubit device (wires=2) at 0x302761c90>', interface='auto', diff_method='best'>
            ] a
        in (b,) }

    """
    return jax.make_jaxpr(partial(to_catalyst_interpreter, plxpr.jaxpr, plxpr.consts))


def to_catalyst_interpreter(jaxpr: jax.core.Jaxpr, consts, *args) -> list:
    """Convert pennylane variant jaxpr to catalyst variant jaxpr.

    See the documentation on
    `Writing custom interpreters in JAX <https://jax.readthedocs.io/en/latest/notebooks/Writing_custom_interpreters_in_Jax.html>`_
    for a walkthrough on the general architecture and behavior of this function.

    Given that ``catalyst.jax_primitives.func_p`` does not define a concrete implementation, this
    function will fail outside of a abstract evaluation call.

    """
    env = {}  # dict mapping var "variables" to val "values"

    # Bind args and consts to environment
    for arg, invar in zip(args, jaxpr.invars):
        env[invar] = arg
    for const, constvar in zip(consts, jaxpr.constvars):
        env[constvar] = const

    # Loop through equations and evaluate primitives using `bind`
    for eqn in jaxpr.eqns:
        # Read inputs to equation from environment
        invals = [_read(invar, env) for invar in eqn.invars]
        if eqn.primitive == qnode_prim:
            if eqn.params["device"].shots != eqn.params["shots"]:
                raise NotImplementedError("catalyst does not yet support dynamic shots")

            f = partial(
                _bind_catalxpr,
                eqn.params["qfunc_jaxpr"].jaxpr,
                eqn.params["qfunc_jaxpr"].consts,
                eqn.params["device"],
            )
            # func_p is a CallPrimitive, so interpreter passed as first arg
            # wrap_init turns the function into a WrappedFun, which can store
            # transformations
            outvals = c_prims.func_p.bind(wrap_init(f), *invals, fn=eqn.params["qnode"])
        else:
            outvals = eqn.primitive.bind(*invals, **eqn.params)
            # Primitives may return multiple outputs or not
        if not eqn.primitive.multiple_results:
            outvals = [outvals]
        # Write the results of the primitive into the environment
        for outvar, outval in zip(eqn.outvars, outvals):
            env[outvar] = outval
    return [env[outvar] for outvar in jaxpr.outvars]


@dataclass
class _InterpreterState:
    """This dataclass stores the mutable variables modified
    over the course of the interpretting plxpr as catalxpr."""

    qreg: c_prims.AbstractQreg
    """The current quantum register.
    """

    env: dict = field(default_factory=dict)
    """A dictionary mapping variable to values."""

    wire_map: dict = field(default_factory=dict)
    """A map from wire values to ``AbstractQbit`` instances.
    
    If a value is not present in this dictionary, it needs to be extracted
    from the ``qreg`` property.
    """

    op_math_cache: dict = field(default_factory=dict)
    """A cache of operations that will be consumed by later operations.
    This is a map from the ``AbstractOperator`` variables to the corresponding
    equation. The equation will be need to be intrepretted when the abstract
    operator is consumed.
    """

    def read(self, var):
        """Extract the value corresponding to a variable."""
        return var.val if type(var) is jax.core.Literal else self.env[var]


def _get_wires(wire_values, state: _InterpreterState):
    """Convert wire values into ``AbstractQbit`` instances."""
    wires = []
    for w in wire_values:
        if w in state.wire_map:
            wires.append(state.wire_map[w])
        else:
            wires.append(c_prims.qextract_p.bind(state.qreg, w))
    return wires


def _return_wires(state: _InterpreterState) -> None:
    """Reinsert all active wires into the quantum register and deallocate the register."""
    for orig_wire, wire in state.wire_map.items():
        state.qreg = c_prims.qinsert_p.bind(state.qreg, orig_wire, wire)
    c_prims.qdealloc_p.bind(state.qreg)


def _operator_eqn(eqn: jax.core.JaxprEqn, state: _InterpreterState) -> None:
    """Interpret a plxpr equation describing an operation as a catalxpr equation."""
    if not isinstance(eqn.outvars[0], jax.core.DropVar):
        state.op_math_cache[eqn.outvars[0]] = eqn
        return

    if "n_wires" not in eqn.params:
        raise NotImplementedError(
            f"Operator {eqn.primitive.name} not yet supported for catalyst conversion."
        )
    n_wires = eqn.params["n_wires"]

    wire_values = [state.read(v) for v in eqn.invars[-n_wires:]]
    wires = _get_wires(wire_values, state)

    invals = [state.read(invar) for invar in eqn.invars[:-n_wires]]
    outvals = c_prims.qinst_p.bind(
        *wires,
        *invals,
        op=eqn.primitive.name,
        qubits_len=eqn.params["n_wires"],
        params_len=len(eqn.invars) - eqn.params["n_wires"],
        ctrl_len=0,
        adjoint=False,
    )

    for wire_values, new_wire in zip(wire_values, outvals):
        state.wire_map[wire_values] = new_wire


def _obs(eqn: jax.core.JaxprEqn, state: _InterpreterState):
    """Interpret the observable equation corresponding to a measurement equation's input."""
    obs_eqn = state.op_math_cache[eqn.invars[0]]
    if "n_wires" not in obs_eqn.params:
        raise NotImplementedError(
            f"to_catalyst can not yet interpret observables of type {obs_eqn.primitive}"
        )

    n_wires = obs_eqn.params["n_wires"]
    wire_values = [state.read(v) for v in obs_eqn.invars[-n_wires:]]
    wires = _get_wires(wire_values, state)
    invals = [state.read(invar) for invar in obs_eqn.invars[:-n_wires]]
    return c_prims.namedobs_p.bind(*wires, *invals, kind=obs_eqn.primitive.name)


def _compbasis_obs(eqn: jax.core.JaxprEqn, state: _InterpreterState, device: "qml.devices.Device"):
    """Add a computational basis samling observable."""
    if eqn.invars:
        w_vals = [state.read(w_var) for w_var in eqn.invars]
    else:
        w_vals = device.wires  # broadcast across all wires
    wires = _get_wires(w_vals, state)
    return c_prims.compbasis_p.bind(*wires)


def _measurement_eqn(eqn: jax.core.JaxprEqn, state: _InterpreterState, device):
    if eqn.primitive.name not in measurement_map:
        raise NotImplementedError(
            f"measurement {eqn.primitive.name} not yet supported for conversion."
        )
    if eqn.params.get("has_eigvals", False):
        raise NotImplementedError

    if "_wires" in eqn.primitive.name:
        obs = _compbasis_obs(eqn, state, device)
    elif "_obs" in eqn.primitive.name:
        obs = _obs(eqn, state)
    else:
        raise NotImplementedError

    shaped_array = _get_shapes_for(
        eqn.outvars[0].aval, shots=device.shots, num_device_wires=len(device.wires)
    )[0]

    primitive = measurement_map[eqn.primitive.name]
    mval = primitive.bind(obs, shape=shaped_array.shape, shots=device.shots.total_shots)
    if shaped_array.dtype != mval.dtype:
        return jax.lax.convert_element_type(mval, shaped_array.dtype)
    return mval


def _bind_catalxpr(jaxpr: jax.core.Jaxpr, consts, device, *args) -> list:
    """Interpret plxpr as jaxpr."""

    c_prims.qdevice_p.bind(**_get_device_kwargs(device))
    qreg = c_prims.qalloc_p.bind(len(device.wires))
    state = _InterpreterState(qreg=qreg)

    for arg, invar in zip(args, jaxpr.invars):
        state.env[invar] = arg
    for const, constvar in zip(consts, jaxpr.constvars):
        state.env[constvar] = const

    measurements = []
    for eqn in jaxpr.eqns:
        if isinstance(eqn.outvars[0].aval, AbstractOperator):
            _operator_eqn(eqn, state)

        elif isinstance(eqn.outvars[0].aval, AbstractMeasurement):
            mval = _measurement_eqn(eqn, state, device)
            state.env[eqn.outvars[0]] = mval
            measurements.append(eqn.outvars[0])
        else:
            invals = [state.read(invar) for invar in eqn.invars]
            outvals = eqn.primitive.bind(*invals, **eqn.params)
            if not eqn.primitive.multiple_results:
                outvals = [outvals]
            for outvar, outval in zip(eqn.outvars, outvals):
                state.env[outvar] = outval

    _return_wires(state)
    # Read the final result of the Jaxpr from the environment
    return [state.read(outvar) for outvar in measurements]
