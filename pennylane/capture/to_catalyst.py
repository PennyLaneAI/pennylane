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
import catalyst
import jax
import numpy as np
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


null_source_info = jax.extend.source_info_util.SourceInfo(
    None, jax.extend.source_info_util.NameStack()
)


def _get_device_kwargs(device: "pennylane.devices.Device") -> dict:
    """Calulcate the params for a device equation."""
    features = catalyst.utils.toml.ProgramFeatures(device.shots is not None)
    capabilities = catalyst.utils.toml.get_device_capabilities(device, features)
    info = catalyst.device.extract_backend_info(device, capabilities)
    # Note that the value of rtd_kwargs is a string version of the info kwargs, not the info kwargs itself!
    return {
        "rtd_kwargs": str(info.kwargs),
        "rtd_lib": info.lpath,
        "rtd_name": info.c_interface_name,
    }


def to_catalyst(plxpr):
    def f(*args):
        return to_catalyst_interpreter(plxpr.jaxpr, plxpr.consts, *args)

    return jax.make_jaxpr(f)


def to_catalyst_interpreter(jaxpr: jax.core.Jaxpr, consts, *args) -> jax.core.Jaxpr:
    """Convert pennylane variant jaxpr to catalyst variant jaxpr.

    Args:
        jaxpr (jax.core.Jaxpr): pennylane variant jaxpr

    Returns:
        jax.core.Jaxpr: catalyst variant jaxpr

    Note that the input jaxpr should be workflow level and contain qnode primitives, rather than
    qfunc level with individual operators.  See ``qfunc_jaxpr_to_catalyst`` for converting the quantum
    function plxpr.

    .. code-block:: python

        qml.capture.enable()

        @qml.qnode(qml.device('lightning.qubit', wires=2))
        def circuit(x):
            qml.RX(x, 0)
            return qml.probs(wires=(0, 1))

        def f(x):
            return circuit(2 * x) ** 2

        jaxpr = jax.make_jaxpr(circuit)(0.5)

        print(qml.capture.to_catalyst(jaxpr))

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
                f:AbstractQbit() = qinst[ctrl_len=0 op=RX params_len=1 qubits_len=1] e
                    c
                g:AbstractQbit() = qextract d 1
                h:AbstractObs(num_qubits=None,primitive=None) = compbasis f g
                i:f64[4] = probs[shots=Shots(total=None)] h
                j:f64[4] = convert_element_type[new_dtype=float64 weak_type=False] i
                k:AbstractQreg() = qinsert d 0 f
                qdealloc k
                in (j,) }
            fn=<QNode: device='<lightning.qubit device (wires=2) at 0x1398172d0>', interface='auto', diff_method='best'>
            ] a
        in (b,) }

    """
    env = {}

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

            call_jaxpr = _qfunc_jaxpr_to_catalyst(
                eqn.params["qfunc_jaxpr"], eqn.params["device"], *invals
            )

            def f(*innervals):
                return bind_catalxpr(
                    eqn.params["qfunc_jaxpr"].jaxpr,
                    eqn.params["qfunc_jaxpr"].consts,
                    eqn.params["device"],
                    *innervals,
                )

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


def _qfunc_jaxpr_to_catalyst(
    plxpr: jax.core.Jaxpr, device: "pennylane.devices.Device", *args
) -> jax.core.Jaxpr:
    """Convert qfunc jaxpr and a device to catalyst variant jaxpr."""

    def f(*inner_args):
        return bind_catalxpr(plxpr.jaxpr, plxpr.consts, device, *inner_args)

    return jax.make_jaxpr(f)(*args)


def _read(var, env):
    return var.val if type(var) is jax.core.Literal else env[var]


def _operator_eqn(eqn, env, wire_map, qreg):
    if "n_wires" not in eqn.params:
        raise NotImplementedError(
            f"Operator {eqn.primitive.name} not yet supported for catalyst conversion."
        )
    n_wires = eqn.params["n_wires"]
    wire_values = [_read(v, env) for v in eqn.invars[-n_wires:]]
    wires = []
    for w in wire_values:
        if w in wire_map:
            wires.append(wire_map[w])
        else:
            wires.append(c_prims.qextract_p.bind(qreg, w))

    invals = [_read(invar, env) for invar in eqn.invars[:-n_wires]]
    outvals = c_prims.qinst_p.bind(
        *wires,
        *invals,
        op=eqn.primitive.name,
        qubits_len=eqn.params["n_wires"],
        params_len=len(eqn.invars) - eqn.params["n_wires"],
        ctrl_len=0,
    )

    for wire_values, new_wire in zip(wire_values, outvals):
        wire_map[wire_values] = new_wire


def _return_wires(wire_map, qreg):
    for orig_wire, wire in wire_map.items():
        qreg = c_prims.qinsert_p.bind(qreg, orig_wire, wire)
    c_prims.qdealloc_p.bind(qreg)


def _measurement_wires_eqn(eqn, env, wire_map, qreg, device):
    if eqn.primitive.name not in measurement_map:
        raise NotImplementedError
    primitive = measurement_map[eqn.primitive.name]
    if eqn.params.get("has_eigvals", False):
        raise NotImplementedError
    wires = []

    if eqn.invars:
        w_vals = [_read(w_var, env) for w_var in eqn.invars]
    else:
        w_vals = device.wires
    for w_val in w_vals:
        if w_val in wire_map:
            wires.append(wire_map[w_val])
        else:
            wires.append(c_prims.qextract_p.bind(qreg, w_val))
    compbasis_obs = c_prims.compbasis_p.bind(*wires)

    shaped_arrays = _get_shapes_for(
        eqn.outvars[0].aval, shots=device.shots, num_device_wires=len(device.wires)
    )

    return primitive.bind(
        compbasis_obs, shape=shaped_arrays[0].shape, shots=device.shots.total_shots
    )


def _measurement_obs_eqn(eqn, env, op_math_cache):
    primitive = measurement_map[eqn.primitive.name]
    raise NotImplementedError


def bind_catalxpr(jaxpr, consts, device, *args):
    # Mapping from variable -> value
    env = {}
    wire_map = {}
    op_math_cache = {}
    measurements = []
    # Bind args and consts to environment
    for arg, invar in zip(args, jaxpr.invars):
        env[invar] = arg
    for const, constvar in zip(consts, jaxpr.constvars):
        env[constvar] = const

    c_prims.qdevice_p.bind(**_get_device_kwargs(device))
    qreg = c_prims.qalloc_p.bind(len(device.wires))

    # Loop through equations and evaluate primitives using `bind`
    for eqn in jaxpr.eqns:
        if isinstance(eqn.outvars[0].aval, AbstractOperator):
            if isinstance(eqn.outvars[0], jax.core.DropVar):
                _operator_eqn(eqn, env, wire_map, qreg)
            else:
                op_math_cache[eqn.outvars[0]] = eqn

        elif isinstance(eqn.outvars[0].aval, AbstractMeasurement):
            if "_wires" in eqn.primitive.name:
                mvals = _measurement_wires_eqn(eqn, env, wire_map, qreg, device)
            else:
                mvals = _measurement_obs_eqn(eqn, env, op_math_cache)
            if not eqn.primitive.multiple_results:
                mvals = [mvals]
            for outvar, outval in zip(eqn.outvars, mvals):
                env[outvar] = outval
                measurements.append(outvar)
        else:
            invals = [_read(invar, env) for invar in eqn.invars]
            outvals = eqn.primitive.bind(*invals, **eqn.params)
            if not eqn.primitive.multiple_results:
                outvals = [outvals]
            for outvar, outval in zip(eqn.outvars, outvals):
                env[outvar] = outval

    _return_wires(wire_map, qreg)
    # Read the final result of the Jaxpr from the environment
    return [_read(outvar, env) for outvar in measurements]
