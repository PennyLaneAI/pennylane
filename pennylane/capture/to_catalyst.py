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
from catalyst import jax_primitives as cat_p

from .capture_qnode import _get_qnode_prim, _get_shapes_for
from .primitives import _get_abstract_measurement, _get_abstract_operator

AbstractOperator = _get_abstract_operator()
AbstractMeasurement = _get_abstract_measurement()

measurement_map = {
    "sample_wires": cat_p.sample_p,
    "sample_obs": cat_p.sample_p,
    "expval_obs": cat_p.expval_p,
    "var_obs": cat_p.var_p,
    "probs_wires": cat_p.probs_p,
    "state_wires": cat_p.state_p,
}
# TODO: figure out what to do about counts


null_source_info = jax.extend.source_info_util.SourceInfo(
    None, jax.extend.source_info_util.NameStack()
)


def _get_device_kwargs(device: "pennylane.devices.Device") -> dict:
    features = catalyst.utils.toml.ProgramFeatures(device.shots is not None)
    capabilities = catalyst.utils.toml.get_device_capabilities(device, features)
    info = catalyst.device.extract_backend_info(device, capabilities)
    # Note that the value of rtd_kwargs is a string version of the info kwargs, not the info kwargs itself !!!
    return {
        "rtd_kwargs": str(info.kwargs),
        "rtd_lib": info.lpath,
        "rtd_name": info.c_interface_name,
    }


def _get_jaxpr_count(jaxpr) -> int:
    count = 0
    for eqn in jaxpr.eqns:
        for var in [*eqn.invars, *eqn.outvars]:
            if isinstance(var, jax.core.Var):
                count = max(count, var.count)
    return count


def to_catalyst(jaxpr: jax.core.Jaxpr) -> jax.core.Jaxpr:
    """Convert pennylane variant jaxpr to catalyst variant jaxpr.

    Args:
        jaxpr (jax.core.Jaxpr): pennylane variant jaxpr

    Returns:
        jax.core.Jaxpr: catalyst variant jaxpr


    Note that the input jaxpr should be workflow level and contain qnode primitives, rather than
    qfunc level with individual operators.  See ``plxpr_to_catalyst`` for converting the quantum
    function plxpr.

    .. code-block:: python

        qml.capture.enable()

        @qml.qnode(qml.device('lightning.qubit', wires=2))
        def circuit(x):
            qml.RX(x,0)
            return qml.probs(wires=(0,1))

        def f(x):
            return circuit(2* x) ** 2

        jaxpr = jax.make_jaxpr(circuit)(0.5)

        print(qml.capture.to_catalyst(jaxpr))

    .. code-block:: none

        { lambda ; a:f64[]. let
            b:f64[4] = func[
            call_jaxpr={ lambda ; c:f64[]. let
                qdevice[
                    rtd_kwargs={'shots': 0, 'mcmc': False}
                    rtd_lib=something something
                    rtd_name=lightning.qubit
                ]
                d:AbstractQreg() = qalloc 2
                e:AbstractQbit() = qextract d 0
                f:AbstractQbit() = qinst[ctrl_len=0 op=RX params_len=1 qubits_len=1] e
                    c
                g:AbstractQbit() = qextract d 1
                h:AbstractObs(num_qubits=None,primitive=None) = compbasis f g
                i:f64[4] = probs[shots=0] h
                j:AbstractQreg() = qinsert d 0 f
                qdealloc j
                in (i,) }
            fn=TODO
            ] a
        in (b,) }

    """
    new_xpr = jax.core.Jaxpr(
        constvars=jaxpr.jaxpr.constvars,
        invars=jaxpr.jaxpr.invars,  # + qreg var?
        outvars=jaxpr.jaxpr.outvars,
        eqns=[],
    )
    for eqn in jaxpr.eqns:
        if eqn.primitive != _get_qnode_prim():
            new_xpr.eqns.append(eqn)
        else:
            invars = eqn.invars
            outvars = eqn.outvars
            primitive = cat_p.func_p
            call_jaxpr = plxpr_to_catalyst(eqn.params["qfunc_jaxpr"].jaxpr, eqn.params["device"])
            params = {"fn": eqn.params["qnode"], "call_jaxpr": call_jaxpr}

            new_eqn = jax.core.JaxprEqn(
                invars,
                outvars,
                primitive,
                params,
                effects=jax.core.no_effects,
                source_info=null_source_info,
            )
            new_xpr.eqns.append(new_eqn)
    return new_xpr


def plxpr_to_catalyst(plxpr: jax.core.Jaxpr, device: "pennylane.devices.Device") -> jax.core.Jaxpr:
    """Convert"""

    converter = CatalystConverter(plxpr)
    converter.add_device_eqn(device)
    converter.add_qreg_eqn(len(device.wires))
    converter.convert_plxpr_eqns()
    converter.return_wires()
    return converter.catalyst_xpr


class CatalystConverter:
    def __init__(self, plxpr):

        self.plxpr = plxpr

        self.count = _get_jaxpr_count(plxpr)

        self.catalyst_xpr = jax.core.Jaxpr(
            constvars=plxpr.constvars,
            invars=plxpr.invars,
            outvars=[],
            eqns=[],
        )

        self._qreg = None
        self._wire_map = {}
        self._op_math_cache = {}
        self.num_device_wires = 0
        self.shots = None

    def add_device_eqn(self, device):
        self.num_device_wires = len(device.wires)
        shots = device.shots.total_shots
        self.shots = device.shots
        params = _get_device_kwargs(device)
        device_eqn = jax.core.JaxprEqn(
            [],
            [],
            cat_p.qdevice_p,
            params,
            effects=jax.core.no_effects,
            source_info=null_source_info,
        )
        self.catalyst_xpr.eqns.append(device_eqn)

    def add_qreg_eqn(self, n_wires):
        invars = [
            jax.core.Literal(val=n_wires, aval=jax.core.ConcreteArray(dtype=int, val=n_wires))
        ]
        qreg = self._make_var(cat_p.AbstractQreg())
        self._qreg = qreg
        qalloc_eqn = jax.core.JaxprEqn(
            invars,
            [qreg],
            cat_p.qalloc_p,
            {},
            effects=jax.core.no_effects,
            source_info=null_source_info,
        )
        self.catalyst_xpr.eqns.append(qalloc_eqn)

    def _make_var(self, aval):
        self.count += 1
        return jax.core.Var(count=self.count, suffix="", aval=aval)

    def _get_wire(self, orig_wire):
        wire = orig_wire.val
        if wire in self._wire_map:
            return self._wire_map[orig_wire.val]

        wire_var = jax.core.Literal(val=wire, aval=jax.core.ConcreteArray(dtype=int, val=wire))
        invars = [self._qreg, wire_var]
        wire = self._make_var(cat_p.AbstractQbit())
        outvar = [wire]
        qextract_eqn = jax.core.JaxprEqn(
            invars,
            outvar,
            cat_p.qextract_p,
            {},
            effects=jax.core.no_effects,
            source_info=null_source_info,
        )
        self.catalyst_xpr.eqns.append(qextract_eqn)
        return wire

    def return_wires(self):
        for orig_wire, wire in self._wire_map.items():
            orig_wire_var = jax.core.Literal(val=orig_wire, aval=orig_wire)
            invars = [self._qreg, orig_wire_var, wire]
            new_qreg = self._make_var(cat_p.AbstractQreg())
            outvars = [new_qreg]

            eqn = jax.core.JaxprEqn(
                invars,
                outvars,
                cat_p.qinsert_p,
                {},
                effects=jax.core.no_effects,
                source_info=null_source_info,
            )
            self.catalyst_xpr.eqns.append(eqn)

            self._qreg = new_qreg

        eqn = jax.core.JaxprEqn(
            [self._qreg],
            [],
            cat_p.qdealloc_p,
            {},
            effects=jax.core.no_effects,
            source_info=null_source_info,
        )
        self.catalyst_xpr.eqns.append(eqn)

    def _add_operator_eqn(self, eqn):
        n_wires = eqn.params["n_wires"]

        orig_wires = eqn.invars[-n_wires:]
        wires = [self._get_wire(w) for w in orig_wires]
        invars = wires + eqn.invars[:-n_wires]

        outvars = [self._make_var(cat_p.AbstractQbit()) for _ in range(eqn.params["n_wires"])]

        for w, outvar in zip(orig_wires, outvars):
            self._wire_map[w.val] = outvar

        primitive = cat_p.qinst_p
        params = {
            "op": eqn.primitive.name,
            "qubits_len": eqn.params["n_wires"],
            "params_len": len(invars) - eqn.params["n_wires"],
            "ctrl_len": 0,
        }
        new_eqn = jax.core.JaxprEqn(
            invars,
            outvars,
            primitive,
            params,
            effects=jax.core.no_effects,
            source_info=null_source_info,
        )
        self.catalyst_xpr.eqns.append(new_eqn)

    def _add_obs_eqn(self, eqn):
        primitive = cat_p.namedobs_p

        n_wires = eqn.params["n_wires"]
        orig_wires = eqn.invars[-n_wires:]
        wires = [self._get_wire(w) for w in orig_wires]
        invars = wires + eqn.invars[:-n_wires]

        outvars = [self._make_var(cat_p.AbstractObs())]
        params = {"kind": eqn.primitive.name}

        new_eqn = jax.core.JaxprEqn(
            invars,
            outvars,
            primitive,
            params,
            effects=jax.core.no_effects,
            source_info=null_source_info,
        )
        self.catalyst_xpr.eqns.append(new_eqn)
        return outvars[0]

    def _add_comp_basis_eqn(self, eqn):
        wires_invars = [self._get_wire(w) for w in eqn.invars]
        outvars = [self._make_var(cat_p.AbstractObs())]
        wires_eqn = jax.core.JaxprEqn(
            wires_invars,
            outvars,
            cat_p.compbasis_p,
            {},
            effects=jax.core.no_effects,
            source_info=null_source_info,
        )
        self.catalyst_xpr.eqns.append(wires_eqn)
        return outvars

    def _add_measurement_eqn(self, eqn):
        primitive = measurement_map[eqn.primitive.name]

        if "_wires" in eqn.primitive.name:
            if eqn.params.get("has_eigvals", False):
                raise NotImplementedError
            invars = self._add_comp_basis_eqn(eqn)
        else:
            invars = []
            for orig_invar in eqn.invars:
                if isinstance(orig_invar, jax.core.Literal):
                    invars.append(orig_invar)
                elif orig_invar in self._op_math_cache:
                    obs = self._add_obs_eqn(self._op_math_cache[orig_invar])
                    invars.append(obs)

        outavals = _get_shapes_for(
            eqn.outvars[0].aval, shots=self.shots, num_device_wires=self.num_device_wires
        )
        convert_element_inds = []
        for i, oa in enumerate(outavals):
            print(oa)
            if oa != jax.numpy.float64:
                convert_element_inds.append((i, oa))
                outavals[i] = jax.core.ShapedArray(oa.shape, dtype=jax.numpy.float64)

        outvars = [self._make_var(oa) for oa in outavals]

        new_eqn = jax.core.JaxprEqn(
            invars,
            outvars,
            primitive,
            {"shots": self.shots},
            effects=jax.core.no_effects,
            source_info=null_source_info,
        )
        self.catalyst_xpr.eqns.append(new_eqn)

        final_outvars = list(outvars)
        for ind, orig_aval in convert_element_inds:
            c_invars = [outvars[ind]]
            c_outvars = [self._make_var(orig_aval)]
            c_eqn = jax.core.JaxprEqn(
                c_invars,
                c_outvars,
                jax.lax.convert_element_type_p,
                {"new_dtype": orig_aval.dtype, "weak_type": False},
                effects=jax.core.no_effects,
                source_info=null_source_info,
            )
            self.catalyst_xpr.eqns.append(c_eqn)
            final_outvars[ind] = c_outvars[0]

        return final_outvars

    def convert_plxpr_eqns(self):
        for eqn in self.plxpr.eqns:
            if isinstance(eqn.outvars[0].aval, AbstractOperator):
                if isinstance(eqn.outvars[0], jax.core.DropVar):
                    self._add_operator_eqn(eqn)
                else:
                    self._op_math_cache[eqn.outvars[0]] = eqn
            elif isinstance(eqn.outvars[0].aval, AbstractMeasurement):
                mp_outaval = self._add_measurement_eqn(eqn)
                self.catalyst_xpr.outvars.extend(mp_outaval)
            else:
                self.catalyst_xpr.eqns.append(eqn)

        return self.catalyst_xpr
