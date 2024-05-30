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

has_jax = True
try:
    import jax
except ImportError:
    has_jax = False

from catalyst.jax_primitives import (
    AbstractObs,
    AbstractQbit,
    AbstractQreg,
    expval_p,
    func_p,
    namedobs_p,
    probs_p,
    qalloc_p,
    qdealloc_p,
    qdevice_p,
    qextract_p,
    qinsert_p,
    qinst_p,
)

from .capture_qnode import _get_qnode_prim, _get_shapes_for
from .primitives import _get_abstract_measurement, _get_abstract_operator

AbstractOperator = _get_abstract_operator()
AbstractMeasurement = _get_abstract_measurement()
qnode_p = _get_qnode_prim()

has_catalyst = True


measurement_map = {"expval_obs": expval_p, "probs_wires": probs_p}


def _get_jaxpr_count(jaxpr) -> int:
    count = 0
    for eqn in jaxpr.eqns:
        for var in [*eqn.invars, *eqn.outvars]:
            if isinstance(var, jax.core.Var):
                count = max(count, var.count)
    return count


def to_catalyst(jaxpr):
    new_xpr = jax.core.Jaxpr(
        constvars=jaxpr.jaxpr.constvars,
        invars=jaxpr.jaxpr.invars,  # + qreg var?
        outvars=jaxpr.jaxpr.outvars,
        eqns=[],
    )
    for eqn in jaxpr.eqns:
        if eqn.primitive != qnode_p:
            new_xpr.eqns.append(eqn)
        else:
            invars = eqn.invars
            outvars = eqn.outvars
            primitive = func_p
            call_jaxpr = plxpr_to_catalyst(eqn.params["qfunc_jaxpr"].jaxpr, eqn.params["device"])
            params = {"fn": "TODO", "call_jaxpr": call_jaxpr}

            new_eqn = jax.core.JaxprEqn(
                invars, outvars, primitive, params, effects=jax.core.no_effects, source_info=None
            )
            new_xpr.eqns.append(new_eqn)
    return new_xpr


def plxpr_to_catalyst(plxpr, device):
    converter = CatalystConverter(plxpr)
    converter.add_device_eqn(device)
    converter.add_qreg_eqn(len(device.wires))
    converter.convert_plxpr_eqns()
    converter.return_wires()
    return converter.catalyst_xpr


class CatalystConverter:
    def __init__(self, plxpr):
        if not has_catalyst:
            raise ImportError("catalyst is required to convert plxpr to catalyst xpr.")

        self.plxpr = plxpr

        self.count = _get_jaxpr_count(plxpr)

        self.catalyst_xpr = jax.core.Jaxpr(
            constvars=plxpr.constvars,
            invars=plxpr.invars,  # + qreg var?
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
        self.shots = 0 if shots is None else shots
        params = {
            "rtd_kwargs": {"shots": self.shots, "mcmc": device._mcmc},
            "rtd_lib": "something something",
            "rtd_name": device.name,
        }
        device_eqn = jax.core.JaxprEqn(
            [], [], qdevice_p, params, effects=jax.core.no_effects, source_info=None
        )
        self.catalyst_xpr.eqns.append(device_eqn)

    def add_qreg_eqn(self, n_wires):
        invars = [
            jax.core.Literal(val=n_wires, aval=jax.core.ConcreteArray(dtype=int, val=n_wires))
        ]
        qreg = self._make_var(AbstractQreg())
        self._qreg = qreg
        qalloc_eqn = jax.core.JaxprEqn(
            invars, [qreg], qalloc_p, {}, effects=jax.core.no_effects, source_info=None
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
        wire = self._make_var(AbstractQbit())
        outvar = [wire]
        qextract_eqn = jax.core.JaxprEqn(
            invars, outvar, qextract_p, {}, effects=jax.core.no_effects, source_info=None
        )
        self.catalyst_xpr.eqns.append(qextract_eqn)
        return wire

    def return_wires(self):
        for orig_wire, wire in self._wire_map.items():
            orig_wire_var = jax.core.Literal(val=orig_wire, aval=orig_wire)
            invars = [self._qreg, orig_wire_var, wire]
            new_qreg = self._make_var(AbstractQreg())
            outvars = [new_qreg]

            eqn = jax.core.JaxprEqn(
                invars, outvars, qinsert_p, {}, effects=jax.core.no_effects, source_info=None
            )
            self.catalyst_xpr.eqns.append(eqn)

            self._qreg = new_qreg

        eqn = jax.core.JaxprEqn(
            [self._qreg], [], qdealloc_p, {}, effects=jax.core.no_effects, source_info=None
        )
        self.catalyst_xpr.eqns.append(eqn)

    def _add_operator_eqn(self, eqn):
        n_wires = eqn.params["n_wires"]

        orig_wires = eqn.invars[-n_wires:]
        wires = [self._get_wire(w) for w in orig_wires]
        invars = wires + eqn.invars[:-n_wires]

        outvars = [self._make_var(AbstractQbit()) for _ in range(eqn.params["n_wires"])]

        for w, outvar in zip(orig_wires, outvars):
            self._wire_map[w.val] = outvar

        primitive = qinst_p
        params = {
            "op": eqn.primitive.name,
            "qubits_len": eqn.params["n_wires"],
            "params_len": len(invars) - eqn.params["n_wires"],
            "ctrl_len": 0,
        }
        new_eqn = jax.core.JaxprEqn(
            invars, outvars, primitive, params, effects=jax.core.no_effects, source_info=None
        )
        self.catalyst_xpr.eqns.append(new_eqn)

    def _add_obs_eqn(self, eqn):
        primitive = namedobs_p

        n_wires = eqn.params["n_wires"]
        orig_wires = eqn.invars[-n_wires:]
        wires = [self._get_wire(w) for w in orig_wires]
        invars = wires + eqn.invars[:-n_wires]

        outvars = [self._make_var(AbstractObs())]
        params = {"kind": eqn.primitive.name}

        new_eqn = jax.core.JaxprEqn(
            invars, outvars, primitive, params, effects=jax.core.no_effects, source_info=None
        )
        self.catalyst_xpr.eqns.append(new_eqn)
        return outvars[0]

    def _add_measurement_eqn(self, eqn):
        primitive = measurement_map[eqn.primitive.name]

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
        outvars = [self._make_var(oa) for oa in outavals]

        new_eqn = jax.core.JaxprEqn(
            invars,
            outvars,
            primitive,
            {"shots": self.shots},
            effects=jax.core.no_effects,
            source_info=None,
        )
        self.catalyst_xpr.eqns.append(new_eqn)
        return outvars

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
