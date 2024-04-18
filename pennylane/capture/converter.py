# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax

from .meta_type import _get_abstract_operator


from catalyst.jax_primitives import (
    qinst_p,
    AbstractQbit,
    qdevice_p,
    qalloc_p,
    AbstractQreg,
    qextract_p,
    qdealloc_p,
    qinsert_p,
)


AbstractOperator = _get_abstract_operator()


def _get_jaxpr_count(jaxpr) -> int:
    count = 0
    for eqn in jaxpr.eqns:
        for var in [*eqn.invars, *eqn.outvars]:
            if isinstance(var, jax.core.Var):
                count = max(count, var.count)
    return count


def to_catalyst(jaxpr, device):
    cc = CatalystConverter(jaxpr, device)
    cc.convert_plxpr_eqns()
    cc.return_wires()
    return cc.catalyst_xpr


class CatalystConverter:
    def __init__(self, plxpr, device):
        self.plxpr = plxpr

        self.count = _get_jaxpr_count(plxpr)

        self.catalyst_xpr = jax.core.Jaxpr(
            constvars=plxpr.constvars,
            invars=plxpr.invars,  # + qreg var?
            outvars=plxpr.outvars,
            eqns=[],
        )

        self._qreg = None
        self._wire_map = {}

        self._add_device_eqn(device)
        self._add_qreg_eqn(len(device.wires))

        self._op_math_cache = []

    def _add_device_eqn(self, device):
        params = {
            "rtd_kwargs": {"shots": device.shots, "mcmc": device._mcmc},
            "rtd_lib": "something something",
            "rtd_name": device.short_name,
        }
        device_eqn = jax.core.JaxprEqn(
            [], [], qdevice_p, params, effects=jax.core.no_effects, source_info=None
        )
        self.catalyst_xpr.eqns.append(device_eqn)

    def _add_qreg_eqn(self, n_wires):
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

    def _add_plxpr_eqn(self, eqn):
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

    def convert_plxpr_eqns(self):
        for eqn in self.plxpr.eqns:
            if isinstance(eqn.outvars[0].aval, AbstractOperator):
                if isinstance(eqn.outvars[0], jax.core.DropVar):
                    self._add_plxpr_eqn(eqn)
                else:
                    self._op_math_cache.append(eqn)
            else:
                self.catalyst_xpr.eqns.append(eqn)

        return self.catalyst_xpr
