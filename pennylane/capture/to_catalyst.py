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

from .capture_qnode import _get_qnode_prim, _get_shapes_for
from .primitives import _get_abstract_measurement, _get_abstract_operator

AbstractOperator = _get_abstract_operator()
AbstractMeasurement = _get_abstract_measurement()

measurement_map = {
    "sample_wires": c_prims.sample_p,
    "expval_obs": c_prims.expval_p,
    "var_obs": c_prims.var_p,
    "probs_wires": c_prims.probs_p,
    "state_wires": c_prims.state_p,
}


null_source_info = jax.core.source_info_util.SourceInfo(None, jax.core.source_info_util.NameStack())


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


def to_catalyst(jaxpr: jax.core.Jaxpr) -> jax.core.Jaxpr:
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
    new_xpr = jax.core.Jaxpr(
        constvars=jaxpr.jaxpr.constvars,
        invars=jaxpr.jaxpr.invars,
        outvars=jaxpr.jaxpr.outvars,
        eqns=[],
    )
    for eqn in jaxpr.eqns:
        if eqn.primitive != _get_qnode_prim():
            new_xpr.eqns.append(eqn)
        else:
            if eqn.params["shots"] != eqn.params["device"].shots:
                raise NotImplementedError("catalyst does not support dynamic shots.")
            invars = eqn.invars
            outvars = eqn.outvars
            primitive = c_prims.func_p
            call_jaxpr = _qfunc_jaxpr_to_catalyst(
                eqn.params["qfunc_jaxpr"].jaxpr, eqn.params["device"]
            )
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

    return jax.core.ClosedJaxpr(new_xpr, jaxpr.consts)


def _qfunc_jaxpr_to_catalyst(
    plxpr: jax.core.Jaxpr, device: "pennylane.devices.Device"
) -> jax.core.Jaxpr:
    """Convert qfunc jaxpr and a device to catalyst variant jaxpr."""

    converter = _CatalystConverter(plxpr.constvars, plxpr.invars)
    converter.add_device_eqn(device)
    converter.add_qreg_eqn()
    for eqn in plxpr.eqns:
        converter.convert_plxpr_eqn(eqn)
    converter.return_wires()
    return converter.catalyst_xpr


class _CatalystConverter:
    """A class that manages the conversion of pennylane variant qfunc jaxpr to catalyst variant jaxpr.

    By using this class, we can manage the state of the conversion and the clean up required
    at the end.

    This class is purely internal to the ``_qfunc_jaxpr_to_catalyst`` helper function. The division into
    public methods and private methods is designed to improve readability and add an additional level
    of detail at each level in the stack.  While the public methods have a very particular order
    to be called in, the tight coupling between ``_CatalystConverter`` and ``_qfunc_jaxpr_to_catalyst``
    insures that we do not need to worry about the methods being called in a different order. We do
    not need to be concerned about this class being used in a different context.

    Stateful variables are:
    * ``catalyst_xpr``
    * ``_wire_map``: map from wire value to ``AbstractQbit`` produced by earlier operations on the wire
    * ``_op_math_cache``: operators that are consumed by later equations
    * ``_count``: number of variables already existing.  Used when creating new variables.
    * ``_classical_var_map``: map from plxpr classical vars to catalyst classical vars.  Different
      by count.

    Constants saved between calls are:
    * ``_qreg``
    * ``_num_device_wires``
    * ``_shots``

    """

    def __init__(self, constvars, invars):
        self._count = len(invars)

        self.catalyst_xpr = jax.core.Jaxpr(
            constvars=constvars,
            invars=invars,
            outvars=[],
            eqns=[],
        )

        self._qreg = None
        self._wire_map = {}
        self._op_math_cache = {}
        self._num_device_wires = 0
        self._shots = None
        self._classical_var_map = {}

    def _make_var(self, aval):
        """Create a variable from an aval.

        Side Effects:
            Increments the ``_count`` variable.
        """
        out = jax.core.Var(count=self._count, suffix="", aval=aval)
        self._count += 1
        return out

    def _get_wire(self, orig_wire):
        """Get the ``AbstractQubit`` corresponding to a given wire label.

        If the wire has already been acted upon, this will retrieve the ``AbstractQbit`` stored
        in ``self._wire_map``.  If the wire has not already been acted upon, it adds
        a qubit extraction equation and returns the ``AbstractQbit`` extracted from the
        register.

        Side Effects:
            Adds an extraction equation if needed.
        """
        wire = orig_wire.val
        if wire in self._wire_map:
            return self._wire_map[wire]

        wire_var = jax.core.Literal(
            val=wire, aval=jax.core.ShapedArray(dtype=int, shape=(), weak_type=True)
        )
        invars = [self._qreg, wire_var]
        c_wire = self._make_var(c_prims.AbstractQbit())
        outvar = [c_wire]
        qextract_eqn = jax.core.JaxprEqn(
            invars,
            outvar,
            c_prims.qextract_p,
            {},
            effects=jax.core.no_effects,
            source_info=null_source_info,
        )
        self.catalyst_xpr.eqns.append(qextract_eqn)
        return c_wire

    def add_device_eqn(self, device):
        """Add the equation for setting a device.

        Should be the first method called when populating the catalyst xpr.
        """
        self._num_device_wires = len(device.wires)
        self._shots = device.shots
        params = _get_device_kwargs(device)
        device_eqn = jax.core.JaxprEqn(
            [],
            [],
            c_prims.qdevice_p,
            params,
            effects=jax.core.no_effects,
            source_info=null_source_info,
        )
        self.catalyst_xpr.eqns.append(device_eqn)

    def add_qreg_eqn(self):
        """Add an equation for extracting a quantum register.

        Should be the second method called, after ``add_device_eqn``.
        """
        invars = [
            jax.core.Literal(
                val=self._num_device_wires,
                aval=jax.core.ShapedArray(dtype=int, shape=(), weak_type=True),
            )
        ]
        qreg = self._make_var(c_prims.AbstractQreg())
        self._qreg = qreg
        qalloc_eqn = jax.core.JaxprEqn(
            invars,
            [qreg],
            c_prims.qalloc_p,
            {},
            effects=jax.core.no_effects,
            source_info=null_source_info,
        )
        self.catalyst_xpr.eqns.append(qalloc_eqn)

    def return_wires(self):
        """Inserts all active qubits back into the original register and de-allocates the register.

        Should be last method called before extracting the final ``catalyst_xpr`` property.

        Side Effects:
            Adds equations for qubit insertion and register de-allocation.

        """
        for orig_wire, wire in self._wire_map.items():
            orig_wire_var = jax.core.Literal(
                val=orig_wire, aval=jax.core.ShapedArray(dtype=int, shape=(), weak_type=True)
            )
            invars = [self._qreg, orig_wire_var, wire]
            new_qreg = self._make_var(c_prims.AbstractQreg())
            outvars = [new_qreg]

            eqn = jax.core.JaxprEqn(
                invars,
                outvars,
                c_prims.qinsert_p,
                {},
                effects=jax.core.no_effects,
                source_info=null_source_info,
            )
            self.catalyst_xpr.eqns.append(eqn)

            self._qreg = new_qreg

        eqn = jax.core.JaxprEqn(
            [self._qreg],
            [],
            c_prims.qdealloc_p,
            {},
            effects=jax.core.no_effects,
            source_info=null_source_info,
        )
        self.catalyst_xpr.eqns.append(eqn)

    def _add_operator_eqn(self, eqn):
        """Adds qinst equation corresponding to the input pl variant equation.

        Dispatched to from ``convert_plxpr_eqn``.

        Side effects:
            Adds a catalyst-variant equation corresponding to the input operation.
            Updates output ``AbstractQbit`` vars to ``self._wire_map``.

        """
        if "n_wires" not in eqn.params:
            raise NotImplementedError(
                f"Operator {eqn.primitive.name} not yet supported for catalyst conversion."
            )
        n_wires = eqn.params["n_wires"]

        orig_wires = eqn.invars[-n_wires:]
        wires = [self._get_wire(w) for w in orig_wires]
        invars = wires + [
            self._classical_var_map.get(invar, invar) for invar in eqn.invars[:-n_wires]
        ]

        outvars = [self._make_var(c_prims.AbstractQbit()) for _ in range(eqn.params["n_wires"])]

        for w, outvar in zip(orig_wires, outvars):
            self._wire_map[w.val] = outvar

        primitive = c_prims.qinst_p
        params = {
            "op": eqn.primitive.name,
            "qubits_len": eqn.params["n_wires"],
            "params_len": len(invars) - eqn.params["n_wires"],
            "ctrl_len": 0,
            "adjoint": False,
        }
        new_eqn = jax.core.JaxprEqn(
            invars,
            outvars,
            primitive,
            params,
            effects=eqn.source_info,
            source_info=null_source_info,
        )
        self.catalyst_xpr.eqns.append(new_eqn)

    def _add_obs_eqn(self, eqn):
        """Adds a named obs equation and returns an ``AbstractObs`` variable.

        Used by ``_add_measurement_eqn``.

        """
        primitive = c_prims.namedobs_p

        if "n_wires" not in eqn.params:
            raise NotImplementedError(
                f"Operator {eqn.primitive} not yet supported for catalyst conversion"
            )
        n_wires = eqn.params["n_wires"]
        orig_wires = eqn.invars[-n_wires:]
        wires = [self._get_wire(w) for w in orig_wires]
        invars = wires + eqn.invars[:-n_wires]

        outvars = [self._make_var(c_prims.AbstractObs())]
        params = {"kind": eqn.primitive.name}

        new_eqn = jax.core.JaxprEqn(
            invars,
            outvars,
            primitive,
            params,
            effects=jax.core.no_effects,
            source_info=eqn.source_info,
        )
        self.catalyst_xpr.eqns.append(new_eqn)
        return outvars[0]

    def _add_comp_basis_eqn(self, eqn):
        """Adds a computational basis observable equation and returns an ``AbstractObs`` variable.

        Used by ``_add_measurement_eqn``.
        """
        if len(eqn.invars) == 0:
            num_wires = self._num_device_wires
            wires_invars = []
            for w in range(self._num_device_wires):
                w_literal = jax.core.Literal(
                    val=w, aval=jax.core.ShapedArray(dtype=int, shape=(), weak_type=True)
                )
                wires_invars.append(self._get_wire(w_literal))
        else:
            num_wires = len(eqn.invars)
            wires_invars = [self._get_wire(w) for w in eqn.invars]
        outvars = [
            self._make_var(c_prims.AbstractObs(num_qubits=num_wires, primitive=c_prims.compbasis_p))
        ]
        wires_eqn = jax.core.JaxprEqn(
            wires_invars,
            outvars,
            c_prims.compbasis_p,
            {},
            effects=jax.core.no_effects,
            source_info=null_source_info,
        )
        self.catalyst_xpr.eqns.append(wires_eqn)
        return outvars

    def _convert_measurement_dtypes(self, invars, final_aval):
        """Adds a dtype conversion equation to convert invars into the type specified by final_aval.

        Used by ``_add_measurement_eqn``.
        """
        c_outvars = [self._make_var(final_aval)]
        c_eqn = jax.core.JaxprEqn(
            invars,
            c_outvars,
            jax.lax.convert_element_type_p,
            {"new_dtype": final_aval.dtype, "weak_type": False},
            effects=jax.core.no_effects,
            source_info=null_source_info,
        )
        self.catalyst_xpr.eqns.append(c_eqn)
        return c_outvars

    def _add_measurement_eqn(self, eqn):
        """Converts a pl variant measurement eqn into a catalyst one and adds it to ``catalyst_xpr``.

        Called by ``convert_plxpr_eqn``.

        """
        if eqn.primitive.name not in measurement_map:
            raise NotImplementedError(
                f"measurement {eqn.primitive.name} not yet implemented for catalyst conversion."
            )
        primitive = measurement_map[eqn.primitive.name]

        if "_wires" in eqn.primitive.name:
            if eqn.params.get("has_eigvals", False):
                raise NotImplementedError(
                    "Measurements with eigvals not yet supported for catalyst conversion."
                )
            invars = self._add_comp_basis_eqn(eqn)
        else:  # obs or mcms
            invars = []
            for orig_invar in eqn.invars:
                if orig_invar not in self._op_math_cache:
                    raise NotImplementedError(
                        "Measurements must either be in the computational basis or of an observable"
                    )

                obs = self._add_obs_eqn(self._op_math_cache[orig_invar])
                invars.append(obs)
        outavals = _get_shapes_for(
            eqn.outvars[0].aval, shots=self._shots, num_device_wires=self._num_device_wires
        )

        # convert all output dtypes to float64
        # _convert_measurement_dtypes will add an equation converting float64 to any non-float64 output
        final_aval = outavals[0]
        if outavals[0].dtype.name not in {"float64", "complex64", "complex128"}:
            outavals = [jax.core.ShapedArray(outavals[0].shape, dtype=jax.numpy.float64)]

        outvars = [self._make_var(oa) for oa in outavals]

        new_eqn = jax.core.JaxprEqn(
            invars,
            outvars,
            primitive,
            {"shots": self._shots.total_shots, "shape": outavals[0].shape},
            effects=jax.core.no_effects,
            source_info=eqn.source_info,
        )
        self.catalyst_xpr.eqns.append(new_eqn)

        if final_aval is outavals[0]:
            return outvars
        return self._convert_measurement_dtypes(outvars, final_aval)

    def _convert_classical_eqn(self, eqn):
        new_invars = [
            (
                invar
                if isinstance(invar, jax.core.Literal)
                else self._classical_var_map.get(invar, invar)
            )
            for invar in eqn.invars
        ]
        new_outvars = [self._make_var(var.aval) for var in eqn.outvars]
        for new_outvar, outvar in zip(new_outvars, eqn.outvars):
            self._classical_var_map[outvar] = new_outvar
        new_eqn = jax.core.JaxprEqn(
            new_invars, new_outvars, eqn.primitive, eqn.params, eqn.effects, eqn.source_info
        )
        self.catalyst_xpr.eqns.append(new_eqn)

    def convert_plxpr_eqn(self, eqn):
        """Converts a pl variant equation into a catalyst variant equation and adds it to ``catalyst_xpr``.

        Dispatches between ``_add_operator_eqn``, ``_op_math_cache``, ``_add_measurement_eqn``, and
        simply adding the equation to ``catalyst_xpr`` based on the type and contents of the equation.

        """
        if isinstance(eqn.outvars[0].aval, AbstractOperator):
            if isinstance(eqn.outvars[0], jax.core.DropVar):
                self._add_operator_eqn(eqn)
            else:
                self._op_math_cache[eqn.outvars[0]] = eqn
        elif isinstance(eqn.outvars[0].aval, AbstractMeasurement):
            mp_outaval = self._add_measurement_eqn(eqn)
            self.catalyst_xpr.outvars.extend(mp_outaval)
        else:
            self._convert_classical_eqn(eqn)
