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
This module tests the to_catalyst conversion function.
"""

import pytest

import pennylane as qml

catalyst = pytest.importorskip("catalyst")
jax = pytest.importorskip("jax")

# needs to be below the importorskip calls
# pylint: disable=wrong-import-position
from pennylane.capture.to_catalyst import to_catalyst

pytestmark = [pytest.mark.external, pytest.mark.catalyst]


def catalyst_execute_jaxpr(jaxpr):

    # pylint: disable=too-few-public-methods
    class JAXPRRunner(catalyst.QJIT):
        def capture(self, args):

            result_treedef = jax.tree_util.tree_structure((0,) * len(jaxpr.out_avals))
            arg_signature = catalyst.tracing.type_signatures.get_abstract_signature(args)

            return jaxpr, result_treedef, arg_signature

    return JAXPRRunner(fn=lambda: None, compile_options=catalyst.CompileOptions())


def compare_call_jaxprs(jaxpr1, jaxpr2):
    """Compares two call jaxprs and validates that they are essentially equal."""
    for inv1, inv2 in zip(jaxpr1.invars, jaxpr2.invars):
        assert inv1.aval == inv2.aval
    for ov1, ov2 in zip(jaxpr1.outvars, jaxpr2.outvars):
        assert ov1.aval == ov2.aval
    assert len(jaxpr1.eqns) == len(jaxpr2.eqns)

    for eqn1, eqn2 in zip(jaxpr1.eqns, jaxpr2.eqns):
        assert eqn1.primitive == eqn2.primitive
        if "shots" not in eqn1.params and "shape" not in eqn1.params:
            assert eqn1.params == eqn2.params

        assert len(eqn1.invars) == len(eqn2.invars)
        for inv1, inv2 in zip(eqn1.invars, eqn2.invars):
            assert type(inv1) == type(inv2)  # pylint: disable=unidiomatic-typecheck
            assert inv1.aval == inv2.aval, f"{eqn1}, {inv1.aval}, {inv2.aval}"
            if hasattr(inv1, "val"):
                assert inv1.val == inv2.val, f"{eqn1}, {inv1.val}, {inv2.val}"
            if hasattr(inv1, "count"):
                assert inv1.count == inv2.count, f"{eqn1}, {inv1.count}, {inv2.count}"

        assert len(eqn1.outvars) == len(eqn2.outvars)
        for ov1, ov2 in zip(eqn1.outvars, eqn2.outvars):
            assert type(ov1) == type(ov2)  # pylint: disable=unidiomatic-typecheck
            assert ov1.aval == ov2.aval
            if hasattr(ov1, "count"):
                assert ov1.count == ov2.count, f"{eqn1}, {ov1.count}, {ov2.count}"


class TestCatalystCompareJaxpr:
    """Test comparing catalyst and pennylane jaxpr for a variety of situations."""

    def test_simple_qnode(self):
        """Test comparison and execution of the jaxpr for a simple qnode."""
        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.Z(0))

        qml.capture.enable()
        plxpr = jax.make_jaxpr(circuit)(0.5)
        qml.capture.disable()
        converted = to_catalyst(plxpr)

        assert converted.eqns[0].primitive == catalyst.jax_primitives.func_p
        assert converted.eqns[0].params["fn"] == circuit

        catalyst_res = catalyst_execute_jaxpr(converted)(0.5)
        assert len(catalyst_res) == 1
        assert qml.math.allclose(catalyst_res[0], jax.numpy.cos(0.5))

        qjit_obj = qml.qjit(circuit)
        qjit_obj(0.5)
        catalxpr = qjit_obj.jaxpr
        call_jaxpr_pl = converted.eqns[0].params["call_jaxpr"]
        call_jaxpr_c = catalxpr.eqns[0].params["call_jaxpr"]

        compare_call_jaxprs(call_jaxpr_pl, call_jaxpr_c)
