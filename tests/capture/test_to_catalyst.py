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

import numpy as np
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

            return jaxpr, None, result_treedef, arg_signature

    return JAXPRRunner(fn=lambda: None, compile_options=catalyst.CompileOptions())


def compare_call_jaxprs(jaxpr1, jaxpr2, skip_eqns=(), skip_counts=False):
    """Compares two call jaxprs and validates that they are essentially equal."""
    for inv1, inv2 in zip(jaxpr1.invars, jaxpr2.invars):
        assert inv1.aval == inv2.aval, f"{inv1.aval}, {inv2.aval}"
    for ov1, ov2 in zip(jaxpr1.outvars, jaxpr2.outvars):
        assert ov1.aval == ov2.aval
    assert len(jaxpr1.eqns) == len(jaxpr2.eqns)

    for i, (eqn1, eqn2) in enumerate(zip(jaxpr1.eqns, jaxpr2.eqns)):
        if i not in skip_eqns:
            compare_eqns(eqn1, eqn2, skip_counts=skip_counts)


def compare_eqns(eqn1, eqn2, skip_counts=False):
    assert eqn1.primitive == eqn2.primitive
    if "shots" not in eqn1.params and "shape" not in eqn1.params:
        assert eqn1.params == eqn2.params

    assert len(eqn1.invars) == len(eqn2.invars)
    for inv1, inv2 in zip(eqn1.invars, eqn2.invars):
        assert type(inv1) == type(inv2)  # pylint: disable=unidiomatic-typecheck
        assert inv1.aval == inv2.aval, f"{eqn1}, {inv1.aval}, {inv2.aval}"
        if hasattr(inv1, "val"):
            assert inv1.val == inv2.val, f"{eqn1}, {inv1.val}, {inv2.val}"
        if not skip_counts and hasattr(inv1, "count"):
            assert inv1.count == inv2.count, f"{eqn1}, {inv1.count}, {inv2.count}"

    assert len(eqn1.outvars) == len(eqn2.outvars)
    for ov1, ov2 in zip(eqn1.outvars, eqn2.outvars):
        assert type(ov1) == type(ov2)  # pylint: disable=unidiomatic-typecheck
        assert ov1.aval == ov2.aval
        if not skip_counts and hasattr(ov1, "count"):
            assert ov1.count == ov2.count, f"{eqn1}, {ov1.count}, {ov2.count}"


class TestErorrs:
    """Test that errors are raised in unsupported situations."""

    def test_dynamic_shots(self):
        """Test that a NotImplementedError is raised is shots do not match device shots."""

        dev = qml.device("lightning.qubit", wires=2, shots=50)

        @qml.qnode(dev)
        def circuit():
            return qml.sample(wires=0)

        def f():
            return circuit(shots=1000)

        qml.capture.enable()
        jaxpr = jax.make_jaxpr(f)()
        qml.capture.disable()

        with pytest.raises(NotImplementedError):
            to_catalyst(jaxpr)()

    def test_operator_without_n_wires(self):
        """Test that a NotImplementedError is raised for an operator without a n_wires parameter."""

        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.adjoint(qml.X(0))
            return qml.expval(qml.Z(0))

        qml.capture.enable()
        jaxpr = jax.make_jaxpr(circuit)()
        qml.capture.disable()

        with pytest.raises(NotImplementedError):
            to_catalyst(jaxpr)()

    def test_observable_without_n_wires(self):
        """Test that a NotImplementedError is raised for an observable without n_wires."""

        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            return qml.expval(qml.X(0) + qml.Y(0))

        qml.capture.enable()
        jaxpr = jax.make_jaxpr(circuit)()
        qml.capture.disable()

        with pytest.raises(NotImplementedError):
            to_catalyst(jaxpr)()

    def test_measuring_eigvals_not_supported(self):
        """Test that a NotImplementedError is raised for converting a measurement specified via eigvals and wires."""

        dev = qml.device("lightning.qubit", wires=2, shots=50)

        @qml.qnode(dev)
        def circuit():
            return qml.measurements.SampleMP(
                wires=qml.wires.Wires((0, 1)), eigvals=np.array([-1.0, -1.0, 1.0, 1.0])
            )

        qml.capture.enable()
        jaxpr = jax.make_jaxpr(circuit)()
        qml.capture.disable()

        with pytest.raises(NotImplementedError):
            to_catalyst(jaxpr)()

    def test_measuring_measurement_values(self):
        """Test that measuring a MeasurementValue raises a NotImplementedError."""

        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            return qml.measurements.ExpectationMP(
                obs=2
            )  # classical value like will be used for mcms

        qml.capture.enable()
        jaxpr = jax.make_jaxpr(circuit)()
        qml.capture.disable()

        with pytest.raises(NotImplementedError):
            to_catalyst(jaxpr)()

    def test_unsupported_measurement(self):
        """Test that a NotImplementedError is raised if a measurement is not yet supported for conversion."""

        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev)
        def circuit():
            return qml.vn_entropy(wires=0)

        qml.capture.enable()
        jaxpr = jax.make_jaxpr(circuit)()
        qml.capture.disable()

        with pytest.raises(NotImplementedError):
            to_catalyst(jaxpr)()


class TestCatalystCompareJaxpr:
    """Test comparing catalyst and pennylane jaxpr for a variety of situations."""

    def test_expval(self):
        """Test comparison and execution of the jaxpr for a simple qnode."""
        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.Z(0))

        qml.capture.enable()
        plxpr = jax.make_jaxpr(circuit)(0.5)
        qml.capture.disable()
        converted = to_catalyst(plxpr)(0.5)

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

    def test_probs(self):
        """Test comparison and execution of a jaxpr containing a probability measurement."""

        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.probs(wires=0)

        qml.capture.enable()
        plxpr = jax.make_jaxpr(circuit)(0.5)
        qml.capture.disable()

        converted = to_catalyst(plxpr)(0.5)

        assert converted.eqns[0].primitive == catalyst.jax_primitives.func_p
        assert converted.eqns[0].params["fn"] == circuit

        catalyst_res = catalyst_execute_jaxpr(converted)(0.5)
        assert len(catalyst_res) == 1
        expected = np.array([np.cos(0.5 / 2) ** 2, np.sin(0.5 / 2) ** 2])
        assert qml.math.allclose(catalyst_res[0], expected)

        qjit_obj = qml.qjit(circuit)
        qjit_obj(0.5)
        catalxpr = qjit_obj.jaxpr
        call_jaxpr_pl = converted.eqns[0].params["call_jaxpr"]
        call_jaxpr_c = catalxpr.eqns[0].params["call_jaxpr"]

        compare_call_jaxprs(call_jaxpr_pl, call_jaxpr_c)

    def test_state(self):
        """Test that the state can be converted to catalxpr."""

        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(phi):
            qml.Hadamard(0)
            qml.IsingXX(phi, wires=(0, 1))
            return qml.state()

        phi = np.array(-0.6234)

        qml.capture.enable()
        plxpr = jax.make_jaxpr(circuit)(phi)
        qml.capture.disable()

        converted = to_catalyst(plxpr)(phi)

        assert converted.eqns[0].primitive == catalyst.jax_primitives.func_p
        assert converted.eqns[0].params["fn"] == circuit

        catalyst_res = catalyst_execute_jaxpr(converted)(phi)
        assert len(catalyst_res) == 1

        x1 = np.cos(phi / 2) / np.sqrt(2)
        x2 = -1j * np.sin(phi / 2) / np.sqrt(2)
        expected = np.array([x1, x2, x1, x2])

        assert qml.math.allclose(catalyst_res[0], expected)

        qjit_obj = qml.qjit(circuit)
        qjit_obj(phi)
        catalxpr = qjit_obj.jaxpr
        call_jaxpr_pl = converted.eqns[0].params["call_jaxpr"]
        call_jaxpr_c = catalxpr.eqns[0].params["call_jaxpr"]

        # confused by the weak_types error here
        compare_call_jaxprs(call_jaxpr_pl, call_jaxpr_c)

    def test_variance(self):
        """Test comparison and execution of a jaxpr containing a probability measurement."""

        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.var(qml.Y(0))

        x = np.array(0.724)

        qml.capture.enable()
        plxpr = jax.make_jaxpr(circuit)(x)
        qml.capture.disable()

        converted = to_catalyst(plxpr)(np.array(0.724))

        assert converted.eqns[0].primitive == catalyst.jax_primitives.func_p
        assert converted.eqns[0].params["fn"] == circuit

        catalyst_res = catalyst_execute_jaxpr(converted)(x)
        assert len(catalyst_res) == 1
        expected = 1 - np.sin(x) ** 2
        assert qml.math.allclose(catalyst_res[0], expected)

        qjit_obj = qml.qjit(circuit)
        qjit_obj(x)
        catalxpr = qjit_obj.jaxpr
        call_jaxpr_pl = converted.eqns[0].params["call_jaxpr"]
        call_jaxpr_c = catalxpr.eqns[0].params["call_jaxpr"]

        compare_call_jaxprs(call_jaxpr_pl, call_jaxpr_c)

    def test_sample(self):
        """Test comparison and execution of a jaxpr containing a probability measurement."""

        dev = qml.device("lightning.qubit", wires=2, shots=50)

        @qml.qnode(dev)
        def circuit():
            qml.X(0)
            return qml.sample()

        qml.capture.enable()
        plxpr = jax.make_jaxpr(circuit)()
        qml.capture.disable()

        converted = to_catalyst(plxpr)()

        assert converted.eqns[0].primitive == catalyst.jax_primitives.func_p
        assert converted.eqns[0].params["fn"] == circuit

        catalyst_res = catalyst_execute_jaxpr(converted)()
        assert len(catalyst_res) == 1
        expected = np.transpose(np.vstack([np.ones(50), np.zeros(50)]))
        assert qml.math.allclose(catalyst_res[0], expected)

        qjit_obj = qml.qjit(circuit)
        qjit_obj()
        catalxpr = qjit_obj.jaxpr
        call_jaxpr_pl = converted.eqns[0].params["call_jaxpr"]
        call_jaxpr_c = catalxpr.eqns[0].params["call_jaxpr"]

        compare_call_jaxprs(call_jaxpr_pl, call_jaxpr_c)

    def test_multiple_measurements(self):
        """Test that we can convert a circuit with multiple measurement returns."""

        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x, y, z):
            qml.Rot(x, y, z, 0)
            return qml.expval(qml.X(0)), qml.expval(qml.Y(0)), qml.probs(wires=0)

        x, y, z = 0.9, 0.2, 0.5

        qml.capture.enable()
        plxpr = jax.make_jaxpr(circuit)(x, y, z)
        qml.capture.disable()

        converted = to_catalyst(plxpr)(x, y, z)

        assert converted.eqns[0].primitive == catalyst.jax_primitives.func_p
        assert converted.eqns[0].params["fn"] == circuit

        catalyst_res = catalyst_execute_jaxpr(converted)(x, y, z)
        assert len(catalyst_res) == 3

        a = np.cos(y / 2) * np.exp(-0.5j * (x + z))
        b = np.sin(y / 2) * np.exp(-0.5j * (x - z))
        state = np.array([a, b])
        expected_probs = np.abs(state) ** 2
        expected_expval_x = np.conj(state) @ qml.X.compute_matrix() @ state
        expected_expval_y = np.conj(state) @ qml.Y.compute_matrix() @ state
        assert qml.math.allclose(catalyst_res[0], expected_expval_x)
        assert qml.math.allclose(catalyst_res[1], expected_expval_y)
        assert qml.math.allclose(catalyst_res[2], expected_probs)

        qjit_obj = qml.qjit(circuit)
        qjit_obj(x, y, z)
        catalxpr = qjit_obj.jaxpr
        call_jaxpr_pl = converted.eqns[0].params["call_jaxpr"]
        call_jaxpr_c = catalxpr.eqns[0].params["call_jaxpr"]

        compare_call_jaxprs(call_jaxpr_pl, call_jaxpr_c)


class TestHybridPrograms:
    """to_catalyst conversion tests for hybrid programs."""

    def test_pre_post_processing(self):
        """Test converting a workflow with pre and post processing."""

        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x, y):
            qml.RX(x, 0)
            qml.RY(3 * y + 1, 1)
            qml.CNOT((0, 1))
            return qml.expval(qml.X(1)), qml.expval(qml.Y(0))

        def workflow(z):
            a, b = circuit(z, 2 * z)
            return a + b

        qml.capture.enable()
        plxpr = jax.make_jaxpr(workflow)(0.5)
        qml.capture.disable()

        converted = to_catalyst(plxpr)(0.5)

        res = catalyst_execute_jaxpr(converted)(0.5)

        x = 0.5
        y = 3 * 2 * 0.5 + 1

        expval_x1 = np.sin(y)
        expval_y0 = -np.sin(x) * np.sin(y)
        expected = expval_x1 + expval_y0

        assert qml.math.allclose(expected, res[0])

        qjit_obj = qml.qjit(workflow)
        qjit_obj(0.5)

        call_jaxpr_pl = converted.eqns[1].params["call_jaxpr"]
        call_jaxpr_c = qjit_obj.jaxpr.eqns[1].params["call_jaxpr"]

        # qubit extraction and classical equations in a slightly different order
        # thus cant check specific equations and have to discard comparing counts
        compare_call_jaxprs(call_jaxpr_pl, call_jaxpr_c, skip_eqns=(4, 5, 6), skip_counts=True)
        compare_eqns(call_jaxpr_pl.eqns[4], call_jaxpr_c.eqns[5], skip_counts=True)
        compare_eqns(call_jaxpr_pl.eqns[5], call_jaxpr_c.eqns[6], skip_counts=True)
        compare_eqns(call_jaxpr_pl.eqns[6], call_jaxpr_c.eqns[4], skip_counts=True)
