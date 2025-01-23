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
"""Unit tests for the ``DecomposeInterpreter`` class"""
# pylint:disable=protected-access,unused-argument, wrong-import-position, no-value-for-parameter,
import pytest

import pennylane as qml

jax = pytest.importorskip("jax")

from pennylane.capture.primitives import (
    cond_prim,
    for_loop_prim,
    qnode_prim,
    while_loop_prim,
)
from pennylane.operation import Operation
from pennylane.transforms.decompose import (
    DynamicDecomposeInterpreter,
)

pytestmark = [pytest.mark.jax, pytest.mark.usefixtures("enable_disable_plxpr")]


class SimpleCustomOp(Operation):
    """Simple custom operation that contains a single gate in its decomposition"""

    num_wires = 1
    num_params = 1

    def _init__(self, phi, wires, id=None):
        super().__init__(phi, wires=wires, id=id)

    def _plxpr_decomposition(self) -> "jax.core.Jaxpr":

        return jax.make_jaxpr(self._compute_plxpr_decomposition)(
            *self.parameters, *self.wires, **self.hyperparameters
        )

    @staticmethod
    def _compute_plxpr_decomposition(phi, wires):
        qml.RX(phi, wires=wires)


class CustomOpCond(Operation):
    """Custom operation that contains a conditional in its decomposition"""

    num_wires = 1
    num_params = 1

    def __init__(self, phi, wires, id=None):
        super().__init__(phi, wires=wires, id=id)

    def _plxpr_decomposition(self) -> "jax.core.Jaxpr":

        return jax.make_jaxpr(self._compute_plxpr_decomposition)(
            *self.parameters, *self.wires, **self.hyperparameters
        )

    @staticmethod
    def _compute_plxpr_decomposition(phi, wires):

        def true_fn(phi, wires):
            qml.RX(phi, wires=wires)

        def false_fn(phi, wires):
            qml.RY(phi, wires=wires)

        qml.cond(phi > 0.5, true_fn, false_fn)(phi, wires)


class CustomOpForLoop(Operation):
    """Custom operation that contains a for loop in its decomposition"""

    num_wires = 1
    num_params = 1

    def __init__(self, phi, wires, id=None):
        super().__init__(phi, wires=wires, id=id)

    def _plxpr_decomposition(self) -> "jax.core.Jaxpr":

        return jax.make_jaxpr(self._compute_plxpr_decomposition)(
            *self.parameters, *self.wires, **self.hyperparameters
        )

    @staticmethod
    def _compute_plxpr_decomposition(phi, wires):

        @qml.for_loop(0, 3, 1)
        def loop_rx(i, phi):
            qml.RX(phi, wires=wires)
            return jax.numpy.sin(phi)

        # pylint: disable=unused-variable
        loop_rx(phi)

        return qml.expval(qml.Z(0))


class CustomOpWhileLoop(Operation):
    """Custom operation that contains a while loop in its decomposition"""

    num_wires = 1
    num_params = 1

    def __init__(self, phi, wires, id=None):
        super().__init__(phi, wires=wires, id=id)

    def _plxpr_decomposition(self) -> "jax.core.Jaxpr":

        return jax.make_jaxpr(self._compute_plxpr_decomposition)(
            *self.parameters, *self.wires, **self.hyperparameters
        )

    @staticmethod
    def _compute_plxpr_decomposition(phi, wires):

        @qml.while_loop(lambda i: i < 3)
        def loop_fn(i):
            qml.RX(phi, wires=wires)
            return i + 1

        _ = loop_fn(0)

        return qml.expval(qml.Z(0))


class TestDynamicDecomposeInterpreter:
    """Tests for the DynamicDecomposeInterpreter class"""

    def test_function_simple(self):
        """Test that a function with a custom operation is correctly decomposed."""

        @DynamicDecomposeInterpreter()
        def f(x):
            qml.RY(x, wires=0)
            SimpleCustomOp(x, wires=0)
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)(0.5)
        assert jaxpr.eqns[0].primitive == qml.RY._primitive
        assert jaxpr.eqns[1].primitive == qml.RX._primitive

    ############################
    ### QNode
    ############################

    @pytest.mark.parametrize("x", [0.2, 0.8])
    def test_qnode_simple(self, x):
        """Test that a QNode with a custom operation is correctly decomposed."""

        @DynamicDecomposeInterpreter()
        @qml.qnode(device=qml.device("default.qubit", wires=2))
        def circuit(x):
            qml.RY(x, wires=0)
            SimpleCustomOp(x, wires=0)
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(circuit)(x)

        assert jaxpr.eqns[0].primitive == qnode_prim
        qfunc_jaxpr = jaxpr.eqns[0].params["qfunc_jaxpr"]
        assert qfunc_jaxpr.eqns[0].primitive == qml.RY._primitive
        assert qfunc_jaxpr.eqns[1].primitive == qml.RX._primitive
        assert qfunc_jaxpr.eqns[2].primitive == qml.PauliZ._primitive
        assert qfunc_jaxpr.eqns[3].primitive == qml.measurements.ExpectationMP._obs_primitive

        result = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, x)

        @qml.qnode(device=qml.device("default.qubit", wires=2))
        def circuit_comparison(x):
            qml.RY(x, wires=0)
            qml.RX(x, wires=0)
            return qml.expval(qml.Z(0))

        assert jax.numpy.allclose(*result, circuit_comparison(x))

    @pytest.mark.parametrize("wire", [0, 1])
    @pytest.mark.parametrize("x", [0.2, 0.8])
    def test_qnode_cond(self, x, wire):
        """Test that a QNode with a conditional custom operation is correctly decomposed."""

        @DynamicDecomposeInterpreter()
        @qml.qnode(device=qml.device("default.qubit", wires=2))
        def circuit(x, wire):
            CustomOpCond(x, wires=wire)
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(circuit)(x, wire)

        assert jaxpr.eqns[0].primitive == qnode_prim
        qfunc_jaxpr = jaxpr.eqns[0].params["qfunc_jaxpr"]
        assert qfunc_jaxpr.eqns[1].primitive == cond_prim
        assert (
            qfunc_jaxpr.eqns[1].params["jaxpr_branches"][0].eqns[0].primitive == qml.RX._primitive
        )
        assert (
            qfunc_jaxpr.eqns[1].params["jaxpr_branches"][1].eqns[0].primitive == qml.RY._primitive
        )
        assert qfunc_jaxpr.eqns[2].primitive == qml.PauliZ._primitive
        assert qfunc_jaxpr.eqns[3].primitive == qml.measurements.ExpectationMP._obs_primitive

        result = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, x, wire)

        @qml.qnode(device=qml.device("default.qubit", wires=2))
        def circuit_comparison(x, wire):
            def true_fn(x, wire):
                qml.RX(x, wires=wire)

            def false_fn(x, wire):
                qml.RY(x, wires=wire)

            qml.cond(x > 0.5, true_fn, false_fn)(x, wire)

            return qml.expval(qml.Z(0))

        assert jax.numpy.allclose(*result, circuit_comparison(x, wire))

    @pytest.mark.parametrize("wire", [0, 1])
    @pytest.mark.parametrize("x", [0.2, 0.8])
    def test_qnode_for_loop(self, x, wire):
        """Test that a QNode with a for loop custom operation is correctly decomposed."""

        @DynamicDecomposeInterpreter()
        @qml.qnode(device=qml.device("default.qubit", wires=2))
        def circuit(x, wire):
            CustomOpForLoop(x, wires=wire)
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(circuit)(x, wire)

        assert jaxpr.eqns[0].primitive == qnode_prim
        qfunc_jaxpr = jaxpr.eqns[0].params["qfunc_jaxpr"]
        assert qfunc_jaxpr.eqns[0].primitive == for_loop_prim
        assert qfunc_jaxpr.eqns[0].params["jaxpr_body_fn"].eqns[0].primitive == qml.RX._primitive
        assert qfunc_jaxpr.eqns[1].primitive == qml.PauliZ._primitive
        assert qfunc_jaxpr.eqns[2].primitive == qml.measurements.ExpectationMP._obs_primitive

        result = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, x, wire)

        @qml.qnode(device=qml.device("default.qubit", wires=2))
        def circuit_comparison(x, wire):
            @qml.for_loop(0, 3, 1)
            def loop_rx(i, phi):
                qml.RX(phi, wires=wire)
                return jax.numpy.sin(phi)

            # pylint: disable=unused-variable
            loop_rx(x)

            return qml.expval(qml.Z(0))

        assert jax.numpy.allclose(*result, circuit_comparison(x, wire))

    @pytest.mark.parametrize("wire", [0, 1])
    @pytest.mark.parametrize("x", [0.2, 0.8])
    def test_qnode_while_loop(self, x, wire):
        """Test that a QNode with a while loop custom operation is correctly decomposed."""

        @DynamicDecomposeInterpreter()
        @qml.qnode(device=qml.device("default.qubit", wires=2))
        def circuit(x, wire):
            CustomOpWhileLoop(x, wires=wire)
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(circuit)(x, wire)

        assert jaxpr.eqns[0].primitive == qnode_prim
        qfunc_jaxpr = jaxpr.eqns[0].params["qfunc_jaxpr"]
        assert qfunc_jaxpr.eqns[0].primitive == while_loop_prim
        assert qfunc_jaxpr.eqns[0].params["jaxpr_body_fn"].eqns[0].primitive == qml.RX._primitive
        assert qfunc_jaxpr.eqns[1].primitive == qml.PauliZ._primitive
        assert qfunc_jaxpr.eqns[2].primitive == qml.measurements.ExpectationMP._obs_primitive

        result = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, x, wire)

        @qml.qnode(device=qml.device("default.qubit", wires=2))
        def circuit_comparison(x, wire):
            @qml.while_loop(lambda i: i < 3)
            def loop_fn(i):
                qml.RX(x, wires=wire)
                return i + 1

            _ = loop_fn(0)

            return qml.expval(qml.Z(0))

        assert jax.numpy.allclose(*result, circuit_comparison(x, wire))
