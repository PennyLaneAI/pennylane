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
# pylint:disable=protected-access,unused-argument, wrong-import-position
import pytest

import pennylane as qml

jax = pytest.importorskip("jax")

from pennylane.capture.primitives import (
    adjoint_transform_prim,
    cond_prim,
    for_loop_prim,
    grad_prim,
    jacobian_prim,
    qnode_prim,
    while_loop_prim,
)
from pennylane.operation import Operation
from pennylane.transforms.decompose import (
    DynamicDecomposeInterpreter,
)

pytestmark = [pytest.mark.jax, pytest.mark.usefixtures("enable_disable_plxpr")]


class SimpleCustomOp(Operation):
    num_wires = 1
    num_params = 0

    def _init__(self, wires, id=None):
        super().__init__(wires=wires, id=id)

    def _plxpr_decomposition(self) -> "jax.core.Jaxpr":

        return jax.make_jaxpr(self._compute_plxpr_decomposition)(
            *self.parameters, wires=tuple(self.wires), **self.hyperparameters
        )

    @staticmethod
    def _compute_plxpr_decomposition(wires):
        qml.RX(0.5, wires=0)


class CustomOpCond(Operation):
    num_wires = 1
    num_params = 1

    def __init__(self, phi, wires, id=None):
        super().__init__(phi, wires=wires, id=id)

    def _plxpr_decomposition(self) -> "jax.core.Jaxpr":

        return jax.make_jaxpr(self._compute_plxpr_decomposition)(
            *self.parameters, wires=tuple(self.wires), **self.hyperparameters
        )

    @staticmethod
    def _compute_plxpr_decomposition(phi, wires):

        def true_fn(phi):
            qml.RX(phi, wires=0)

        def false_fn(phi):
            qml.RY(phi, wires=0)

        qml.cond(phi > 0.5, true_fn, false_fn)(phi)


class CustomOpForLoop(Operation):
    num_wires = 1
    num_params = 1

    def __init__(self, phi, wires, id=None):
        super().__init__(phi, wires=wires, id=id)

    def _plxpr_decomposition(self) -> "jax.core.Jaxpr":

        return jax.make_jaxpr(self._compute_plxpr_decomposition)(
            *self.parameters, wires=tuple(self.wires), **self.hyperparameters
        )

    @staticmethod
    def _compute_plxpr_decomposition(phi, wires):

        @qml.for_loop(0, 3, 1)
        def loop_rx(i, phi):
            qml.RX(phi, wires=0)
            return jax.numpy.sin(phi)

        final_x = loop_rx(phi)

        return qml.expval(qml.Z(0))


class CustomOpWhileLoop(Operation):
    num_wires = 1
    num_params = 1

    def __init__(self, phi, wires, id=None):
        super().__init__(phi, wires=wires, id=id)

    def _plxpr_decomposition(self) -> "jax.core.Jaxpr":

        return jax.make_jaxpr(self._compute_plxpr_decomposition)(
            *self.parameters, wires=tuple(self.wires), **self.hyperparameters
        )

    @staticmethod
    def _compute_plxpr_decomposition(phi, wires):

        @qml.while_loop(lambda i: i < 3)
        def loop_rx(phi):
            qml.RX(phi, wires=0)
            return jax.numpy.sin(phi)

        loop_rx(phi)

        return qml.expval(qml.Z(0))


class TestDynamicDecomposeInterpreter:

    def test_function_simple(self):
        """ """

        @DynamicDecomposeInterpreter()
        def f(x):
            qml.RY(x, wires=0)
            SimpleCustomOp(wires=0)
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)(0.5)
        assert jaxpr.eqns[0].primitive == qml.RY._primitive
        assert jaxpr.eqns[1].primitive == qml.RX._primitive

    ############################
    ### QNode
    ############################

    def test_qnode_simple(self):
        """ """

        @DynamicDecomposeInterpreter()
        @qml.qnode(device=qml.device("default.qubit", wires=2))
        def circuit(x):
            qml.RY(x, wires=0)
            SimpleCustomOp(wires=0)
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(circuit)(0.5)

        assert jaxpr.eqns[0].primitive == qnode_prim
        qfunc_jaxpr = jaxpr.eqns[0].params["qfunc_jaxpr"]
        assert qfunc_jaxpr.eqns[0].primitive == qml.RY._primitive
        assert qfunc_jaxpr.eqns[1].primitive == qml.RX._primitive
        assert qfunc_jaxpr.eqns[2].primitive == qml.PauliZ._primitive
        assert qfunc_jaxpr.eqns[3].primitive == qml.measurements.ExpectationMP._obs_primitive

    def test_qnode_cond(self):
        """ """

        @DynamicDecomposeInterpreter()
        @qml.qnode(device=qml.device("default.qubit", wires=2))
        def f(x):
            CustomOpCond(x, wires=0)
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)(0.5)

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

    def test_qnode_for_loop(self):
        """ """

        @DynamicDecomposeInterpreter()
        @qml.qnode(device=qml.device("default.qubit", wires=2))
        def f(x):
            CustomOpForLoop(x, wires=0)
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)(0.5)

        assert jaxpr.eqns[0].primitive == qnode_prim
        qfunc_jaxpr = jaxpr.eqns[0].params["qfunc_jaxpr"]
        assert qfunc_jaxpr.eqns[0].primitive == for_loop_prim
        assert qfunc_jaxpr.eqns[0].params["jaxpr_body_fn"].eqns[0].primitive == qml.RX._primitive
        assert qfunc_jaxpr.eqns[1].primitive == qml.PauliZ._primitive
        assert qfunc_jaxpr.eqns[2].primitive == qml.measurements.ExpectationMP._obs_primitive

    def test_qnode_while_loop(self):
        """ """

        @DynamicDecomposeInterpreter()
        @qml.qnode(device=qml.device("default.qubit", wires=2))
        def f(x):
            CustomOpWhileLoop(x, wires=0)
            return qml.expval(qml.Z(0))

        jaxpr = jax.make_jaxpr(f)(0.5)

        assert jaxpr.eqns[0].primitive == qnode_prim
        qfunc_jaxpr = jaxpr.eqns[0].params["qfunc_jaxpr"]
        assert qfunc_jaxpr.eqns[0].primitive == while_loop_prim
        assert qfunc_jaxpr.eqns[0].params["jaxpr_body_fn"].eqns[0].primitive == qml.RX._primitive
        assert qfunc_jaxpr.eqns[1].primitive == qml.PauliZ._primitive
        assert qfunc_jaxpr.eqns[2].primitive == qml.measurements.ExpectationMP._obs_primitive
