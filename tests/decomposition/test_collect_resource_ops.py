# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the CollectResourceOps interpreter."""

# pylint: disable=wrong-import-position,wrong-import-order

import pytest

import pennylane as qml

pytestmark = [pytest.mark.jax, pytest.mark.capture]

jax = pytest.importorskip("jax")

from pennylane.decomposition.collect_resource_ops import CollectResourceOps


class TestCollectResourceOps:
    """Unit tests for the CollectResourceOps interpreter."""

    @pytest.mark.unit
    def test_flat_body_fn(self):
        """Tests a function without classical structure."""

        def f(x, wires):
            qml.RX(x, wires=wires[0])
            qml.CNOT(wires=wires[:2])
            qml.MultiRZ(x * 2, wires=wires[1:])
            qml.CNOT(wires=wires[:2])
            qml.RX(x * 2, wires=wires[1])
            qml.MultiRZ(x * 2, wires=wires[2:])

        jaxpr = jax.make_jaxpr(f)(0.5, [0, 1, 2, 3])
        collector = CollectResourceOps()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, 0.5, *[0, 1, 2, 3])
        ops = collector.state["ops"]
        assert len(ops) == 4
        assert ops == {
            qml.resource_rep(qml.RX),
            qml.resource_rep(qml.CNOT),
            qml.resource_rep(qml.MultiRZ, num_wires=2),
            qml.resource_rep(qml.MultiRZ, num_wires=3),
        }

    @pytest.mark.unit
    def test_for_loop(self):
        """Tests a function with a for loop."""

        class CustomOp(qml.operation.Operator):  # pylint: disable=too-few-public-methods

            resource_keys = {"x"}

            @property
            def resource_params(self) -> dict:
                return {"x": self.parameters[0]}

        def f(x, wires):

            wires = qml.math.array(wires, like="jax")

            @qml.for_loop(3)
            def loop(i):
                qml.RX(x, wires=wires[i])
                qml.MultiRZ(x * 2, wires=[wires[i], wires[i + 1], wires[i + 2]])
                CustomOp(x * i, wires=wires[1])

            loop()

        jaxpr = jax.make_jaxpr(f)(0.5, [0, 1, 2, 3, 4])
        collector = CollectResourceOps()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, 0.5, *[0, 1, 2, 3, 4])
        ops = collector.state["ops"]
        assert len(ops) == 5
        assert ops == {
            qml.resource_rep(qml.RX),
            qml.resource_rep(qml.MultiRZ, num_wires=3),
            qml.resource_rep(CustomOp, x=0),
            qml.resource_rep(CustomOp, x=0.5),
            qml.resource_rep(CustomOp, x=1),
        }

    @pytest.mark.unit
    def test_ctrl(self):
        """Tests circuits containing control transforms."""

        def circuit(x, wires):

            def f():
                qml.X(wires[0])
                qml.RX(x, wires=wires[0])
                qml.MultiRZ(x * 2, wires=[wires[1], wires[2], wires[3]])

            qml.ctrl(f, control=wires[4])()

        jaxpr = jax.make_jaxpr(circuit)(0.5, [0, 1, 2, 3, 4])
        collector = CollectResourceOps()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, 0.5, *[0, 1, 2, 3, 4])
        ops = collector.state["ops"]
        assert len(ops) == 3
        assert ops == {
            qml.decomposition.controlled_resource_rep(qml.X, {}, 1, 0, 0),
            qml.decomposition.controlled_resource_rep(qml.RX, {}, 1, 0, 0),
            qml.decomposition.controlled_resource_rep(qml.MultiRZ, {"num_wires": 3}, 1, 0, 0),
        }

    @pytest.mark.unit
    def test_adjoint(self):
        """Tests circuits containing adjoint transforms."""

        def circuit(x, wires):

            def f():
                qml.X(wires[0])
                qml.RX(x, wires=wires[0])
                qml.MultiRZ(x * 2, wires=[wires[1], wires[2], wires[3]])

            qml.adjoint(f)()

        jaxpr = jax.make_jaxpr(circuit)(0.5, [0, 1, 2, 3, 4])
        collector = CollectResourceOps()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, 0.5, *[0, 1, 2, 3, 4])
        ops = collector.state["ops"]
        assert len(ops) == 3
        assert ops == {
            qml.decomposition.adjoint_resource_rep(qml.X, {}),
            qml.decomposition.adjoint_resource_rep(qml.RX, {}),
            qml.decomposition.adjoint_resource_rep(qml.MultiRZ, {"num_wires": 3}),
        }

    @pytest.mark.unit
    def test_cond(self):
        """Tests that all branches of a `cond` are explored."""

        def circuit(x, wires):

            wires = qml.math.array(wires, like="jax")

            def true_fn():
                qml.CRX(x, wires=wires)

            def false_fn():
                qml.CRZ(x, wires=wires)

            qml.cond(x > 0.5, true_fn, false_fn)()
            qml.cond(x > 0.5, qml.RX, qml.RY, elifs=(x > 0.2, qml.RZ))(x, wires=wires[0])

        jaxpr = jax.make_jaxpr(circuit)(0.5, [0, 1])
        collector = CollectResourceOps()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, 0.5, *[0, 1])
        ops = collector.state["ops"]
        assert len(ops) == 5
        assert ops == {
            qml.resource_rep(qml.CRX),
            qml.resource_rep(qml.CRZ),
            qml.resource_rep(qml.RX),
            qml.resource_rep(qml.RY),
            qml.resource_rep(qml.RZ),
        }
