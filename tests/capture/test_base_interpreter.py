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
This submodule tests strategy structure for defining custom plxpr interpreters
"""

import pytest

import pennylane as qml

jax = pytest.importorskip("jax")

from pennylane.capture.base_interpreter import PlxprInterpreter

pytestmark = pytest.mark.jax


@pytest.fixture(autouse=True)
def enable_disable_plxpr():
    """Enable and disable the PennyLane JAX capture context manager."""
    qml.capture.enable()
    yield
    qml.capture.disable()


class SimplifyInterpreter(PlxprInterpreter):

    def interpret_operation(self, op):
        new_op = op.simplify()
        if new_op is op:
            # if new op isn't queued, need to requeue op.
            data, struct = jax.tree_util.tree_flatten(new_op)
            new_op = jax.tree_util.tree_unflatten(struct, data)
        return new_op


def test_env_and_state_initialized():
    """Test that env and state are initialized at the start."""

    interpreter = SimplifyInterpreter()
    assert interpreter._env == {}
    assert interpreter.state is None


def test_primitive_registrations():
    """Test that child primitive registrations dict's are not copied and do
    not effect PlxprInterpreeter."""

    assert (
        SimplifyInterpreter._primitive_registrations
        is not PlxprInterpreter._primitive_registrations
    )

    @SimplifyInterpreter.register_primitive(qml.X._primitive)
    def _(self, *invals, **params):
        print("in custom interpreter")
        return qml.Z(*invals)

    assert qml.X._primitive in SimplifyInterpreter._primitive_registrations
    assert qml.X._primitive not in PlxprInterpreter._primitive_registrations

    @SimplifyInterpreter()
    def f():
        qml.X(0)
        qml.Y(5)

    jaxpr = jax.make_jaxpr(f)()

    with qml.queuing.AnnotatedQueue() as q:
        jax.core.eval_jaxpr(jaxpr.jaxpr, [])

    print(jaxpr)
    print(q.queue)
    qml.assert_equal(q.queue[0], qml.Z(0))  # turned into a Y
    qml.assert_equal(q.queue[1], qml.Y(5))  # mapped wire

    # restore simplify interpreter to its previous state
    SimplifyInterpreter._primitive_registrations.pop(qml.X._primitive)


class TestHigherOrderPrimitiveRegistrations:

    @pytest.mark.parametrize("lazy", (True, False))
    def test_adjoint_transform(self, lazy):
        """Test the higher order adjoint transform."""

        @SimplifyInterpreter()
        def f(x):
            def g(y):
                qml.RX(y, 0) ** 3

            qml.adjoint(g, lazy=lazy)(x)

        jaxpr = jax.make_jaxpr(f)(0.5)

        assert jaxpr.eqns[0].params["lazy"] == lazy
        # assert jaxpr.eqns[0].primitive == adjoint_transform_primitive
        inner_jaxpr = jaxpr.eqns[0].params["jaxpr"]
        # first eqn mul, second RX
        assert inner_jaxpr.eqns[1].primitive == qml.RX._primitive
        assert len(inner_jaxpr.eqns) == 2

        with qml.queuing.AnnotatedQueue() as q:
            jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 0.5)

        if lazy:
            qml.assert_equal(q.queue[0], qml.adjoint(qml.RX(jax.numpy.array(1.5), 0)))
        else:
            qml.assert_equal(q.queue[0], qml.RX(jax.numpy.array(-1.5), 0))
