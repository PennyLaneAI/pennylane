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


class MapWiresInterpreter(PlxprInterpreter):

    def interpret_operation(self, op):
        # fairly limited use case, but good enough for testing.
        wire_map = {0: 5, 1: 6, 2: 7}
        return type(op)(*op.data, wires=tuple(wire_map[w] for w in op.wires))


def test_env_and_state_initialized():
    """Test that env and state are initialized at the start."""

    interpreter = MapWiresInterpreter()
    assert interpreter._env == {}
    assert interpreter.state is None


def test_primitive_registrations():
    """Test that child primitive registrations dict's are not copied and do
    not effect PlxprInterpreeter."""

    assert (
        MapWiresInterpreter._primitive_registrations
        is not PlxprInterpreter._primitive_registrations
    )

    @MapWiresInterpreter.register_primitive(qml.X._primitive)
    def _(self, *invals, **params):
        return qml.X(*invals)

    assert qml.X._primitive in MapWiresInterpreter._primitive_registrations
    assert qml.X._primitive not in PlxprInterpreter._primitive_registrations

    @MapWiresInterpreter()
    def f():
        qml.X(0)
        qml.Y(0)

    jaxpr = jax.make_jaxpr(f)()

    with qml.queuing.AnnotatedQueue() as q:
        jax.core.eval_jaxpr(jaxpr.jaxpr, [])

    qml.assert_equal(q.queue[0], qml.X(0))  # not mapped due to primitive registration
    qml.assert_equal(q.queue[1], qml.Y(5))  # mapped wire
