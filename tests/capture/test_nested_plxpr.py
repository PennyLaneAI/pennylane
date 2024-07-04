# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
Tests for capturing nested plxpr.
"""
# pylint: disable=protected-access
import pytest

import pennylane as qml

pytestmark = pytest.mark.jax

jax = pytest.importorskip("jax")


@pytest.fixture(autouse=True)
def enable_disable_plxpr():
    qml.capture.enable()
    yield
    qml.capture.disable()


def test_adjoint_qfunc():
    """Test that the adjoint qfunc transform can be captured."""

    def workflow(x):
        qml.adjoint(qml.PauliRot)(x, pauli_word="XY", wires=(0, 1))

    plxpr = jax.make_jaxpr(workflow)(0.5)

    assert len(plxpr.eqns) == 1
    # assert plxpr.eqns[0].primitive == adjoint_transform.primitive

    nested_jaxpr = plxpr.eqns[0].params["jaxpr"]
    assert nested_jaxpr.eqns[0].primitive == qml.PauliRot._primitive
    assert nested_jaxpr.eqns[0].params == {"id": None, "n_wires": 2, "pauli_word": "XY"}

    assert plxpr.eqns[0].params["lazy"] is True

    with qml.queuing.AnnotatedQueue() as q:
        jax.core.eval_jaxpr(plxpr.jaxpr, plxpr.consts, 1.2)

    assert len(q) == 1
    qml.assert_equal(q.queue[0], qml.adjoint(qml.PauliRot(1.2, "XY", wires=(0, 1))))


def test_adjoint_qfunc_eager():
    """Test eager execution can be captured by the qfunc transform."""

    def workflow(x, y, z):
        qml.adjoint(qml.Rot, lazy=False)(x, y, z, 0)

    plxpr = jax.make_jaxpr(workflow)(0.5, 0.7, 0.8)

    assert len(plxpr.eqns) == 1
    # assert plxpr.eqns[0].primitive == adjoint_transform.primitive

    nested_jaxpr = plxpr.eqns[0].params["jaxpr"]
    assert nested_jaxpr.eqns[0].primitive == qml.Rot._primitive
    assert nested_jaxpr.eqns[0].params == {"n_wires": 1}

    assert plxpr.eqns[0].params["lazy"] is False

    with qml.queuing.AnnotatedQueue() as q:
        jax.core.eval_jaxpr(plxpr.jaxpr, plxpr.consts, -1.0, -2.0, -3.0)

    assert len(q) == 1
    qml.assert_equal(q.queue[0], qml.Rot(3.0, 2.0, 1.0, 0))


def test_nested_adjoint():
    """Test that adjoint can be nested multiple times."""

    def workflow(w):
        return qml.adjoint(qml.adjoint(qml.X))(w)

    plxpr = jax.make_jaxpr(workflow)(10)

    # assert plxpr.eqns[0].primitive == adjoint_transform.primitive
    # assert plxpr.eqns[0].params["jaxpr"].eqns[0].primitive == adjoint_transform.primitive
    assert (
        plxpr.eqns[0].params["jaxpr"].eqns[0].params["jaxpr"].eqns[0].primitive
        == qml.PauliX._primitive
    )

    with qml.queuing.AnnotatedQueue() as q:
        out = jax.core.eval_jaxpr(plxpr.jaxpr, plxpr.consts, 10)[0]

    qml.assert_equal(out, qml.adjoint(qml.adjoint(qml.X(10))))
    qml.assert_equal(q.queue[0], out)
