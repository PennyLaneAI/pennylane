# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the `qml.workflow.resolution._resolve_interface` helper function"""

import pytest

import pennylane as qml
from pennylane import numpy as pnp
from pennylane.math import Interface
from pennylane.tape import QuantumScript
from pennylane.workflow import _resolve_interface


def test_auto_with_numpy():
    """Test that 'auto' interface resolves to numpy correctly."""
    tapes = [
        QuantumScript([qml.RX(0.5, wires=0)], [qml.expval(qml.PauliZ(0))]),
    ]
    resolved_interface = _resolve_interface("auto", tapes)
    assert resolved_interface == Interface.NUMPY


@pytest.mark.tf
def test_auto_with_tf():
    """Test that 'auto' interface resolves to 'tf' correctly."""
    try:
        # pylint: disable=import-outside-toplevel
        import tensorflow as tf
    except ImportError:
        pytest.skip("TensorFlow is not installed.")
    tapes = [
        QuantumScript([qml.RX(tf.Variable(0.5), wires=0)], [qml.expval(qml.PauliZ(0))]),
    ]
    resolved_interface = _resolve_interface("auto", tapes)
    assert resolved_interface == Interface.TF


@pytest.mark.autograd
def test_auto_with_autograd():
    """Test that 'auto' interface resolves to 'autograd' correctly."""

    x = pnp.array([0.5], requires_grad=True)
    tapes = [
        QuantumScript([qml.RX(x, wires=0)], [qml.expval(qml.PauliZ(0))]),
    ]
    resolved_interface = _resolve_interface("auto", tapes)
    assert resolved_interface == Interface.AUTOGRAD


@pytest.mark.jax
def test_auto_with_jax():
    """Test that 'auto' interface resolves to 'jax' correctly.."""
    try:
        # pylint: disable=import-outside-toplevel
        import jax.numpy as jnp
    except ImportError:
        pytest.skip("JAX not installed.")

    tapes = [
        QuantumScript([qml.RX(jnp.array(0.5), wires=0)], [qml.expval(qml.PauliZ(0))]),
    ]
    resolved_interface = _resolve_interface("auto", tapes)
    assert resolved_interface == Interface.JAX


def test_auto_with_unsupported_interface():
    """Test that 'auto' interface resolves to None correctly."""
    # pylint: disable=import-outside-toplevel
    import networkx as nx

    # pylint: disable=too-few-public-methods
    class DummyCustomGraphOp(qml.operation.Operation):
        """Dummy custom operation for testing purposes."""

        def __init__(self, graph: nx.Graph):
            super().__init__(graph, wires=graph.nodes)

        def decomposition(self) -> list:
            return []

    graph = nx.complete_graph(3)
    tape = qml.tape.QuantumScript([DummyCustomGraphOp(graph)], [qml.expval(qml.PauliZ(0))])

    assert _resolve_interface("auto", [tape]) == Interface.NUMPY


@pytest.mark.tf
def test_tf_autograph():
    """Test that 'tf' interface resolves to 'tf-autograph' in graph mode."""
    try:
        # pylint: disable=import-outside-toplevel
        import tensorflow as tf
    except ImportError:
        pytest.skip("TensorFlow is not installed.")

    # pylint: disable=not-context-manager
    with tf.Graph().as_default():
        tapes = [
            QuantumScript([qml.RX(tf.constant(0.5), wires=0)], [qml.expval(qml.PauliZ(0))]),
        ]
        resolved_interface = _resolve_interface("tf", tapes)

    assert resolved_interface == Interface.TF_AUTOGRAPH


@pytest.mark.jax
def test_jax():
    """Test that non-abstract JAX parameters in tapes resolve to 'jax'."""
    try:
        # pylint: disable=import-outside-toplevel
        import jax.numpy as jnp
    except ImportError:
        pytest.skip("JAX not installed.")

    x = jnp.pi / 2
    tapes = [
        QuantumScript([qml.RX(x, wires=0)], [qml.expval(qml.PauliZ(0))]),
    ]
    assert not qml.math.is_abstract(x)

    resolved_interface_abstract = _resolve_interface("jax", tapes)
    assert resolved_interface_abstract == Interface.JAX


@pytest.mark.jax
def test_jax_jit():
    """Test that abstract JAX parameters in tapes resolve to 'jax-jit'."""
    try:
        # pylint: disable=import-outside-toplevel
        import jax
        import jax.numpy as jnp
    except ImportError:
        pytest.skip("JAX not installed.")

    param = jnp.array(0.5)
    assert not qml.math.is_abstract(param)

    @jax.jit
    def abstract_func(x):
        assert qml.math.is_abstract(x)
        tapes = [
            QuantumScript([qml.RX(x, wires=0)], [qml.expval(qml.PauliZ(0))]),
        ]
        assert _resolve_interface("jax", tapes) == Interface.JAX_JIT

    abstract_func(param)


def test_unsupported():
    """Test that an unsupported interface raises an error."""
    tapes = [
        QuantumScript([qml.RX(0.5, wires=0)], [qml.expval(qml.PauliZ(0))]),
    ]
    with pytest.raises(ValueError, match="'.*' is not a valid Interface."):
        _resolve_interface("unsupported_interface", tapes)
