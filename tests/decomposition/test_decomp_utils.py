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

"""
Unit tests for utility functions in the ``decomposition`` module.
"""

import pytest

import pennylane as qp
from pennylane.core.operator import abstractify
from pennylane.decomposition import (
    initialize_signature_registry,
    register_signature,
    signature_registry,
)
from pennylane.decomposition.utils import translate_op_alias
from pennylane.typing import Float, Wire
from tests.core.operator.operator2_utils import (
    CompilableDynOp,
    OneWireDynOp,
    ParametrizedHybridOp,
)


@pytest.mark.unit
def test_toggle_graph_decomposition():
    """Test that the graph-based decomposition system can be toggled."""

    assert not qp.decomposition.enabled_graph()

    qp.decomposition.enable_graph()
    assert qp.decomposition.enabled_graph()

    qp.decomposition.disable_graph()
    assert not qp.decomposition.enabled_graph()

    qp.decomposition.enable_graph()
    assert qp.decomposition.enabled_graph()

    qp.decomposition.disable_graph()
    assert not qp.decomposition.enabled_graph()

    qp.decomposition.enable_graph()
    assert qp.decomposition.enabled_graph()

    qp.decomposition.disable_graph()
    assert not qp.decomposition.enabled_graph()


@pytest.mark.usefixtures("enable_and_disable_graph_decomp")
def test_graph_ctx():
    """Test that the context manager for toggling the graph works."""

    original_status = qp.decomposition.enabled_graph()

    with qp.decomposition.toggle_graph_ctx(True):
        assert qp.decomposition.enabled_graph()

    assert qp.decomposition.enabled_graph() == original_status

    with qp.decomposition.toggle_graph_ctx(False):
        assert not qp.decomposition.enabled_graph()

    assert qp.decomposition.enabled_graph() == original_status


@pytest.mark.unit
@pytest.mark.parametrize(
    "base_op_alias, expected_op_name",
    [
        ("X", "PauliX"),
        ("Y", "PauliY"),
        ("Z", "PauliZ"),
        ("I", "Identity"),
        ("H", "Hadamard"),
    ],
)
def test_translate_op_alias(base_op_alias, expected_op_name):
    """Test that operator aliases are translated to their proper names."""

    assert translate_op_alias(base_op_alias) == expected_op_name
    assert translate_op_alias(f"C({base_op_alias})") == f"C({expected_op_name})"
    assert translate_op_alias(f"Controlled({base_op_alias})") == f"C({expected_op_name})"
    assert translate_op_alias(f"Adjoint({base_op_alias})") == f"Adjoint({expected_op_name})"
    assert translate_op_alias(f"Pow({base_op_alias})") == f"Pow({expected_op_name})"


def test_translate_op_error():
    """Test that an error is raised when the symbolic operator name is not valid."""

    with pytest.raises(ValueError, match="'Adj' is not a valid name for a symbolic operator"):
        translate_op_alias("Adj(X)")


@pytest.mark.unit
class TestSignatureRegistration:
    """Tests for ``register_signature``, ``signature_registry`` and
    ``initialize_signature_registry``."""

    def test_register_operator_type(self):
        """Test that registering an operator type stores its abstract signature built from
        ``arg_specs``."""
        register_signature(OneWireDynOp)
        assert abstractify(OneWireDynOp(Float, Wire[1])) in signature_registry()[OneWireDynOp]

    def test_register_operator_instance(self):
        """Test that registering an operator instance stores its abstractified signature."""
        register_signature(OneWireDynOp(Float, Wire[1]))
        assert abstractify(OneWireDynOp(Float, Wire[1])) in signature_registry()[OneWireDynOp]

    def test_register_with_kwargs_override(self):
        """Test that keyword arguments override entries in ``arg_specs`` when registering a type."""
        register_signature(CompilableDynOp, word="XY")
        register_signature(CompilableDynOp, word="ZZ")

        registered = signature_registry()[CompilableDynOp]
        assert abstractify(CompilableDynOp(Float, "XY", Wire[1])) in registered
        assert abstractify(CompilableDynOp(Float, "ZZ", Wire[1])) in registered

    def test_registration_deduplicates(self):
        """Test that registering equivalent signatures does not create duplicate entries."""
        register_signature(OneWireDynOp)
        count = len(signature_registry()[OneWireDynOp])

        register_signature(OneWireDynOp)  # same type
        register_signature(OneWireDynOp(Float, Wire[1]))  # equivalent instance
        register_signature(OneWireDynOp(0.5, wires=0))  # concrete, abstractifies to the same

        assert len(signature_registry()[OneWireDynOp]) == count

    def test_initialize_registers_fixed_sig_operators(self):
        """Test that ``initialize_signature_registry`` registers every operator with a fixed
        signature."""
        initialize_signature_registry()
        assert abstractify(qp.Hadamard(Wire[1])) in signature_registry()[qp.Hadamard]

    def test_registry_is_read_only(self):
        """Test that the returned registry is read-only."""
        register_signature(OneWireDynOp)
        registry = signature_registry()

        with pytest.raises(TypeError, match="does not support item deletion"):
            del registry[OneWireDynOp]

        with pytest.raises(TypeError, match="does not support item assignment"):
            registry[OneWireDynOp] = 0

    def test_error_hybrid_or_static_args(self):
        """Test that signatures cannot be registered for operators with hybrid or
        static arguments."""
        with pytest.raises(ValueError, match="hybrid or non-compilable static arguments"):
            register_signature(ParametrizedHybridOp)

    def test_error_instance_with_kwargs(self):
        """Test that keyword arguments cannot be provided together with an operator instance."""
        with pytest.raises(ValueError, match="Keyword arguments can only be provided"):
            register_signature(OneWireDynOp(Float, Wire[1]), phi=Float)

    def test_invalid_signature_raises_at_construction(self):
        """Test that an invalid signature is rejected when the operator instance is constructed.
        For example, an incompatible wire count raises via the operator constructor."""
        # OneWireDynOp declares a single wire (Wire[1]).
        with pytest.raises(ValueError, match="Incorrect number of wires"):
            register_signature(OneWireDynOp, wires=Wire[2])
