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
from pennylane.decomposition import register_signature, signature_registry
from pennylane.decomposition.utils import _init_signature_registration, translate_op_alias
from pennylane.typing import Float, Wire
from tests.core.operator.operator2_utils import CompilableDynOp, OneWireDynOp


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
    """Tests for ``register_signature`` and ``signature_registry``.

    Tests use a fresh, isolated registry via ``_init_signature_registration`` to isolate the registrations
    done inside the tests
    """

    def test_register_operator_type(self):
        """Test that registering an operator type stores its abstract signature built from
        ``arg_specs``."""
        register, registry = _init_signature_registration()
        register(OneWireDynOp)
        assert abstractify(OneWireDynOp(Float, Wire[1])) in registry()[OneWireDynOp]

    def test_register_operator_instance(self):
        """Test that registering a fully abstract operator instance stores its signature."""
        register, registry = _init_signature_registration()
        register(OneWireDynOp(Float, Wire[1]))
        assert abstractify(OneWireDynOp(Float, Wire[1])) in registry()[OneWireDynOp]

    def test_register_with_kwargs_override(self):
        """Test that keyword arguments override entries in ``arg_specs`` when registering a type."""
        register, registry = _init_signature_registration()
        register(CompilableDynOp, word="XY")
        register(CompilableDynOp, word="ZZ")

        registered = registry()[CompilableDynOp]
        assert abstractify(CompilableDynOp(Float, "XY", Wire[1])) in registered
        assert abstractify(CompilableDynOp(Float, "ZZ", Wire[1])) in registered

    def test_registration_deduplicates(self):
        """Test that registering equivalent signatures does not create duplicate entries, whether
        registered as a type or as an instance."""
        register, registry = _init_signature_registration()
        register(OneWireDynOp)  # type
        register(OneWireDynOp)  # same type again
        register(OneWireDynOp(Float, Wire[1]))  # equivalent instance

        assert len(registry()[OneWireDynOp]) == 1

    def test_registry_materialization_is_idempotent(self):
        """Test that accessing the registry repeatedly does not duplicate or drop signatures."""
        register, registry = _init_signature_registration()
        register(OneWireDynOp)
        first = set(registry()[OneWireDynOp])
        assert set(registry()[OneWireDynOp]) == first

    def test_fixed_sig_operators_registered_automatically(self):
        """Test that operators with a fixed signature are auto-registered in the global registry."""
        assert abstractify(qp.Hadamard(Wire[1])) in signature_registry()[qp.Hadamard]

    def test_registry_is_read_only(self):
        """Test that the returned registry is read-only."""
        register, registry = _init_signature_registration()
        register(OneWireDynOp)
        materialized = registry()

        with pytest.raises(TypeError, match="does not support item deletion"):
            del materialized[OneWireDynOp]

        with pytest.raises(TypeError, match="does not support item assignment"):
            materialized[OneWireDynOp] = 0

    def test_error_instance_with_kwargs(self):
        """Test that keyword arguments cannot be provided together with an operator instance."""
        with pytest.raises(ValueError, match="Keyword arguments can only be provided"):
            register_signature(OneWireDynOp(Float, Wire[1]), phi=Float)

    def test_error_non_abstract_instance(self):
        """Test that a non-fully-abstract operator instance cannot be registered."""
        with pytest.raises(ValueError, match="fully abstract operator instances"):
            register_signature(OneWireDynOp(0.5, wires=0))

    def test_error_incomplete_specs(self):
        """Test that a registration must cover every operator argument."""
        # CompilableDynOp's compilable ``word`` argument is not in ``arg_specs`` nor provided here.
        with pytest.raises(ValueError, match="must cover all operator arguments"):
            register_signature(CompilableDynOp)

    def test_error_non_abstract_kwarg(self):
        """Test that dynamic and wire arguments overridden via keyword must be abstract types."""
        with pytest.raises(ValueError, match="must be fully abstract"):
            register_signature(OneWireDynOp, phi=0.5)

    def test_invalid_signature_raises_at_materialization(self):
        """Test that a signature that is only invalid at construction time (e.g. an incompatible
        wire count) is rejected when the registry is materialized, not when it is registered."""
        register, registry = _init_signature_registration()
        register(OneWireDynOp, wires=Wire[2])  # cheap validation passes; error is deferred
        with pytest.raises(ValueError, match="Incorrect number of wires"):
            registry()
