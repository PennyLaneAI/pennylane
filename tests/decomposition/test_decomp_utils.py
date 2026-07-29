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
from pennylane.decomposition import register_signature, signature_registry
from pennylane.decomposition.utils import translate_op_alias
from pennylane.typing import Float, Wire
from tests.core.operator.operator2_utils import (
    CompilableDynOp,
    OneWireDynOp,
    ParametrizedHybridOp,
)


@pytest.mark.unit
def test_toggle_graph_decomposition():
    """Test the toggling of the graph-based decomposition system."""

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
    """Tests the context manager for toggling graph."""

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
    """Test the translation of operator aliases to their proper names."""

    assert translate_op_alias(base_op_alias) == expected_op_name
    assert translate_op_alias(f"C({base_op_alias})") == f"C({expected_op_name})"
    assert translate_op_alias(f"Controlled({base_op_alias})") == f"C({expected_op_name})"
    assert translate_op_alias(f"Adjoint({base_op_alias})") == f"Adjoint({expected_op_name})"
    assert translate_op_alias(f"Pow({base_op_alias})") == f"Pow({expected_op_name})"


def test_translate_op_error():
    """Tests that an error is raised when the symbolic operator name is not valid."""

    with pytest.raises(ValueError, match="'Adj' is not a valid name for a symbolic operator"):
        translate_op_alias("Adj(X)")


@pytest.mark.unit
class TestSignatureRegistration:
    """Tests for ``register_signature`` and ``signature_registry``."""

    def test_fixed_signature_auto_registered(self):
        """An operator with a fixed signature is registered automatically on definition."""
        assert OneWireDynOp.has_fixed_sig
        assert signature_registry()[OneWireDynOp] == (OneWireDynOp.arg_specs,)

    def test_register_operator_with_compilable_arg(self):
        """A signature can be registered manually for an operator with a compilable argument."""
        assert not CompilableDynOp.has_fixed_sig
        assert CompilableDynOp not in signature_registry()

        register_signature(CompilableDynOp, word="XY")
        assert signature_registry()[CompilableDynOp] == (
            {**CompilableDynOp.arg_specs, "word": "XY"},
        )

        register_signature(CompilableDynOp, word="ZZ")
        assert signature_registry()[CompilableDynOp] == (
            {**CompilableDynOp.arg_specs, "word": "XY"},
            {**CompilableDynOp.arg_specs, "word": "ZZ"},
        )

    def test_registry_returns_shallow_copy(self):
        """Mutating the returned registry does not affect the underlying registry."""
        registry = signature_registry()
        del registry[OneWireDynOp]
        assert OneWireDynOp in signature_registry()

    def test_error_hybrid_or_static_args(self):
        """Signatures cannot be registered for operators with hybrid or static arguments."""
        with pytest.raises(ValueError, match="hybrid or non-compilable static arguments"):
            register_signature(ParametrizedHybridOp)

    def test_error_incomplete_signature(self):
        """The registered signature must cover all of the operator's arguments."""
        with pytest.raises(ValueError, match="must cover all operator arguments"):
            register_signature(CompilableDynOp)  # 'word' is missing

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"phi": "not_abstract"}, "Expected an abstract type for 'phi'"),
            ({"phi": Float[-1]}, "dynamic data has fixed shapes"),
            ({"wires": "not_wires"}, "Expected an abstract type for 'wires'"),
            ({"wires": Wire[-1]}, "all wire arguments have fixed shapes"),
        ],
    )
    def test_error_invalid_arg_spec(self, kwargs, match):
        """Dynamic and wire arguments must be given fixed-shape abstract types."""

        with pytest.raises(ValueError, match=match):
            register_signature(OneWireDynOp, **kwargs)
