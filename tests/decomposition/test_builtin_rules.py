# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests the initialization and compatibility exports of the built-in rule package."""

import importlib
import inspect
import subprocess
import sys

import pytest

import pennylane as qp
from pennylane.decomposition.decomposition_rule import _decompositions_var
from pennylane.decomposition_rules import _load_builtin_rules


def test_builtin_rule_implementations_have_moved():
    """No main-package built-in rule implementation remains in its old package."""
    original_packages = (
        "pennylane.ops.",
        "pennylane.templates.",
        "pennylane.drawer.",
        "pennylane.decomposition.",
    )
    implementations = [
        inspect.unwrap(rule).__module__
        for rules in _decompositions_var.get().values()
        for rule in rules
    ]
    assert any(module.startswith("pennylane.decomposition_rules.") for module in implementations)
    assert not any(module.startswith(original_packages) for module in implementations)


def test_loading_preserves_custom_registrations_and_exclusions():
    """Repeated imports preserve custom rules and the active registry context."""
    before = list(qp.list_decomps(qp.RX))

    @qp.register_resources({})
    def custom_rx(phi, wires):
        del phi, wires

    qp.add_decomps(qp.RX, custom_rx)
    qp.exclude_decomps(qp.RX, before[0].name)
    expected = before[1:] + [custom_rx]

    _load_builtin_rules()
    _load_builtin_rules()

    assert list(qp.list_decomps(qp.RX)) == expected


@pytest.mark.parametrize(
    ("original_module", "canonical_module", "name"),
    [
        ("decomposition", "decomposition_rules._common", "null_decomp"),
        ("decomposition.decomposition_rule", "decomposition_rules._common", "null_decomp"),
        (
            "decomposition.symbolic_decomposition",
            "decomposition_rules.op_math.symbolicop2",
            "self_adjoint",
        ),
        ("ops.op_math.adjoint2", "decomposition_rules.op_math.adjoint2", "adjoint_rotation"),
        ("ops.op_math.pow2", "decomposition_rules.op_math.pow2", "pow_rotation"),
        (
            "transforms.decompositions",
            "decomposition_rules.transforms.decompositions.rz_phase_gradient",
            "make_rz_to_phase_gradient_decomp",
        ),
        (
            "transforms.decompositions.crz_phase_gradient",
            "decomposition_rules.transforms.decompositions.crz_phase_gradient",
            "make_crz_to_phase_gradient_decomp",
        ),
        (
            "transforms.decompositions.select_pauli_rot_phase_gradient",
            "decomposition_rules.transforms.decompositions.select_pauli_rot_phase_gradient",
            "make_selectpaulirot_to_phase_gradient_decomp",
        ),
        (
            "transforms.decompositions.phase_gradient_state_prep_decomp",
            "decomposition_rules.transforms.decompositions.phase_gradient_state_prep_decomp",
            "make_phase_gradient_distillation_decomp",
        ),
    ],
)
def test_existing_public_exports(original_module, canonical_module, name):
    """Existing public paths resolve to the same relocated objects."""
    original = importlib.import_module(f"pennylane.{original_module}")
    canonical = importlib.import_module(f"pennylane.{canonical_module}")
    assert getattr(original, name) is getattr(canonical, name)


@pytest.mark.parametrize(
    "entrypoint",
    [
        "pennylane",
        "pennylane.decomposition",
        "pennylane.decomposition_rules.qubit.parametric_ops_single_qubit",
        "pennylane.ops.op_math.controlled2",
        "pennylane.templates.subroutines.qrom",
    ],
)
def test_fresh_import_registers_builtin_rules(entrypoint):
    """Each import entry point initializes operators, templates, and symbolic rules."""
    source = f"""
import importlib
importlib.import_module({entrypoint!r})
import pennylane as qp
assert qp.list_decomps(qp.RX)
assert qp.list_decomps(qp.QFT)
assert qp.list_decomps(qp.QROM)
assert qp.list_decomps(qp.adjoint(qp.RX(0.5, 0)))
assert qp.list_decomps(qp.ctrl(qp.H(0), control=1))
"""
    subprocess.run([sys.executable, "-B", "-c", source], check=True, capture_output=True, text=True)
