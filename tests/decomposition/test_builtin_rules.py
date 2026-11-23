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
"""Tests initialization and registration of the built-in rule package."""

import inspect
import subprocess
import sys

import pytest

import pennylane as qp
from pennylane.decomposition.decomposition_rule import _decompositions_var
from pennylane.decomposition_rules import _load_builtin_rules


def test_builtin_rule_implementations_have_moved():
    """Built-in rules have moved, except the drawing-specific LabelledOp rule."""
    original_packages = (
        "pennylane.ops.",
        "pennylane.templates.",
        "pennylane.drawer.",
        "pennylane.decomposition.",
    )
    registry = _decompositions_var.get()
    assert len(registry["LabelledOp"]) == 1
    assert inspect.unwrap(registry["LabelledOp"][0]).__module__ == "pennylane.drawer.label"
    implementations = [
        inspect.unwrap(rule).__module__
        for name, rules in registry.items()
        if name != "LabelledOp"
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


def test_loading_preserves_drawer_rule_exclusions():
    """The drawer initializer preserves its registered rule when it is excluded."""
    from pennylane.drawer.label import register_label_decomposition

    before = list(_decompositions_var.get()["LabelledOp"])
    qp.exclude_decomps("LabelledOp", "unwrap_base")

    register_label_decomposition()
    register_label_decomposition()
    _load_builtin_rules()

    assert not qp.list_decomps("LabelledOp")
    assert list(_decompositions_var.get()["LabelledOp"]) == before


@pytest.mark.parametrize(
    "name",
    ["_CRZPhaseGradientCompute", "_SelectPauliRotPhaseGradientCompute"],
)
def test_loading_preserves_phase_gradient_subcircuits(name):
    """Repeated loading preserves helper classes without restoring excluded rules."""
    operator = getattr(qp.decomposition_rules, name)
    rules = list(qp.list_decomps(operator))
    qp.exclude_decomps(operator, rules[0].name)

    _load_builtin_rules()
    _load_builtin_rules()

    assert getattr(qp.decomposition_rules, name) is operator
    assert not qp.list_decomps(operator)


def test_package_exports_preserve_rule_identity():
    """Package imports expose the registered objects and their original implementations."""
    from pennylane.decomposition_rules import (
        make_rz_to_phase_gradient_decomp,
        mottonen_decomp,
        null_decomp,
    )
    from pennylane.decomposition_rules.common import null_decomp as defined_null_decomp
    from pennylane.decomposition_rules.qubit.parametric_ops_single_qubit import (
        make_rz_to_phase_gradient_decomp as defined_rz_factory,
    )
    from pennylane.decomposition_rules.templates.state_preparations.mottonen import (
        mottonen_decomp as defined_mottonen_decomp,
    )

    assert null_decomp is defined_null_decomp
    assert mottonen_decomp is defined_mottonen_decomp
    assert make_rz_to_phase_gradient_decomp is defined_rz_factory


@pytest.mark.parametrize(
    "name, module",
    [
        ("make_rz_to_phase_gradient_decomp", "qubit.parametric_ops_single_qubit"),
        ("make_crz_to_phase_gradient_decomp", "op_math.controlled_ops"),
        (
            "make_selectpaulirot_to_phase_gradient_decomp",
            "templates.subroutines.select_pauli_rot",
        ),
        (
            "make_phase_gradient_distillation_decomp",
            "templates.state_preparations.phase_gradient",
        ),
        ("_CRZPhaseGradientCompute", "op_math.controlled_ops"),
        ("_SelectPauliRotPhaseGradientCompute", "templates.subroutines.select_pauli_rot"),
    ],
)
def test_phase_gradient_rules_follow_operator_modules(name, module):
    """Phase-gradient factories and helpers live beside the corresponding operator rules."""
    value = getattr(qp.decomposition_rules, name)
    assert value.__module__ == f"pennylane.decomposition_rules.{module}"


@pytest.mark.parametrize(
    "entrypoint",
    [
        "pennylane",
        "pennylane.decomposition",
        "pennylane.decomposition_rules.common",
        "pennylane.decomposition_rules.qubit.parametric_ops_single_qubit",
        "pennylane.decomposition_rules.op_math.controlled_ops",
        "pennylane.decomposition_rules.templates.subroutines.select_pauli_rot",
        "pennylane.decomposition_rules.templates.state_preparations.phase_gradient",
        "pennylane.transforms",
        "pennylane.transforms.rz_phase_gradient",
        "pennylane.drawer.label",
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
from pennylane.decomposition_rules import null_decomp, make_rz_to_phase_gradient_decomp
from pennylane.transforms.rz_phase_gradient import rz_phase_gradient
assert qp.transforms.rz_phase_gradient is rz_phase_gradient
assert [rule.name for rule in qp.list_decomps("LabelledOp")] == ["unwrap_base"]
assert null_decomp.name == "null_decomp"
assert callable(make_rz_to_phase_gradient_decomp)
assert qp.list_decomps(qp.RX)
assert qp.list_decomps(qp.QFT)
assert qp.list_decomps(qp.QROM)
assert qp.list_decomps(qp.adjoint(qp.RX(0.5, 0)))
assert qp.list_decomps(qp.ctrl(qp.H(0), control=1))
expected = ["to_unitary", "many_zeroed_work_wires", "single_work_wire"]
assert [rule.name for rule in qp.list_decomps("ControlledOp")] == expected
assert [rule.name for rule in qp.list_decomps("C(UnregisteredOp)")] == expected
assert [rule.name for rule in qp.list_decomps("Controlled(UnregisteredOp)")] == expected
"""
    subprocess.run([sys.executable, "-B", "-c", source], check=True, capture_output=True, text=True)
