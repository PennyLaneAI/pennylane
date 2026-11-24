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
"""Test the independently importable partial rule migration."""
import importlib
import inspect
import subprocess
import sys

import pytest
import pennylane as qp


@pytest.mark.parametrize("name", ['Hadamard', 'RX', 'QubitUnitary'])
def test_migrated_rule_owner(name):
    rules = list(qp.list_decomps(name))
    assert rules
    assert all(inspect.unwrap(rule).__module__.startswith("pennylane.decomposition_rules.") for rule in rules)


def test_repeated_import_preserves_registry_context():
    before = list(qp.list_decomps(qp.RX))
    with qp.exclude_decomps(qp.RX, before[0].name):
        expected = list(qp.list_decomps(qp.RX))
        importlib.import_module("pennylane.decomposition_rules")
        assert list(qp.list_decomps(qp.RX)) == expected


def test_fresh_import():
    source = "import pennylane as qp; assert qp.list_decomps(qp.RX); assert qp.list_decomps(qp.QFT)"
    subprocess.run([sys.executable, "-B", "-c", source], check=True, capture_output=True, text=True)
