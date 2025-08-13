# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

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
Unit tests for the `zx.todd` transform.
"""
import sys

import pytest

import pennylane as qml
from pennylane.tape import QuantumScript

pytest.importorskip("pyzx")


def test_import_pyzx_error(monkeypatch):
    """Test that a ModuleNotFoundError is raised by the todd transform
    when the pyzx external package is not installed."""

    with monkeypatch.context() as m:
        m.setitem(sys.modules, "pyzx", None)

        qs = QuantumScript(ops=[], measurements=[])

        with pytest.raises(ModuleNotFoundError, match="The `pyzx` package is required."):
            qml.transforms.zx.push_hadamards(qs)


@pytest.mark.external
class TestTODD:

    @pytest.mark.parametrize(
        "num_gates, expected_ops",
        (
            (1, [qml.S(0)]),
            (2, [qml.Z(0)]),
            (4, []),
        ),
    )
    def test_S_gate_simplification(self, num_gates, expected_ops):
        """Test S gate simplification/cancellation."""
        ops = [qml.S(0)] * num_gates

        qs = QuantumScript(ops)
        (new_qs,), _ = qml.transforms.zx.todd(qs)

        assert new_qs.operations == expected_ops

    @pytest.mark.parametrize(
        "num_gates, expected_ops",
        (
            (1, [qml.T(0)]),
            (2, [qml.S(0)]),
            (4, [qml.Z(0)]),
            (8, []),
        ),
    )
    def test_T_gate_simplification(self, num_gates, expected_ops):
        """Test T gate simplification/cancellation."""
        ops = [qml.T(0)] * num_gates

        qs = QuantumScript(ops)
        (new_qs,), _ = qml.transforms.zx.todd(qs)

        assert new_qs.operations == expected_ops

    @pytest.mark.parametrize(
        "gate",
        (
            # non-Clifford or T gates
            qml.RX(0.5, wires=0),
            qml.RY(0.5, wires=0),
            qml.RZ(0.5, wires=0),
            qml.U1(0.1, wires=0),
            qml.U2(0.1, 0.2, wires=0),
            qml.U3(0.1, 0.2, 0.3, wires=0),
            qml.CRX(0.5, wires=[0, 1]),
            qml.CRY(0.5, wires=[0, 1]),
            qml.CRZ(0.5, wires=[0, 1]),
        ),
    )
    def test_non_clifford_or_T_gates_error(self, gate):
        qs = QuantumScript(ops=[gate])

        with pytest.raises(
            TypeError,
            match=r"The input quantum circuit must be a Clifford \+ T circuit.",
        ):
            qml.transforms.zx.todd(qs)
