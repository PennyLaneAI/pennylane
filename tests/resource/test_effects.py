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
"""Unit tests for the declared resource effects of compilation passes."""

from math import log2

import pytest

from pennylane.resource.effects import (
    FunctionResources,
    Op,
    Wires,
    apply_effect,
    get_effect,
    substitution_effect,
)
from pennylane.resource.parsing import parse_resources_json


def function(ops, calls=None, qnode=False, **num_qubits):
    """The resources of a function in the JSON format of the resource analysis, with the
    ``alloc``, ``arg``, ``peak`` and ``live_at_call`` qubit counts given by ``num_qubits``."""
    quantum_operations = {}
    for (name, wires), count in ops.items():
        quantum_operations.setdefault(str(wires), {})[name] = count
    alloc, arg = num_qubits.get("alloc", 0), num_qubits.get("arg", 0)
    return {
        "metadata": {"qnode": qnode, "has_branches": False, "device_name": None},
        "num_qubits": {
            "alloc": alloc,
            "arg": arg,
            "total": alloc + arg,
            "peak": num_qubits.get("peak", 0),
            "live_at_call": num_qubits.get("live_at_call", {}),
        },
        "classical_instructions": {},
        "quantum_operations": quantum_operations,
        "function_calls": {"static": calls or {}, "dynamic": {}},
        "measurement_processes": {},
        "extended_fields": {},
    }


def ops_of(data):
    return {(op.name, op.wires): count for op, count in FunctionResources(data).ops.items()}


class TestEffects:
    """Test the resource effect classes."""

    def test_substitution(self):
        """Test that a substitution replaces each operator, keeps the others, and accounts for
        the allocations of the replacements."""

        def rule(op, options):
            if op.name == "A":
                return {Op("B", 1): options["n"]}, Wires(alloc=1, peak=1)
            return None

        data = {"f": function({("A", 1): 3, ("C", 2): 1}, alloc=5, peak=5)}
        new_data = apply_effect(data, substitution_effect(rule), {"n": 2})

        assert ops_of(new_data["f"]) == {("B", 1): 6, ("C", 2): 1}
        assert new_data["f"]["num_qubits"]["alloc"] == 8
        assert new_data["f"]["num_qubits"]["total"] == 8
        assert new_data["f"]["num_qubits"]["peak"] == 6
        # The input is not modified
        assert ops_of(data["f"]) == {("A", 1): 3, ("C", 2): 1}

    def test_substitution_peak_is_the_largest(self):
        """Test that the extra qubits of different operators do not add up, since each operator
        releases them."""

        def rule(op, _options):
            return {}, Wires(alloc=1, peak={"A": 1, "B": 2}[op.name])

        data = {"f": function({("A", 1): 3, ("B", 1): 1}, peak=4)}
        assert apply_effect(data, substitution_effect(rule), {})["f"]["num_qubits"]["peak"] == 6

    def test_function_effect(self):
        """Test that an effect is applied to each function."""

        def effect(resources, _options):
            resources.ops = {op: 2 * count for op, count in resources.ops.items()}

        data = {"f": function({("A", 1): 1}), "g": function({("B", 2): 3})}
        new_data = apply_effect(data, effect, {})
        assert ops_of(new_data["f"]) == {("A", 1): 2}
        assert ops_of(new_data["g"]) == {("B", 2): 6}


class TestPassEffects:
    """Test the declared resource effects of the built-in passes."""

    def test_to_ppr(self):
        """Test that Clifford+T gates become PPRs with a global phase each."""
        data = {
            "f": function(
                {
                    ("Hadamard", 1): 1,
                    ("Adjoint(T)", 1): 2,
                    ("CNOT", 2): 1,
                    ("PauliX", 1): 1,
                    ("MidCircuitMeasure", 1): 1,
                    ("Identity", 1): 4,
                    ("GlobalPhase", 0): 1,
                }
            )
        }
        new_data = apply_effect(data, get_effect("to-ppr"), {})
        assert ops_of(new_data["f"]) == {
            ("PPR-pi/4", 1): 5,
            ("PPR-pi/8", 1): 2,
            ("PPR-pi/4", 2): 1,
            ("PPR-pi/2", 1): 1,
            ("PPM", 1): 1,
            ("GlobalPhase", 0): 6,
        }

    def test_ppr_to_ppm(self):
        """Test that each non-Pauli PPR becomes a gadget with an ancilla."""
        data = {"f": function({("PPR-pi/8", 1): 2, ("PPR-pi/4", 2): 3, ("PPR-pi/2", 1): 1}, peak=3)}
        new_data = apply_effect(data, get_effect("ppr-to-ppm"), {})
        assert ops_of(new_data["f"]) == {
            ("pbc.fabricate", 0): 2,
            ("PPM", 2): 2,
            ("PPM", 1): 5,
            ("PPR-pi/2", 1): 3,
            ("PPM", 3): 3,
            ("PPR-pi/2", 2): 3,
        }
        assert new_data["f"]["num_qubits"]["alloc"] == 5
        assert new_data["f"]["num_qubits"]["peak"] == 4

    def test_ppr_to_ppm_unsupported_options(self):
        """Test that an error is raised for the options without a declared effect."""
        data = {"f": function({("PPR-pi/8", 1): 1})}
        with pytest.raises(NotImplementedError, match="only declared for"):
            apply_effect(data, get_effect("ppr-to-ppm"), {"avoid_y_measure": True})

    def test_combine_global_phases(self):
        """Test that the global phases of each function are combined into one."""
        data = {
            "f": function({("GlobalPhase", 0): 3, ("Adjoint(GlobalPhase)", 0): 1}),
            "g": function({("Hadamard", 1): 1}),
        }
        new_data = apply_effect(data, get_effect("combine-global-phases"), {})
        assert ops_of(new_data["f"]) == {("GlobalPhase", 0): 1}
        assert ops_of(new_data["g"]) == {("Hadamard", 1): 1}

    @pytest.mark.parametrize("ppr_basis", [False, True])
    def test_gridsynth_deterministic(self, ppr_basis):
        """Test that each rotation becomes the expected gate sequence of the runtime, whose
        number of T gates is linear in log2(1/epsilon)."""
        epsilon = 1e-6
        data = {"f": function({("RZ", 1): 2, ("Hadamard", 1): 1})}
        options = {"epsilon": epsilon, "ppr_basis": ppr_basis, "method": "deterministic"}
        new_ops = ops_of(apply_effect(data, get_effect("gridsynth"), options)["f"])

        num_t = new_ops[("PPR-pi/8", 1)] if ppr_basis else new_ops[("T", 1)]
        assert num_t == pytest.approx(2 * (3.0590 * log2(1 / epsilon) + 1.1278))
        assert new_ops[("GlobalPhase", 0)] == 2
        assert ("RZ", 1) not in new_ops
        # The sequences in the PPR basis are only pi/8 PPRs, and in the Clifford+T basis they
        # also have Clifford gates
        assert (new_ops[("Hadamard", 1)] > 1) != ppr_basis

    def test_gridsynth_mixed_fallback(self):
        """Test that the mixed fallback method uses an ancilla per rotation."""
        data = {"f": function({("RZ", 1): 3}, peak=2)}
        options = {"epsilon": 1e-4, "ppr_basis": False, "method": "mixed-fallback"}
        new_data = apply_effect(data, get_effect("gridsynth"), options)
        new_ops = ops_of(new_data["f"])
        assert new_ops[("CY", 2)] == 6
        assert new_ops[("MidCircuitMeasure", 1)] == 3
        assert new_data["f"]["num_qubits"]["alloc"] == 3
        assert new_data["f"]["num_qubits"]["peak"] == 3


def test_peak_wires():
    """Test that the peak number of live qubits accounts for the qubits live at each call."""
    data = {
        "main": function(
            {("Hadamard", 1): 1},
            alloc=3,
            peak=3,
            arg=1,
            calls={"g": 2, "h": 1},
            live_at_call={"g": 3, "h": 1},
            qnode=True,
        ),
        "g": function({("CNOT", 2): 1}, alloc=1, peak=1),
        "h": function({}, alloc=4, peak=4),
    }
    (resources,) = parse_resources_json(data, peak_wires=True)
    # max(3, 3 + 1, 1 + 4) live qubits, plus the argument qubit
    assert resources.extra["qubit_liveness"] == {"peak_wires": 6}
    # The total number of allocations is unchanged
    assert resources.num_wires == 4 + 2 * 1 + 4
