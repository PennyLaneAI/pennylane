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
"""Unit tests for the JSON parsing helpers for the specs transform"""

import pytest

from pennylane.resource import PBCSpecsResources, SpecsResources
from pennylane.resource.expression import Expression
from pennylane.resource.parsing import (
    _generate_display_name_for_symbolic_var,
    _mlir_resources_to_specs_resources,
    parse_resources_json,
)


def test_generate_display_name_for_symbolic_var():
    display_names = {}

    assert _generate_display_name_for_symbolic_var("x", display_names) == "a"
    assert _generate_display_name_for_symbolic_var("y", display_names) == "b"
    assert _generate_display_name_for_symbolic_var("x", display_names) == "a"
    assert display_names == {"x": "a", "y": "b"}


class TestAnalysisPassConversion:
    @pytest.fixture
    def example_loop_analysis_pass_result(self) -> dict[str, dict]:
        """
        This test uses a snapshot from a real result of the resource analysis pass from the following snippet:

        :: code-block:: python

            @qp.qjit(autograph=True)
            @qp.qnode(qp.device("lightning.qubit", wires=10))
            def circuit(x):
                qp.Hadamard(wires=0)
                for _ in range(2):
                    qp.Hadamard(wires=0)
                    for _ in range(3):
                        qp.PauliZ(wires=0)
                    for _ in range(x):
                        qp.PauliX(wires=0)
                return qp.expval(qp.PauliZ(0))

            res = qp.specs(circuit, level=0)(x=5)
        """

        return {
            "circuit": {
                "classical_instructions": {
                    "arith.constant": 5,
                    "func.return": 1,
                    "scf.for": 1,
                    "tensor.from_elements": 1,
                },
                "extended_fields": {},
                "function_calls": {"dynamic": {}, "static": {"for_loop_2": 2}},
                "measurement_processes": {"expval(PauliZ)": 1},
                "metadata": {
                    "auto_qubit_management": False,
                    "device_name": "LightningSimulator",
                    "has_branches": False,
                    "qnode": True,
                },
                "num_qubits": {"alloc": 10, "arg": 0, "total": 10},
                "quantum_operations": {"1": {"Hadamard": 1}},
            },
            "dyn_for_loop_1": {
                "classical_instructions": {"scf.yield": 1},
                "extended_fields": {},
                "function_calls": {"dynamic": {}, "static": {}},
                "measurement_processes": {},
                "metadata": {
                    "auto_qubit_management": None,
                    "device_name": "",
                    "has_branches": False,
                    "qnode": False,
                },
                "num_qubits": {"alloc": 0, "arg": 0, "total": 0},
                "quantum_operations": {"1": {"PauliX": 1}},
            },
            "for_loop_1": {
                "classical_instructions": {"scf.yield": 1},
                "extended_fields": {},
                "function_calls": {"dynamic": {}, "static": {}},
                "measurement_processes": {},
                "metadata": {
                    "auto_qubit_management": None,
                    "device_name": "",
                    "has_branches": False,
                    "qnode": False,
                },
                "num_qubits": {"alloc": 0, "arg": 0, "total": 0},
                "quantum_operations": {"1": {"PauliZ": 1}},
            },
            "for_loop_2": {
                "classical_instructions": {
                    "arith.index_cast": 1,
                    "scf.for": 2,
                    "scf.yield": 1,
                    "tensor.extract": 1,
                },
                "extended_fields": {},
                "function_calls": {
                    "dynamic": {"dyn_for_loop_1": "0xf30441eef5432233"},
                    "static": {"for_loop_1": 3},
                },
                "measurement_processes": {},
                "metadata": {
                    "auto_qubit_management": None,
                    "device_name": "",
                    "has_branches": False,
                    "qnode": False,
                },
                "num_qubits": {"alloc": 0, "arg": 0, "total": 0},
                "quantum_operations": {"1": {"Hadamard": 1}},
            },
        }

    def test_parse_resources_json(self, example_loop_analysis_pass_result):
        actual = parse_resources_json(example_loop_analysis_pass_result)

        var = _generate_display_name_for_symbolic_var("a", {})

        assert actual == [
            SpecsResources(
                counts={"Hadamard": 3, "PauliX": Expression({(var,): 2}), "PauliZ": 6},
                measurement_processes={"expval(PauliZ)": 1},
                num_wires=10,
                circuit_depth=None,
            ),
        ]

    def test_parse_resources_json_warns_for_branches(self, example_loop_analysis_pass_result):
        example_loop_analysis_pass_result["circuit"]["metadata"]["has_branches"] = True

        with pytest.warns(UserWarning, match="branches"):
            parse_resources_json(example_loop_analysis_pass_result)

    def test_parse_resources_json_warns_for_self_recursion(self, example_loop_analysis_pass_result):
        example_loop_analysis_pass_result["circuit"]["function_calls"]["static"]["circuit"] = 1

        with pytest.warns(UserWarning, match="recursion"):
            parse_resources_json(example_loop_analysis_pass_result)

    def test_parse_resources_json_warns_for_paired_recursion(
        self, example_loop_analysis_pass_result
    ):
        example_loop_analysis_pass_result["for_loop_1"]["function_calls"]["static"][
            "for_loop_2"
        ] = 1
        example_loop_analysis_pass_result["for_loop_2"]["function_calls"]["static"][
            "for_loop_1"
        ] = 1

        with pytest.warns(UserWarning, match="recursion"):
            parse_resources_json(example_loop_analysis_pass_result)

    def test_parse_resources_json_warns_for_auto_management(
        self, example_loop_analysis_pass_result
    ):
        example_loop_analysis_pass_result["circuit"]["metadata"]["auto_qubit_management"] = True

        with pytest.warns(UserWarning, match="automatic qubit management"):
            parse_resources_json(example_loop_analysis_pass_result)

    def test_parse_resources_json_misc(self, example_loop_analysis_pass_result):
        """Extra tests for features that aren't tested in the main test"""

        # Force both a PPR and PPM to exist
        example_loop_analysis_pass_result["circuit"]["quantum_operations"]["3"] = {}
        example_loop_analysis_pass_result["circuit"]["quantum_operations"]["3"]["PPR-pi/2"] = 1
        example_loop_analysis_pass_result["circuit"]["quantum_operations"]["3"]["PPM"] = 1

        # Force a measurement inside a subroutine
        example_loop_analysis_pass_result["dyn_for_loop_1"]["measurement_processes"] = {}
        example_loop_analysis_pass_result["dyn_for_loop_1"]["measurement_processes"][
            "expval(PauliZ)"
        ] = 1

        var = _generate_display_name_for_symbolic_var("a", {})
        actual = parse_resources_json(example_loop_analysis_pass_result)

        assert actual == [
            SpecsResources(
                counts={
                    "Hadamard": 3,
                    "PPM-w3": 1,
                    "PPR-pi/2-w3": 1,
                    "PauliX": Expression({(var,): 2}),
                    "PauliZ": 6,
                },
                measurement_processes={"expval(PauliZ)": Expression({(var,): 2, (): 1})},
                num_wires=10,
                circuit_depth=None,
            ),
        ]

    def test_same_op_name_multiple_widths(self):
        """A single op name at multiple qubit widths must accumulate in counts,
        not overwrite. Regression for the 'Inconsistent counts' ValueError."""
        actual = parse_resources_json(
            {
                "circuit": {
                    "classical_instructions": {},
                    "metadata": {
                        "auto_qubit_management": False,
                        "device_name": "NullQubit",
                        "has_branches": False,
                        "qnode": True,
                    },
                    "measurement_processes": {},
                    "num_qubits": {
                        "alloc": 4,
                        "arg": 0,
                        "total": 4,
                    },
                    "quantum_operations": {
                        "1": {"Hadamard": 2},
                        "2": {"MultiControlledX": 5},
                        "3": {"MultiControlledX": 7},
                    },
                    "function_calls": {"dynamic": {}, "static": {}},
                    "extended_fields": {},
                }
            }
        )

        assert actual == [
            SpecsResources(
                counts={"Hadamard": 2, "MultiControlledX": 12},
                measurement_processes={},
                num_wires=4,
                circuit_depth=None,
            )
        ]

    def test_fractional_operation_counts_from_branch_probabilities(self):
        """Probabilistic branch weighting (from cond ``estimated_probability`` hints) can
        produce fractional operation counts, which must be preserved."""
        with pytest.warns(UserWarning, match="branch"):
            actual = parse_resources_json(
                {
                    "circuit": {
                        "classical_instructions": {},
                        "extended_fields": {},
                        "function_calls": {"dynamic": {}, "static": {}},
                        "measurement_processes": {"probs()": 1},
                        "metadata": {
                            "auto_qubit_management": False,
                            "device_name": "NullQubit",
                            "has_branches": True,
                            "qnode": True,
                        },
                        "num_qubits": {"alloc": 4, "arg": 0, "total": 4},
                        "quantum_operations": {"1": {"RY": 0.3, "RX": 0.2}},
                    }
                }
            )

        assert actual == [
            SpecsResources(
                counts={"RX": 0.2, "RY": 0.3},
                measurement_processes={"probs()": 1},
                num_wires=4,
                circuit_depth=None,
            )
        ]

    def test_fractional_function_call_count(self):
        """A float call count (e.g. a probability-weighted branch) must be treated as a
        numeric factor, not as a symbolic trip-count variable."""
        actual = parse_resources_json(
            {
                "circuit": {
                    "classical_instructions": {},
                    "extended_fields": {},
                    "function_calls": {"dynamic": {}, "static": {"sub": 2.5}},
                    "measurement_processes": {},
                    "metadata": {
                        "auto_qubit_management": False,
                        "device_name": "NullQubit",
                        "has_branches": False,
                        "qnode": True,
                    },
                    "num_qubits": {"alloc": 4, "arg": 0, "total": 4},
                    "quantum_operations": {"1": {"Hadamard": 1}},
                },
                "sub": {
                    "classical_instructions": {},
                    "extended_fields": {},
                    "function_calls": {"dynamic": {}, "static": {}},
                    "measurement_processes": {},
                    "metadata": {
                        "auto_qubit_management": None,
                        "device_name": "",
                        "has_branches": False,
                        "qnode": False,
                    },
                    "num_qubits": {"alloc": 0, "arg": 0, "total": 0},
                    "quantum_operations": {"1": {"RX": 1}, "2": {"CNOT": 1}},
                },
            }
        )

        assert actual == [
            SpecsResources(
                counts={"CNOT": 2.5, "Hadamard": 1, "RX": 2.5},
                measurement_processes={},
                num_wires=4,
                circuit_depth=None,
            )
        ]

    def test_mlir_resources_to_specs_resources(self, example_loop_analysis_pass_result):
        fn_resources = {}
        display_names = {}
        _mlir_resources_to_specs_resources(
            example_loop_analysis_pass_result,
            focus="dyn_for_loop_1",
            fn_resources=fn_resources,
            display_names=display_names,
        )
        assert fn_resources["dyn_for_loop_1"] == SpecsResources(
            counts={"PauliX": 1},
            measurement_processes={},
            num_wires=0,
            circuit_depth=None,
        )

        # This should should also resolve the recursive call to for_loop_1
        _mlir_resources_to_specs_resources(
            example_loop_analysis_pass_result,
            focus="for_loop_2",
            fn_resources=fn_resources,
            display_names=display_names,
        )

        assert fn_resources["for_loop_1"] == SpecsResources(
            counts={"PauliZ": 1},
            measurement_processes={},
            num_wires=0,
            circuit_depth=None,
        )

        assert len(display_names) == 1
        var_name = next(iter(display_names.values()))

        a = fn_resources["for_loop_2"]
        b = SpecsResources(
            counts={"PauliZ": 3, "Hadamard": 1, "PauliX": Expression({(var_name,): 1})},
            measurement_processes={},
            num_wires=0,
            circuit_depth=None,
        )

        assert a == b

    def test_extra_depth_info(self, example_loop_analysis_pass_result):
        """Test that PBC depth information is correctly extracted from the analysis pass result."""
        example_loop_analysis_pass_result["for_loop_2"]["extended_fields"]["pbc_depth"] = {
            "any_commuting_depth": 5,
            "qubit_disjoint_depth": 0,
        }
        example_loop_analysis_pass_result["for_loop_1"]["extended_fields"]["pbc_depth"] = {
            "any_commuting_depth": 2,
            "qubit_disjoint_depth": 3,
        }

        actual = parse_resources_json(example_loop_analysis_pass_result)

        var = _generate_display_name_for_symbolic_var("a", {})

        expected = [
            PBCSpecsResources(
                counts={"Hadamard": 3, "PauliX": Expression({(var,): 2}), "PauliZ": 6},
                measurement_processes={"expval(PauliZ)": 1},
                num_wires=10,
                any_commuting_depth=22,
                qubit_disjoint_depth=18,
            ),
        ]

        assert actual == expected

    def test_unknown_extended_fields(self, example_loop_analysis_pass_result):
        """Test that unknown extended fields are ignored with a warning."""
        example_loop_analysis_pass_result["for_loop_2"]["extended_fields"]["unknown_field"] = {
            "some_key": 42
        }
        example_loop_analysis_pass_result["circuit"]["extended_fields"]["unknown_field"] = {
            "foo": 15
        }

        with pytest.warns(
            UserWarning, match="Specs detected unknown extended fields in the resource data:"
        ):
            res = parse_resources_json(example_loop_analysis_pass_result)

        assert "unknown_field" in res[0].extra

        # The sub-function call is not expected to propagate since how to do this is undefined
        assert "some_key" not in res[0].extra["unknown_field"]

        # The top-level unknown field should be preserved
        assert "foo" in res[0].extra["unknown_field"]
        assert res[0].extra["unknown_field"]["foo"] == 15
