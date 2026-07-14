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
"""Unit tests for the MLIR helpers for the specs transform"""

import pytest

from pennylane.resource import SpecsResources
from pennylane.resource.expression import Expression
from pennylane.resource.mlir_specs import (
    _generate_display_name_for_symbolic_var,
    _get_resources_from_analysis_pass,
    _mlir_resources_to_specs_resources,
    make_level_name_unique,
)


def test_make_level_name_unique():
    existing_levels = {"foo", "foo-2", "bar"}

    assert make_level_name_unique("foo", existing_levels) == "foo-3"
    assert make_level_name_unique("bar", existing_levels) == "bar-2"
    assert make_level_name_unique("baz", existing_levels) == "baz"


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
                "auto_qubit_management": False,
                "classical_instructions": {
                    "arith.index_cast": 3,
                    "func.return": 1,
                    "scf.for": 1,
                    "stablehlo.constant": 4,
                    "tensor.extract": 8,
                    "tensor.from_elements": 1,
                },
                "device_name": "LightningSimulator",
                "function_calls": {"for_loop_2": 2},
                "has_branches": False,
                "measurements": {"expval(PauliZ)": 1},
                "num_alloc_qubits": 10,
                "num_arg_qubits": 0,
                "num_qubits": 10,
                "operations": {"Hadamard(1)": 1},
                "qnode": True,
                "var_function_calls": {},
            },
            "dyn_for_loop_1": {
                "classical_instructions": {
                    "arith.index_cast": 1,
                    "scf.yield": 1,
                    "stablehlo.constant": 1,
                    "tensor.extract": 2,
                    "tensor.from_elements": 1,
                },
                "device_name": "",
                "function_calls": {},
                "has_branches": False,
                "measurements": {},
                "num_alloc_qubits": 0,
                "num_arg_qubits": 0,
                "num_qubits": 0,
                "operations": {"PauliX(1)": 1},
                "qnode": False,
                "var_function_calls": {},
            },
            "for_loop_1": {
                "classical_instructions": {
                    "arith.index_cast": 1,
                    "scf.yield": 1,
                    "stablehlo.constant": 1,
                    "tensor.extract": 2,
                    "tensor.from_elements": 1,
                },
                "device_name": "",
                "function_calls": {},
                "has_branches": False,
                "measurements": {},
                "num_alloc_qubits": 0,
                "num_arg_qubits": 0,
                "num_qubits": 0,
                "operations": {"PauliZ(1)": 1},
                "qnode": False,
                "var_function_calls": {},
            },
            "for_loop_2": {
                "classical_instructions": {
                    "arith.index_cast": 7,
                    "scf.for": 2,
                    "scf.yield": 1,
                    "stablehlo.constant": 3,
                    "tensor.extract": 8,
                    "tensor.from_elements": 1,
                },
                "device_name": "",
                "function_calls": {"for_loop_1": 3},
                "has_branches": False,
                "measurements": {},
                "num_alloc_qubits": 0,
                "num_arg_qubits": 0,
                "num_qubits": 0,
                "operations": {"Hadamard(1)": 1},
                "qnode": False,
                "var_function_calls": {"dyn_for_loop_1": "a"},
            },
        }

    def test_get_resources_from_analysis_pass(self, example_loop_analysis_pass_result):
        actual = _get_resources_from_analysis_pass(example_loop_analysis_pass_result)

        var = _generate_display_name_for_symbolic_var("a", {})

        assert actual == [
            SpecsResources(
                counts={"Hadamard": 3, "PauliX": Expression({(var,): 2}), "PauliZ": 6},
                gate_sizes={1: Expression({(var,): 2, (): 9})},
                measurements={"expval(PauliZ)": 1},
                num_allocs=10,
                circuit_depth=None,
            ),
        ]

    def test_get_resources_from_analysis_pass_warns_for_branches(
        self, example_loop_analysis_pass_result
    ):
        example_loop_analysis_pass_result["circuit"]["has_branches"] = True

        with pytest.warns(UserWarning, match="branches"):
            _get_resources_from_analysis_pass(example_loop_analysis_pass_result)

    def test_get_resources_from_analysis_pass_warns_for_self_recursion(
        self, example_loop_analysis_pass_result
    ):
        example_loop_analysis_pass_result["circuit"]["function_calls"]["circuit"] = 1

        with pytest.warns(UserWarning, match="recursion"):
            _get_resources_from_analysis_pass(example_loop_analysis_pass_result)

    def test_get_resources_from_analysis_pass_warns_for_paired_recursion(
        self, example_loop_analysis_pass_result
    ):
        example_loop_analysis_pass_result["for_loop_1"]["function_calls"]["for_loop_2"] = 1
        example_loop_analysis_pass_result["for_loop_2"]["function_calls"]["for_loop_1"] = 1

        with pytest.warns(UserWarning, match="recursion"):
            _get_resources_from_analysis_pass(example_loop_analysis_pass_result)

    def test_get_resources_from_analysis_pass_warns_for_auto_management(
        self, example_loop_analysis_pass_result
    ):
        example_loop_analysis_pass_result["circuit"]["auto_qubit_management"] = True

        with pytest.warns(UserWarning, match="automatic qubit management"):
            _get_resources_from_analysis_pass(example_loop_analysis_pass_result)

    def test_get_resources_from_analysis_pass_misc(self, example_loop_analysis_pass_result):
        """Extra tests for features that aren't tested in the main test"""

        # Force both a PPR and PPM to exist
        example_loop_analysis_pass_result["circuit"]["operations"]["PPR-pi/2(3)"] = 1
        example_loop_analysis_pass_result["circuit"]["operations"]["PPM(3)"] = 1

        # Force a measurement inside a subroutine
        example_loop_analysis_pass_result["dyn_for_loop_1"]["measurements"]["expval(PauliZ)"] = 1

        var = _generate_display_name_for_symbolic_var("a", {})
        actual = _get_resources_from_analysis_pass(example_loop_analysis_pass_result)

        assert actual == [
            SpecsResources(
                counts={
                    "Hadamard": 3,
                    "PPM-w3": 1,
                    "PPR-pi/2-w3": 1,
                    "PauliX": Expression({(var,): 2}),
                    "PauliZ": 6,
                },
                gate_sizes={1: Expression({(var,): 2, (): 9}), 3: 2},
                measurements={"expval(PauliZ)": Expression({(var,): 2, (): 1})},
                num_allocs=10,
                circuit_depth=None,
            ),
        ]

    def test_same_op_name_multiple_widths(self):
        """A single op name at multiple qubit widths must accumulate in counts,
        not overwrite. Regression for the 'Inconsistent gate counts' ValueError."""
        actual = _get_resources_from_analysis_pass(
            {
                "circuit": {
                    "auto_qubit_management": False,
                    "classical_instructions": {},
                    "device_name": "NullQubit",
                    "function_calls": {},
                    "has_branches": False,
                    "measurements": {},
                    "num_alloc_qubits": 4,
                    "num_arg_qubits": 0,
                    "num_qubits": 4,
                    "operations": {
                        "MultiControlledX(2)": 5,
                        "MultiControlledX(3)": 7,
                        "Hadamard(1)": 2,
                    },
                    "qnode": True,
                    "var_function_calls": {},
                }
            }
        )

        assert actual == [
            SpecsResources(
                counts={"Hadamard": 2, "MultiControlledX": 12},
                gate_sizes={1: 2, 2: 5, 3: 7},
                measurements={},
                num_allocs=4,
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
            gate_sizes={1: 1},
            measurements={},
            num_allocs=0,
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
            gate_sizes={1: 1},
            measurements={},
            num_allocs=0,
            circuit_depth=None,
        )

        assert len(display_names) == 1
        var_name = next(iter(display_names.values()))

        a = fn_resources["for_loop_2"]
        b = SpecsResources(
            counts={"PauliZ": 3, "Hadamard": 1, "PauliX": Expression({(var_name,): 1})},
            gate_sizes={1: Expression({(var_name,): 1, (): 4})},
            measurements={},
            num_allocs=0,
            circuit_depth=None,
        )

        assert a == b
