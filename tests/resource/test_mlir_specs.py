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
"""Unit tests for the specs transform"""

# pylint: disable=invalid-sequence-index

from pathlib import Path
from types import SimpleNamespace

import pytest

import pennylane as qp
from pennylane.resource import mlir_specs
from pennylane.resource.mlir_specs import resources_from_analysis_pass

catalyst = pytest.importorskip("catalyst")

pytestmark = pytest.mark.catalyst


@pytest.mark.parametrize(
    "level",
    [
        2,
        3,
        [3, 4],
    ],
)
def test_resources_from_analysis_pass_invalid_levels(level):
    """Test that resources_from_analysis_pass raises an error when invalid levels are provided."""

    @qp.qjit
    @qp.qnode(qp.device("lightning.qubit", wires=1))
    def circuit():
        qp.RX(0.5, 0)
        qp.Hadamard(wires=0)
        return qp.expval(qp.PauliZ(0))

    with pytest.raises(ValueError, match="Requested specs levels .* not found"):
        resources_from_analysis_pass(
            circuit,
            original_qnode=circuit.original_function,
            level=level,
            level_to_markers={},
            level_to_name={},
        )


@pytest.mark.parametrize("has_native_evaluator", [True, False])
def test_native_evaluation_options_at_all_levels(monkeypatch, has_native_evaluator):
    """All analysis stages receive native paths only when Catalyst supports them."""
    native_options = {"libQPD_path": "/compiler/lib", "libpython_path": "/python/library"}

    def import_options(name):
        assert name == "catalyst.utils.pbc_classical"
        if not has_native_evaluator:
            raise ImportError("Older Catalyst has no native classical evaluator")
        return SimpleNamespace(resource_analysis_options=lambda: native_options)

    monkeypatch.setattr(mlir_specs, "import_module", import_options)
    recorded_options = []

    def fake_qjit(qnode, compile_options):
        assert compile_options.target == "mlir_opt"

        def compile_circuit():
            for comp_pass in (*qnode.compile_pipeline, *qnode.post_device_pipeline):
                if comp_pass.pass_name == "resource-analysis":
                    recorded_options.append(comp_pass.kwargs)
                    Path(comp_pass.kwargs["output_fname"]).write_text("{}", encoding="utf-8")

        return SimpleNamespace(aot_compile=compile_circuit)

    monkeypatch.setattr(catalyst, "QJIT", fake_qjit)

    @qp.transforms.cancel_inverses
    @qp.qnode(qp.device("null.qubit", wires=1))
    def circuit():
        return qp.expval(qp.Z(0))

    results = resources_from_analysis_pass(
        SimpleNamespace(compile_options=SimpleNamespace(pipelines=None)),
        original_qnode=circuit,
        level=[0, 1, 2],
        level_to_markers={},
        level_to_name={},
    )
    assert len(results) == len(recorded_options) == 3
    expected = native_options if has_native_evaluator else {}
    for options in recorded_options:
        assert options["output_json"] is True
        assert {
            k: v for k, v in options.items() if k not in ("output_json", "output_fname")
        } == expected
