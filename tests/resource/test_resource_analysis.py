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
"""Tests for the resource_analysis transform."""

import json

import pytest

import pennylane as qp
from pennylane.resource import resource_analysis


class TestResourceAnalysis:
    """Tests for :func:`~.resource.resource_analysis`."""

    def test_not_implemented_on_tape(self):
        """Test that NotImplementedError is raised when applying to a tape."""
        tape = qp.tape.QuantumScript([qp.Hadamard(0)])

        with pytest.raises(
            NotImplementedError,
            match=r"Transform <transform: resource-analysis> has no defined tape implementation",
        ):
            resource_analysis(tape)

    def test_pass_name(self):
        """Test that the pass name is set on the transform."""
        assert resource_analysis.pass_name == "resource-analysis"

    def test_setup_inputs_defaults(self):
        """Test that output_json defaults to True when only a filename is given."""
        args, kwargs = resource_analysis.setup_inputs("out.json")
        assert args == ()
        assert kwargs == {"output_json": True, "output_fname": "out.json"}

        bound_t = resource_analysis("out.json")
        assert bound_t.args == ()
        assert bound_t.kwargs == {"output_json": True, "output_fname": "out.json"}

    def test_output_fname_required(self):
        """Test that output_fname is a required argument."""
        with pytest.raises(TypeError, match="output_fname"):
            resource_analysis.setup_inputs()

        with pytest.raises(TypeError, match="output_fname"):
            resource_analysis(output_json=True)

    def test_setup_inputs_to_kwargs(self):
        """Test that positional inputs are promoted to kwargs."""
        bound_t = resource_analysis("out.json", True)
        assert bound_t.args == ()
        assert bound_t.kwargs == {"output_json": True, "output_fname": "out.json"}

        bound_t = resource_analysis(output_json=False, output_fname="/tmp/resources.json")
        assert bound_t.kwargs == {"output_json": False, "output_fname": "/tmp/resources.json"}

    def test_bad_inputs(self):
        """Test that invalid option types raise ValueError."""
        with pytest.raises(TypeError, match="output_json must be of type bool"):
            resource_analysis("out.json", output_json="yes")

        with pytest.raises(TypeError, match="output_fname must be of type str"):
            resource_analysis(output_fname=None)


@pytest.mark.catalyst
def test_writes_resource_json_for_qjit_workflow(tmp_path):
    """Test that compiling a qjit workflow with resource_analysis writes the expected
    resource counts to the requested JSON file.

    ``tmp_path`` is a pytest fixture: it gives this test its own temporary directory and
    removes it afterwards, so the JSON file written by the pass is cleaned up automatically.
    """
    fname = str(tmp_path / "resources.json")

    @qp.qjit(capture=True, target="mlir")
    @resource_analysis(fname)
    @qp.qnode(qp.device("null.qubit", wires=2))
    def circuit():
        qp.Hadamard(0)
        qp.CNOT([0, 1])
        return qp.expval(qp.Z(0))

    circuit()

    with open(fname, encoding="utf-8") as f:
        data = json.load(f)

    # The JSON reports one entry per analyzed function; find the qnode's entry and check
    # that the two gates applied above were counted.
    qnode_resources = next(r for r in data.values() if r["metadata"]["qnode"])
    gate_counts = {
        gate: count
        for layer in qnode_resources["quantum_operations"].values()
        for gate, count in layer.items()
    }
    assert gate_counts == {"Hadamard": 1, "CNOT": 1}
