# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test for gridsynth (not implemented in tape)."""

import pytest

import pennylane as qp
from pennylane.transforms.decompositions import gridsynth


class TestGridsynth:
    def test_not_implemented(self):
        """Test that NotImplementedError is raised when trying to use gridsynth on tape."""

        tape = qp.tape.QuantumScript([qp.RZ(0.5, wires=0), qp.PhaseShift(0.2, wires=0)])

        with pytest.raises(
            NotImplementedError,
            match=r"Transform <transform: gridsynth> has no defined tape implementation, and can only be applied when decorating the entire workflow with '@qp.qjit' and when it is placed after all transforms that only have a tape implementation.",
        ):
            gridsynth(tape)

    def test_pass_name(self):
        """Test the pass name is set on the gridsynth transform."""
        assert gridsynth.pass_name == "gridsynth"

    def test_setup_inputs_to_kwargs(self):
        """Test that positional inputs are promoted to kwargs."""

        bound_t = gridsynth(1e-6)
        assert bound_t.args == ()
        assert bound_t.kwargs == {"epsilon": 1e-6, "ppr_basis": False, "method": "deterministic"}

        bound_t = gridsynth(1e-6, True, "mixed")
        assert bound_t.kwargs == {"epsilon": 1e-6, "ppr_basis": True, "method": "mixed"}

        bound_t = gridsynth(1e-6, False, "mixed-fallback")
        assert bound_t.kwargs == {"epsilon": 1e-6, "ppr_basis": False, "method": "mixed-fallback"}

    def test_bad_inputs(self):
        """Test that bad inputs raise errors."""

        with pytest.raises(ValueError, match="ppr_basis must be of type bool"):
            gridsynth(ppr_basis="a")

        with pytest.raises(ValueError, match="epsilon must be of type float."):
            gridsynth(epsilon="a")

        with pytest.raises(ValueError, match="method must be 'deterministic', 'mixed' or"):
            gridsynth(method="rus")


def _specs_t_count(method):
    """T-count that specs reports for one rotation synthesized with ``method``."""

    @qp.qjit(capture=True, target="mlir")
    @qp.transforms.gridsynth(epsilon=1e-6, method=method)
    @qp.qnode(qp.device("null.qubit", wires=1))
    def circuit(x: float):
        qp.RZ(x, 0)
        return qp.expval(qp.Z(0))

    resources = qp.specs(circuit, level="user")(1.1).resources.quantum_operations
    assert "RZ" not in resources
    return resources["T"]


@pytest.mark.catalyst
def test_mixed_method_halves_t_count_in_specs():
    """Test that specs reports roughly half the T gates for the mixed method at the same
    epsilon, using the resource hints of the gridsynth pass."""
    assert _specs_t_count("mixed") < 0.6 * _specs_t_count("deterministic")


@pytest.mark.catalyst
def test_mixed_fallback_method_reduces_t_count_in_specs():
    """Test that specs reports roughly half the T gates for the mixed fallback method compared
    to the mixed method at the same epsilon."""
    assert _specs_t_count("mixed-fallback") < 0.6 * _specs_t_count("mixed")
