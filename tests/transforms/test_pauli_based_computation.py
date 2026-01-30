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
"""Tests for to_ppr, commute_ppr, merge_ppr_ppm, ppm_compilation, ppr_to_ppm, and reduce_t_depth (not implemented with tape)."""


import pytest

import pennylane as qml
from pennylane.transforms.decompositions import (
    commute_ppr,
    decompose_arbitrary_ppr,
    merge_ppr_ppm,
    ppm_compilation,
    ppr_to_ppm,
    reduce_t_depth,
    to_ppr,
)

PBC_TRANSFORMS = [
    to_ppr,
    commute_ppr,
    merge_ppr_ppm,
    ppm_compilation,
    ppr_to_ppm,
    reduce_t_depth,
    decompose_arbitrary_ppr,
]
PASS_NAMES = [
    "to-ppr",
    "commute-ppr",
    "merge-ppr-ppm",
    "ppm-compilation",
    "ppr-to-ppm",
    "reduce-t-depth",
    "decompose-arbitrary-ppr",
]
PBC_TRANSFORM_DATA = list(zip(PBC_TRANSFORMS, PASS_NAMES))


@qml.qnode(qml.device("null.qubit", wires=1))
def dummy_qnode():
    return qml.probs()


@pytest.mark.parametrize("pbc_transform, pass_name", PBC_TRANSFORM_DATA)
class TestPauliBasedComputationTransforms:

    # pylint: disable=unused-argument
    def test_not_implemented(self, pbc_transform, pass_name):
        """Test that NotImplementedError is raised when trying to use Pauli-based computation
        transforms on a tape."""

        tape = qml.tape.QuantumScript([qml.T(wires=0), qml.PauliRot(0.1, "Y", wires=0)])

        with pytest.raises(
            NotImplementedError,
            match=f"Transform {pbc_transform} has no defined tape implementation",
        ):
            pbc_transform(tape)

    def test_pass_name(self, pbc_transform, pass_name):
        """Test the pass name is set on the given PBC transform."""
        assert pbc_transform.pass_name == pass_name


class TestTransformsSetup:

    def test_to_ppr_setup(self):
        """Test that to_ppr has no arguments."""

        transformed = to_ppr(dummy_qnode)
        bound_t = transformed.compile_pipeline[0]
        assert bound_t.args == ()
        assert bound_t.kwargs == {}
        assert bound_t.pass_name == "to-ppr"

    def test_commute_ppr_setup(self):
        """Test that commute_ppr has a default max_pauli_size=0."""

        transformed = commute_ppr(dummy_qnode)
        bound_t = transformed.compile_pipeline[0]
        assert bound_t.args == ()
        assert bound_t.kwargs == {"max_pauli_size": 0}
        assert bound_t.pass_name == "commute-ppr"

        with pytest.raises(ValueError, match="max_pauli_size must be an int and >= 0."):
            commute_ppr(max_pauli_size="a")
        with pytest.raises(ValueError, match="max_pauli_size must be an int and >= 0."):
            commute_ppr(max_pauli_size=-1)

    def test_merge_ppr_ppm_setup(self):
        """Test that merge_ppr_ppm has a default max_pauli_size=0."""

        transformed = merge_ppr_ppm(dummy_qnode)
        bound_t = transformed.compile_pipeline[0]
        assert bound_t.args == ()
        assert bound_t.kwargs == {"max_pauli_size": 0}
        assert bound_t.pass_name == "merge-ppr-ppm"

        with pytest.raises(ValueError, match="max_pauli_size must be an int and >= 0."):
            merge_ppr_ppm(max_pauli_size="a")
        with pytest.raises(ValueError, match="max_pauli_size must be an int and >= 0."):
            merge_ppr_ppm(max_pauli_size=-1)

    def test_ppr_to_ppm_setup(self):
        """Test that ppr_to_ppm default setup."""

        transformed = ppr_to_ppm(dummy_qnode)
        bound_t = transformed.compile_pipeline[0]
        assert bound_t.args == ()
        assert bound_t.kwargs == {"decompose_method": "pauli-corrected", "avoid_y_measure": False}
        assert bound_t.pass_name == "ppr-to-ppm"

    def test_ppm_compilation_setup(self):
        """Test the ppm_compilation default setup."""

        transformed = ppm_compilation(dummy_qnode)
        bound_t = transformed.compile_pipeline[0]
        assert bound_t.args == ()
        assert bound_t.kwargs == {
            "decompose_method": "pauli-corrected",
            "avoid_y_measure": False,
            "max_pauli_size": 0,
        }
        assert bound_t.pass_name == "ppm-compilation"

        with pytest.raises(ValueError, match="max_pauli_size must be an int and >= 0."):
            ppm_compilation(max_pauli_size="a")
        with pytest.raises(ValueError, match="max_pauli_size must be an int and >= 0."):
            ppm_compilation(max_pauli_size=-1)

    def test_reduce_t_depth_setup(self):
        """Test that ppr_to_ppm default setup."""

        transformed = reduce_t_depth(dummy_qnode)
        bound_t = transformed.compile_pipeline[0]
        assert bound_t.args == ()
        assert bound_t.kwargs == {}
        assert bound_t.pass_name == "reduce-t-depth"

    def test_decompose_arbitrary_ppr_setup(self):
        """Test that ppr_to_ppm default setup."""

        transformed = decompose_arbitrary_ppr(dummy_qnode)
        bound_t = transformed.compile_pipeline[0]
        assert bound_t.args == ()
        assert bound_t.kwargs == {}
        assert bound_t.pass_name == "decompose-arbitrary-ppr"
