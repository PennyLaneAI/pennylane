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

import pennylane as qp
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


@pytest.mark.parametrize("pbc_transform, pass_name", PBC_TRANSFORM_DATA)
class TestPauliBasedComputationTransforms:

    # pylint: disable=unused-argument
    def test_not_implemented(self, pbc_transform, pass_name):
        """Test that NotImplementedError is raised when trying to use Pauli-based computation
        transforms on a tape."""

        tape = qp.tape.QuantumScript([qp.T(wires=0), qp.PauliRot(0.1, "Y", wires=0)])

        with pytest.raises(
            NotImplementedError,
            match=f"The '{pbc_transform.__name__}' compilation pass has no tape implementation",
        ):
            pbc_transform(tape)

    def test_pass_name(self, pbc_transform, pass_name):
        """Test the pass name is set on the given PBC transform."""
        assert pbc_transform.pass_name == pass_name
