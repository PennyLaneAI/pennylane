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
"""Tests for disentangle_cnot and disentangle_swap (not implemented with tape)."""


import pytest

import pennylane as qml
from pennylane.transforms import (
    disentangle_cnot,
    disentangle_swap,
)

TRANSFORMS = [
    disentangle_cnot,
    disentangle_swap,
]
PASS_NAMES = [
    "disentangle-cnot",
    "disentangle-swap",
]
TRANSFORM_DATA = list(zip(TRANSFORMS, PASS_NAMES))


@qml.qnode(qml.device("null.qubit", wires=1))
def dummy_qnode():
    return qml.probs()


@pytest.mark.parametrize("transform, pass_name", TRANSFORM_DATA)
class TestDisentangleTransforms:

    # pylint: disable=unused-argument
    def test_not_implemented(self, transform, pass_name):
        """Test that NotImplementedError is raised when trying to use disentangle_cnot/swap
        transforms on a tape."""

        tape = qml.tape.QuantumScript([qml.X(wires=0), qml.CNOT((0, 1))])

        with pytest.raises(
            NotImplementedError,
            match=f"Transform {transform} has no defined tape implementation",
        ):
            transform(tape)

    def test_pass_name(self, transform, pass_name):
        """Test the pass name is set on the given transform."""
        assert transform.pass_name == pass_name


class TestTransformsSetup:

    def test_disentangle_cnot_setup(self):
        """Test that disentangle_cnot has no arguments."""

        transformed = disentangle_cnot(dummy_qnode)
        bound_t = transformed.compile_pipeline[0]
        assert bound_t.args == ()
        assert bound_t.kwargs == {}
        assert bound_t.pass_name == "disentangle-cnot"

    def test_disentangle_swap_setup(self):
        """Test that disentangle_swap has no arguments."""

        transformed = disentangle_swap(dummy_qnode)
        bound_t = transformed.compile_pipeline[0]
        assert bound_t.args == ()
        assert bound_t.kwargs == {}
        assert bound_t.pass_name == "disentangle-swap"
