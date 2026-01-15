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

import pennylane as qml
from pennylane.transforms.decompositions import gridsynth


class TestGridsynth:
    def test_not_implemented(self):
        """Test that NotImplementedError is raised when trying to use gridsynth on tape."""

        tape = qml.tape.QuantumScript([qml.RZ(0.5, wires=0), qml.PhaseShift(0.2, wires=0)])

        with pytest.raises(
            NotImplementedError,
            match=r"Transform <transform: gridsynth> has no defined tape implementation, and can only be applied when decorating the entire worfklow with '@qml.qjit' and when it is placed after all transforms that only have a tape implementation.",
        ):
            gridsynth(tape)

    def test_pass_name(self):
        """Test the pass name is set on the gridsynth transform."""
        assert gridsynth.pass_name == "gridsynth"

    def test_setup_inputs_to_kwargs(self):
        """Test that positional inputs are promoted to kwargs."""

        bound_t = gridsynth(1e-6)
        assert bound_t.args == ()
        assert bound_t.kwargs == {"epsilon": 1e-6, "ppr_basis": False}
