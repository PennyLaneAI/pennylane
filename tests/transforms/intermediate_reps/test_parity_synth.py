# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for parity_synth (not implemented with tape)."""


import pytest

import pennylane as qp
from pennylane.transforms.intermediate_reps import parity_synth


class TestParitySynth:
    """Tests for Catalyst pass ``parity_synth``."""

    # pylint: disable=unused-argument
    def test_not_implemented(self):
        """Test that NotImplementedError is raised when trying to use ``parity_synth``
        on a tape."""

        tape = qml.tape.QuantumScript([qml.CNOT(wires=[0, 1]), qml.RZ(0.1, wires=0)])

        with pytest.raises(
            NotImplementedError,
            match="The parity_synth compilation pass has no tape implementation",
        ):
            parity_synth(tape)

    def test_pass_name(self):
        """Test the pass name is set on the given PBC transform."""
        assert parity_synth.pass_name == "parity-synth"
