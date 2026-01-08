# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Unit tests for the IPython display integration in the Estimator module.
"""
from collections import defaultdict
import pytest
from pennylane.estimator.resources_base import Resources

# pylint: disable=no-self-use, too-few-public-methods


class TestResourcesDisplay:
    """Test the IPython display integration for Estimator Resources."""

    def test_ipython_display(self, capsys):
        """Test that _ipython_display_ prints the string representation."""

        # Helper class to simulate a Gate object (required by Resources API)
        class FakeGate:
            def __init__(self, name):
                self.name = name
                self.params = {}

        # Create gate types using objects, NOT strings
        gate_types = {FakeGate("Hadamard"): 5, FakeGate("CNOT"): 3}

        resources = Resources(
            zeroed_wires=2, any_state_wires=1, algo_wires=3, gate_types=gate_types
        )

        # Call the method
        resources._ipython_display_()

        # Capture output and verify
        captured = capsys.readouterr()
        assert captured.out == str(resources) + "\n"

    def test_ipython_display_empty_resources(self, capsys):
        """Test _ipython_display_ with empty Resources object."""

        resources = Resources(zeroed_wires=0, any_state_wires=0, algo_wires=0, gate_types={})

        resources._ipython_display_()
        captured = capsys.readouterr()

        assert captured.out == str(resources) + "\n"
