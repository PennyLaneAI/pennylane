# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

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
Unit tests for the sum function.
"""
import pytest
from pennylane.ops.functions import sum


def test_that_paulisum_returned_for_paulis():
    """Test that the sum function constructs a PauliSum instance when the summands are all Pauli ops."""
    

def test_that_hamiltonian_returned_for_observables():
    """Test that the sum function constructs a Hamiltonian instance when the summands are all observables."""


def test_that_sum_operator_returned_in_general():
    """Test that the sum function constructs a Sum instance in general."""


