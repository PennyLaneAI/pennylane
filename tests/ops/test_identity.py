# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the Identity Operator.  Others are currently in
tests/ops/qubit/test_observables.py"""

from pennylane import Identity
import numpy as np


def test_decomposition():
    """Test the decomposition of the identity operation."""

    assert Identity.compute_decomposition(wires=0) == tuple()
    assert Identity(wires=0).decomposition() == tuple()


def test_identity_eigvals(tol):
    """Test identity eigenvalues are correct"""
    res = Identity._eigvals()
    expected = np.array([1, 1])
    assert np.allclose(res, expected, atol=tol, rtol=0)


def test_label_method():
    """Test the label method for the Identity Operator"""
    assert Identity(wires=0).label() == "I"
