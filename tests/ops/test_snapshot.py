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
"""Unit tests for the Snapshot operation."""
from pennylane import Snapshot
import pennylane as qml

def test_decomposition():
    """Test the decomposition of the Snapshot operation."""

    assert Snapshot.compute_decomposition() == []
    assert Snapshot().decomposition() == []


def test_label_method():
    """Test the label method for the Snapshot operation."""
    assert Snapshot().label() == "|S|"
    assert Snapshot("my_label").label() == "|S|"


def test_control():
    """Test the _controlled method for the Snapshot operation."""
    assert isinstance(Snapshot()._controlled(0), Snapshot)
    assert Snapshot("my_label")._controlled(0).tag == Snapshot("my_label").tag


def test_adjoint():
    """Test the adjoint method for the Snapshot operation."""
    assert isinstance(Snapshot().adjoint(), Snapshot)
    assert Snapshot("my_label").adjoint().tag == Snapshot("my_label").tag

def test_snapshot_no_empty_wire_list_error():
    """Test that Snapshot does not raise an empty wire error."""
    snapshot = qml.Snapshot()
    assert isinstance(snapshot, qml.Snapshot)
