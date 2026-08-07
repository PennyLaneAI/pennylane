# Copyright 2018-2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the specs transform"""

# pylint: disable=invalid-sequence-index
from functools import partial

import pytest

import pennylane as qp
from pennylane import numpy as pnp
from pennylane.core.shots import Shots
from pennylane.resource import SpecsResources

catalyst = pytest.importorskip("catalyst")

pytestmark = pytest.mark.catalyst


@pytest.mark.usefixtures("enable_and_disable_graph_decomp")
class TestSpecsTransform:
    """Tests for the transform specs using the QNode"""

    @pytest.mark.catalyst
    @pytest.mark.parametrize("level", [0, "device"])
    def test_qjit_partial(self, level):
        """Test specs for a partial-wrapped Catalyst jitted QNode."""

        @qp.qjit
        @qp.qnode(qp.device("lightning.qubit", wires=1))
        def circuit(x, y, z):
            qp.RX(x, wires=0)
            qp.RY(y, wires=0)
            qp.RZ(z, wires=0)
            return qp.expval(qp.Z(0))

        resources = qp.specs(partial(circuit, 0.1, z=0.3), level=level)(0.2)["resources"]

        assert resources.counts == {"RX": 1, "RY": 1, "RZ": 1}
        assert resources.total_quantum_operations == 3

    @pytest.mark.catalyst
    def test_qjit_partial_all_levels(self):
        """Test all-level specs for a partial-wrapped Catalyst jitted QNode."""

        @qp.qjit
        @qp.qnode(qp.device("lightning.qubit", wires=1))
        def circuit(x, y, z):
            qp.RX(x, wires=0)
            qp.RY(y, wires=0)
            qp.RZ(z, wires=0)
            return qp.expval(qp.Z(0))

        specs = qp.specs(partial(circuit, 0.1, z=0.3), level="all")(0.2)

        assert specs["level"] == {0: "Before MLIR Passes"}
        resources = specs["resources"]["Before MLIR Passes"]
        assert resources.counts == {"RX": 1, "RY": 1, "RZ": 1}
        assert resources.total_quantum_operations == 3

    def test_error_with_non_qnode(self):
        """Test that a helpful error message is raised if the input is not a QNode."""

        def f():
            return 0

        with pytest.raises(
            ValueError, match="qp.specs can only be applied to a QNode or qjit'd QNode"
        ):
            qp.specs(f)()
