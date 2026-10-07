# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the track transform"""

from functools import partial

import pytest

import pennylane as qp
from pennylane.core.shots import Shots
from pennylane.resource import CircuitSpecs, SpecsResources

catalyst = pytest.importorskip("catalyst")

pytestmark = pytest.mark.catalyst


class TestTrack:
    """Test qp.track()"""

    @pytest.fixture
    def circuit(self):
        """Fixture for a qjit'd circuit."""

        @qp.qjit
        @qp.qnode(qp.device("null.qubit", wires=2))
        def circuit(x, y):
            qp.RX(x, wires=0)
            qp.RY(y, wires=1)
            qp.CNOT(wires=[0, 1])
            return qp.probs()

        return circuit

    def test_results_and_specs(self, circuit):
        """Test that track returns both the execution result and the circuit specs."""

        results, specs = qp.track(circuit)(0.1, 0.2)

        assert results.shape == (4,)
        assert specs == CircuitSpecs(
            device_name="null.qubit",
            num_device_wires=2,
            shots=Shots(None),
            level="device",
            resources=SpecsResources(
                counts={"RX": 1, "RY": 1, "CNOT": 1},
                measurement_processes={"probs(all wires)": 1},
                num_wires=2,
                circuit_depth=2,
            ),
        )

    @pytest.mark.usefixtures("enable_and_disable_capture")
    def test_with_and_without_capture(self):
        """Test that track matches device-level specs with program capture enabled and disabled."""

        @qp.qjit(capture="global")
        @qp.qnode(qp.device("null.qubit", wires=2))
        def circuit(x):
            qp.RX(x, wires=0)
            qp.CNOT(wires=[0, 1])
            return qp.expval(qp.PauliZ(0))

        results, specs = qp.track(circuit)(0.1)

        assert results.shape == ()
        assert specs.resources.quantum_operations == {"RX": 1, "CNOT": 1}
        assert specs == qp.specs(circuit, level="device")(0.1)

    @pytest.mark.usefixtures("enable_graph_decomposition")
    def test_with_catalyst_passes_and_capture(self):
        """Test that track counts resources after a pipeline of Catalyst passes is applied to a
        program-captured workflow."""

        pipeline = qp.CompilePipeline(
            catalyst.passes.cancel_inverses, catalyst.passes.merge_rotations
        )

        @qp.qjit(capture=True)
        @pipeline
        @qp.qnode(qp.device("null.qubit", wires=2))
        def circuit(x):
            qp.Hadamard(wires=0)
            qp.Hadamard(wires=0)
            qp.RX(x, wires=0)
            qp.RX(x, wires=0)
            qp.CNOT(wires=[0, 1])
            return qp.expval(qp.PauliZ(0))

        results, specs = qp.track(circuit)(0.5)

        assert results.shape == ()
        assert specs.resources.quantum_operations == {"RX": 1, "CNOT": 1}
        assert specs.resources.circuit_depth == 2
        assert specs == qp.specs(circuit, level="device")(0.5)

    @pytest.mark.usefixtures("enable_graph_decomposition")
    def test_pbc_pipeline(self):
        """Test that track counts device-level resources after a Clifford+T → PPR → PPM
        compilation pipeline, including H/T from lowered magic-state fabrication."""

        @qp.qjit(capture=True)
        @qp.transforms.ppr_to_ppm
        @qp.transforms.to_ppr
        @catalyst.passes.graph_decomposition(gate_set=qp.gate_sets.CLIFFORD_T)
        @qp.qnode(qp.device("null.qubit", wires=3))
        def qfunc():
            qp.Toffoli([0, 1, 2])
            qp.Hadamard(0)
            qp.Hadamard(0)
            return qp.expval(qp.Z(0))

        results, specs = qp.track(qfunc)()

        assert results.shape == ()
        assert specs.level == "device"
        assert specs.resources.counts == {
            "T": 7,
            "Hadamard": 7,
            "PauliMeasure-w2": 31,
            "PauliMeasure-w3": 6,
            "PauliRot-pi-w1": 12,
            "PauliMeasure-w1": 37,
            "PauliRot-pi-w2": 6,
            "GlobalPhase": 17,
        }
        assert specs.resources.num_wires == 4
        assert specs.resources.circuit_depth == 40

    def test_partial(self, circuit):
        """Test track for a partial-wrapped Catalyst jitted QNode."""

        results, specs = qp.track(partial(circuit, 0.1))(0.2)

        assert results.shape == (4,)
        assert specs.resources.counts == {"RX": 1, "RY": 1, "CNOT": 1}

    @pytest.mark.parametrize("level", [0, None, "top", "user", "all"])
    def test_unsupported_level(self, circuit, level):
        """Test that a helpful error message is raised for levels other than 'device'."""

        with pytest.raises(NotImplementedError, match="qp.track only supports level='device'"):
            qp.track(circuit, level=level)(0.1, 0.2)

    def test_error_with_non_qjit(self):
        """Test that a helpful error message is raised if the input is not QJIT'd."""

        @qp.qnode(qp.device("null.qubit", wires=1))
        def circuit():
            qp.Hadamard(0)
            return qp.expval(qp.PauliZ(0))

        with pytest.raises(ValueError, match="qp.track can only be applied to a qjit'd QNode"):
            qp.track(circuit)()
