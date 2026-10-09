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
"""Unit tests for the estimate transform"""

from functools import partial

import pytest

import pennylane as qp
from pennylane.core.shots import Shots
from pennylane.resource import CircuitSpecs, SpecsResources

catalyst = pytest.importorskip("catalyst")

pytestmark = pytest.mark.catalyst

TARGET = {"Hadamard", "CNOT", "RX"}


@pytest.mark.capture
class TestEstimate:
    """Test qp.estimate()"""

    @pytest.fixture
    def circuit(self):
        """Fixture for a qjit'd circuit with a user transform."""

        @qp.qjit
        @qp.transforms.cancel_inverses
        @qp.qnode(qp.device("lightning.qubit", wires=2))
        def circuit(x):
            qp.Hadamard(0)
            qp.Hadamard(0)
            qp.CZ([0, 1])
            qp.RX(x, wires=1)
            return qp.probs()

        return circuit

    def test_specs(self, circuit):
        """Test that estimate ignores the user transforms and decomposes the circuit into the
        target gate set."""

        specs = qp.estimate(circuit, target=TARGET)(0.5)

        assert specs == CircuitSpecs(
            device_name="lightning.qubit",
            num_device_wires=2,
            shots=Shots(None),
            level="top",
            resources=SpecsResources(
                counts={"Hadamard": 4, "CNOT": 1, "RX": 1},
                measurement_processes={"probs(all wires)": 1},
                num_wires=2,
            ),
        )

    @pytest.mark.parametrize(
        "target",
        [
            ["Hadamard", "CNOT", "RX"],
            {qp.Hadamard, qp.CNOT, qp.RX},
            {qp.Hadamard: 1, "CNOT": 2.0, qp.RX: 1},
        ],
    )
    def test_target_formats(self, circuit, target):
        """Test that estimate accepts the gate set formats supported by qp.decompose."""

        specs = qp.estimate(circuit, target=target)(0.5)

        assert specs.resources.quantum_operations == {"Hadamard": 4, "CNOT": 1, "RX": 1}

    def test_predefined_gate_set(self, circuit):
        """Test that estimate accepts a predefined gate set as target."""

        specs = qp.estimate(circuit, target=qp.gate_sets.ROTATIONS_PLUS_CNOT)(0.5)

        assert set(specs.resources.quantum_operations) <= {"RX", "RY", "RZ", "CNOT", "GlobalPhase"}

    def test_operations_in_target(self, circuit):
        """Test that operations already in the target gate set are kept."""

        specs = qp.estimate(circuit, target={"Hadamard", "CZ", "RX"})(0.5)

        assert specs.resources.quantum_operations == {"Hadamard": 2, "CZ": 1, "RX": 1}

    def test_original_circuit_unchanged(self, circuit):
        """Test that estimate does not remove the user transforms from the original circuit."""

        qp.estimate(circuit, target=TARGET)(0.5)

        specs = qp.analyze(circuit, level="user")(0.5)

        assert specs.resources.quantum_operations == {"CZ": 1, "RX": 1}

    def test_symbolic_resources(self):
        """Test that estimate counts the resources inside a loop with a number of iterations that
        is not known at compile time symbolically."""

        @qp.qjit
        @qp.qnode(qp.device("lightning.qubit", wires=2))
        def circuit(n):
            @qp.for_loop(n)
            def loop(_):
                qp.CZ([0, 1])

            loop()
            return qp.probs()

        resources = qp.estimate(circuit, target={"Hadamard", "CNOT"})(5).resources

        assert resources.is_symbolic
        assert len(resources.vars) == 1

        concrete = resources.subs({var: 5 for var in resources.vars})
        assert not concrete.is_symbolic
        assert concrete.counts == {"Hadamard": 10, "CNOT": 5}

    def test_partial(self, circuit):
        """Test that estimate supports functools.partial wrappers."""

        specs = qp.estimate(partial(circuit, 0.5), target=TARGET)()

        assert specs == qp.estimate(circuit, target=TARGET)(0.5)

    def test_global_capture(self):
        """Test that estimate supports capture="global" when program capture is enabled."""

        @qp.qjit(capture="global")
        @qp.qnode(qp.device("lightning.qubit", wires=2))
        def circuit():
            qp.CZ([0, 1])
            return qp.probs()

        specs = qp.estimate(circuit, target={"Hadamard", "CNOT"})()

        assert specs.resources.quantum_operations == {"Hadamard": 2, "CNOT": 1}

    def test_level_zero(self, circuit):
        """Test that level=0 is the same as level="top"."""

        specs = qp.estimate(circuit, level=0, target=TARGET)(0.5)

        assert specs == qp.estimate(circuit, level="top", target=TARGET)(0.5)

    @pytest.mark.parametrize("level", [1, False, "user", "device", "all"])
    def test_unsupported_level(self, circuit, level):
        """Test that an error is raised for levels other than "top"."""

        with pytest.raises(NotImplementedError, match="qp.estimate only supports level='top'"):
            qp.estimate(circuit, level=level, target=TARGET)(0.5)

    @pytest.mark.parametrize("target", ["device", "user"])
    def test_unsupported_target(self, circuit, target):
        """Test that an error is raised for string targets."""

        with pytest.raises(
            NotImplementedError, match=f"qp.estimate does not support target='{target}'"
        ):
            qp.estimate(circuit, target=target)(0.5)

    def test_error_with_non_qjit(self):
        """Test that an error is raised if the QNode is not qjit'd."""

        @qp.qnode(qp.device("lightning.qubit", wires=1))
        def circuit():
            return qp.probs()

        with pytest.raises(ValueError, match="qp.estimate can only be applied to a qjit'd QNode"):
            qp.estimate(circuit, target=TARGET)()


@pytest.mark.usefixtures("disable_capture")
@pytest.mark.parametrize("capture", [False, "global"])
def test_error_without_capture(capture):
    """Test that an error is raised if the QNode is compiled without program capture."""

    @qp.qjit(capture=capture)
    @qp.qnode(qp.device("lightning.qubit", wires=1))
    def circuit():
        qp.Hadamard(0)
        return qp.probs()

    with pytest.raises(ValueError, match=r"qp.estimate requires program capture"):
        qp.estimate(circuit, target=TARGET)()
