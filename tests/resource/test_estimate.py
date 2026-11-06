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

import importlib
from functools import partial

import numpy as np
import pytest

import pennylane as qp
from pennylane.core.shots import Shots
from pennylane.resource import CircuitSpecs, SpecsResources

catalyst = pytest.importorskip("catalyst")

pytestmark = pytest.mark.catalyst

TARGET = {"Hadamard", "CNOT", "RX"}


@qp.register_resources({qp.Z: 2, qp.RY: 2, qp.CNOT: 1})
def cz_with_ry(wires, **_):
    """Decomposition rule for CZ that writes each Hadamard as Z followed by RY(pi/2)."""
    qp.Z(wires[1])
    qp.RY(np.pi / 2, wires[1])
    qp.CNOT(wires)
    qp.Z(wires[1])
    qp.RY(np.pi / 2, wires[1])


@qp.register_resources({qp.RY: 1, qp.X: 1})
def h_with_x(wires, **_):
    """Decomposition rule for Hadamard as RY(pi/2) followed by X."""
    qp.RY(np.pi / 2, wires)
    qp.X(wires)


@pytest.mark.capture
class TestEstimate:  # pylint: disable=too-many-public-methods
    """Test qp.estimate()"""

    @pytest.fixture
    def qnode(self):
        """Fixture for a QNode with gates that cancel and a gate outside the target gate set."""

        @qp.qnode(qp.device("lightning.qubit", wires=2))
        def circuit(x):
            qp.Hadamard(0)
            qp.Hadamard(0)
            qp.CZ([0, 1])
            qp.RX(x, wires=1)
            return qp.probs()

        return circuit

    @pytest.fixture
    def circuit(self, qnode):
        """Fixture for the qjit'd QNode with a user transform."""
        return qp.qjit(qp.transforms.cancel_inverses(qnode))

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

    @pytest.mark.parametrize("num_wires", [1, 3])
    @pytest.mark.parametrize("measurement", ["probs", "sample", "counts", "expval"])
    def test_abstract_wire_measurements(self, num_wires, measurement):
        """Estimate supports terminal measurements on abstract registers."""
        wires = qp.typing.Wire[num_wires]

        @qp.qjit(capture=True)
        @qp.set_shots(50 if measurement in ("sample", "counts") else None)
        @qp.qnode(qp.device("null.qubit", wires=num_wires))
        def circuit():
            qp.X(wires=0)
            if measurement == "expval":
                return qp.expval(qp.Z(wires=wires[0]))
            if measurement == "counts":
                return qp.counts(wires=wires, all_outcomes=True)
            return getattr(qp, measurement)(wires=wires)

        specs = qp.estimate(circuit, target={"PauliX": 1})()

        assert specs.resources.quantum_operations == {"PauliX": 1}
        measured = "PauliZ" if measurement == "expval" else f"{num_wires} wires"
        assert specs.resources.measurement_processes == {f"{measurement}({measured})": 1}

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

    def test_fixed_decomps(self, qnode):
        """Test that the fixed_decomps of a decompose transform in the pipeline replace the
        existing decomposition rules."""

        target = TARGET | {"RY", "PauliZ"}
        circuit = qp.transforms.decompose(qnode, gate_set=target, fixed_decomps={qp.CZ: cz_with_ry})

        specs = qp.estimate(qp.qjit(circuit), target=target)(0.5)

        assert specs.resources.quantum_operations == {
            "Hadamard": 2,
            "PauliZ": 2,
            "RY": 2,
            "CNOT": 1,
            "RX": 1,
        }

    def test_alt_decomps(self, qnode):
        """Test that the alt_decomps of a decompose transform in the pipeline are used if they
        lead to a cheaper decomposition."""

        target = {"Hadamard": 10, "CNOT": 1, "RX": 1, "RY": 1, "PauliZ": 1}
        circuit = qp.transforms.decompose(qnode, gate_set=target, alt_decomps={qp.CZ: [cz_with_ry]})

        specs = qp.estimate(qp.qjit(circuit), target=target)(0.5)

        assert specs.resources.quantum_operations == {
            "Hadamard": 2,
            "PauliZ": 2,
            "RY": 2,
            "CNOT": 1,
            "RX": 1,
        }

    def test_decomps_from_several_decompose_transforms(self, qnode):
        """Test that the rules of every decompose transform in the pipeline are used."""

        target = {"CNOT", "RX", "RY", "PauliX", "PauliZ"}
        circuit = qp.transforms.decompose(
            qnode, gate_set=target | {"Hadamard"}, fixed_decomps={qp.CZ: cz_with_ry}
        )
        circuit = qp.transforms.decompose(
            circuit, gate_set=target, fixed_decomps={qp.Hadamard: h_with_x}
        )

        specs = qp.estimate(qp.qjit(circuit), target=target)(0.5)

        assert specs.resources.quantum_operations == {
            "PauliX": 2,
            "PauliZ": 2,
            "RY": 4,
            "CNOT": 1,
            "RX": 1,
        }

    def test_unreachable_target(self, circuit):
        """Test that a short error naming the operator is raised if the circuit cannot be
        decomposed into the target gate set."""

        with pytest.raises(ValueError, match="Decomposition rule not found for operator RX"):
            qp.estimate(circuit, target={"Hadamard", "CNOT"})(0.5)

    def test_other_compile_errors(self, circuit, monkeypatch):
        """Test that compile errors unrelated to a missing decomposition rule are raised as they
        are."""

        def fail(*_, **__):
            raise catalyst.utils.exceptions.CompileError("some other compile error")

        estimate_module = importlib.import_module("pennylane.resource.estimate")
        monkeypatch.setattr(estimate_module, "resources_from_analysis_pass", fail)

        with pytest.raises(catalyst.utils.exceptions.CompileError, match="some other"):
            qp.estimate(circuit, target=TARGET)(0.5)

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

    def test_marker_levels(self, qnode):
        """Test that a marker before every pass is the same as level="top", and that a marker
        after a pass is not supported."""

        circuit = qp.marker(qnode, "start")
        circuit = qp.transforms.cancel_inverses(circuit)
        circuit = qp.qjit(qp.marker(circuit, "end"))

        specs = qp.estimate(circuit, level="start", target=TARGET)(0.5)
        assert specs == qp.estimate(circuit, level="top", target=TARGET)(0.5)

        with pytest.raises(NotImplementedError, match="qp.estimate only supports level='top'"):
            qp.estimate(circuit, level="end", target=TARGET)(0.5)

    @pytest.mark.parametrize("level", [1, "user", "all", [0]])
    def test_unsupported_level(self, circuit, level):
        """Test that an error is raised for levels other than "top"."""

        with pytest.raises(NotImplementedError, match="qp.estimate only supports level='top'"):
            qp.estimate(circuit, level=level, target=TARGET)(0.5)

    def test_device_level(self, circuit):
        """Test that an error is raised for level="device"."""

        with pytest.raises(NotImplementedError, match="does not support level='device' yet"):
            qp.estimate(circuit, level="device", target=TARGET)(0.5)

    @pytest.mark.parametrize("level", [0.0, False])
    def test_invalid_level(self, circuit, level):
        """Test that levels that are neither int nor str are rejected."""

        with pytest.raises(ValueError, match="Invalid level"):
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
