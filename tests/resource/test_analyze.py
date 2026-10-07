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
"""Unit tests for the analyze transform"""

from functools import partial

import pytest

import pennylane as qp
from pennylane.core.shots import Shots
from pennylane.devices.capabilities import DeviceCapabilities, OperatorProperties
from pennylane.resource import CircuitSpecs, SpecsResources

catalyst = pytest.importorskip("catalyst")

pytestmark = pytest.mark.catalyst


class CapabilitiesDevice(qp.devices.NullQubit):
    """Device that allows setting capabilities on a per-instance basis."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._capabilities = super().capabilities

    @property
    def capabilities(self) -> DeviceCapabilities:
        """Capabilities."""
        return self._capabilities

    @capabilities.setter
    def capabilities(self, obj: DeviceCapabilities):
        """Capabilities setter."""
        self._capabilities = obj
        self._capabilities.qjit_compatible = True

    @property
    def qjit_capabilities(self):
        """Alias used by Catalyst instead of loading a TOML file."""
        return self._capabilities


class TestAnalyze:
    """Test qp.analyze() with capture enabled and disabled."""

    @pytest.fixture
    def circuit(self):
        """Fixture for a qjit'd circuit with two transforms and a marker between them."""

        dev = CapabilitiesDevice(wires=2)
        dev.capabilities = DeviceCapabilities(
            non_commuting_observables=True,
            observables={
                "PauliX": OperatorProperties(),
                "PauliY": OperatorProperties(),
                "PauliZ": OperatorProperties(),
                "Hadamard": OperatorProperties(),
            },
            measurement_processes={"ExpectationMP": [], "SampleMP": [], "CountsMP": []},
        )

        @qp.qjit(capture=True)
        @qp.transforms.merge_rotations
        @qp.marker("cancelled")
        @qp.transforms.cancel_inverses
        @qp.qnode(dev)
        def circuit(x):
            qp.RX(x, wires=0)
            qp.RX(x, wires=0)
            qp.X(0)
            qp.X(0)
            qp.CNOT(wires=[0, 1])
            return qp.expval(qp.sum(qp.H(0), qp.Z(1)))

        return circuit

    # Gate counts of the ``circuit`` fixture at each level
    TOP_COUNTS = {"RX": 2, "PauliX": 2, "CNOT": 1}
    CANCELLED_COUNTS = {"RX": 2, "CNOT": 1}
    USER_COUNTS = {"RX": 1, "CNOT": 1}
    DEVICE_COUNTS = {"RX": 1, "CNOT": 1}

    # Measurement processes of the ``circuit`` fixture at each level
    TOP_MEASUREMENT_PROCESSES = {"expval(Hamiltonian(num_terms=2))": 1}
    CANCELLED_MEASUREMENT_PROCESSES = {"expval(Hamiltonian(num_terms=2))": 1}
    USER_MEASUREMENT_PROCESSES = {"expval(Hamiltonian(num_terms=2))": 1}
    DEVICE_MEASUREMENT_PROCESSES = {"expval(Hadamard)": 1, "expval(PauliZ)": 1}

    @pytest.mark.parametrize("level", ["all", "all-user", range(3)])
    def test_resources_at_each_level(self, circuit, level):
        """Test that analyze counts the resources left after each transform."""

        specs = qp.analyze(circuit, level=level)(0.1)

        expected_level = {0: "Before MLIR Passes", 1: "cancelled", 2: "merge-rotations"}
        expected_resources = {
            "Before MLIR Passes": SpecsResources(
                counts=self.TOP_COUNTS,
                measurement_processes=self.TOP_MEASUREMENT_PROCESSES,
                num_wires=2,
            ),
            "cancelled": SpecsResources(
                counts=self.CANCELLED_COUNTS,
                measurement_processes=self.CANCELLED_MEASUREMENT_PROCESSES,
                num_wires=2,
            ),
            "merge-rotations": SpecsResources(
                counts=self.USER_COUNTS,
                measurement_processes=self.USER_MEASUREMENT_PROCESSES,
                num_wires=2,
            ),
        }
        if level == "all":
            expected_level[3] = "Device Preprocessing"
            expected_resources["Device Preprocessing"] = SpecsResources(
                counts=self.DEVICE_COUNTS,
                measurement_processes=self.DEVICE_MEASUREMENT_PROCESSES,
                num_wires=2,
            )

        assert specs == CircuitSpecs(
            device_name="null.qubit",
            num_device_wires=2,
            shots=Shots(None),
            level=expected_level,
            resources=expected_resources,
        )

    @pytest.mark.parametrize(
        "level, expected_counts, expected_mps",
        [
            (0, TOP_COUNTS, TOP_MEASUREMENT_PROCESSES),
            ("top", TOP_COUNTS, TOP_MEASUREMENT_PROCESSES),
            (1, CANCELLED_COUNTS, CANCELLED_MEASUREMENT_PROCESSES),
            ("cancelled", CANCELLED_COUNTS, CANCELLED_MEASUREMENT_PROCESSES),
            (2, USER_COUNTS, USER_MEASUREMENT_PROCESSES),
            ("user", USER_COUNTS, USER_MEASUREMENT_PROCESSES),
            ("device", DEVICE_COUNTS, DEVICE_MEASUREMENT_PROCESSES),
        ],
    )
    def test_single_level(self, circuit, level, expected_counts, expected_mps):
        """Test that analyze counts the resources at a single level."""

        specs = qp.analyze(circuit, level=level)(0.1)

        assert specs.resources.counts == expected_counts
        assert specs.resources.measurement_processes == expected_mps

    def test_default_level_is_user(self, circuit):
        """Test that analyze defaults to the resources after all user transforms."""

        assert qp.analyze(circuit)(0.1) == qp.analyze(circuit, level="user")(0.1)

    @pytest.mark.parametrize(
        "level, expected_counts",
        [
            ([0, "cancelled"], {"Before MLIR Passes": TOP_COUNTS, "cancelled": CANCELLED_COUNTS}),
            ((0, 2), {"Before MLIR Passes": TOP_COUNTS, "merge-rotations": USER_COUNTS}),
        ],
    )
    def test_multiple_levels(self, circuit, level, expected_counts):
        """Test that analyze counts the resources at each of the given levels only."""

        specs = qp.analyze(circuit, level=level)(0.1)

        assert {name: res.counts for name, res in specs.resources.items()} == expected_counts

    def test_symbolic_resources(self):
        """Test that analyze counts the resources inside a loop with a number of iterations that
        is not known at compile time symbolically."""

        @qp.qjit(autograph=True, capture="global")
        @qp.qnode(qp.device("null.qubit", wires=1))
        def circuit(n):
            qp.Hadamard(0)
            for _ in range(n):
                qp.PauliX(0)
            return qp.expval(qp.Z(0))

        resources = qp.analyze(circuit, level=0)(5).resources

        assert resources.is_symbolic
        assert len(resources.vars) == 1

        concrete = resources.subs({var: 5 for var in resources.vars})
        assert not concrete.is_symbolic
        assert concrete.counts == {"Hadamard": 1, "PauliX": 5}

    @pytest.mark.parametrize(
        "level", [0, "top", 2, "user", "cancelled", [0, "cancelled"], (0, 2), range(3), "all"]
    )
    def test_matches_specs(self, circuit, level):
        """Test that analyze returns the same specs as qp.specs for the same level."""

        assert qp.analyze(circuit, level=level)(0.1) == qp.specs(circuit, level=level)(0.1)

    @pytest.mark.usefixtures("enable_graph_decomposition")
    def test_with_catalyst_passes(self):
        """Test that analyze counts the resources after each pass of a pipeline of Catalyst passes,
        with program capture enabled and disabled."""

        pipeline = qp.CompilePipeline(
            catalyst.passes.cancel_inverses, catalyst.passes.merge_rotations
        )

        @qp.qjit(capture="global")
        @pipeline
        @qp.qnode(qp.device("null.qubit", wires=2))
        def circuit(x):
            qp.Hadamard(wires=0)
            qp.Hadamard(wires=0)
            qp.RX(x, wires=0)
            qp.RX(x, wires=0)
            qp.CNOT(wires=[0, 1])
            return qp.expval(qp.PauliZ(0))

        if qp.capture.enabled():
            specs = qp.analyze(circuit, level="all")(0.5)

            assert specs.level == {
                0: "Before MLIR Passes",
                1: "cancel-inverses",
                2: "merge-rotations",
                3: "Device Preprocessing",
            }
            assert [resources.counts for resources in specs.resources.values()] == [
                {"Hadamard": 2, "RX": 2, "CNOT": 1},
                {"RX": 2, "CNOT": 1},
                {"RX": 1, "CNOT": 1},
                {"RX": 1, "CNOT": 1},
            ]
        else:
            with pytest.raises(
                ValueError, match="Device level is only supported when capture is enabled"
            ):
                specs = qp.analyze(circuit, level="all")(0.5)

    def test_partial(self, circuit):
        """Test analyze for a partial-wrapped Catalyst jitted QNode."""

        specs = qp.analyze(partial(circuit, 0.1), level="user")()

        assert specs.resources.counts == self.USER_COUNTS

    @pytest.mark.parametrize("level", ["all-user", "all", "device"])
    def test_device_level_requires_capture(self, level):
        """Device-containing levels require capture"""

        @qp.qjit(capture=False)
        @qp.qnode(qp.device("null.qubit", wires=1))
        def circuit():
            qp.Hadamard(0)
            return qp.expval(qp.Z(0))

        if level == "all-user":
            qp.analyze(circuit, level=level)()
        else:
            with pytest.raises(
                ValueError, match="Device level is only supported when capture is enabled"
            ):
                qp.analyze(circuit, level=level)()

    @pytest.mark.parametrize("level", [None, 1.5])
    def test_unsupported_level(self, circuit, level):
        """Test that a helpful error message is raised for levels of an unsupported type."""

        with pytest.raises(ValueError, match="Invalid level"):
            qp.analyze(circuit, level=level)(0.1)

    def test_error_with_non_qjit(self):
        """Test that a helpful error message is raised if the input is not QJIT'd."""

        @qp.qnode(qp.device("null.qubit", wires=1))
        def circuit():
            qp.Hadamard(0)
            return qp.expval(qp.PauliZ(0))

        with pytest.raises(ValueError, match="qp.analyze can only be applied to a qjit'd QNode"):
            qp.analyze(circuit, level=0)()
