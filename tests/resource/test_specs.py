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
from jax import numpy as jnp

import pennylane as qp
from pennylane import qjit
from pennylane.core.shots import Shots
from pennylane.resource import CircuitSpecs, PBCSpecsResources, SpecsResources

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


class TestDeviceLevelSpecs:
    """Test qp.specs() at device level"""

    def test_with_passes(self):
        """Test that device-level specs count resources *after* all passes are applied"""

        dev = qp.device("lightning.qubit", wires=2)

        @qjit
        @qp.transforms.merge_rotations
        @qp.transforms.cancel_inverses
        @qp.qnode(dev)
        def circuit():
            qp.Hadamard(wires=0)
            qp.Hadamard(wires=0)
            qp.CNOT(wires=[0, 1])
            qp.CNOT(wires=[0, 1])
            qp.RX(1.2, wires=0)
            qp.RX(1.2, wires=0)
            return qp.expval(qp.PauliZ(0))

        specs = qp.specs(circuit, level="device")()

        assert specs.resources.total_quantum_operations == 1
        assert specs.resources.quantum_operations == {"RX": 1}

    def test_simple(self):
        """Test a simple case of qp.specs() against PennyLane"""

        dev = qp.device("lightning.qubit", wires=1)

        @qp.qnode(dev)
        def circuit():
            qp.Hadamard(wires=0)
            return qp.expval(qp.PauliZ(0))

        specs = qp.specs(qjit(circuit), level="device")()

        assert specs == CircuitSpecs(
            device_name="lightning.qubit",
            num_device_wires=1,
            shots=Shots(None),
            level="device",
            resources=SpecsResources(
                counts={"Hadamard": 1},
                measurement_processes={"expval(PauliZ)": 1},
                num_allocs=1,
                circuit_depth=1,
            ),
        )

    def test_complex(self):
        """Test a complex case of qp.specs() against PennyLane"""

        dev = qp.device("lightning.qubit", wires=4)
        U = 1 / jnp.sqrt(2) * jnp.array([[1, 1], [1, -1]], dtype=jnp.complex128)

        @qp.qnode(dev)
        def circuit():
            qp.PauliX(0)
            qp.adjoint(qp.T)(0)
            qp.ctrl(op=qp.S, control=[1], control_values=[1])(0)
            qp.ctrl(op=qp.S, control=[1, 2], control_values=[1, 0])(0)
            qp.ctrl(op=qp.adjoint(qp.Y), control=[2], control_values=[1])(0)
            qp.CNOT([0, 1])

            qp.QubitUnitary(U, wires=0)
            qp.ControlledQubitUnitary(U, control_values=[1], wires=[1, 0])
            qp.adjoint(qp.QubitUnitary(U, wires=0))
            qp.adjoint(qp.ControlledQubitUnitary(U, control_values=[1, 1], wires=[1, 2, 0]))

            return qp.probs()

        specs = qp.specs(qjit(circuit), level="device")()

        assert specs == CircuitSpecs(
            device_name="lightning.qubit",
            num_device_wires=4,
            shots=Shots(None),
            level="device",
            resources=SpecsResources(
                counts={
                    "PauliX": 1,
                    "Adjoint(T)": 1,
                    "C(S)": 1,
                    "2C(S)": 1,
                    "CY": 1,
                    "CNOT": 1,
                    "QubitUnitary": 1,
                    "ControlledQubitUnitary": 2,
                    "Adjoint(QubitUnitary)": 1,
                },
                measurement_processes={"probs(all wires)": 1},
                num_allocs=4,
                circuit_depth=10,
            ),
        )

    @pytest.mark.capture
    def test_paulirot_and_measure(self):
        """Test that PauliRot and PauliMeasure are tracked at the device level."""

        dev = qp.device("null.qubit", wires=2)

        @qjit
        @qp.qnode(dev)
        def circuit():
            qp.PauliRot(0.42, pauli_word="Y", wires=0)  # arbitrary angle
            qp.PauliRot(jnp.pi / 2, pauli_word="YZ", wires=[0, 1])  # pi/2 angle
            qp.PauliRot(2 * jnp.pi, pauli_word="X", wires=0)  # identity
            qp.pauli_measure("X", wires=0)
            return qp.probs()

        specs = qp.specs(circuit, level="device")()

        assert specs.resources.total_quantum_operations == 4
        assert specs.resources.quantum_operations == {
            "PauliRot-pi/2-w2": 1,
            "PauliRot-identity-w1": 1,
            "PauliRot-Phi-w1": 1,
            "PauliMeasure-w1": 1,
        }

    def test_measurements_simple(self):
        """Test that measurements are tracked correctly at device level."""

        @qp.set_shots(1)
        @qp.qnode(qp.device("null.qubit", wires=3))
        def circuit():
            return (
                qp.expval(qp.PauliX(0)),
                qp.expval(qp.PauliZ(0)),
                qp.expval(qp.PauliZ(1)),
                qp.probs(),
                qp.probs(wires=[0]),
                qp.sample(),
                qp.counts(),
                qp.counts(wires=[1]),
            )

        specs = qp.specs(qjit(circuit), level="device")()

        expected_measurements = {
            "expval(PauliX)": 1,
            "expval(PauliZ)": 2,
            "probs(all wires)": 1,
            "probs(1 wires)": 1,
            "sample(all wires)": 1,
            "counts(all wires)": 1,
            "counts(1 wires)": 1,
        }

        assert specs.resources.measurement_processes == expected_measurements

    def test_measurements_complex(self):
        """Test that measurements are tracked correctly at device level."""

        @qp.qnode(qp.device("null.qubit", wires=3))
        def circuit_complex():
            coeffs = [0.2, -0.543]
            obs = [qp.X(0) @ qp.Z(1), qp.Z(0) @ qp.Hadamard(2)]
            ham = qp.ops.LinearCombination(coeffs, obs)
            return (
                qp.expval(qp.PauliZ(0) @ qp.PauliX(1)),
                qp.expval(ham),
                qp.state(),
                qp.var(qp.PauliX(0) @ qp.PauliY(1) @ qp.PauliZ(2)),
            )

        specs = qp.specs(qjit(circuit_complex), level="device")()
        expected_measurements = {
            "expval(Prod(num_terms=2))": 1,
            "expval(Hamiltonian(num_terms=2))": 1,
            "state(all wires)": 1,
            "var(Prod(num_terms=3))": 1,
        }
        assert specs.resources.measurement_processes == expected_measurements


class TestPassByPassSpecs:
    """Test qp.specs() pass-by-pass specs"""

    @pytest.fixture
    def simple_circuit(self):
        """Fixture for a circuit."""

        @qp.qnode(qp.device("lightning.qubit", wires=2))
        def circ():
            qp.RX(1.0, 0)
            qp.RX(2.0, 0)
            qp.RZ(3.0, 1)
            qp.RZ(4.0, 1)
            qp.Hadamard(0)
            qp.Hadamard(0)
            qp.CNOT([0, 1])
            qp.CNOT([0, 1])
            return qp.probs()

        return circ

    def test_invalid_levels(self, simple_circuit):
        """Test invalid inputs."""

        no_passes = qjit(simple_circuit)
        with pytest.raises(
            ValueError,
            match=r"The 'level' argument to .*\.specs for QJIT'd QNodes is out of "
            "bounds, got -5.",
        ):
            qp.specs(no_passes, level=-5)()

        with pytest.raises(
            ValueError,
            match=r"The 'level' argument to .*\.specs for QJIT'd "
            "QNodes is out of bounds, got 10.",
        ):
            qp.specs(no_passes, level=10)()

        with pytest.raises(
            ValueError,
            match=r"The 'level' argument to .*\.specs for QJIT'd "
            "QNodes is out of bounds, got 10.",
        ):
            qp.specs(no_passes, level=[10, 11])()

    def test_basic_passes_multi_level(self, simple_circuit):
        """Test that when passes are applied, the circuit resources are updated accordingly."""

        simple_circuit = qp.transforms.cancel_inverses(simple_circuit)
        simple_circuit = qp.transforms.merge_rotations(simple_circuit)

        simple_circuit = qjit(simple_circuit)

        expected = CircuitSpecs(
            device_name="lightning.qubit",
            num_device_wires=2,
            shots=Shots(None),
            level=dict(
                enumerate(
                    (
                        "Before MLIR Passes",
                        "cancel-inverses",
                        "merge-rotations",
                    )
                )
            ),
            resources={
                "Before MLIR Passes": SpecsResources(
                    counts={"RX": 2, "RZ": 2, "Hadamard": 2, "CNOT": 2},
                    measurement_processes={"probs(all wires)": 1},
                    num_allocs=2,
                ),
                "cancel-inverses": SpecsResources(
                    counts={"RX": 2, "RZ": 2},
                    measurement_processes={"probs(all wires)": 1},
                    num_allocs=2,
                ),
                "merge-rotations": SpecsResources(
                    counts={"RX": 1, "RZ": 1},
                    measurement_processes={"probs(all wires)": 1},
                    num_allocs=2,
                ),
            },
        )

        actual = qp.specs(simple_circuit, level="all")()

        assert actual == expected

        # Test resources at each level match individual specs calls
        for i, res in enumerate(actual["resources"].values()):
            single_level_specs = qp.specs(simple_circuit, level=i)()
            assert res == single_level_specs.res

    def test_user_level(self, simple_circuit):
        """Test that 'user' level is handled correctly."""

        simple_circuit = qp.transform(pass_name="cancel-inverses")(simple_circuit)
        simple_circuit = qp.transform(pass_name="merge-rotations")(simple_circuit)
        simple_circuit = qp.qjit(simple_circuit)

        specs = qp.specs(simple_circuit, level="user")()
        assert specs.level == "merge-rotations"
        assert specs.resources == SpecsResources(
            counts={"RX": 1, "RZ": 1},
            measurement_processes={"probs(all wires)": 1},
            num_allocs=2,
        )

    def test_duplicate_level_names(self, simple_circuit):
        """Test that duplicate pass names are handled gracefully."""

        simple_circuit = qp.transforms.cancel_inverses(simple_circuit)
        simple_circuit = qp.transform(pass_name="cancel-inverses")(simple_circuit)
        simple_circuit = qjit(simple_circuit)

        before_res = SpecsResources(
            counts={"RX": 2, "RZ": 2, "Hadamard": 2, "CNOT": 2},
            measurement_processes={"probs(all wires)": 1},
            num_allocs=2,
        )

        canceled_res = SpecsResources(
            counts={"RX": 2, "RZ": 2},
            measurement_processes={"probs(all wires)": 1},
            num_allocs=2,
        )

        expected = CircuitSpecs(
            device_name="lightning.qubit",
            num_device_wires=2,
            shots=Shots(None),
            level=dict(
                enumerate(
                    (
                        "Before MLIR Passes",
                        "cancel-inverses",
                        "cancel-inverses-2",
                    )
                )
            ),
            resources={
                "Before MLIR Passes": before_res,
                "cancel-inverses": canceled_res,
                "cancel-inverses-2": canceled_res,
            },
        )

        actual = qp.specs(simple_circuit, level="all")()

        assert actual == expected

    def test_basic_passes_multi_level(self, simple_circuit):
        """Test that when passes are applied, the circuit resources are updated accordingly."""

        simple_circuit = qp.transforms.cancel_inverses(simple_circuit)
        simple_circuit = qp.transforms.merge_rotations(simple_circuit)

        simple_circuit = qjit(simple_circuit)

        expected = CircuitSpecs(
            device_name="lightning.qubit",
            num_device_wires=2,
            shots=Shots(None),
            level=dict(
                enumerate(
                    (
                        "Before MLIR Passes",
                        "cancel-inverses",
                        "merge-rotations",
                    )
                )
            ),
            resources={
                "Before MLIR Passes": SpecsResources(
                    counts={"RX": 2, "RZ": 2, "Hadamard": 2, "CNOT": 2},
                    measurement_processes={"probs(all wires)": 1},
                    num_allocs=2,
                ),
                "cancel-inverses": SpecsResources(
                    counts={"RX": 2, "RZ": 2},
                    measurement_processes={"probs(all wires)": 1},
                    num_allocs=2,
                ),
                "merge-rotations": SpecsResources(
                    counts={"RX": 1, "RZ": 1},
                    measurement_processes={"probs(all wires)": 1},
                    num_allocs=2,
                ),
            },
        )

        actual = qp.specs(simple_circuit, level="all")()

        assert actual == expected

    def test_circuit_with_args(self):
        """Test circuits with positional args"""

        @qp.qnode(qp.device("lightning.qubit", wires=2))
        def circ(x):
            qp.RX(x * 1.0, 0)
            qp.RX(x * 2.0, 0)
            qp.RZ(x * 3.0, 1)
            qp.RZ(x * 4.0, 1)
            qp.Hadamard(0)
            qp.Hadamard(0)
            qp.CNOT([0, 1])
            qp.CNOT([0, 1])
            return qp.probs()

        circ = qp.transforms.cancel_inverses(circ)
        circ = qp.transforms.merge_rotations(circ)  # Can be applied as an MLIR pass

        circ = qjit(circ)

        actual = qp.specs(circ, level="all")(3)
        expected = CircuitSpecs(
            device_name="lightning.qubit",
            num_device_wires=2,
            shots=Shots(None),
            level=dict(
                enumerate(
                    (
                        "Before MLIR Passes",
                        "cancel-inverses",
                        "merge-rotations",
                    )
                )
            ),
            resources={
                "Before MLIR Passes": SpecsResources(
                    counts={"CNOT": 2, "Hadamard": 2, "RX": 2, "RZ": 2},
                    measurement_processes={"probs(all wires)": 1},
                    num_allocs=2,
                ),
                "cancel-inverses": SpecsResources(
                    counts={"RX": 2, "RZ": 2},
                    measurement_processes={"probs(all wires)": 1},
                    num_allocs=2,
                ),
                "merge-rotations": SpecsResources(
                    counts={"RX": 1, "RZ": 1},
                    measurement_processes={"probs(all wires)": 1},
                    num_allocs=2,
                ),
            },
        )

        assert actual == expected

    def test_all_mlir(self, simple_circuit):
        """Test using "all" level"""

        simple_circuit = qp.transforms.cancel_inverses(simple_circuit)
        simple_circuit = qp.transforms.merge_rotations(simple_circuit)

        simple_circuit = qjit(simple_circuit)

        actual = qp.specs(simple_circuit, level="all")()
        expected = CircuitSpecs(
            device_name="lightning.qubit",
            num_device_wires=2,
            shots=Shots(None),
            level={
                0: "Before MLIR Passes",
                1: "cancel-inverses",
                2: "merge-rotations",
            },
            resources={
                "Before MLIR Passes": SpecsResources(
                    counts={"RX": 2, "RZ": 2, "Hadamard": 2, "CNOT": 2},
                    measurement_processes={"probs(all wires)": 1},
                    num_allocs=2,
                ),
                "cancel-inverses": SpecsResources(
                    counts={"RX": 2, "RZ": 2},
                    measurement_processes={"probs(all wires)": 1},
                    num_allocs=2,
                ),
                "merge-rotations": SpecsResources(
                    counts={"RX": 1, "RZ": 1},
                    measurement_processes={"probs(all wires)": 1},
                    num_allocs=2,
                ),
            },
        )

        assert actual == expected

    def test_advanced_measurements(self):
        """Test that advanced measurements such as LinearCombination are handled correctly."""

        dev = qp.device("lightning.qubit", wires=7)

        @qjit
        @qp.qnode(dev, shots=10)
        def circ():
            coeffs = [0.2, -0.543]
            obs = [qp.X(0) @ qp.Z(1), qp.Z(0) @ qp.Hadamard(2)]
            ham = qp.ops.LinearCombination(coeffs, obs)

            return (
                qp.expval(ham),
                qp.expval(qp.PauliZ(0) @ qp.PauliZ(1)),
                qp.sample(wires=3),
                qp.sample(),
            )

        # Representations are slightly different from plain PL -- wire counts are missing
        info = qp.specs(circ, level=0, compute_depth=False)()

        assert info.resources.measurement_processes == {
            "expval(Hamiltonian(num_terms=2))": 1,
            "expval(Prod(num_terms=2))": 1,
            "sample(1 wires)": 1,
            "sample(all wires)": 1,
        }

    @pytest.mark.capture
    def test_conditionals(self):
        """Test that conditionals are handled correctly."""

        @qp.qjit(autograph=True)
        @qp.qnode(qp.device("null.qubit", wires=1))
        def circuit(x):
            if x > 0.5:
                qp.Hadamard(0)
                qp.PauliX(0)
            else:
                qp.PauliX(0)
                if x < 2:
                    qp.PauliX(0)
                else:
                    qp.PauliZ(0)

            return qp.expval(qp.PauliX(0))

        with pytest.warns(
            UserWarning,
            match="Specs was unable to determine the branch of a conditional or switch statement.",
        ):
            actual = qp.specs(circuit, level=0)(3)
        expected = CircuitSpecs(
            device_name="null.qubit",
            num_device_wires=1,
            shots=Shots(None),
            level="Before MLIR Passes",
            resources=SpecsResources(
                counts={"Hadamard": 1, "PauliX": 2, "PauliZ": 1},
                measurement_processes={"expval(PauliX)": 1},
                num_allocs=1,
            ),
        )

        assert actual == expected

    @pytest.mark.capture
    def test_loops(self):
        """Test that static loops are handled correctly and that resources are counted
        according to the number of iterations (including nested loops)."""

        @qp.qjit(autograph=True)
        @qp.qnode(qp.device("null.qubit", wires=1))
        def circuit():
            for _ in range(5):
                qp.PauliX(0)
                for _ in range(3):
                    qp.Hadamard(0)
            return qp.expval(qp.PauliX(0))

        actual = qp.specs(circuit, level=0)()
        expected = CircuitSpecs(
            device_name="null.qubit",
            num_device_wires=1,
            shots=Shots(None),
            level="Before MLIR Passes",
            resources=SpecsResources(
                counts={"Hadamard": 15, "PauliX": 5},
                measurement_processes={"expval(PauliX)": 1},
                num_allocs=1,
            ),
        )

        assert actual == expected

    def test_split_non_commuting_mlir(self):
        """Test that split-non-commuting works as expected"""

        @qp.transforms.cancel_inverses
        @qp.transform(pass_name="split-non-commuting")  # Applies as MLIR pass
        @qp.qnode(qp.device("null.qubit", wires=3))
        def circuit():
            qp.H(0)
            qp.X(0)
            qp.X(0)
            return qp.expval(qp.X(0)), qp.expval(qp.Y(0)), qp.expval(qp.Z(0))

        actual = qp.specs(qjit(circuit), level=[1, 2])()
        expected = CircuitSpecs(
            device_name="null.qubit",
            num_device_wires=3,
            shots=Shots(None),
            level={1: "split-non-commuting", 2: "cancel-inverses"},
            resources={
                "split-non-commuting": [
                    SpecsResources(
                        counts={"Hadamard": 1, "PauliX": 2},
                        measurement_processes={"expval(PauliX)": 1},
                        num_allocs=3,
                    ),
                    SpecsResources(
                        counts={"Hadamard": 1, "PauliX": 2},
                        measurement_processes={"expval(PauliY)": 1},
                        num_allocs=3,
                    ),
                    SpecsResources(
                        counts={"Hadamard": 1, "PauliX": 2},
                        measurement_processes={"expval(PauliZ)": 1},
                        num_allocs=3,
                    ),
                ],
                "cancel-inverses": [  # The split should remain throughout subsequent passes
                    SpecsResources(
                        counts={"Hadamard": 1},
                        measurement_processes={"expval(PauliX)": 1},
                        num_allocs=3,
                    ),
                    SpecsResources(
                        counts={"Hadamard": 1},
                        measurement_processes={"expval(PauliY)": 1},
                        num_allocs=3,
                    ),
                    SpecsResources(
                        counts={"Hadamard": 1},
                        measurement_processes={"expval(PauliZ)": 1},
                        num_allocs=3,
                    ),
                ],
            },
        )

        assert actual == expected

    @pytest.mark.capture
    def test_subroutine(self):
        """Test qp.specs when there is a Catalyst subroutine"""
        dev = qp.device("lightning.qubit", wires=3)

        @qp.capture.subroutine
        def subroutine():
            qp.Hadamard(wires=0)

        @qp.qjit(autograph=True)
        @qp.qnode(dev)
        def circuit():
            qp.PauliX(wires=1)

            for _ in range(3):
                subroutine()

            return qp.probs()

        actual = qp.specs(circuit, level=0)()
        expected = CircuitSpecs(
            device_name="lightning.qubit",
            num_device_wires=3,
            shots=Shots(None),
            level="Before MLIR Passes",
            resources=SpecsResources(
                counts={"Hadamard": 3, "PauliX": 1},
                measurement_processes={"probs(all wires)": 1},
                num_allocs=3,
            ),
        )

        assert actual == expected

    @pytest.mark.capture
    def test_operator2(self):
        """Test that specs works with operator2 classes."""

        # pylint: disable=useless-parent-delegation
        class DummyOp(qp.core.Operator2):
            """Dummy Local Operator."""

            dynamic_argnames = ("phi",)
            wire_argnames = ("reg1", "reg2")
            compilable_argnames = ("metadata",)

            def __init__(self, phi, reg1, reg2, metadata):
                super().__init__(phi, reg1, reg2, metadata)

        @qp.qjit(target="mlir")
        @qp.transforms.merge_rotations
        @qp.qnode(qp.device("null.qubit", wires=10))
        def c():
            DummyOp(0.5, (0, 1), (2, 3, 4), metadata="word")
            DummyOp(0.5, (2, 3, 4), (0,), metadata="word")
            return qp.state()

        for level in [0, 1]:
            resources = qp.specs(c, level=level)().resources

            assert resources.quantum_operations == {"DummyOp": 2}

    @pytest.mark.capture
    def test_symbolic_array(self):
        """Test using specs with symbolic_array."""

        @qp.qjit(target="mlir")
        @qp.transforms.merge_rotations
        @qp.qnode(qp.device("null.qubit", wires=1))
        def c():
            x = qp.capture.symbolic_array((), float)
            qp.RX(x, 0)
            qp.RX(2 * x, 0)
            return qp.probs()

        counts = qp.specs(c, level=0)().resources.quantum_operations
        assert counts == {"RX": 2}

        counts1 = qp.specs(c, level=1)().resources.quantum_operations
        assert counts1 == {"RX": 1}

        with pytest.raises(catalyst.utils.exceptions.CompileError, match="is a placeholder op"):
            qp.specs(c, level="device")()


class TestSpecsWithPPR:
    """Tests for using qp.specs with PPRs"""

    def test_ppr(self):
        """Test that PPRs are handled correctly."""

        @qp.qjit(target="mlir")
        @catalyst.passes.to_ppr
        @qp.qnode(qp.device("null.qubit", wires=2))
        def circ():
            qp.H(0)
            qp.T(0)

        expected = CircuitSpecs(
            device_name="null.qubit",
            num_device_wires=2,
            shots=Shots(None),
            level="to-ppr",
            resources=PBCSpecsResources(
                counts={"GlobalPhase": 2, "PPR-pi/4-w1": 3, "PPR-pi/8-w1": 1},
                measurement_processes={},
                num_allocs=2,
                any_commuting_depth=3,
                qubit_disjoint_depth=4,
            ),
        )

        actual = qp.specs(circ, level=1)()
        assert actual == expected

    @pytest.mark.capture
    def test_arbitrary_ppr(self):
        """Test that PPRs are handled correctly."""

        @qp.qjit(target="mlir")
        @qp.transforms.decompose_arbitrary_ppr
        @qp.transforms.to_ppr
        @qp.qnode(qp.device("null.qubit", wires=3))
        def circ():
            qp.PauliRot(0.1, pauli_word="XY", wires=[0, 1])

        expected = CircuitSpecs(
            device_name="null.qubit",
            num_device_wires=3,
            shots=Shots(None),
            level="decompose-arbitrary-ppr",
            resources=PBCSpecsResources(
                counts={
                    "pbc.prepare": 1,
                    "PPM-w3": 1,
                    "PPM-w1": 1,
                    "PPR-pi/2-w1": 1,
                    "PPR-pi/2-w2": 1,
                    "PPR-Phi-w1": 1,
                },
                measurement_processes={},
                num_allocs=4,
                any_commuting_depth=4,
                qubit_disjoint_depth=4,
            ),
        )

        actual = qp.specs(circ, level=2)()
        assert actual == expected


class TestSymbolicSpecs:
    """Tests for using qp.specs with dynamic loops whose bounds are not known at compile time"""

    @pytest.mark.capture
    def test_dynamic_loop(self):
        """Test specs with a dynamic loop that can't be resolved at compile time"""

        @qp.qjit(autograph=True)
        @qp.qnode(qp.device("lightning.qubit", wires=1))
        def circuit(x):
            qp.Hadamard(0)
            qp.PauliX(0)
            for _ in range(x):
                qp.PauliX(0)
            return qp.expval(qp.PauliX(0))

        s = qp.specs(circuit, level=0)(5)
        assert s.level == "Before MLIR Passes"
        assert s.device_name == "lightning.qubit"
        res = s.resources
        assert res.is_symbolic
        assert len(res.vars) == 1

        concrete_res = res.subs({var: 5 for var in res.vars})
        assert isinstance(concrete_res, SpecsResources) and not concrete_res.is_symbolic

        expected_res = SpecsResources(
            counts={"Hadamard": 1, "PauliX": 6},
            measurement_processes={"expval(PauliX)": 1},
            num_allocs=1,
        )
        assert concrete_res == expected_res

    @pytest.mark.capture
    def test_dynamic_loop_and_static_loop(self):
        """
        Test specs with a dynamic loop that can't be resolved at compile time and
        a static loop nested inside it
        """

        @qp.qjit(autograph=True)
        @qp.qnode(qp.device("lightning.qubit", wires=1))
        def circuit(x):
            qp.Hadamard(0)
            qp.PauliX(0)
            for _ in range(x):
                qp.PauliX(0)
                for _ in range(3):
                    qp.PauliY(0)
                for _ in range(5):
                    qp.PauliZ(0)

            return qp.expval(qp.PauliX(0))

        s = qp.specs(circuit, level=0)(5)
        assert s.level == "Before MLIR Passes"
        assert s.device_name == "lightning.qubit"

        res = s.resources
        assert res.is_symbolic
        assert len(res.vars) == 1

        concrete_res = res.subs({var: 5 for var in res.vars})
        assert isinstance(concrete_res, SpecsResources) and not concrete_res.is_symbolic

        expected_res = SpecsResources(
            counts={"Hadamard": 1, "PauliX": 6, "PauliY": 15, "PauliZ": 25},
            measurement_processes={"expval(PauliX)": 1},
            num_allocs=1,
        )
        assert concrete_res == expected_res

    @pytest.mark.capture
    def test_dynamic_loop_and_static_loop2(self):
        """
        Test specs with a static loop and a dynamic loop that can't be resolved at compile time
        nested inside it
        """

        @qp.qjit(autograph=True)
        @qp.qnode(qp.device("lightning.qubit", wires=1))
        def circuit(x):
            qp.Hadamard(0)
            qp.PauliX(0)
            for _ in range(3):
                qp.PauliZ(0)
                for _ in range(x):
                    qp.PauliX(0)

            return qp.expval(qp.PauliX(0))

        s = qp.specs(circuit, level=0)(5)
        assert s.level == "Before MLIR Passes"
        assert s.device_name == "lightning.qubit"

        res = s.resources
        assert res.is_symbolic
        assert len(res.vars) == 1

        concrete_res = res.subs({var: 5 for var in res.vars})
        assert isinstance(concrete_res, SpecsResources) and not concrete_res.is_symbolic

        expected_res = SpecsResources(
            counts={"Hadamard": 1, "PauliX": 16, "PauliZ": 3},
            measurement_processes={"expval(PauliX)": 1},
            num_allocs=1,
        )
        assert concrete_res == expected_res

    @pytest.mark.capture
    def test_nested_dynamic_loop(self):
        """Test specs with a nested dynamic loops that can't be resolved at compile time"""

        @qp.qjit(autograph=True)
        @qp.qnode(qp.device("lightning.qubit", wires=1))
        def circuit(x, y):
            qp.Hadamard(0)
            for _ in range(x):
                qp.PauliX(0)
                for _ in range(y):
                    qp.Hadamard(0)
            return qp.expval(qp.PauliX(0))

        s = qp.specs(circuit, level=0)(5, 3)
        assert s.level == "Before MLIR Passes"
        assert s.device_name == "lightning.qubit"
        res = s.resources
        assert res.is_symbolic
        assert len(res.vars) == 2

        for n in [2, 3]:
            concrete_res = res.subs({var: n for var in res.vars})
            assert isinstance(concrete_res, SpecsResources) and not concrete_res.is_symbolic
            expected_res = SpecsResources(
                counts={"Hadamard": 1 + n * n, "PauliX": n},
                measurement_processes={"expval(PauliX)": 1},
                num_allocs=1,
            )
            assert concrete_res == expected_res

    @pytest.mark.capture
    def test_dynamic_loops_multi_level(self):
        """Test smulti-level specs with dynamic loops"""

        @qp.qjit(autograph=True)
        @qp.transforms.cancel_inverses
        @qp.qnode(qp.device("lightning.qubit", wires=1))
        def circuit(x, y):
            qp.Hadamard(0)
            for _ in range(x):
                qp.Hadamard(0)
                qp.PauliX(0)
                for _ in range(y):
                    qp.Hadamard(0)
            return qp.expval(qp.PauliX(0))

        s = qp.specs(circuit, level="all")(3, 5)
        assert s.level == {0: "Before MLIR Passes", 1: "cancel-inverses"}
        assert s.device_name == "lightning.qubit"
        all_res = s.resources

        assert isinstance(all_res, dict)
        for res in all_res.values():
            assert res.is_symbolic
            assert len(res.vars) == 2

        for n in [2, 3]:
            for res in all_res.values():
                concrete_res = res.subs({var: n for var in res.vars})

                assert concrete_res == SpecsResources(
                    counts={"Hadamard": n * n + n + 1, "PauliX": n},
                    measurement_processes={"expval(PauliX)": 1},
                    num_allocs=1,
                )

    @pytest.mark.capture
    def test_symbolic_array_inside_loop(self):
        """Test dynamic loop with symbolic_array in a loop."""

        @qp.qjit
        @qp.qnode(qp.device("null.qubit", wires=1))
        def c(n):

            # pylint: disable=unused-argument
            @qp.for_loop(n)
            def loop(i):
                x = qp.capture.symbolic_array((), float)
                qp.RX(x, 0)

            loop()  # pylint: disable=no-value-for-parameter

            return qp.state()

        r = qp.specs(c, level=0)(2).resources
        assert r.subs({var: 10 for var in r.vars}).quantum_operations["RX"] == 10

    @pytest.mark.capture
    def test_symbolic_array_loop_arguemtn(self):
        """Test dynamic loop with a symbolic array as a loop argument."""

        @qp.qjit
        @qp.qnode(qp.device("null.qubit", wires=1))
        def c(n):

            # pylint: disable=unused-argument
            @qp.for_loop(n)
            def loop(i, x):
                qp.RX(x, 0)
                return x

            y = qp.capture.symbolic_array((), float)
            loop(y)  # pylint: disable=no-value-for-parameter

            return qp.state()

        r = qp.specs(c, level=0)(2).resources
        assert r.subs({var: 10 for var in r.vars}).quantum_operations["RX"] == 10


class TestMarkerIntegration:
    """Tests the integration with qp.marker."""

    @pytest.fixture
    def simple_circuit(self):
        """Fixture for a circuit."""

        @qp.qnode(qp.device("lightning.qubit", wires=2))
        def circ():
            qp.RX(1.0, 0)
            qp.RX(2.0, 0)
            qp.RZ(3.0, 1)
            qp.RZ(4.0, 1)
            qp.Hadamard(0)
            qp.Hadamard(0)
            qp.CNOT([0, 1])
            qp.CNOT([0, 1])
            return qp.probs()

        return circ

    def test_multi_marker(self, simple_circuit):
        """Tests that markers work with level=<iterable>."""

        simple_circuit = qp.marker(simple_circuit, "before")
        simple_circuit = qp.transforms.cancel_inverses(simple_circuit)
        simple_circuit = qp.marker(simple_circuit, "between")
        simple_circuit = qp.transforms.merge_rotations(simple_circuit)
        simple_circuit = qp.marker(simple_circuit, "after")

        assert len(simple_circuit.compile_pipeline.markers) == 3

        qjit_circuit = qp.qjit(simple_circuit)

        expected = CircuitSpecs(
            device_name="lightning.qubit",
            num_device_wires=2,
            shots=Shots(None),
            level={0: "before", 1: "between", 2: "after"},
            resources={
                "before": SpecsResources(
                    counts={"RX": 2, "RZ": 2, "Hadamard": 2, "CNOT": 2},
                    measurement_processes={"probs(all wires)": 1},
                    num_allocs=2,
                ),
                "between": SpecsResources(
                    counts={"RX": 2, "RZ": 2},
                    measurement_processes={"probs(all wires)": 1},
                    num_allocs=2,
                ),
                "after": SpecsResources(
                    counts={"RX": 1, "RZ": 1},
                    measurement_processes={"probs(all wires)": 1},
                    num_allocs=2,
                ),
            },
        )

        actual = qp.specs(qjit_circuit, level=["before", "between", "after"])()

        assert actual == expected

    def test_multi_marker_all(self, simple_circuit):
        """Tests that markers work with level="all"."""

        simple_circuit = qp.marker(simple_circuit, "before")
        simple_circuit = qp.transforms.cancel_inverses(simple_circuit)
        simple_circuit = qp.marker(simple_circuit, "between")
        simple_circuit = qp.transforms.merge_rotations(simple_circuit)
        simple_circuit = qp.marker(simple_circuit, "after")

        assert len(simple_circuit.compile_pipeline.markers) == 3

        qjit_circuit = qp.qjit(simple_circuit)

        expected = CircuitSpecs(
            device_name="lightning.qubit",
            num_device_wires=2,
            shots=Shots(None),
            level={0: "before", 1: "between", 2: "after"},
            resources={
                "before": SpecsResources(
                    counts={"RX": 2, "RZ": 2, "Hadamard": 2, "CNOT": 2},
                    measurement_processes={"probs(all wires)": 1},
                    num_allocs=2,
                ),
                "between": SpecsResources(
                    counts={"RX": 2, "RZ": 2},
                    measurement_processes={"probs(all wires)": 1},
                    num_allocs=2,
                ),
                "after": SpecsResources(
                    counts={"RX": 1, "RZ": 1},
                    measurement_processes={"probs(all wires)": 1},
                    num_allocs=2,
                ),
            },
        )

        actual = qp.specs(qjit_circuit, level="all")()

        assert actual == expected

    @pytest.mark.capture
    def test_redundant_marker(self, simple_circuit):
        """Test that two markers on the same level generate the same specs."""

        simple_circuit = partial(qp.marker, label="m0")(simple_circuit)
        simple_circuit = qp.transforms.cancel_inverses(simple_circuit)
        simple_circuit = partial(qp.marker, label="m1")(simple_circuit)
        simple_circuit = partial(qp.marker, label="m1-duplicate")(simple_circuit)

        simple_circuit = qjit(simple_circuit)

        expected = CircuitSpecs(
            device_name="lightning.qubit",
            num_device_wires=2,
            shots=Shots(None),
            level={0: "m0", 1: "m1, m1-duplicate"},
            resources={
                "m0": SpecsResources(
                    counts={"RX": 2, "RZ": 2, "Hadamard": 2, "CNOT": 2},
                    measurement_processes={"probs(all wires)": 1},
                    num_allocs=2,
                ),
                "m1, m1-duplicate": SpecsResources(
                    counts={"RX": 2, "RZ": 2},
                    measurement_processes={"probs(all wires)": 1},
                    num_allocs=2,
                ),
            },
        )

        with pytest.warns(
            UserWarning,
            match=r"The 'level' argument to .*\.specs for QJIT'd QNodes has been sorted to be "
            "in ascending order with no duplicate levels.",
        ):
            actual = qp.specs(simple_circuit, level=["m0", "m1", "m1-duplicate"])()

        assert actual == expected

    @pytest.mark.capture
    def test_marker(self, simple_circuit):
        """Test that qp.marker can be used appropriately."""

        simple_circuit = partial(qp.marker, label="m0")(simple_circuit)
        simple_circuit = qp.transforms.cancel_inverses(simple_circuit)
        simple_circuit = partial(qp.marker, label="m1")(simple_circuit)
        simple_circuit = qp.transforms.merge_rotations(simple_circuit)
        simple_circuit = partial(qp.marker, label="m2")(simple_circuit)

        simple_circuit = qjit(simple_circuit)

        expected = CircuitSpecs(
            device_name="lightning.qubit",
            num_device_wires=2,
            shots=Shots(None),
            level={0: "m0", 1: "m1", 2: "m2"},
            resources={
                "m0": SpecsResources(
                    counts={"RX": 2, "RZ": 2, "Hadamard": 2, "CNOT": 2},
                    measurement_processes={"probs(all wires)": 1},
                    num_allocs=2,
                ),
                "m1": SpecsResources(
                    counts={"RX": 2, "RZ": 2},
                    measurement_processes={"probs(all wires)": 1},
                    num_allocs=2,
                ),
                "m2": SpecsResources(
                    counts={"RX": 1, "RZ": 1},
                    measurement_processes={"probs(all wires)": 1},
                    num_allocs=2,
                ),
            },
        )

        actual = qp.specs(simple_circuit, level=["m0", "m1", "m2"])()

        assert actual == expected
