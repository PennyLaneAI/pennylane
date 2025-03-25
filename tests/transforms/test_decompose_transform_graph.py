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

"""Tests the ``decompose`` transform with the new experimental graph-based decomposition system."""

# pylint: disable=no-name-in-module, too-few-public-methods

import numpy as np
import pytest

import pennylane as qml


@pytest.mark.unit
def test_fixed_alt_decomps_not_available():
    """Test that a TypeError is raised when graph is disabled and
    fixed_decomps or alt_decomps is used."""

    @qml.register_resources({qml.H: 2, qml.CZ: 1})
    def my_cnot(*_, **__):
        raise NotImplementedError

    tape = qml.tape.QuantumScript([])

    with pytest.raises(TypeError, match="The keyword arguments fixed_decomps and alt_decomps"):
        qml.transforms.decompose(tape, fixed_decomps={qml.CNOT: my_cnot})

    with pytest.raises(TypeError, match="The keyword arguments fixed_decomps and alt_decomps"):
        qml.transforms.decompose(tape, alt_decomps={qml.CNOT: [my_cnot]})


@pytest.mark.usefixtures("enable_graph_decomposition")
class TestDecomposeGraphEnabled:
    """Tests the decompose transform with graph enabled."""

    @pytest.mark.unit
    def test_callable_gate_set_not_available(self):
        """Tests that a callable gate set is not available with graph enabled."""

        tape = qml.tape.QuantumScript([])
        with pytest.raises(TypeError, match="Specifying gate_set as a function"):
            qml.transforms.decompose(tape, gate_set=lambda op: True)

    @pytest.mark.integration
    def test_mixed_gate_set_specification(self):
        """Tests that the gate_set can be specified as both a type and a string."""

        tape = qml.tape.QuantumScript([qml.RX(0.5, wires=[0]), qml.CNOT(wires=[0, 1])])
        [new_tape], _ = qml.transforms.decompose(tape, gate_set={"RX", qml.CNOT})
        assert new_tape.operations == tape.operations

    @pytest.mark.integration
    def test_gate_set_targeted_decompositions(self):
        """Tests that a simple circuit is correctly decomposed into different gate sets."""

        tape = qml.tape.QuantumScript(
            [
                qml.H(0),  # non-parametric op
                qml.Rot(0.1, 0.2, 0.3, wires=[0]),  # parametric single-qubit op
                qml.MultiRZ(0.5, wires=[0, 1, 2]),  # parametric multi-qubit op
            ]
        )

        [new_tape], _ = qml.transforms.decompose(tape, gate_set={"Hadamard", "CNOT", "RZ", "RY"})
        assert new_tape.operations == [
            # H is in the target gate set
            qml.H(0),
            # Rot decomposes to ZYZ
            qml.RZ(0.1, wires=[0]),
            qml.RY(0.2, wires=[0]),
            qml.RZ(0.3, wires=[0]),
            # Decomposition of MultiRZ
            qml.CNOT(wires=[2, 1]),
            qml.CNOT(wires=[1, 0]),
            qml.RZ(0.5, wires=[0]),
            qml.CNOT(wires=[1, 0]),
            qml.CNOT(wires=[2, 1]),
        ]

        [new_tape], _ = qml.transforms.decompose(tape, gate_set={"RY", "RZ", "CZ", "GlobalPhase"})
        assert new_tape.operations == [
            # The H decomposes to RZ and RY
            qml.RZ(np.pi, wires=[0]),
            qml.RY(np.pi / 2, wires=[0]),
            qml.GlobalPhase(-np.pi / 2),
            # Rot decomposes to ZYZ
            qml.RZ(0.1, wires=[0]),
            qml.RY(0.2, wires=[0]),
            qml.RZ(0.3, wires=[0]),
            # CNOT decomposes to H and CZ, where H decomposes to RZ and RY
            qml.RZ(np.pi, wires=[1]),
            qml.RY(np.pi / 2, wires=[1]),
            qml.GlobalPhase(-np.pi / 2),
            qml.CZ(wires=[2, 1]),
            qml.RZ(np.pi, wires=[1]),
            qml.RY(np.pi / 2, wires=[1]),
            qml.GlobalPhase(-np.pi / 2),
            # second CNOT
            qml.RZ(np.pi, wires=[0]),
            qml.RY(np.pi / 2, wires=[0]),
            qml.GlobalPhase(-np.pi / 2),
            qml.CZ(wires=[1, 0]),
            qml.RZ(np.pi, wires=[0]),
            qml.RY(np.pi / 2, wires=[0]),
            qml.GlobalPhase(-np.pi / 2),
            # The middle RZ
            qml.RZ(0.5, wires=[0]),
            # The last two CNOTs
            qml.RZ(np.pi, wires=[0]),
            qml.RY(np.pi / 2, wires=[0]),
            qml.GlobalPhase(-np.pi / 2),
            qml.CZ(wires=[1, 0]),
            qml.RZ(np.pi, wires=[0]),
            qml.RY(np.pi / 2, wires=[0]),
            qml.GlobalPhase(-np.pi / 2),
            qml.RZ(np.pi, wires=[1]),
            qml.RY(np.pi / 2, wires=[1]),
            qml.GlobalPhase(-np.pi / 2),
            qml.CZ(wires=[2, 1]),
            qml.RZ(np.pi, wires=[1]),
            qml.RY(np.pi / 2, wires=[1]),
            qml.GlobalPhase(-np.pi / 2),
        ]

    @pytest.mark.integration
    def test_fixed_decomp(self):
        """Tests that a fixed decomposition rule is used instead of the stock ones."""

        @qml.register_resources({qml.RY: 2, qml.CZ: 1, qml.Z: 2})
        def my_cnot(wires, **__):
            qml.RY(np.pi / 2, wires[1])
            qml.Z(wires[1])
            qml.CZ(wires=wires)
            qml.RY(np.pi / 2, wires[1])
            qml.Z(wires[1])

        tape = qml.tape.QuantumScript([qml.CNOT(wires=[1, 0])])
        [new_tape], _ = qml.transforms.decompose(
            tape,
            gate_set={"RY", "RZ", "CZ", "Hadamard", "GlobalPhase"},
            fixed_decomps={qml.CNOT: my_cnot},
        )
        assert new_tape.operations == [
            qml.RY(np.pi / 2, wires=[0]),
            qml.RZ(np.pi, wires=[0]),
            qml.GlobalPhase(-np.pi / 2),
            qml.CZ(wires=[1, 0]),
            qml.RY(np.pi / 2, wires=[0]),
            qml.RZ(np.pi, wires=[0]),
            qml.GlobalPhase(-np.pi / 2),
        ]

    @pytest.mark.integration
    def test_alt_decomp_not_used(self):
        """Tests that alt_decomp isn't necessarily used if it's not efficient."""

        @qml.register_resources({qml.RY: 2, qml.CZ: 1, qml.Z: 2})
        def my_cnot(wires, **__):
            qml.RY(np.pi / 2, wires[1])
            qml.Z(wires[1])
            qml.CZ(wires=wires)
            qml.RY(np.pi / 2, wires[1])
            qml.Z(wires[1])

        tape = qml.tape.QuantumScript([qml.CNOT(wires=[1, 0])])
        [new_tape], _ = qml.transforms.decompose(
            tape,
            gate_set={"RY", "RZ", "CZ", "Hadamard", "GlobalPhase"},
            alt_decomps={qml.CNOT: [my_cnot]},
        )
        assert new_tape.operations == [
            qml.H(0),
            qml.CZ(wires=[1, 0]),
            qml.H(0),
        ]

    @pytest.mark.integration
    def test_alt_decomp(self):
        """Tests that alternative decomposition rules are used when applicable."""

        @qml.register_resources({qml.RY: 2, qml.CZ: 1, qml.Z: 2})
        def my_cnot(wires, **__):
            qml.RY(np.pi / 2, wires[1])
            qml.Z(wires[1])
            qml.CZ(wires=wires)
            qml.RY(np.pi / 2, wires[1])
            qml.Z(wires[1])

        tape = qml.tape.QuantumScript([qml.CNOT(wires=[1, 0])])
        [new_tape], _ = qml.transforms.decompose(
            tape,
            gate_set={"RY", "RZ", "CZ", "PauliZ", "GlobalPhase"},
            alt_decomps={qml.CNOT: [my_cnot]},
        )
        assert new_tape.operations == [
            qml.RY(np.pi / 2, wires=[0]),
            qml.Z(0),
            qml.CZ(wires=[1, 0]),
            qml.RY(np.pi / 2, wires=[0]),
            qml.Z(0),
        ]

    @pytest.mark.integration
    def test_fall_back(self):
        """Tests that op.decompose() is used for ops unsolved in the graph."""

        class CustomOp(qml.operation.Operation):  # pylint: disable=too-few-public-methods
            """Dummy custom op."""

            resource_keys = set()

            @property
            def resource_params(self):
                return {}

            def decomposition(self):
                return [qml.H(self.wires[1]), qml.CNOT(self.wires), qml.H(self.wires[1])]

        @qml.register_resources({qml.CZ: 1})
        def my_decomp(wires, **__):
            qml.CZ(wires=wires)

        tape = qml.tape.QuantumScript([CustomOp(wires=[0, 1])])
        [new_tape], _ = qml.transforms.decompose(
            tape, gate_set={"CNOT", "Hadamard"}, fixed_decomps={CustomOp: my_decomp}
        )
        assert new_tape.operations == [qml.H(1), qml.CNOT(wires=[0, 1]), qml.H(1)]

    @pytest.mark.integration
    def test_controlled_decomp(self):
        """Tests decomposing a controlled operation."""

        # The C(MultiRZ) is decomposed by applying control on the base decomposition.
        # The decomposition of MultiRZ contains two CNOTs
        # So this also tests applying control on an PauliX based operation
        # The decomposition of MultiRZ also contains an RZ gate
        # So this also tests logic involving custom controlled operators.
        ops = [qml.ctrl(qml.MultiRZ(0.5, wires=[0, 1]), control=[2])]
        tape = qml.tape.QuantumScript(ops)
        [new_tape], _ = qml.transforms.decompose(tape, gate_set={"RZ", "CNOT", "Toffoli"})
        assert new_tape.operations == [
            # Decomposition of C(CNOT)
            qml.Toffoli(wires=[2, 1, 0]),
            # Decomposition of C(RZ) -> CRZ
            qml.RZ(0.25, wires=[0]),
            qml.CNOT(wires=[2, 0]),
            qml.RZ(-0.25, wires=[0]),
            qml.CNOT(wires=[2, 0]),
            # Decomposition of C(CNOT)
            qml.Toffoli(wires=[2, 1, 0]),
        ]

    @pytest.mark.integration
    def test_adjoint_decomp(self):
        """Tests decomposing an adjoint operation."""

        class CustomOp(qml.operation.Operator):  # pylint: disable=too-few-public-methods

            resource_keys = set()

            @property
            def resource_params(self) -> dict:
                return {}

        @qml.register_resources({qml.RX: 1, qml.RY: 1, qml.RZ: 1})
        def custom_decomp(theta, phi, omega, wires):
            qml.RX(theta, wires[0])
            qml.RY(phi, wires[0])
            qml.RZ(omega, wires[0])

        tape = qml.tape.QuantumScript(
            [
                qml.adjoint(qml.RX(0.5, wires=[0])),
                qml.adjoint(qml.adjoint(qml.MultiRZ(0.5, wires=[0, 1]))),
                qml.adjoint(CustomOp(0.1, 0.2, 0.3, wires=[0])),
            ]
        )
        [new_tape], _ = qml.transforms.decompose(
            tape, gate_set={"CNOT", "RX", "RY", "RZ"}, fixed_decomps={CustomOp: custom_decomp}
        )
        assert new_tape.operations == [
            qml.RX(-0.5, wires=[0]),
            qml.CNOT(wires=[1, 0]),
            qml.RZ(0.5, wires=[0]),
            qml.CNOT(wires=[1, 0]),
            qml.RZ(-0.3, wires=[0]),
            qml.RY(-0.2, wires=[0]),
            qml.RX(-0.1, wires=[0]),
        ]
