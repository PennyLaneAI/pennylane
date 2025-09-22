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
from collections import defaultdict
from functools import partial

import numpy as np
import pytest

import pennylane as qml
from pennylane.operation import Operation
from pennylane.transforms.decompose import _resolve_gate_set


@pytest.mark.unit
def test_weighted_graph_handles_negative_weight():
    """Tests a DecompositionGraph raises a ValueError when given negative weights."""

    tape = qml.tape.QuantumScript([])

    # edge case: negative gate weight
    with pytest.raises(ValueError, match="Negative gate weights"):
        qml.transforms.decompose(tape, gate_set={"CNOT": -10.0, "RZ": 1.0})


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


class CustomOpDynamicWireDecomp(Operation):  # pylint: disable=too-few-public-methods
    """A custom operation."""

    resource_keys = set()

    @property
    def resource_params(self):
        return {}


@qml.register_resources({qml.Toffoli: 2, qml.CRot: 1}, work_wires={"burnable": 2})
def _decomp_with_work_wire(wires, **__):
    with qml.allocation.allocate(2, state="zero", restored=False) as work_wires:
        qml.Toffoli(wires=[wires[0], wires[1], work_wires[0]])
        qml.Toffoli(wires=[wires[1], work_wires[0], work_wires[1]])
        qml.CRot(0.1, 0.2, 0.3, wires=[work_wires[1], wires[2]])


@qml.register_resources({qml.Toffoli: 4, qml.CRot: 3})
def _decomp_without_work_wire(wires, **__):
    qml.Toffoli(wires=wires)
    qml.CRot(0.1, 0, 0, wires=[wires[0], wires[1]])
    qml.Toffoli(wires=wires[::-1])
    qml.CRot(0, 0.2, 0, wires=[wires[1], wires[2]])
    qml.Toffoli(wires=wires)
    qml.CRot(0, 0, 0.3, wires=[wires[2], wires[0]])
    qml.Toffoli(wires=wires[::1])


class LargeOpDynamicWireDecomp(Operation):  # pylint: disable=too-few-public-methods
    """A larger custom operation."""

    resource_keys = set()

    @property
    def resource_params(self):
        return {}


@qml.register_resources({qml.Toffoli: 2, CustomOpDynamicWireDecomp: 2}, work_wires={"zeroed": 1})
def _decomp2_with_work_wire(wires, **__):
    with qml.allocation.allocate(1, state="zero", restored=True) as work_wires:
        qml.Toffoli(wires=[wires[0], wires[1], work_wires[0]])
        CustomOpDynamicWireDecomp(wires=[work_wires[0], wires[2], wires[3]])
        qml.Toffoli(wires=[wires[0], wires[1], work_wires[0]])
        CustomOpDynamicWireDecomp(wires=[wires[1], wires[2], wires[3]])


@qml.register_resources({qml.Toffoli: 4, CustomOpDynamicWireDecomp: 2})
def _decomp2_without_work_wire(wires, **__):
    qml.Toffoli(wires=[wires[0], wires[1], wires[2]])
    CustomOpDynamicWireDecomp(wires=[wires[1], wires[2], wires[3]])
    qml.Toffoli(wires=[wires[1], wires[2], wires[3]])
    qml.Toffoli(wires=[wires[2], wires[3], wires[4]])
    CustomOpDynamicWireDecomp(wires=[wires[2], wires[3], wires[4]])
    qml.Toffoli(wires=[wires[2], wires[1], wires[0]])


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

        @qml.register_resources({qml.CRZ: 1})
        def my_decomp(wires, **__):
            qml.CRZ(np.pi, wires=wires)

        tape = qml.tape.QuantumScript([CustomOp(wires=[0, 1])])

        with pytest.warns(UserWarning, match="The graph-based decomposition system is unable"):
            [new_tape], _ = qml.transforms.decompose(
                [tape],
                gate_set={"CNOT", "Hadamard"},
                fixed_decomps={CustomOp: my_decomp},
            )

        assert new_tape.operations == [qml.H(1), qml.CNOT(wires=[0, 1]), qml.H(1)]

    @pytest.mark.integration
    def test_global_phase_warning(self):
        """Tests that a sensible warning is raised when the graph fails to find a solution
        due to GlobalPhase not being part of the gate set."""

        tape = qml.tape.QuantumScript([qml.X(0)])

        with pytest.warns(UserWarning, match="GlobalPhase is not assumed"):
            with pytest.warns(UserWarning, match="The graph-based decomposition system is unable"):
                [new_tape], _ = qml.transforms.decompose([tape], gate_set={"RX"})

        assert new_tape.operations == [qml.RX(np.pi, wires=0), qml.GlobalPhase(-np.pi / 2, wires=0)]

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

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "num_work_wires, expected_gate_count",
        [
            (
                None,
                {
                    qml.Toffoli: 2 + 2 * 2 + 2,
                    qml.RZ: 2 * 3 + 3,
                    qml.RY: 2 * 2 + 2,
                    qml.CNOT: 2 * 2 + 2,
                },
            ),
            (
                0,
                {
                    qml.Toffoli: 4 + 2 * 4 + 4,
                    qml.RZ: 2 * 3 * 3 + 3 * 3,
                    qml.RY: 2 * 3 * 2 + 3 * 2,
                    qml.CNOT: 2 * 3 * 2 + 3 * 2,
                },
            ),
            (
                1,
                {
                    qml.Toffoli: 2 + 2 * 4 + 4,
                    qml.RZ: 2 * 3 * 3 + 3 * 3,
                    qml.RY: 2 * 3 * 2 + 3 * 2,
                    qml.CNOT: 2 * 3 * 2 + 3 * 2,
                },
            ),
            (
                2,
                {
                    qml.Toffoli: 4 + 2 * 2 + 2,
                    qml.RZ: 2 * 3 + 3,
                    qml.RY: 2 * 2 + 2,
                    qml.CNOT: 2 * 2 + 2,
                },
            ),
            (
                3,
                {
                    qml.Toffoli: 2 + 2 * 2 + 2,
                    qml.RZ: 2 * 3 + 3,
                    qml.RY: 2 * 2 + 2,
                    qml.CNOT: 2 * 2 + 2,
                },
            ),
        ],
    )
    def test_dynamic_work_wire_allocation(self, num_work_wires, expected_gate_count):
        """Tests that the decompose transform supports dynamic wire allocation."""

        op1 = LargeOpDynamicWireDecomp(wires=[0, 1, 2, 3, 4])
        op2 = CustomOpDynamicWireDecomp(wires=[0, 1, 2])
        tape = qml.tape.QuantumScript([op1, op2])

        [decomp], _ = qml.transforms.decompose(
            [tape],
            gate_set={qml.Toffoli, qml.RZ, qml.RY, qml.CNOT},
            max_work_wires=num_work_wires,
            alt_decomps={
                CustomOpDynamicWireDecomp: [_decomp_without_work_wire, _decomp_with_work_wire],
                LargeOpDynamicWireDecomp: [_decomp2_without_work_wire, _decomp2_with_work_wire],
            },
        )
        if num_work_wires is None:
            [result], _ = qml.transforms.resolve_dynamic_wires([decomp], min_int=5)
        else:
            [result], _ = qml.transforms.resolve_dynamic_wires(
                [decomp], zeroed=range(5, 5 + num_work_wires)
            )

        gate_counts = defaultdict(int)
        for op in result.operations:
            if isinstance(op, qml.measurements.MidMeasureMP):
                continue
            gate_counts[type(op)] += 1
        assert gate_counts == expected_gate_count


@pytest.mark.capture
@pytest.mark.system
@pytest.mark.usefixtures("enable_graph_decomposition")
def test_decompose_qnode():
    """Tests that the decompose transform works with a QNode."""

    @partial(qml.transforms.decompose, gate_set={"CZ", "Hadamard"})
    @qml.qnode(qml.device("default.qubit", wires=2))
    def circuit():
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0))

    res = circuit()
    assert qml.math.allclose(res, 1.0)


@pytest.mark.unit
@pytest.mark.usefixtures("enable_graph_decomposition")
def test_stopping_condition_graph_enabled():
    """Tests that the stopping condition is resolved correctly when the graph is disabled."""

    def _stopping_condition(op):
        return op.num_params > 0 and all(qml.math.allclose(p, 0) for p in op.parameters)

    gate_set, stopping_condition = _resolve_gate_set({"RX", "RY"}, _stopping_condition)
    assert stopping_condition(qml.RX(0.1, wires=0))
    assert stopping_condition(qml.RY(0.1, wires=0))
    assert not stopping_condition(qml.RZ(0.1, wires=0))
    assert stopping_condition(qml.RZ(0, wires=1))
    assert gate_set == {"RX", "RY"}


@pytest.mark.integration
@pytest.mark.usefixtures("enable_graph_decomposition")
def test_stopping_condition():
    """Tests that the stopping condition is respected."""

    # Prepare a unitary matrix that we want to decompose
    U = qml.matrix(qml.Rot(0.1, 0.2, 0.3, wires=0) @ qml.Identity(wires=1))

    def stopping_condition(op):

        if isinstance(op, qml.QubitUnitary):
            identity = qml.math.eye(2 ** len(op.wires))
            return qml.math.allclose(op.matrix(), identity)

        return False

    tape = qml.tape.QuantumScript([qml.QubitUnitary(U, wires=[0, 1])])

    [decomp], _ = qml.transforms.decompose(
        tape,
        gate_set={qml.RZ, qml.RY, qml.GlobalPhase, qml.CNOT},
        stopping_condition=stopping_condition,
    )

    assert decomp.operations == [
        qml.RZ(0.1, wires=[0]),
        qml.RY(0.2, wires=[0]),
        qml.RZ(0.3, wires=[0]),
        qml.QubitUnitary(qml.math.eye(2), wires=[1]),
    ]
