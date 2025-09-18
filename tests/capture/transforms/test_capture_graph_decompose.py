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

"""
Tests the ``DecomposeInterpreter`` with the new graph-based decomposition system enabled.
"""

# pylint: disable=no-name-in-module, too-few-public-methods, wrong-import-position, protected-access

from collections import defaultdict

import numpy as np
import pytest

import pennylane as qml
from pennylane.operation import Operation

jax = pytest.importorskip("jax")
from pennylane.tape.plxpr_conversion import CollectOpsandMeas
from pennylane.transforms.decompose import DecomposeInterpreter
from pennylane.transforms.resolve_dynamic_wires import resolve_dynamic_wires

pytestmark = [pytest.mark.jax, pytest.mark.capture]


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
class TestDecomposeInterpreterGraphEnabled:
    """Tests the DecomposeInterpreter with the new graph-based decomposition system enabled."""

    @pytest.mark.unit
    def test_gate_set_contains(self):
        """Tests specifying the target gate set."""

        interpreter = DecomposeInterpreter(gate_set={qml.RX, "RZ", "CNOT"})
        assert interpreter.stopping_condition(qml.RX(1.5, 0))
        assert interpreter.stopping_condition(qml.RZ(1.5, 0))
        assert interpreter.stopping_condition(qml.CNOT(wires=[0, 1]))
        assert not interpreter.stopping_condition(qml.Hadamard(0))

    @pytest.mark.unit
    def test_callable_gate_set_not_supported(self):
        """Tests that specifying the gate_set as a function raises an error."""

        with pytest.raises(TypeError, match="Specifying gate_set as a function"):
            DecomposeInterpreter(gate_set=lambda op: op.name in {"RX", "RZ", "CNOT"})

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

        @DecomposeInterpreter(gate_set={"CNOT", "Hadamard"}, fixed_decomps={CustomOp: my_decomp})
        def f():
            CustomOp(wires=[0, 1])

        jaxpr = jax.make_jaxpr(f)()
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)
        assert collector.state["ops"] == [qml.H(1), qml.CNOT(wires=[0, 1]), qml.H(1)]

    @pytest.mark.integration
    def test_gate_set_targeted_decompositions(self):
        """Tests that a simple circuit is correctly decomposed into different gate sets."""

        def f(x, y, z):
            qml.H(0)
            qml.Rot(x, y, z, wires=0)
            qml.MultiRZ(x, wires=[0, 1, 2])

        decomposed_f = DecomposeInterpreter(gate_set={"Hadamard", "CNOT", "RZ", "RY"})(f)
        jaxpr = jax.make_jaxpr(decomposed_f)(0.1, 0.2, 0.3)
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, 0.1, 0.2, 0.3)
        assert collector.state["ops"] == [
            # H is in the target gate set
            qml.H(0),
            # Rot decomposes to ZYZ
            qml.RZ(0.1, wires=[0]),
            qml.RY(0.2, wires=[0]),
            qml.RZ(0.3, wires=[0]),
            # Decomposition of MultiRZ
            qml.CNOT(wires=[2, 1]),
            qml.CNOT(wires=[1, 0]),
            qml.RZ(0.1, wires=[0]),
            qml.CNOT(wires=[1, 0]),
            qml.CNOT(wires=[2, 1]),
        ]

        decomposed_f = DecomposeInterpreter(gate_set={"RY", "RZ", "CZ", "GlobalPhase"})(f)
        jaxpr = jax.make_jaxpr(decomposed_f)(0.1, 0.2, 0.3)
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, 0.1, 0.2, 0.3)
        assert collector.state["ops"] == [
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
            qml.RZ(0.1, wires=[0]),
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
    def test_decompose_controlled(self):
        """Tests that controlled decomposition works."""

        # The C(MultiRZ) is decomposed by applying control on the base decomposition.
        # The decomposition of MultiRZ contains two CNOTs
        # So this also tests applying control on an PauliX based operation
        # The decomposition of MultiRZ also contains an RZ gate
        # So this also tests logic involving custom controlled operators.
        @DecomposeInterpreter(gate_set={"RZ", "CNOT", "Toffoli"})
        def f(x):
            qml.ctrl(qml.MultiRZ(x, wires=[0, 1]), control=[2])

        jaxpr = jax.make_jaxpr(f)(0.5)
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, 0.5)
        assert collector.state["ops"] == [
            # Decomposition of C(CNOT)
            qml.Toffoli(wires=[2, 1, 0]),
            # Decomposition of C(RZ) -> CRZ
            qml.RZ(qml.math.array(0.25, like="jax"), wires=[0]),
            qml.CNOT(wires=[2, 0]),
            qml.RZ(qml.math.array(-0.25, like="jax"), wires=[0]),
            qml.CNOT(wires=[2, 0]),
            # Decomposition of C(CNOT)
            qml.Toffoli(wires=[2, 1, 0]),
        ]

    @pytest.mark.integration
    def test_control_transform(self):
        """Tests that a controlled transform can be decomposed correctly."""

        def inner_f(theta, wires):
            qml.RZ(theta, wires=wires[0])
            qml.CNOT(wires=wires[:2])
            qml.X(wires=wires[0])

        @DecomposeInterpreter(gate_set={"RZ", "CNOT", "Toffoli"})
        def f(x):
            qml.ctrl(inner_f, control=[2])(x, wires=[0, 1])

        jaxpr = jax.make_jaxpr(f)(0.5)
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, 0.5)
        assert collector.state["ops"] == [
            qml.RZ(qml.math.array(0.25, like="jax"), wires=[0]),
            qml.CNOT(wires=[2, 0]),
            qml.RZ(qml.math.array(-0.25, like="jax"), wires=[0]),
            qml.CNOT(wires=[2, 0]),
            qml.Toffoli(wires=[2, 0, 1]),
            qml.CNOT(wires=[2, 0]),
        ]

    @pytest.mark.xfail(reason="DecomposeInterpreter cannot handle adjoint transforms [sc-87096]")
    @pytest.mark.integration
    def test_decompose_adjoint(self):
        """Tests that an adjoint operation is decomposed."""

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

        @DecomposeInterpreter(
            gate_set={"CNOT", "RX", "RY", "RZ"}, fixed_decomps={CustomOp: custom_decomp}
        )
        def f(x, y, z):
            qml.adjoint(qml.RX(x, wires=[0]))
            qml.adjoint(qml.adjoint(qml.MultiRZ(x, wires=[0, 1])))
            qml.adjoint(CustomOp(x, y, z, wires=[0]))

        jaxpr = jax.make_jaxpr(f)(0.1, 0.2, 0.3)
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, 0.1, 0.2, 0.3)
        assert collector.state["ops"] == [
            qml.RX(qml.math.array(-0.1, like="jax"), wires=[0]),
            qml.CNOT(wires=[1, 0]),
            qml.RZ(0.1, wires=[0]),
            qml.CNOT(wires=[1, 0]),
            qml.RZ(qml.math.array(-0.3, like="jax"), wires=[0]),
            qml.RY(qml.math.array(-0.2, like="jax"), wires=[0]),
            qml.RX(qml.math.array(-0.1, like="jax"), wires=[0]),
        ]

    @pytest.mark.xfail(reason="DecomposeInterpreter cannot handle adjoint transforms [sc-87096]")
    @pytest.mark.integration
    def test_adjoint_transform(self):
        """Tests that an adjoint transform can be decomposed correctly."""

        def inner_f(theta, phi, omega, wires):
            qml.RX(theta, wires[0])
            qml.RY(phi, wires[0])
            qml.RZ(omega, wires[0])

        @DecomposeInterpreter(gate_set={"CNOT", "RX", "RY", "RZ"})
        def f(x, y, z):
            qml.adjoint(inner_f)(x, y, z, wires=[0])

        jaxpr = jax.make_jaxpr(f)(0.1, 0.2, 0.3)
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, 0.1, 0.2, 0.3)
        assert collector.state["ops"] == [
            qml.RZ(qml.math.array(-0.3, like="jax"), wires=[0]),
            qml.RY(qml.math.array(-0.2, like="jax"), wires=[0]),
            qml.RX(qml.math.array(-0.1, like="jax"), wires=[0]),
        ]

    @pytest.mark.integration
    def test_cond(self):
        """Tests that a circuit containing conditionals can be decomposed correctly."""

        @DecomposeInterpreter(gate_set={"CNOT", "RX", "RY", "RZ"})
        def f(x, wires):

            def true_fn():
                qml.CRX(x, wires=wires)

            def false_fn():
                qml.CRZ(x, wires=wires)

            qml.cond(x > 0.5, true_fn, false_fn)()

        # The PLxPR is constructed with the true_fn branch
        jaxpr = jax.make_jaxpr(f)(0.6, [0, 1])
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, 0.6, *[0, 1])
        assert collector.state["ops"] == [
            qml.RZ(np.pi / 2, wires=[1]),
            qml.RY(qml.math.array(0.6 / 2, like="jax"), wires=[1]),
            qml.CNOT(wires=[0, 1]),
            qml.RY(qml.math.array(-0.6 / 2, like="jax"), wires=[1]),
            qml.CNOT(wires=[0, 1]),
            qml.RZ(-np.pi / 2, wires=[1]),
        ]
        # Tests that the false_fn branch is also decomposed
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, 0.2, *[0, 1])
        assert collector.state["ops"] == [
            qml.RZ(qml.math.array(0.1, like="jax"), wires=[1]),
            qml.CNOT(wires=[0, 1]),
            qml.RZ(qml.math.array(-0.1, like="jax"), wires=[1]),
            qml.CNOT(wires=[0, 1]),
        ]

    @pytest.mark.integration
    def test_ctrl_cond(self):
        """Tests that a circuit containing a cond nested in a ctrl is decomposed correctly."""

        @DecomposeInterpreter(gate_set={"CNOT", "RX", "RY", "RZ"})
        def f(x, wires):

            def true_fn():
                qml.RX(x, wires=wires[1])

            def false_fn():
                qml.RZ(x, wires=wires[1])

            qml.ctrl(qml.cond(x > 0.5, true_fn, false_fn), control=wires[0])()

        # The PLxPR is constructed with the true_fn branch
        jaxpr = jax.make_jaxpr(f)(0.6, [0, 1])
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, 0.6, *[0, 1])
        assert collector.state["ops"] == [
            qml.RZ(np.pi / 2, wires=[1]),
            qml.RY(qml.math.array(0.6 / 2, like="jax"), wires=[1]),
            qml.CNOT(wires=[0, 1]),
            qml.RY(qml.math.array(-0.6 / 2, like="jax"), wires=[1]),
            qml.CNOT(wires=[0, 1]),
            qml.RZ(-np.pi / 2, wires=[1]),
        ]
        # Tests that the false_fn branch is also decomposed
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, 0.2, *[0, 1])
        assert collector.state["ops"] == [
            qml.RZ(qml.math.array(0.1, like="jax"), wires=[1]),
            qml.CNOT(wires=[0, 1]),
            qml.RZ(qml.math.array(-0.1, like="jax"), wires=[1]),
            qml.CNOT(wires=[0, 1]),
        ]

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "num_work_wires, expected_gate_count",
        [
            (
                None,
                # Use the most efficient decomposition for both.
                {
                    qml.Toffoli: 2 + 2 * 2 + 2,
                    qml.RZ: 2 * 3 + 3,
                    qml.RY: 2 * 2 + 2,
                    qml.CNOT: 2 * 2 + 2,
                },
            ),
            (
                0,
                # Use the versions that don't use work wires for both.
                {
                    qml.Toffoli: 4 + 2 * 4 + 4,
                    qml.RZ: 2 * 3 * 3 + 3 * 3,
                    qml.RY: 2 * 3 * 2 + 3 * 2,
                    qml.CNOT: 2 * 3 * 2 + 3 * 2,
                },
            ),
            (
                1,
                # When there's only one available, use it to decompose the larger op
                {
                    qml.Toffoli: 2 + 2 * 4 + 4,
                    qml.RZ: 2 * 3 * 3 + 3 * 3,
                    qml.RY: 2 * 3 * 2 + 3 * 2,
                    qml.CNOT: 2 * 3 * 2 + 3 * 2,
                },
            ),
            (
                2,
                # When there are only 2 available, use 1 to decompose the larger op
                {
                    qml.Toffoli: 4 + 2 * 2 + 2,
                    qml.RZ: 2 * 3 + 3,
                    qml.RY: 2 * 2 + 2,
                    qml.CNOT: 2 * 2 + 2,
                },
            ),
            (
                3,
                # Enough wires to choose the most efficient decomposition for both.
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

        @DecomposeInterpreter(
            gate_set={qml.Toffoli, qml.RZ, qml.RY, qml.CNOT},
            max_work_wires=num_work_wires,
            alt_decomps={
                CustomOpDynamicWireDecomp: [_decomp_without_work_wire, _decomp_with_work_wire],
                LargeOpDynamicWireDecomp: [_decomp2_without_work_wire, _decomp2_with_work_wire],
            },
        )
        def f():
            LargeOpDynamicWireDecomp(wires=[0, 1, 2, 3, 4])
            CustomOpDynamicWireDecomp(wires=[0, 1, 2])

        plxpr = jax.make_jaxpr(f)()
        tape = qml.tape.plxpr_to_tape(plxpr.jaxpr, plxpr.consts)

        if num_work_wires is None:
            [result], _ = resolve_dynamic_wires([tape], min_int=5)
        else:
            [result], _ = resolve_dynamic_wires([tape], zeroed=range(5, 5 + num_work_wires))

        gate_counts = defaultdict(int)
        for op in result.operations:
            if isinstance(op, qml.measurements.MidMeasureMP):
                continue
            gate_counts[type(op)] += 1
        assert gate_counts == expected_gate_count
