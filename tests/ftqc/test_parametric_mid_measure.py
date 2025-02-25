# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the mid_measure module"""

from functools import partial

import numpy as np
import pytest

import pennylane as qml
from pennylane.devices.qubit import measure as apply_qubit_measurement
from pennylane.ftqc import ParametricMidMeasureMP, diagonalize_mcms
from pennylane.measurements import MeasurementValue, MidMeasureMP
from pennylane.wires import Wires

# pylint: disable=too-few-public-methods, too-many-public-methods


class TestParametricMidMeasure:
    """Tests for the measure function"""

    def test_hash(self):
        """Test that the hash for `ParametricMidMeasureMP` is defined correctly."""
        m1 = ParametricMidMeasureMP(Wires(0), angle=1.23, id="m1", plane="XY")
        m2 = ParametricMidMeasureMP(Wires(1), angle=1.23, id="m1", plane="XY")
        m3 = ParametricMidMeasureMP(Wires(0), angle=2.45, id="m1", plane="XY")
        m4 = ParametricMidMeasureMP(Wires(0), angle=1.23, id="m2", plane="XY")
        m5 = ParametricMidMeasureMP(Wires(0), angle=1.23, id="m1", plane="YZ")
        m6 = ParametricMidMeasureMP(Wires(0), angle=1.23, id="m1", plane="ZX")
        m7 = ParametricMidMeasureMP(Wires(0), angle=1.23, id="m1", plane="XY")

        assert m1.hash != m2.hash
        assert m1.hash != m3.hash
        assert m1.hash != m4.hash
        assert m1.hash != m5.hash
        assert m1.hash != m6.hash
        assert m1.hash == m7.hash

    def test_flatten_unflatten(self):
        """Test that we can flatten and unflatten the ParametricMidMeasureMP"""

        op = ParametricMidMeasureMP(Wires(0), angle=1.23, id="m1", plane="XY")
        data, metadata = op._flatten()  # pylint: disable = protected-access

        assert hash(metadata)  # metadata must be hashable

        unflattened_op = ParametricMidMeasureMP._unflatten(  # pylint: disable = protected-access
            data, metadata
        )
        assert op.hash == unflattened_op.hash

    @pytest.mark.jax
    def test_flatten_unflatten_jax(self):
        """Test that jax.tree_util can flatten and unflatten the ParametricMidMeasureMP"""

        import jax

        op = ParametricMidMeasureMP(Wires(0), angle=1.23, id="m1", plane="XY")

        leaves, struct = jax.tree_util.tree_flatten(op)
        unflattened_op = jax.tree_util.tree_unflatten(struct, leaves)

        assert op.hash == unflattened_op.hash

    @pytest.mark.parametrize(
        "plane, angle, wire, expected",
        [
            ("XY", 1.2, 0, "measure_xy(wires=[0], angle=1.2)"),
            ("ZX", 1.2, 0, "measure_zx(wires=[0], angle=1.2)"),
            ("YZ", 1.2, 0, "measure_yz(wires=[0], angle=1.2)"),
            ("XY", np.pi, 0, "measure_xy(wires=[0], angle=3.141592653589793)"),
            ("XY", 2.345, 1, "measure_xy(wires=[1], angle=2.345)"),
        ],
    )
    def test_repr(self, plane, angle, wire, expected):
        """Test the repr for ParametricMidMeasureMP is correct"""
        mp = ParametricMidMeasureMP(wires=Wires([wire]), angle=angle, plane=plane)
        assert repr(mp) == expected

    @pytest.mark.parametrize(
        "plane, postselect, reset, expected",
        [
            ("XY", None, False, "┤↗ˣʸ├"),
            ("XY", None, True, "┤↗ˣʸ│  │0⟩"),
            ("XY", 0, False, "┤↗ˣʸ₀├"),
            ("XY", 0, True, "┤↗ˣʸ₀│  │0⟩"),
            ("XY", 1, False, "┤↗ˣʸ₁├"),
            ("XY", 1, True, "┤↗ˣʸ₁│  │0⟩"),
            ("ZX", None, False, "┤↗ᶻˣ├"),
            ("ZX", None, True, "┤↗ᶻˣ│  │0⟩"),
            ("ZX", 0, False, "┤↗ᶻˣ₀├"),
            ("ZX", 0, True, "┤↗ᶻˣ₀│  │0⟩"),
            ("ZX", 1, False, "┤↗ᶻˣ₁├"),
            ("ZX", 1, True, "┤↗ᶻˣ₁│  │0⟩"),
            ("YZ", None, False, "┤↗ʸᶻ├"),
            ("YZ", None, True, "┤↗ʸᶻ│  │0⟩"),
            ("YZ", 0, False, "┤↗ʸᶻ₀├"),
            ("YZ", 0, True, "┤↗ʸᶻ₀│  │0⟩"),
            ("YZ", 1, False, "┤↗ʸᶻ₁├"),
            ("YZ", 1, True, "┤↗ʸᶻ₁│  │0⟩"),
        ],
    )
    def test_label_no_decimals(self, plane, postselect, reset, expected):
        """Test that the label for a ParametricMidMeasureMP is correct"""
        mp = ParametricMidMeasureMP(
            Wires([0]), angle=1.23, postselect=postselect, reset=reset, plane=plane
        )

        label = mp.label()
        assert label == expected

    @pytest.mark.parametrize(
        "decimals, postselect, expected",
        [
            (2, None, "┤↗ˣʸ(0.79)├"),
            (2, 0, "┤↗ˣʸ(0.79)₀├"),
            (4, None, "┤↗ˣʸ(0.7854)├"),
            (4, 0, "┤↗ˣʸ(0.7854)₀├"),
        ],
    )
    def test_label_with_decimals(self, decimals, postselect, expected):
        """Test that the label for a ParametricMidMeasureMP is correct when
        decimals are specified in the label function"""
        mp = ParametricMidMeasureMP(Wires([0]), angle=np.pi / 4, postselect=postselect, plane="XY")

        label = mp.label(decimals=decimals)
        assert label == expected

    @pytest.mark.parametrize(
        "plane, measurement_angle, corresponding_obs",
        [
            # XY measurements
            ("XY", 0, qml.X(0)),
            ("XY", np.pi / 4, qml.X(0) + qml.Y(0)),
            ("XY", np.pi / 2, qml.Y(0)),
            ("XY", 3 * np.pi / 4, qml.Y(0) - qml.X(0)),
            ("XY", np.pi, -qml.X(0)),
            ("XY", 5 * np.pi / 4, -qml.X(0) - qml.Y(0)),
            ("XY", -3 * np.pi / 4, -qml.X(0) - qml.Y(0)),
            # ZX measurements
            ("ZX", 0, qml.Z(0)),
            ("ZX", np.pi / 4, qml.X(0) + qml.Z(0)),
            ("ZX", np.pi / 2, qml.X(0)),
            ("ZX", 3 * np.pi / 4, qml.X(0) - qml.Z(0)),
            ("ZX", np.pi, -qml.Z(0)),
            ("ZX", 5 * np.pi / 4, -qml.X(0) - qml.Z(0)),
            ("ZX", -3 * np.pi / 4, -qml.X(0) - qml.Z(0)),
            # YZ measurements
            ("YZ", 0, qml.Z(0)),
            ("YZ", np.pi / 4, -qml.Y(0) + qml.Z(0)),
            ("YZ", np.pi / 2, -qml.Y(0)),
            ("YZ", 3 * np.pi / 4, -qml.Y(0) - qml.Z(0)),
            ("YZ", np.pi, -qml.Z(0)),
            ("YZ", 5 * np.pi / 4, qml.Y(0) - qml.Z(0)),
            ("YZ", -3 * np.pi / 4, qml.Y(0) - qml.Z(0)),
        ],
    )
    def test_diagonalizing_gates(self, plane, measurement_angle, corresponding_obs):
        """Test that diagonalizing a parametrized mid-circuit measurement and measuring
        in the computational basis corresponds to the expected observable"""

        dev = qml.device("default.qubit")

        @diagonalize_mcms
        @qml.qnode(dev, mcm_method="tree-traversal")
        def circ(state, angle):
            qml.StatePrep(state, wires=0)
            mp = ParametricMidMeasureMP([0], angle=angle, plane=plane)
            assert mp.has_diagonalizing_gates
            return qml.expval(qml.Z(0))

        input_state = np.random.random(2) + 1j * np.random.random(2)
        input_state = input_state / np.linalg.norm(input_state)

        res = circ(input_state, measurement_angle)

        expected_res = apply_qubit_measurement(qml.expval(corresponding_obs), input_state)
        if isinstance(corresponding_obs, qml.ops.Sum):
            expected_res = expected_res / np.sqrt(2)

        assert np.allclose(res, expected_res)

    def test_invalid_plane_raises_error(self):
        """Test that an error is raised at diagonalization if a plane other than
        XY, YZ, or ZX are passed."""

        mp = ParametricMidMeasureMP([0], angle=1.23, plane="AB")

        with pytest.raises(NotImplementedError, match="plane not implemented"):
            mp.diagonalizing_gates()


class TestDrawParametricMidMeasure:

    @pytest.mark.matplotlib
    def test_draw_mpl_label(self):
        """Test that the plane label is added to the MCM in a mpl drawing"""

        from matplotlib import pyplot as plt

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circ():
            ParametricMidMeasureMP(Wires([0]), angle=np.pi / 4, plane="XY")
            return qml.expval(qml.Z(0))

        _, ax = qml.draw_mpl(circ)()

        assert len(ax.texts) == 2  # one wire label, 1 box label on the MCM

        assert ax.texts[0].get_text() == "0"
        assert ax.texts[1].get_text() == "XY"

        plt.close()

    @pytest.mark.matplotlib
    def test_draw_mpl_reset(self):
        """Test that the reset is added after the MCM as expected in a mpl drawing"""

        from matplotlib import pyplot as plt

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circ():
            ParametricMidMeasureMP(Wires([0]), angle=np.pi / 4, plane="XY", reset=True)
            return qml.expval(qml.Z(0))

        _, ax = qml.draw_mpl(circ)()

        assert len(ax.texts) == 3  # one wire label, 1 box label on the MCM, one reset box

        assert ax.texts[0].get_text() == "0"
        assert ax.texts[1].get_text() == "XY"
        assert ax.texts[2].get_text() == "|0⟩"

        plt.close()

    def test_text_drawer(self):
        """Test that the text drawer works as expected"""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, mcm_method="tree-traversal")
        def circ():
            ParametricMidMeasureMP(Wires([0]), angle=np.pi / 4, plane="XY")
            return qml.expval(qml.Z(0))

        assert qml.draw(circ)() == "0: ──┤↗ˣʸ(0.79)├─┤  <Z>"


class TestDiagonalizeMCMs:
    def test_diagonalize_mcm_with_no_parametrized_mcms(self):
        """Test that the diagonalize_mcms transform leaves standard operations
        and MidMeasureMP on the tape untouched"""

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(1.2, 0)
            m = qml.measure(0)
            qml.cond(m, qml.RY, qml.RX)(1.2, 0)

        original_tape = qml.tape.QuantumScript.from_queue(q)
        (new_tape,), _ = diagonalize_mcms(qml.tape.QuantumScript.from_queue(q))
        assert qml.equal(original_tape, new_tape)

    def test_diagonalize_mcm_transform(self):
        """Test that the diagonalize_mcm transform works as expected on a tape
        containing ParametricMidMeasureMPs"""

        tape = qml.tape.QuantumScript(
            [qml.RY(np.pi / 4, 0), ParametricMidMeasureMP(Wires([0]), angle=np.pi, plane="XY")]
        )
        diagonalizing_gates = tape.operations[1].diagonalizing_gates()

        (new_tape,), _ = diagonalize_mcms(tape)
        assert len(new_tape.operations) == 4

        assert new_tape.operations[1] == diagonalizing_gates[0]
        assert new_tape.operations[2] == diagonalizing_gates[1]

        assert isinstance(new_tape.operations[3], MidMeasureMP)
        assert not isinstance(new_tape.operations[3], ParametricMidMeasureMP)
        assert new_tape.operations[3].wires == tape.operations[1].wires

    def test_diagonalize_mcm_in_cond(self):
        """Test that the diagonalize_mcm transform works as expected on a tape
        containing a conditional with a ParametricMidMeasureMP"""

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(1.2, 0)
            m = qml.measure(0)
            qml.cond(m == 1, partial(ParametricMidMeasureMP, angle=1.2))(wires=2, plane="XY")

        original_tape = qml.tape.QuantumScript.from_queue(q)
        base_diagonalizing_gates = ParametricMidMeasureMP(
            Wires([2]), angle=1.2, plane="XY"
        ).diagonalizing_gates()

        (new_tape,), _ = diagonalize_mcms(qml.tape.QuantumScript.from_queue(q))
        assert len(new_tape.operations) == 5
        diagonalizing_gates = new_tape.operations[2:4]
        measurement = new_tape.operations[4]

        # conditional diagonalizing gate
        for gate, expected_base in zip(diagonalizing_gates, base_diagonalizing_gates):
            assert isinstance(gate, qml.ops.Conditional)
            assert gate.base == expected_base
            assert gate.wires == original_tape.operations[2].wires

        # conditional diagonalized ParametricMidMeasureMP
        assert isinstance(measurement, qml.ops.Conditional)
        assert measurement.wires == original_tape.operations[2].wires
        assert isinstance(measurement.base, MidMeasureMP)
        assert not isinstance(measurement.base, ParametricMidMeasureMP)

        # cond(diagonalizing gate) and cond(mcm) rely on same measurement in the same way
        assert (
            diagonalizing_gates[0].meas_val.measurements[0] == measurement.meas_val.measurements[0]
        )
        assert diagonalizing_gates[0].meas_val.processing_fn == measurement.meas_val.processing_fn

    def test_diagonalize_mcm_cond_two_outcomes(self):
        """Test that the diagonalize_mcm transform works as expected on a tape
        containing a conditional with two ParametricMidMeasureMPs as the true
        and false condition respectively"""

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(1.2, 0)
            m = qml.measure(0)
            qml.cond(
                m == 1,
                partial(ParametricMidMeasureMP, angle=1.2),
                partial(ParametricMidMeasureMP, angle=2.4),
            )(wires=2, plane="ZX")
        original_tape = qml.tape.QuantumScript.from_queue(q)
        true_diag_gate = ParametricMidMeasureMP(2, angle=1.2, plane="ZX").diagonalizing_gates()[0]
        false_diag_gate = ParametricMidMeasureMP(2, angle=2.4, plane="ZX").diagonalizing_gates()[0]

        (new_tape,), _ = diagonalize_mcms(qml.tape.QuantumScript.from_queue(q))
        assert len(new_tape.operations) == 6

        for idx1, idx2, base_diagonalizing_gate in [
            (2, 3, true_diag_gate),
            (4, 5, false_diag_gate),
        ]:
            diagonalizing_gate = new_tape.operations[idx1]
            measurement = new_tape.operations[idx2]

            # conditional diagonalizing gate
            assert isinstance(diagonalizing_gate, qml.ops.Conditional)
            assert diagonalizing_gate.base == base_diagonalizing_gate
            assert diagonalizing_gate.wires == original_tape.operations[2].wires

            # conditional diagonalized ParametricMidMeasureMP
            assert isinstance(measurement, qml.ops.Conditional)
            assert measurement.wires == original_tape.operations[2].wires
            assert isinstance(measurement.base, MidMeasureMP)
            assert not isinstance(measurement.base, ParametricMidMeasureMP)

        # all 4 conditionals rely on the same measurement value
        for gate in new_tape.operations[3:]:
            assert gate.meas_val.measurements[0] == new_tape.operations[2].meas_val.measurements[0]

        # diagonalized gates each have processing functions matching their respective cond(mcm)
        assert (
            new_tape.operations[2].meas_val.processing_fn
            == new_tape.operations[3].meas_val.processing_fn
        )
        assert (
            new_tape.operations[4].meas_val.processing_fn
            == new_tape.operations[5].meas_val.processing_fn
        )
        assert (
            new_tape.operations[2].meas_val.processing_fn
            != new_tape.operations[4].meas_val.processing_fn
        )

    def test_diagonalizing_measurements_cond_and_op(self):
        """Test that when calling diagonalize_mcms, references to previously
        diagonalized measurements that are stored in conditions on Conditional
        operators are updated to track the measurement on the tape following
        diagonalization, rather than the original object (for conditional
        with MCM condition and MCM applied op)"""

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(1.2, 0)
            mp = ParametricMidMeasureMP(0, angle=1.2, plane="YZ")
            mv = MeasurementValue([mp], processing_fn=lambda v: v)
            qml.cond(mv == 0, partial(ParametricMidMeasureMP, angle=1.2))(wires=2, plane="XY")

        original_tape = qml.tape.QuantumScript.from_queue(q)
        old_mp = original_tape.operations[1]
        assert isinstance(old_mp, ParametricMidMeasureMP)

        (new_tape,), _ = diagonalize_mcms(original_tape)
        assert len(new_tape.operations) == 6
        for op in new_tape.operations[3:]:
            assert isinstance(op, qml.ops.Conditional)

        new_mp = new_tape.operations[2]
        assert not isinstance(new_mp, ParametricMidMeasureMP)
        assert isinstance(new_mp, MidMeasureMP)

        assert new_tape.operations[3].meas_val.measurements == [new_mp]

    def test_diagonalizing_measurements_cond(self):
        """Test that when calling diagonalize_mcms, references to previously
        diagonalized measurements that are stored in conditions on Conditional
        operators are updated to track the measurement on the tape following
        diagonalization, rather than the original object (for conditional
        with MCM condition and non-MCM op)"""

        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(1.2, 0)
            mp = ParametricMidMeasureMP(0, angle=1.2, plane="YZ")
            mv = MeasurementValue([mp], processing_fn=lambda v: v)
            qml.cond(mv == 0, partial(qml.RX, 1.2))(wires=2)

        original_tape = qml.tape.QuantumScript.from_queue(q)
        old_mp = original_tape.operations[1]
        assert isinstance(old_mp, ParametricMidMeasureMP)

        (new_tape,), _ = diagonalize_mcms(original_tape)
        assert len(new_tape.operations) == 4
        for op in new_tape.operations[3:]:
            assert isinstance(op, qml.ops.Conditional)

        new_mp = new_tape.operations[2]
        assert not isinstance(new_mp, ParametricMidMeasureMP)
        assert isinstance(new_mp, MidMeasureMP)

        assert new_tape.operations[3].meas_val.measurements == [new_mp]


class TestWorkflows:
    @pytest.mark.parametrize("mcm_method, shots", [("tree-traversal", None), ("one-shot", 10000)])
    def test_execution(self, mcm_method, shots):
        """Test that we can execute a QNode with a ParametricMidMeasureMP and produce
        an accurate result"""

        dev = qml.device("default.qubit", shots=shots)

        @diagonalize_mcms
        @qml.qnode(dev, mcm_method=mcm_method)
        def circ():
            qml.RX(2.345, 0)
            ParametricMidMeasureMP(0, angle=np.pi / 2, plane="XY")
            return qml.expval(qml.Z(0))

        if shots:
            assert np.isclose(circ(), -np.sin(2.345), atol=0.03)
        else:
            assert np.isclose(circ(), -np.sin(2.345))

    @pytest.mark.xfail(reason="bug with all MidMeasureMPs in cond for both mcm_methods, ")
    @pytest.mark.parametrize("mcm_method, shots", [("tree-traversal", None), ("one-shot", 10000)])
    def test_execution_in_cond(self, mcm_method, shots):
        """Test that we can execute a QNode with a ParametricMidMeasureMP applied in a conditional,
        and produce an accurate result"""

        dev = qml.device("default.qubit", shots=shots)

        @diagonalize_mcms
        @qml.qnode(dev, mcm_method=mcm_method)
        def circ():
            qml.RX(np.pi, 0)
            m = qml.measure(0)  # always 1

            qml.RX(2.345, 1)
            qml.cond(m == 1, ParametricMidMeasureMP)(1, angle=np.pi / 2, plane="XY")
            return qml.expval(qml.Z(0))

        assert np.isclose(circ(), -np.sin(2.345))
