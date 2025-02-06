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

import pytest

import pennylane as qml
import pennylane.numpy as np
from pennylane.ftqc import ParametricMidMeasureMP, diagonalize_mcms
from pennylane.wires import Wires

# ToDo: probably move this into the relevant functions instead of skipping the whole file
mpl = pytest.importorskip("matplotlib")
plt = pytest.importorskip("matplotlib.pyplot")

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
        m6 = ParametricMidMeasureMP(Wires(0), angle=1.23, id="m1", plane="XY")

        assert m1.hash != m2.hash
        assert m1.hash != m3.hash
        assert m1.hash != m4.hash
        assert m1.hash != m5.hash
        assert m1.hash == m6.hash

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
            ("XZ", 1.2, 0, "measure_xz(wires=[0], angle=1.2)"),
            ("YZ", 1.2, 0, "measure_yz(wires=[0], angle=1.2)"),
            ("XY", np.pi, 0, "measure_xy(wires=[0], angle=3.141592653589793)"),
            ("XY", 2.345, 1, "measure_xy(wires=[1], angle=2.345)"),
        ],
    )
    def test_repr(self, plane, angle, wire, expected):
        """Test the repr for ParametricMidMeasureMP is correct"""
        mp = ParametricMidMeasureMP(wires=wire, angle=angle, plane=plane)
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
            ("XZ", None, False, "┤↗ˣᶻ├"),
            ("XZ", None, True, "┤↗ˣᶻ│  │0⟩"),
            ("XZ", 0, False, "┤↗ˣᶻ₀├"),
            ("XZ", 0, True, "┤↗ˣᶻ₀│  │0⟩"),
            ("XZ", 1, False, "┤↗ˣᶻ₁├"),
            ("XZ", 1, True, "┤↗ˣᶻ₁│  │0⟩"),
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
        mp = ParametricMidMeasureMP(0, angle=1.23, postselect=postselect, reset=reset, plane=plane)

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
        mp = ParametricMidMeasureMP(0, angle=np.pi / 4, postselect=postselect, plane="XY")

        label = mp.label(decimals=decimals)
        assert label == expected

    @pytest.mark.parametrize("angle", [1.2])
    def test_diagonalizing_gates_xy(self, angle):
        op = ParametricMidMeasureMP([0], angle=angle, plane="XY")

        assert op.has_diagonalizing_gates

        raise RuntimeError

    @pytest.mark.parametrize("angle", [1.2])
    def test_diagonalizing_gates_yz(self, angle):
        op = ParametricMidMeasureMP([0], angle=angle, plane="YZ")

        assert op.has_diagonalizing_gates

        raise RuntimeError

    @pytest.mark.parametrize("angle", [1.2])
    def test_diagonalizing_gates_xz(self, angle):
        op = ParametricMidMeasureMP([0], angle=angle, plane="XZ")

        assert op.has_diagonalizing_gates

        raise RuntimeError


class TestDrawParametricMidMeasure:
    def test_draw_mpl_label(self):
        """Test that the plane label is added to the MCM in a mpl drawing"""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circ():
            ParametricMidMeasureMP(0, angle=np.pi / 4, plane="XY")
            return qml.expval(qml.Z(0))

        _, ax = qml.draw_mpl(circ)()

        assert len(ax.texts) == 2  # one wire label, 1 box label on the MCM

        assert ax.texts[0].get_text() == "0"
        assert ax.texts[1].get_text() == "XY"

        plt.close()

    def test_draw_mpl_reset(self):
        """Test that the reset is added after the MCM as expected in a mpl drawing"""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circ():
            ParametricMidMeasureMP(0, angle=np.pi / 4, plane="XY", reset=True)
            return qml.expval(qml.Z(0))

        _, ax = qml.draw_mpl(circ)()

        assert len(ax.texts) == 3  # one wire label, 1 box label on the MCM, one reset box

        assert ax.texts[0].get_text() == "0"
        assert ax.texts[1].get_text() == "XY"
        assert ax.texts[2].get_text() == "|0⟩"

        plt.close()

    def test_text_drawer(self):
        """Test that the text drawer works as expected"""
        # ToDo: is this redundant because we already tested the label works above,
        #  and elsewhere in the test suite we (presumably) test that the object
        #  label is used in the drawer?

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, mcm_method="tree-traversal")
        def circ():
            ParametricMidMeasureMP(0, angle=np.pi / 4, plane="XY")
            return qml.expval(qml.Z(0))

        assert qml.draw(circ)() == "0: ──┤↗ˣʸ(0.79)├─┤  <Z>"


class TestDiagonalization:
    def test_diagonalize_mcm_transform(self):
        """Test that the diagonalize_mcm transform works as expected on a tape
        containing ParametricMidMeasureMPs"""

        tape = qml.tape.QuantumScript(
            [qml.RY(np.pi / 4, 0), ParametricMidMeasureMP(0, angle=np.pi, plane="XY")]
        )

        (new_tape,), _ = diagonalize_mcms(tape)

        assert len(new_tape.operations) == 3
        assert new_tape.operations[0] == qml.RY(np.pi / 4, 0)
        assert new_tape.operations[1] == tape.operations[1].diagonalizing_gates()[0]

        assert isinstance(new_tape.operations[2], ParametricMidMeasureMP)
        assert new_tape.operations[2].wires == tape.operations[1].wires
        assert new_tape.operations[2].angle == 0
        assert new_tape.operations[2].plane == "YZ"
        assert np.allclose(
            new_tape.operations[2].diagonalizing_gates()[0].matrix(), qml.I(0).matrix()
        )

    def test_diagonalize_mcm_in_cond(self):
        raise RuntimeError


class TestIntegration:
    # ToDo: ask Lee for input on testing practices - we want to know whether the
    #  combination of these MCM strategies and the MCM diagonalization play
    #  nicely together to give a meaningful result. This feels complicated
    #  to test without just testing the full workflow, but also seems like something
    #  he would say should be an integration test - and it could be, if I were
    #  confident that I knew the expected output for applying the transform programs
    #  to a plxpr to make something executable.

    # parametrize over single-branch-statistics, dynamic-one-shot, tree-traversal

    def test_execution(self):
        raise RuntimeError

    def test_execution_in_cond(self):
        raise RuntimeError
