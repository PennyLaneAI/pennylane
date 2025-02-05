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
# pylint: disable=too-few-public-methods, too-many-public-methods


class TestParametricMidMeasure:
    """Tests for the measure function"""

    def test_shortname(self):
        return RuntimeError

    def test_flatten_unflatten(self):
        return RuntimeError

    def test_hash(self):
        """Test that the hash for `MidMeasureMP` is defined correctly."""
        m1 = ParametricMidMeasureMP(Wires(0), id="m1")
        m2 = ParametricMidMeasureMP(Wires(0), id="m2")
        m3 = ParametricMidMeasureMP(Wires(1), id="m1")
        m4 = ParametricMidMeasureMP(Wires(0), id="m1")

        assert m1.hash != m2.hash
        assert m1.hash != m3.hash
        assert m1.hash == m4.hash

        raise RuntimeError

    def test_repr(self):
        raise RuntimeError

    @pytest.mark.parametrize(
        "postselect, reset, expected",
        [
            (None, False, "┤↗├"),
            (None, True, "┤↗│  │0⟩"),
            (0, False, "┤↗₀├"),
            (0, True, "┤↗₀│  │0⟩"),
            (1, False, "┤↗₁├"),
            (1, True, "┤↗₁│  │0⟩"),
        ],
    )
    def test_label(self, postselect, reset, expected):
        """Test that the label for a MidMeasureMP is correct"""
        mp = ParametricMidMeasureMP(0, postselect=postselect, reset=reset)

        label = mp.label()
        assert label == expected

    def test_diagonalizing_gates(self):

        op = ParametricMidMeasureMP([0], angle=np.pi, plane="XY")

        assert op.has_diagonalizing_gates

        raise RuntimeError


class TestDrawParametricMidMeasure():

    def test_draw_mpl(self):
        raise RuntimeError

    def test_text_drawer(self):
        raise RuntimeError


class TestDiagonalization():

    def test_diagonalize_mcm_transform(self):
        raise RuntimeError

    def test_diagonalize_mcm_in_cond(self):
        raise RuntimeError


class TestIntegration():

    # parametrize over single-branch-statistics, dynamic-one-shot, tree-traversal

    def test_execution(self):
        raise RuntimeError

    def test_execution_in_cond(self):
        raise RuntimeError
