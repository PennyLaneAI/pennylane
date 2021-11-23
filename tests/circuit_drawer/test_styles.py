# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
Unit tests for the pennylane.circuit_drawer.styles` module.
"""

import pytest
from pennylane.circuit_drawer import styles

plt = pytest.importorskip("matplotlib.pyplot")


def test_black_white_style():
    """Tests the black white style sets ``plt.rcParams`` with correct values"""

    styles.black_white()

    assert plt.rcParams["patch.facecolor"] == "white"
    assert plt.rcParams["patch.edgecolor"] == "black"
    assert plt.rcParams["patch.linewidth"] == 2
    assert plt.rcParams["patch.force_edgecolor"]  # = True
    assert plt.rcParams["lines.color"] == "black"
    assert plt.rcParams["text.color"] == "black"


def test_black_white_style_dark():
    """Tests the black white style dark sets ``plt.rcParams`` with correct values"""

    styles.black_white_dark()

    almost_black = "#151515"
    assert plt.rcParams["figure.facecolor"] == almost_black
    assert plt.rcParams["axes.facecolor"] == almost_black
    assert plt.rcParams["patch.edgecolor"] == "white"
    assert plt.rcParams["patch.facecolor"] == almost_black
    assert plt.rcParams["patch.force_edgecolor"]
    assert plt.rcParams["lines.color"] == "white"
    assert plt.rcParams["text.color"] == "white"
