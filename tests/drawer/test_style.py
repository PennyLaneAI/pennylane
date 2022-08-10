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
Unit tests for the pennylane.drawer.style` module.
"""

import pytest
import pennylane as qml

plt = pytest.importorskip("matplotlib.pyplot")


def test_available_styles():
    """Assert ``available_styles`` returns tuple of available styles."""

    assert qml.drawer.available_styles() == (
        "black_white",
        "black_white_dark",
        "sketch",
        "sketch_dark",
        "solarized_light",
        "solarized_dark",
        "default",
    )


def test_black_white_style():
    """Tests the black white style sets ``plt.rcParams`` with correct values"""

    qml.drawer.use_style("black_white")
    assert plt.rcParams["savefig.facecolor"] == "white"
    assert plt.rcParams["figure.facecolor"] == "white"
    assert plt.rcParams["axes.facecolor"] == "white"
    assert plt.rcParams["patch.facecolor"] == "white"
    assert plt.rcParams["patch.edgecolor"] == "black"
    assert plt.rcParams["patch.linewidth"] == 3.0
    assert plt.rcParams["patch.force_edgecolor"]  # = True
    assert plt.rcParams["lines.color"] == "black"
    assert plt.rcParams["text.color"] == "black"
    assert plt.rcParams["path.sketch"] == None

    plt.style.use("default")


def test_black_white_style_dark():
    """Tests the black white style dark sets ``plt.rcParams`` with correct values"""

    qml.drawer.use_style("black_white_dark")

    almost_black = "#151515"
    assert plt.rcParams["savefig.facecolor"] == almost_black
    assert plt.rcParams["figure.facecolor"] == almost_black
    assert plt.rcParams["axes.facecolor"] == almost_black
    assert plt.rcParams["patch.edgecolor"] == "white"
    assert plt.rcParams["patch.facecolor"] == almost_black
    assert plt.rcParams["patch.force_edgecolor"]  # = True
    assert plt.rcParams["lines.color"] == "white"
    assert plt.rcParams["text.color"] == "white"
    assert plt.rcParams["path.sketch"] == None

    plt.style.use("default")


def test_sketch_style():
    """Tests the sketch style sets ``plt.rcParams`` with correct values"""

    qml.drawer.use_style("sketch")

    assert plt.rcParams["figure.facecolor"] == "white"
    assert plt.rcParams["savefig.facecolor"] == "white"
    assert plt.rcParams["axes.facecolor"] == "#D6F5E2"
    assert plt.rcParams["patch.facecolor"] == "#FFEED4"
    assert plt.rcParams["patch.edgecolor"] == "black"
    assert plt.rcParams["patch.linewidth"] == 3.0
    assert plt.rcParams["patch.force_edgecolor"]  # = True
    assert plt.rcParams["lines.color"] == "black"
    assert plt.rcParams["text.color"] == "black"
    assert plt.rcParams["font.weight"] == "bold"
    assert plt.rcParams["path.sketch"] == (1, 100, 2)

    plt.style.use("default")


def test_sketch_style_dark():
    """Tests the sketch style dark sets ``plt.rcParams`` with correct values"""

    qml.drawer.use_style("sketch_dark")

    almost_black = "#151515"  # less harsh than full black
    assert plt.rcParams["figure.facecolor"] == almost_black
    assert plt.rcParams["savefig.facecolor"] == almost_black
    assert plt.rcParams["axes.facecolor"] == "#EBAAC1"
    assert plt.rcParams["patch.facecolor"] == "#B0B5DC"
    assert plt.rcParams["patch.edgecolor"] == "white"
    assert plt.rcParams["patch.linewidth"] == 3.0
    assert plt.rcParams["patch.force_edgecolor"]  # = True
    assert plt.rcParams["lines.color"] == "white"
    assert plt.rcParams["text.color"] == "white"
    assert plt.rcParams["font.weight"] == "bold"
    assert plt.rcParams["path.sketch"] == (1, 100, 2)
    plt.style.use("default")


def test_solarized_light_style():
    """Tests the solarized light style sets ``plt.rcParams`` with correct values"""

    qml.drawer.use_style("solarized_light")
    assert plt.rcParams["patch.linewidth"] == 3.0
    assert plt.rcParams["savefig.facecolor"] == "#fdf6e3"
    assert plt.rcParams["figure.facecolor"] == "#fdf6e3"
    assert plt.rcParams["axes.facecolor"] == "#eee8d5"
    assert plt.rcParams["patch.edgecolor"] == "#93a1a1"
    assert plt.rcParams["patch.facecolor"] == "#eee8d5"
    assert plt.rcParams["lines.color"] == "#657b83"
    assert plt.rcParams["text.color"] == "#586e75"
    assert plt.rcParams["patch.force_edgecolor"]  # = True
    assert plt.rcParams["path.sketch"] == None

    plt.style.use("default")


def test_solarized_dark_style():
    """Tests the solarized dark style sets ``plt.rcParams`` with correct values"""

    qml.drawer.use_style("solarized_dark")
    assert plt.rcParams["patch.linewidth"] == 3.0
    assert plt.rcParams["savefig.facecolor"] == "#002b36"
    assert plt.rcParams["figure.facecolor"] == "#002b36"
    assert plt.rcParams["axes.facecolor"] == "#002b36"
    assert plt.rcParams["patch.edgecolor"] == "#268bd2"
    assert plt.rcParams["patch.facecolor"] == "#073642"
    assert plt.rcParams["lines.color"] == "#839496"
    assert plt.rcParams["text.color"] == "#2aa198"
    assert plt.rcParams["patch.force_edgecolor"]  # = True
    assert plt.rcParams["path.sketch"] == None

    plt.style.use("default")


def test_default():
    """Tests default option resets the same as ``plt.style.use``."""

    plt.style.use("default")
    initial = plt.rcParams.copy()

    qml.drawer.use_style("black_white")
    qml.drawer.use_style("default")
    new = plt.rcParams.copy()

    # make sure dictionaries are the same
    assert sum(1 for key, value in initial.items() if new[key] != value) == 0


def test_style_none_error():
    """Tests proper error raised when style doesn't exist."""

    with pytest.raises(TypeError, match="style 'none' provided to ``qml.drawer.use_style``"):
        qml.drawer.use_style("none")
