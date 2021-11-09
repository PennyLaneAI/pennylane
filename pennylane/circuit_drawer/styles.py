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
This module contains styles for using matplotlib graphics
"""

has_mpl = True
try:
    import matplotlib.pyplot as plt
except (ModuleNotFoundError, ImportError) as e:
    has_mpl = False


def black_white_style(inverted=False):
    """Apply the black and white style to matplotlib's configuration. This functions modifies ``plt.rcParams``.

    Args:
        inverted (bool): If ``True``, plots white on black instead of black on white.

    The style can be reset with ``plt.style.use('default')``.
    """

    if not has_mpl:
        raise ImportError(
            "``black_white_style`` requires matplotlib."
            "You can install matplotlib via \n\n   pip install matplotlib"
        )

    if inverted:
        almost_black = "#151515"  # less harsh than full black
        plt.rcParams["figure.facecolor"] = almost_black
        plt.rcParams["axes.facecolor"] = almost_black
        plt.rcParams["patch.edgecolor"] = "white"
        plt.rcParams["patch.facecolor"] = almost_black
        plt.rcParams["patch.force_edgecolor"] = True
        plt.rcParams["lines.color"] = "white"
        plt.rcParams["text.color"] = "white"
    else:
        plt.rcParams["patch.facecolor"] = "white"
        plt.rcParams["patch.edgecolor"] = "black"
        plt.rcParams["patch.linewidth"] = 2
        plt.rcParams["patch.force_edgecolor"] = True
        plt.rcParams["lines.color"] = "black"
        plt.rcParams["text.color"] = "black"
