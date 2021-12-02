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
This module contains styles for using matplotlib graphics.

To add a new style:
* create a private function that modifies ``plt.rcParams``.
* add an entry to the private dictionary ``_style_map``.
* Add an entry to ``doc/code/qml.drawer.rst``
* Add a test in ``tests/drawer/test_style.py``

Use the decorator ``_needs_mpl`` on style functions to raise appropriate
errors if ``matplotlib`` is not installed.
"""


_has_mpl = True  # pragma: no cover
try:  # pragma: no cover
    import matplotlib.pyplot as plt
except (ModuleNotFoundError, ImportError) as e:  # pragma: no cover
    _has_mpl = False

# pragma: no cover
def _needs_mpl(func):
    def wrapper():
        if not _has_mpl:  # pragma: no cover
            raise ImportError(
                "The drawer style module requires matplotlib."
                "You can install matplotlib via \n\n   pip install matplotlib"
            )
        func()

    return wrapper


@_needs_mpl
def _black_white():
    """Apply the black and white style to matplotlib's configuration. This function
    modifies ``plt.rcParams``.
    """
    plt.rcParams["patch.facecolor"] = "white"
    plt.rcParams["patch.edgecolor"] = "black"
    plt.rcParams["patch.linewidth"] = 2
    plt.rcParams["patch.force_edgecolor"] = True
    plt.rcParams["lines.color"] = "black"
    plt.rcParams["text.color"] = "black"


@_needs_mpl
def _black_white_dark():
    """Apply the black and white dark style to matplotlib's configuration. This functions modifies ``plt.rcParams``."""
    almost_black = "#151515"  # less harsh than full black
    plt.rcParams["figure.facecolor"] = almost_black
    plt.rcParams["axes.facecolor"] = almost_black
    plt.rcParams["patch.edgecolor"] = "white"
    plt.rcParams["patch.facecolor"] = almost_black
    plt.rcParams["patch.force_edgecolor"] = True
    plt.rcParams["lines.color"] = "white"
    plt.rcParams["text.color"] = "white"


_styles_map = {
    "black_white": _black_white,
    "black_white_dark": _black_white_dark,
    "default": _needs_mpl(lambda: plt.style.use("default")),
}


def available_styles():
    """Get available style specification strings.

    Returns:
        tuple(str)
    """
    return tuple(_styles_map)


def use_style(style: str):
    """Set a style setting. Reset to default style using ``plt.style.use('default')``

    Args:
        style (str): A style specification.

    Current styles:

    * ``'default'``
    * ``'black_white'``
    * ``'black_white_dark'``

    **Example**:

    .. code-block:: python

        qml.style.use_style('black_white')

        @qml.qnode(qml.device('lightning.qubit', wires=(0,1,2,3)))
        def circuit(x, z):
            qml.QFT(wires=(0,1,2,3))
            qml.Toffoli(wires=(0,1,2))
            qml.CSWAP(wires=(0,2,3))
            qml.RX(x, wires=0)
            qml.CRZ(z, wires=(3,0))
            return qml.expval(qml.PauliZ(0))


        fig, ax = qml.draw_mpl(circuit)(1.2345,1.2345)
        fig.show()

    .. figure:: ../../_static/style/black_white_style.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

    """
    if style in _styles_map:
        _styles_map[style]()
    else:
        raise TypeError(
            f"style '{style}' provided to ``qml.drawer.use_style``"
            f" does not exist.  Available options are {available_styles()}"
        )
