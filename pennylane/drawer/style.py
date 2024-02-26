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
* update the docstrings for ``use_style`` and ``draw_mpl``.
* Add an entry to ``doc/code/qml_drawer.rst``
* Add a test in ``tests/drawer/test_style.py``

Use the decorator ``_needs_mpl`` on style functions to raise appropriate
errors if ``matplotlib`` is not installed.
"""


_has_mpl = True  # pragma: no cover
try:  # pragma: no cover
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
except (ModuleNotFoundError, ImportError) as e:  # pragma: no cover
    _has_mpl = False


# pragma: no cover
def _needs_mpl(func):
    def wrapper():
        if not _has_mpl:  # pragma: no cover
            raise ImportError(
                "The drawer style module requires matplotlib. "
                "You can install matplotlib via \n\n   pip install matplotlib"
            )
        func()

    return wrapper


@_needs_mpl
def _black_white():
    """Apply the black and white style to matplotlib's configuration. This function
    modifies ``plt.rcParams``.
    """
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["patch.facecolor"] = "white"
    plt.rcParams["patch.edgecolor"] = "black"
    plt.rcParams["patch.linewidth"] = 3.0
    plt.rcParams["patch.force_edgecolor"] = True
    plt.rcParams["lines.color"] = "black"
    plt.rcParams["text.color"] = "black"
    plt.rcParams["path.sketch"] = None
    plt.rcParams["font.weight"] = "normal"
    plt.rcParams["lines.linewidth"] = 1.5


@_needs_mpl
def _black_white_dark():
    """Apply the black and white dark style to matplotlib's configuration. This functions modifies ``plt.rcParams``."""
    almost_black = "#151515"  # less harsh than full black
    plt.rcParams["savefig.facecolor"] = almost_black
    plt.rcParams["figure.facecolor"] = almost_black
    plt.rcParams["axes.facecolor"] = almost_black
    plt.rcParams["patch.edgecolor"] = "white"
    plt.rcParams["patch.facecolor"] = almost_black
    plt.rcParams["patch.force_edgecolor"] = True
    plt.rcParams["lines.color"] = "white"
    plt.rcParams["text.color"] = "white"
    plt.rcParams["path.sketch"] = None


@_needs_mpl
def _solarized_light():
    """Apply the solarized light style to matplotlib's configuration. This function
    modifies ``plt.rcParams``.
    """
    plt.rcParams["savefig.facecolor"] = "#fdf6e3"
    plt.rcParams["figure.facecolor"] = "#fdf6e3"
    plt.rcParams["axes.facecolor"] = "#eee8d5"
    plt.rcParams["patch.edgecolor"] = "#93a1a1"
    plt.rcParams["patch.linewidth"] = 3.0
    plt.rcParams["patch.facecolor"] = "#eee8d5"
    plt.rcParams["lines.color"] = "#657b83"
    plt.rcParams["text.color"] = "#586e75"
    plt.rcParams["patch.force_edgecolor"] = True
    plt.rcParams["path.sketch"] = None


@_needs_mpl
def _solarized_dark():
    """Apply the solarized light style to matplotlib's configuration. This function
    modifies ``plt.rcParams``.
    """
    plt.rcParams["savefig.facecolor"] = "#002b36"
    plt.rcParams["figure.facecolor"] = "#002b36"
    plt.rcParams["axes.facecolor"] = "#002b36"
    plt.rcParams["patch.edgecolor"] = "#268bd2"
    plt.rcParams["patch.linewidth"] = 3.0
    plt.rcParams["patch.facecolor"] = "#073642"
    plt.rcParams["lines.color"] = "#839496"
    plt.rcParams["text.color"] = "#2aa198"
    plt.rcParams["patch.force_edgecolor"] = True
    plt.rcParams["path.sketch"] = None


@_needs_mpl
def _sketch():
    """Apply the sketch style to matplotlib's configuration. This function
    modifies ``plt.rcParams``.
    """
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "#D6F5E2"
    plt.rcParams["patch.facecolor"] = "#FFEED4"
    plt.rcParams["patch.edgecolor"] = "black"
    plt.rcParams["patch.linewidth"] = 3.0
    plt.rcParams["patch.force_edgecolor"] = True
    plt.rcParams["lines.color"] = "black"
    plt.rcParams["text.color"] = "black"
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["path.sketch"] = (1, 100, 2)


@_needs_mpl
def _pennylane():
    """Apply the PennyLane style to matplotlib's configuration. This function
    modifies ``plt.rcParams``.
    """
    almost_black = "#151515"
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "#FFB5F1"
    plt.rcParams["patch.facecolor"] = "#D5F0FD"
    plt.rcParams["patch.edgecolor"] = almost_black
    plt.rcParams["patch.linewidth"] = 2.0
    plt.rcParams["patch.force_edgecolor"] = True
    plt.rcParams["lines.color"] = "black"
    plt.rcParams["text.color"] = "black"
    if "Quicksand" in {font.name for font in fm.FontManager().ttflist}:  # pragma: no cover
        plt.rcParams["font.family"] = "Quicksand"
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["path.sketch"] = None


@_needs_mpl
def _pennylane_sketch():
    """Apply the PennyLane-Sketch style to matplotlib's configuration. This function
    modifies ``plt.rcParams``.
    """
    _pennylane()
    plt.rcParams["path.sketch"] = (1, 250, 1)


@_needs_mpl
def _sketch_dark():
    """Apply the sketch dark style to matplotlib's configuration. This function
    modifies ``plt.rcParams``.
    """
    almost_black = "#151515"  # less harsh than full black
    plt.rcParams["figure.facecolor"] = almost_black
    plt.rcParams["savefig.facecolor"] = almost_black
    plt.rcParams["axes.facecolor"] = "#EBAAC1"
    plt.rcParams["patch.facecolor"] = "#B0B5DC"
    plt.rcParams["patch.edgecolor"] = "white"
    plt.rcParams["patch.linewidth"] = 3.0
    plt.rcParams["patch.force_edgecolor"] = True
    plt.rcParams["lines.color"] = "white"
    plt.rcParams["text.color"] = "white"
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["path.sketch"] = (1, 100, 2)


_styles_map = {
    "black_white": _black_white,
    "black_white_dark": _black_white_dark,
    "sketch": _sketch,
    "pennylane": _pennylane,
    "pennylane_sketch": _pennylane_sketch,
    "sketch_dark": _sketch_dark,
    "solarized_light": _solarized_light,
    "solarized_dark": _solarized_dark,
    "default": _needs_mpl(lambda: plt.style.use("default")),
}

__current_style_fn = _black_white


def available_styles():
    """Get available style specification strings.

    Returns:
        tuple(str)
    """
    return tuple(_styles_map)


def use_style(style: str):
    """Set a style setting. Reset to default style using ``use_style('black_white')``

    Args:
        style (str): A style specification.

    Current styles:

    * ``'default'``
    * ``'black_white'``
    * ``'black_white_dark'``
    * ``'sketch'``
    * ``'pennylane'``
    * ``'pennylane_sketch'``
    * ``'sketch_dark'``
    * ``'solarized_light'``
    * ``'solarized_dark'``

    **Example**:

    .. code-block:: python

        qml.drawer.use_style('black_white')

        @qml.qnode(qml.device('lightning.qubit', wires=(0,1,2,3)))
        def circuit(x, z):
            qml.QFT(wires=(0,1,2,3))
            qml.Toffoli(wires=(0,1,2))
            qml.CSWAP(wires=(0,2,3))
            qml.RX(x, wires=0)
            qml.CRZ(z, wires=(3,0))
            return qml.expval(qml.Z(0))


        fig, ax = qml.draw_mpl(circuit)(1.2345,1.2345)
        fig.show()

    .. figure:: ../../_static/style/black_white_style.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

    """
    global __current_style_fn  # pylint:disable=global-statement
    if style in _styles_map:
        __current_style_fn = _styles_map[style]
    else:
        raise TypeError(
            f"style '{style}' provided to ``qml.drawer.use_style`` "
            f"does not exist.  Available options are {available_styles()}"
        )


def _set_style(style: str = None):
    """
    Execute a style function to change the current rcParams.

    Args:
        style (Optional[str]): A style specification. If no style is provided,
            the latest style set with ``use_style`` is used instead.
    """
    if not style:
        __current_style_fn()
    elif style in _styles_map:
        _styles_map[style]()
    else:
        raise TypeError(
            f"style '{style}' provided to ``_set_style`` "
            f"does not exist.  Available options are {available_styles()}"
        )
