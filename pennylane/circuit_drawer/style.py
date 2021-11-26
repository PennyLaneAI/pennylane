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

See available styles in the variable ``qml.style.available``. Styles can be reset
with ``plt.style.use('default')``.

This module matches the interfaces in
`matplotlib's style module <https://matplotlib.org/stable/api/style_api.html>`__ ,
 but provides extra styling options.
"""

import contextlib

_has_mpl = True   # pragma: no cover
try:  # pragma: no cover
    import matplotlib.pyplot as plt
    from matplotlib import rc_context, rcdefaults
except (ModuleNotFoundError, ImportError) as e:
    _has_mpl = False

available = ["black_white", "black_white_dark"]
"""Lists all available styling functions"""


def black_white():
    """Apply the black and white style to matplotlib's configuration. This function
    modifies ``plt.rcParams``.

    The style can be reset with ``qml.style.use('default')``.

    **Example**:

    .. code-block:: python

        qml.style.black_white()

        dev = qml.device('lightning.qubit', wires=(0,1,2,3))
        @qml.qnode(dev)
        def circuit(x, z):
            qml.QFT(wires=(0,1,2,3))
            qml.Toffoli(wires=(0,1,2))
            qml.CSWAP(wires=(0,2,3))
            qml.RX(x, wires=0)
            qml.CRZ(z, wires=(3,0))
            return qml.expval(qml.PauliZ(0))


        fig, ax = qml.draw_mpl(circuit)(1.2345,1.2345)
        fig.show()

    .. figure:: ../../_static/styles/black_white_style.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

    """

    if not _has_mpl:  # pragma: no cover
        raise ImportError(
            "``black_white_style`` requires matplotlib."
            "You can install matplotlib via \n\n   pip install matplotlib"
        )

    plt.rcParams["patch.facecolor"] = "white"
    plt.rcParams["patch.edgecolor"] = "black"
    plt.rcParams["patch.linewidth"] = 2
    plt.rcParams["patch.force_edgecolor"] = True
    plt.rcParams["lines.color"] = "black"
    plt.rcParams["text.color"] = "black"


def black_white_dark():
    """Apply the black and white style to matplotlib's configuration. This functions modifies ``plt.rcParams``.

    The style can be reset with ``qml.style.use('default')``.

    **Example**:

    .. code-block:: python

        qml.style.black_white_dark()

        dev = qml.device('lightning.qubit', wires=(0,1,2,3))
        @qml.qnode(dev)
        def circuit(x, z):
            qml.QFT(wires=(0,1,2,3))
            qml.Toffoli(wires=(0,1,2))
            qml.CSWAP(wires=(0,2,3))
            qml.RX(x, wires=0)
            qml.CRZ(z, wires=(3,0))
            return qml.expval(qml.PauliZ(0))


        fig, ax = qml.draw_mpl(circuit)(1.2345,1.2345)
        fig.show()

    .. figure:: ../../_static/styles/black_white_style_dark.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

    """

    if not _has_mpl:  # pragma: no cover
        raise ImportError(
            "``black_white_style`` requires matplotlib."
            "You can install matplotlib via \n\n   pip install matplotlib"
        )

    almost_black = "#151515"  # less harsh than full black
    plt.rcParams["figure.facecolor"] = almost_black
    plt.rcParams["axes.facecolor"] = almost_black
    plt.rcParams["patch.edgecolor"] = "white"
    plt.rcParams["patch.facecolor"] = almost_black
    plt.rcParams["patch.force_edgecolor"] = True
    plt.rcParams["lines.color"] = "white"
    plt.rcParams["text.color"] = "white"

def use(style):
    """Set a style setting. If not a PennyLane style, then the function
    defers to matplotlib's ``plt.style.use``.

    Args:
        style (Union[str, dict, Path, or list]): A style specification. 

    **Example**:

    .. code-block:: python

        qml.style.use('black_white')

        dev = qml.device('lightning.qubit', wires=(0,1,2,3))
        @qml.qnode(dev)
        def circuit(x, z):
            qml.QFT(wires=(0,1,2,3))
            qml.Toffoli(wires=(0,1,2))
            qml.CSWAP(wires=(0,2,3))
            qml.RX(x, wires=0)
            qml.CRZ(z, wires=(3,0))
            return qml.expval(qml.PauliZ(0))


        fig, ax = qml.draw_mpl(circuit)(1.2345,1.2345)
        fig.show()

    .. figure:: ../../_static/styles/black_white_style.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

    """
    if not _has_mpl:  # pragma: no cover
        raise ImportError(
            "``black_white_style`` requires matplotlib."
            "You can install matplotlib via \n\n   pip install matplotlib"
        )

    map = {'black_white': black_white,
        'black_white_dark': black_white_dark}

    if style in map:
        map[style]()
    else:
        plt.style.use(style)

@contextlib.contextmanager
def context(style, after_reset=False):
    """A context manager for setting a style temporarily. If not a PennyLane style,
    then the function defers to matplotlib's ``plt.style.use``.
    
    Args:
        style (Union[str, dict, Path, list]): A style specification. 

    Keyword Args:
        after_reset (Bool): If True, apply style after resetting to their default

    **Example**:

    .. code-block:: python

        with qml.style.context('black_white'):

            dev = qml.device('lightning.qubit', wires=(0,1,2,3))
            @qml.qnode(dev)
            def circuit(x, z):
                qml.QFT(wires=(0,1,2,3))
                qml.Toffoli(wires=(0,1,2))
                qml.CSWAP(wires=(0,2,3))
                qml.RX(x, wires=0)
                qml.CRZ(z, wires=(3,0))
                return qml.expval(qml.PauliZ(0))


            fig, ax = qml.draw_mpl(circuit)(1.2345,1.2345)
            fig.show()

    .. figure:: ../../_static/styles/black_white_style.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

    """
    if not _has_mpl:  # pragma: no cover
        raise ImportError(
            "``black_white_style`` requires matplotlib."
            "You can install matplotlib via \n\n   pip install matplotlib"
        )
    with rc_context():
        if after_reset:
            rcdefaults()
        use(style)
        yield