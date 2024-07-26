# pylint: disable=too-many-arguments

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
Contains the drawing function.
"""
import warnings
from functools import wraps

import pennylane as qml
from pennylane.data.base.typing_util import UNSET

from .tape_mpl import tape_mpl
from .tape_text import tape_text


def _determine_draw_level(kwargs, qnode=None):
    level = kwargs.get("level", UNSET)
    expansion_strategy = kwargs.get("expansion_strategy", UNSET)

    if all(val != UNSET for val in (level, expansion_strategy)):
        raise ValueError("Either 'level' or 'expansion_strategy' need to be set, but not both.")

    if expansion_strategy != UNSET:
        warnings.warn(
            "The 'expansion_strategy' argument is deprecated and will be removed in "
            "version 0.39. Instead, use the 'level' argument which offers more flexibility "
            "and options.",
            qml.PennyLaneDeprecationWarning,
        )

    if level == UNSET:
        if expansion_strategy == UNSET:
            return qnode.expansion_strategy if qnode else UNSET
        return expansion_strategy
    return level


def catalyst_qjit(qnode):
    """A method checking whether a qnode is compiled by catalyst.qjit"""
    return qnode.__class__.__name__ == "QJIT" and hasattr(qnode, "user_function")


def draw(
    qnode,
    wire_order=None,
    show_all_wires=False,
    decimals=2,
    max_length=100,
    show_matrices=True,
    **kwargs,
):
    r"""Create a function that draws the given qnode or quantum function.

    Args:
        qnode (.QNode or Callable): the input QNode or quantum function that is to be drawn.
        wire_order (Sequence[Any]): the order (from top to bottom) to print the wires of the circuit.
           If not provided, the wire order defaults to the device wires. If device wires are not
           available, the circuit wires are sorted if possible.
        show_all_wires (bool): If True, all wires, including empty wires, are printed.
        decimals (int): How many decimal points to include when formatting operation parameters.
            ``None`` will omit parameters from operation labels.
        max_length (int): Maximum string width (columns) when printing the circuit
        show_matrices=False (bool): show matrix valued parameters below all circuit diagrams

    Keyword Args:
        level (None, str, int, slice): An indication of what transforms to apply before drawing.
            Check :func:`~.workflow.get_transform_program` for more information on the allowed values and usage details of
            this argument.
        expansion_strategy (str): The strategy to use when circuit expansions or decompositions
            are required.

            - ``gradient``: The QNode will attempt to decompose
              the internal circuit such that all circuit operations are supported by the gradient
              method.

            - ``device``: The QNode will attempt to decompose the internal circuit
              such that all circuit operations are natively supported by the device.

    Returns:
        A function that has the same argument signature as ``qnode``. When called,
        the function will draw the QNode/qfunc.

    .. note::

        At most, one of ``level`` or ``expansion_strategy`` needs to be provided. If neither is provided,
        ``qnode.expansion_strategy`` will be used instead. Users are encouraged to predominantly use ``level``,
        as it allows for the same values as ``expansion_strategy`` and offers more flexibility in choosing
        the desired transforms/expansions.

    .. warning::
        The ``expansion_strategy`` argument is deprecated and will be removed in version 0.39. Use the ``level``
        argument instead to specify the resulting tape you want.

    **Example**

    .. code-block:: python3

        @qml.qnode(qml.device('lightning.qubit', wires=2))
        def circuit(a, w):
            qml.Hadamard(0)
            qml.CRX(a, wires=[0, 1])
            qml.Rot(*w, wires=[1], id="arbitrary")
            qml.CRX(-a, wires=[0, 1])
            return qml.expval(qml.Z(0) @ qml.Z(1))

    >>> print(qml.draw(circuit)(a=2.3, w=[1.2, 3.2, 0.7]))
    0: ──H─╭●─────────────────────────────────────────╭●─────────┤ ╭<Z@Z>
    1: ────╰RX(2.30)──Rot(1.20,3.20,0.70,"arbitrary")─╰RX(-2.30)─┤ ╰<Z@Z>

    .. details::
        :title: Usage Details

        By specifying the ``decimals`` keyword, parameters are displayed to the specified precision.

        >>> print(qml.draw(circuit, decimals=4)(a=2.3, w=[1.2, 3.2, 0.7]))
        0: ──H─╭●─────────────────────────────────────────────────╭●───────────┤ ╭<Z@Z>
        1: ────╰RX(2.3000)──Rot(1.2000,3.2000,0.7000,"arbitrary")─╰RX(-2.3000)─┤ ╰<Z@Z>

        Parameters can be omitted by requesting ``decimals=None``:

        >>> print(qml.draw(circuit, decimals=None)(a=2.3, w=[1.2, 3.2, 0.7]))
        0: ──H─╭●────────────────────╭●──┤ ╭<Z@Z>
        1: ────╰RX──Rot("arbitrary")─╰RX─┤ ╰<Z@Z>

        If the parameters are not acted upon by classical processing like ``-a``, then
        ``qml.draw`` can handle string-valued parameters as well:

        >>> @qml.qnode(qml.device('lightning.qubit', wires=1))
        ... def circuit2(x):
        ...     qml.RX(x, wires=0)
        ...     return qml.expval(qml.Z(0))
        >>> print(qml.draw(circuit2)("x"))
        0: ──RX(x)─┤  <Z>

        When requested with ``show_matrices=True`` (the default), matrix valued parameters
        are printed below the circuit. For ``show_matrices=False``, they are not printed:

        >>> @qml.qnode(qml.device('default.qubit', wires=2))
        ... def circuit3():
        ...     qml.QubitUnitary(np.eye(2), wires=0)
        ...     qml.QubitUnitary(-np.eye(4), wires=(0,1))
        ...     return qml.expval(qml.Hermitian(np.eye(2), wires=1))
        >>> print(qml.draw(circuit3)())
        0: ──U(M0)─╭U(M1)─┤
        1: ────────╰U(M1)─┤  <𝓗(M0)>
        M0 =
        [[1. 0.]
        [0. 1.]]
        M1 =
        [[-1. -0. -0. -0.]
        [-0. -1. -0. -0.]
        [-0. -0. -1. -0.]
        [-0. -0. -0. -1.]]
        >>> print(qml.draw(circuit3, show_matrices=False)())
        0: ──U(M0)─╭U(M1)─┤
        1: ────────╰U(M1)─┤  <𝓗(M0)>

        The ``max_length`` keyword warps long circuits:

        .. code-block:: python

            rng = np.random.default_rng(seed=42)
            shape = qml.StronglyEntanglingLayers.shape(n_wires=3, n_layers=3)
            params = rng.random(shape)

            @qml.qnode(qml.device('lightning.qubit', wires=3))
            def longer_circuit(params):
                qml.StronglyEntanglingLayers(params, wires=range(3))
                return [qml.expval(qml.Z(i)) for i in range(3)]

        >>> print(qml.draw(longer_circuit, max_length=60, level="device")(params))
        0: ──Rot(0.77,0.44,0.86)─╭●────╭X──Rot(0.45,0.37,0.93)─╭●─╭X
        1: ──Rot(0.70,0.09,0.98)─╰X─╭●─│───Rot(0.64,0.82,0.44)─│──╰●
        2: ──Rot(0.76,0.79,0.13)────╰X─╰●──Rot(0.23,0.55,0.06)─╰X───
        ───Rot(0.83,0.63,0.76)──────────────────────╭●────╭X─┤  <Z>
        ──╭X────────────────────Rot(0.35,0.97,0.89)─╰X─╭●─│──┤  <Z>
        ──╰●────────────────────Rot(0.78,0.19,0.47)────╰X─╰●─┤  <Z>

        The ``wire_order`` keyword specifies the order of the wires from
        top to bottom:

        >>> print(qml.draw(circuit, wire_order=[1,0])(a=2.3, w=[1.2, 3.2, 0.7]))
        1: ────╭RX(2.30)──Rot(1.20,3.20,0.70,"arbitrary")─╭RX(-2.30)─┤ ╭<Z@Z>
        0: ──H─╰●─────────────────────────────────────────╰●─────────┤ ╰<Z@Z>

        If the device or ``wire_order`` has wires not used by operations, those wires are omitted
        unless requested with ``show_all_wires=True``

        >>> empty_qfunc = lambda : qml.expval(qml.Z(0))
        >>> empty_circuit = qml.QNode(empty_qfunc, qml.device('lightning.qubit', wires=3))
        >>> print(qml.draw(empty_circuit, show_all_wires=True)())
        0: ───┤  <Z>
        1: ───┤
        2: ───┤

        Drawing also works on batch transformed circuits:

        .. code-block:: python

            from functools import partial
            from pennylane import numpy as pnp

            @partial(qml.gradients.param_shift, shifts=[(0.1,)])
            @qml.qnode(qml.device('default.qubit', wires=1))
            def transformed_circuit(x):
                qml.RX(x, wires=0)
                return qml.expval(qml.Z(0))

        >>> print(qml.draw(transformed_circuit)(pnp.array(1.0, requires_grad=True)))
        0: ──RX(1.10)─┤  <Z>
        0: ──RX(0.90)─┤  <Z>

        The function also accepts quantum functions rather than QNodes. This can be especially
        helpful if you want to visualize only a part of a circuit that may not be convertible into
        a QNode, such as a sub-function that does not return any measurements.

        >>> def qfunc(x):
        ...     qml.RX(x, wires=[0])
        ...     qml.CNOT(wires=[0, 1])
        >>> print(qml.draw(qfunc)(1.1))
        0: ──RX(1.10)─╭●─┤
        1: ───────────╰X─┤

        **Levels:**

        The ``level`` keyword argument allows one to select a subset of the transforms to apply on the ``QNode``
        before carrying out any drawing. Take, for example, this circuit:

        .. code-block:: python

            @qml.transforms.merge_rotations
            @qml.transforms.cancel_inverses
            @qml.qnode(qml.device("default.qubit"), diff_method="parameter-shift")
            def circ(weights, order):
                qml.RandomLayers(weights, wires=(0, 1))
                qml.Permute(order, wires=(0, 1, 2))
                qml.PauliX(0)
                qml.PauliX(0)
                qml.RX(0.1, wires=0)
                qml.RX(-0.1, wires=0)
                return qml.expval(qml.PauliX(0))

            order = [2, 1, 0]
            weights = qml.numpy.array([[1.0, 20]])

        One can print the circuit without any transforms applied by passing ``level="top"`` or ``level=0``:

        >>> print(qml.draw(circ, level="top")(weights, order))
        0: ─╭RandomLayers(M0)─╭Permute──X──X──RX(0.10)──RX(-0.10)─┤  <X>
        1: ─╰RandomLayers(M0)─├Permute────────────────────────────┤
        2: ───────────────────╰Permute────────────────────────────┤
        M0 =
        [[ 1. 20.]]

        Or print the circuit after applying the transforms manually applied on the QNode (``merge_rotations`` and ``cancel_inverses``):

        >>> print(qml.draw(circ, level="user", show_matrices=False)(weights, order))
        0: ─╭RandomLayers(M0)─╭Permute─┤  <X>
        1: ─╰RandomLayers(M0)─├Permute─┤
        2: ───────────────────╰Permute─┤

        To apply all of the transforms, including those carried out by the differentiation method and the device, use ``level=None``:

        >>> print(qml.draw(circ, level=None, show_matrices=False)(weights, order))
        0: ──RY(1.00)──╭SWAP─┤  <X>
        1: ──RX(20.00)─│─────┤
        2: ────────────╰SWAP─┤

        Slices can also be passed to the ``level`` argument. So one can, for example, request that only the ``merge_rotations`` transform is applied:

        >>> print(qml.draw(circ, level=slice(1, 2), show_matrices=False)(weights, order))
        0: ─╭RandomLayers(M0)─╭Permute──X──X─┤  <X>
        1: ─╰RandomLayers(M0)─├Permute───────┤
        2: ───────────────────╰Permute───────┤

    """
    if catalyst_qjit(qnode):
        qnode = qnode.user_function

    if hasattr(qnode, "construct"):
        return _draw_qnode(
            qnode,
            wire_order=wire_order,
            show_all_wires=show_all_wires,
            decimals=decimals,
            max_length=max_length,
            show_matrices=show_matrices,
            level=_determine_draw_level(kwargs, qnode),
        )

    if _determine_draw_level(kwargs) != UNSET:
        warnings.warn(
            "When the input to qml.draw is not a QNode, the expansion_strategy and level arguments are ignored.",
            UserWarning,
        )

    @wraps(qnode)
    def wrapper(*args, **kwargs):
        tape = qml.tape.make_qscript(qnode)(*args, **kwargs)

        if wire_order:
            _wire_order = wire_order
        else:
            try:
                _wire_order = sorted(tape.wires)
            except TypeError:
                _wire_order = tape.wires

        return tape_text(
            tape,
            wire_order=_wire_order,
            show_all_wires=show_all_wires,
            decimals=decimals,
            show_matrices=show_matrices,
            max_length=max_length,
        )

    return wrapper


def _draw_qnode(
    qnode,
    wire_order=None,
    show_all_wires=False,
    decimals=2,
    max_length=100,
    show_matrices=True,
    level=None,
):
    @wraps(qnode)
    def wrapper(*args, **kwargs):
        tapes, _ = qml.workflow.construct_batch(qnode, level=level)(*args, **kwargs)

        if wire_order:
            _wire_order = wire_order
        elif qnode.device.wires:
            _wire_order = qnode.device.wires
        else:
            try:
                _wire_order = sorted(tapes[0].wires)
            except TypeError:
                _wire_order = tapes[0].wires

        if tapes is not None:
            cache = {"tape_offset": 0, "matrices": []}
            res = [
                tape_text(
                    t,
                    wire_order=_wire_order,
                    show_all_wires=show_all_wires,
                    decimals=decimals,
                    show_matrices=False,
                    max_length=max_length,
                    cache=cache,
                )
                for t in tapes
            ]
            if show_matrices and cache["matrices"]:
                mat_str = ""
                for i, mat in enumerate(cache["matrices"]):
                    if qml.math.requires_grad(mat) and hasattr(mat, "detach"):
                        mat = mat.detach()
                    mat_str += f"\nM{i} = \n{mat}"
                if mat_str:
                    mat_str = "\n" + mat_str
                return "\n\n".join(res) + mat_str
            return "\n\n".join(res)

        return tape_text(
            qnode.qtape,
            wire_order=_wire_order,
            show_all_wires=show_all_wires,
            decimals=decimals,
            show_matrices=show_matrices,
            max_length=max_length,
        )

    return wrapper


def draw_mpl(
    qnode,
    wire_order=None,
    show_all_wires=False,
    decimals=None,
    style=None,
    *,
    fig=None,
    **kwargs,
):
    r"""Draw a qnode with matplotlib

    Args:
        qnode (.QNode or Callable): the input QNode/quantum function that is to be drawn.
        wire_order (Sequence[Any]): the order (from top to bottom) to print the wires of the circuit.
           If not provided, the wire order defaults to the device wires. If device wires are not
           available, the circuit wires are sorted if possible.
        show_all_wires (bool): If True, all wires, including empty wires, are printed.
        decimals (int): How many decimal points to include when formatting operation parameters.
            Default ``None`` will omit parameters from operation labels.
        style (str): visual style of plot. Valid strings are ``{'black_white', 'black_white_dark', 'sketch',
            'pennylane', 'pennylane_sketch', 'sketch_dark', 'solarized_light', 'solarized_dark', 'default'}``.
            If no style is specified, the global style set with :func:`~.use_style` will be used, and the
            initial default is 'black_white'. If you would like to use your environment's current rcParams,
            set ``style`` to "rcParams". Setting style does not modify matplotlib global plotting settings.

    Keyword Args:
        fig (None or matplotlib.Figure): Matplotlib figure to plot onto. If None, then create a new figure
        fontsize (float or str): fontsize for text. Valid strings are
            ``{'xx-small', 'x-small', 'small', 'medium', large', 'x-large', 'xx-large'}``.
            Default is ``14``.
        wire_options (dict): matplotlib formatting options for the wire lines
        label_options (dict): matplotlib formatting options for the wire labels
        active_wire_notches (bool): whether or not to add notches indicating active wires.
            Defaults to ``True``.
        level (None, str, int, slice): An indication of what transforms to apply before drawing.
            Check :func:`~.workflow.get_transform_program` for more information on the allowed values and usage details of
            this argument.
        expansion_strategy (str): The strategy to use when circuit expansions or decompositions
            are required.

            - ``gradient``: The QNode will attempt to decompose
              the internal circuit such that all circuit operations are supported by the gradient
              method.

            - ``device``: The QNode will attempt to decompose the internal circuit
              such that all circuit operations are natively supported by the device.

    Returns:
        A function that has the same argument signature as ``qnode``. When called,
        the function will draw the QNode as a tuple of (``matplotlib.figure.Figure``,
        ``matplotlib.axes._axes.Axes``)

    .. note::

        At most, one of ``level`` or ``expansion_strategy`` needs to be provided. If neither is provided,
        ``qnode.expansion_strategy`` will be used instead. Users are encouraged to predominantly use ``level``,
        as it allows for the same values as ``expansion_strategy`` and offers more flexibility in choosing
        the desired transforms/expansions.

    .. warning::
        The ``expansion_strategy`` argument is deprecated and will be removed in version 0.39. Use the ``level``
        argument instead to specify the resulting tape you want.

    .. warning::

        Unlike :func:`~.draw`, this function can not draw the full result of a tape-splitting transform. In such cases,
        only the tape generated first will be plotted.

    **Example**:

    .. code-block:: python

        dev = qml.device('lightning.qubit', wires=(0,1,2,3))

        @qml.qnode(dev)
        def circuit(x, z):
            qml.QFT(wires=(0,1,2,3))
            qml.IsingXX(1.234, wires=(0,2))
            qml.Toffoli(wires=(0,1,2))
            mcm = qml.measure(1)
            mcm_out = qml.measure(2)
            qml.CSWAP(wires=(0,2,3))
            qml.RX(x, wires=0)
            qml.cond(mcm, qml.RY)(np.pi / 4, wires=3)
            qml.CRZ(z, wires=(3,0))
            return qml.expval(qml.Z(0)), qml.probs(op=mcm_out)


        fig, ax = qml.draw_mpl(circuit)(1.2345,1.2345)
        fig.show()

    .. figure:: ../../_static/draw_mpl/main_example.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

    .. details::
        :title: Usage Details

        **Decimals:**

        The keyword ``decimals`` controls how many decimal points to include when labelling the operations.
        The default value ``None`` omits parameters for brevity.

        .. code-block:: python

            @qml.qnode(dev)
            def circuit2(x, y):
                qml.RX(x, wires=0)
                qml.Rot(*y, wires=0)
                return qml.expval(qml.Z(0))

            fig, ax = qml.draw_mpl(circuit2, decimals=2)(1.23456, [1.2345,2.3456,3.456])
            fig.show()

        .. figure:: ../../_static/draw_mpl/decimals.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

        **Wires:**

        The keywords ``wire_order`` and ``show_all_wires`` control the location of wires from top to bottom.

        .. code-block:: python

            fig, ax = qml.draw_mpl(circuit, wire_order=[3,2,1,0])(1.2345,1.2345)
            fig.show()

        .. figure:: ../../_static/draw_mpl/wire_order.png
                :align: center
                :width: 60%
                :target: javascript:void(0);

        If a wire is in ``wire_order``, but not in the ``tape``, it will be omitted by default.  Only by selecting
        ``show_all_wires=True`` will empty wires be displayed.

        .. code-block:: python

            fig, ax = qml.draw_mpl(circuit, wire_order=["aux"], show_all_wires=True)(1.2345,1.2345)
            fig.show()

        .. figure:: ../../_static/draw_mpl/show_all_wires.png
                :align: center
                :width: 60%
                :target: javascript:void(0);

        **Integration with matplotlib:**

        This function returns matplotlib figure and axes objects. Using these objects,
        users can perform further customization of the graphic.

        .. code-block:: python

            fig, ax = qml.draw_mpl(circuit)(1.2345,1.2345)
            fig.suptitle("My Circuit", fontsize="xx-large")

            options = {'facecolor': "white", 'edgecolor': "#f57e7e", "linewidth": 6, "zorder": -1}
            box1 = plt.Rectangle((-0.5, -0.5), width=3.0, height=4.0, **options)
            ax.add_patch(box1)

            ax.annotate("CSWAP", xy=(5, 2.5), xycoords='data', xytext=(5.8,1.5), textcoords='data',
                        arrowprops={'facecolor': 'black'}, fontsize=14)

            ax.annotate("classical control flow", xy=(3.5, 4.2), xycoords='data', xytext=(0.8,4.2),
                        textcoords='data', arrowprops={'facecolor': 'blue'}, fontsize=14,
                        va="center")
            fig.show()

        .. figure:: ../../_static/draw_mpl/postprocessing.png
                :align: center
                :width: 60%
                :target: javascript:void(0);

        **Formatting:**

        PennyLane has inbuilt styles for controlling the appearance of the circuit drawings.
        All available styles can be determined by evaluating ``qml.drawer.available_styles()``.
        Any available string can then be passed via the kwarg ``style`` to change the settings for
        that plot. This will not affect style settings for subsequent matplotlib plots.

        .. code-block:: python

            fig, ax = qml.draw_mpl(circuit, style='sketch')(1.2345,1.2345)
            fig.show()


        .. figure:: ../../_static/draw_mpl/sketch_style.png
                :align: center
                :width: 60%
                :target: javascript:void(0);

        You can also control the appearance with matplotlib's provided tools, see the
        `matplotlib docs <https://matplotlib.org/stable/tutorials/introductory/customizing.html>`_ .
        For example, we can customize ``plt.rcParams``. To use a customized appearance based on matplotlib's
        ``plt.rcParams``, ``qml.draw_mpl`` must be run with ``style="rcParams"``:

        .. code-block:: python

            plt.rcParams['patch.facecolor'] = 'mistyrose'
            plt.rcParams['patch.edgecolor'] = 'maroon'
            plt.rcParams['text.color'] = 'maroon'
            plt.rcParams['font.weight'] = 'bold'
            plt.rcParams['patch.linewidth'] = 4
            plt.rcParams['patch.force_edgecolor'] = True
            plt.rcParams['lines.color'] = 'indigo'
            plt.rcParams['lines.linewidth'] = 2
            plt.rcParams['figure.facecolor'] = 'ghostwhite'

            fig, ax = qml.draw_mpl(circuit, style="rcParams")(1.2345,1.2345)
            fig.show()

        .. figure:: ../../_static/draw_mpl/rcparams.png
                :align: center
                :width: 60%
                :target: javascript:void(0);

        The wires and wire labels can be manually formatted by passing in dictionaries of
        keyword-value pairs of matplotlib options. ``wire_options`` accepts options for lines,
        and ``label_options`` accepts text options.

        .. code-block:: python

            fig, ax = qml.draw_mpl(circuit, wire_options={'color':'teal', 'linewidth': 5},
                        label_options={'size': 20})(1.2345,1.2345)
            fig.show()

        .. figure:: ../../_static/draw_mpl/wires_labels.png
                :align: center
                :width: 60%
                :target: javascript:void(0);

        **Levels:**

        The ``level`` keyword argument allows one to select a subset of the transforms to apply on the ``QNode``
        before carrying out any drawing. Take, for example, this circuit:

        .. code-block:: python

            @qml.transforms.merge_rotations
            @qml.transforms.cancel_inverses
            @qml.qnode(qml.device("default.qubit"), diff_method="parameter-shift")
            def circ():
                qml.RandomLayers([[1.0, 20]], wires=(0, 1))
                qml.Permute([2, 1, 0], wires=(0, 1, 2))
                qml.PauliX(0)
                qml.PauliX(0)
                qml.RX(0.1, wires=0)
                qml.RX(-0.1, wires=0)
                return qml.expval(qml.PauliX(0))

        One can plot the circuit without any transforms applied by passing ``level="top"`` or ``level=0``:

        .. code-block:: python

            fig, ax = qml.draw_mpl(circ, level="top")()
            fig.show()

        .. figure:: ../../_static/draw_mpl/level_top.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

        Or plot the circuit after applying the transforms manually applied on the QNode (``merge_rotations`` and ``cancel_inverses``):

        .. code-block:: python

            fig, ax = qml.draw_mpl(circ, level="user")()
            fig.show()

        .. figure:: ../../_static/draw_mpl/level_user.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

        To apply all of the transforms, including those carried out by the differentiation method and the device, use ``level=None``:

        .. code-block:: python

            fig, ax = qml.draw_mpl(circ, level=None)()
            fig.show()

        .. figure:: ../../_static/draw_mpl/level_none.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

        Slices can also be passed to the ``level`` argument. So one can, for example, request that only the ``merge_rotations`` transform is applied:

        .. code-block:: python

            fig, ax = qml.draw_mpl(circ, level=slice(1, 2))()
            fig.show()

        .. figure:: ../../_static/draw_mpl/level_slice.png
            :align: center
            :width: 60%
            :target: javascript:void(0);


    """
    if catalyst_qjit(qnode):
        qnode = qnode.user_function

    if hasattr(qnode, "construct"):
        resolved_level = _determine_draw_level(kwargs, qnode)

        kwargs.pop("expansion_strategy", None)
        kwargs.pop("level", None)

        return _draw_mpl_qnode(
            qnode,
            wire_order=wire_order,
            show_all_wires=show_all_wires,
            decimals=decimals,
            level=resolved_level,
            style=style,
            fig=fig,
            **kwargs,
        )

    if _determine_draw_level(kwargs) != UNSET:
        warnings.warn(
            "When the input to qml.draw is not a QNode, the expansion_strategy and level arguments are ignored.",
            UserWarning,
        )

    @wraps(qnode)
    def wrapper(*args, **kwargs):
        tape = qml.tape.make_qscript(qnode)(*args, **kwargs)
        if wire_order:
            _wire_order = wire_order
        else:
            try:
                _wire_order = sorted(tape.wires)
            except TypeError:
                _wire_order = tape.wires

        return tape_mpl(
            tape,
            wire_order=_wire_order,
            show_all_wires=show_all_wires,
            decimals=decimals,
            style=style,
            fig=fig,
            **kwargs,
        )

    return wrapper


def _draw_mpl_qnode(
    qnode,
    wire_order=None,
    show_all_wires=False,
    decimals=None,
    level=None,
    style="black_white",
    *,
    fig=None,
    **kwargs,
):
    @wraps(qnode)
    def wrapper(*args, **kwargs_qnode):
        tapes, _ = qml.workflow.construct_batch(qnode, level=level)(*args, **kwargs_qnode)

        if len(tapes) > 1:
            warnings.warn(
                "Multiple tapes constructed, but only displaying the first one.", UserWarning
            )

        tape = tapes[0]

        if wire_order:
            _wire_order = wire_order
        elif qnode.device.wires:
            _wire_order = qnode.device.wires
        else:
            try:
                _wire_order = sorted(tape.wires)
            except TypeError:
                _wire_order = tape.wires

        return tape_mpl(
            tape,
            wire_order=_wire_order,
            show_all_wires=show_all_wires,
            decimals=decimals,
            style=style,
            fig=fig,
            **kwargs,
        )

    return wrapper
