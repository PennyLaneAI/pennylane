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
from functools import wraps

from pennylane.drawer import tape_mpl, tape_text
from pennylane.wires import Wires


def draw_old(
    qnode,
    charset="unicode",
    wire_order=None,
    show_all_wires=False,
    max_length=None,
    expansion_strategy=None,
):
    """Create a function that draws the given qnode.

    Args:
        qnode (.QNode): the input QNode that is to be drawn.
        charset (str, optional): The charset that should be used. Currently, "unicode" and
            "ascii" are supported.
        wire_order (Sequence[Any]): the order (from top to bottom) to print the wires of the circuit
        show_all_wires (bool): If True, all wires, including empty wires, are printed.
        max_length (int, optional): Maximum string width (columns) when printing the circuit to the CLI.
        expansion_strategy (str): The strategy to use when circuit expansions or decompositions
            are required.

            - ``gradient``: The QNode will attempt to decompose
              the internal circuit such that all circuit operations are supported by the gradient
              method.

            - ``device``: The QNode will attempt to decompose the internal circuit
              such that all circuit operations are natively supported by the device.


    Returns:
        A function that has the same argument signature as ``qnode``. When called,
        the function will draw the QNode.

    **Example**

    Given the following definition of a QNode,

    .. code-block:: python3

        dev = qml.device('default.qubit', wires=2)

        @qml.qnode(dev)
        def circuit(a, w):
            qml.Hadamard(0)
            qml.CRX(a, wires=[0, 1])
            qml.Rot(*w, wires=[1])
            qml.CRX(-a, wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    We can draw the it like such:

    >>> drawer = qml.draw(circuit)
    >>> print(drawer(a=2.3, w=[1.2, 3.2, 0.7]))
    0: ──H──╭C────────────────────────────╭C─────────╭┤ ⟨Z ⊗ Z⟩
    1: ─────╰RX(2.3)──Rot(1.2, 3.2, 0.7)──╰RX(-2.3)──╰┤ ⟨Z ⊗ Z⟩

    Circuit drawing works with devices with custom wire labels:

    .. code-block:: python3

        dev = qml.device('default.qubit', wires=["a", -1, "q2"])

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=-1)
            qml.CNOT(wires=["a", "q2"])
            qml.RX(0.2, wires="a")
            return qml.expval(qml.PauliX(wires="q2"))

    When printed, the wire order matches the order defined on the device:

    >>> drawer = qml.draw(circuit)
    >>> print(drawer())
      a: ─────╭C──RX(0.2)──┤
     -1: ──H──│────────────┤
     q2: ─────╰X───────────┤ ⟨X⟩

    We can use the ``wire_order`` argument to change the wire order:

    >>> drawer = qml.draw(circuit, wire_order=["q2", "a", -1])
    >>> print(drawer())
     q2: ──╭X───────────┤ ⟨X⟩
      a: ──╰C──RX(0.2)──┤
     -1: ───H───────────┤
    """

    @wraps(qnode)
    def wrapper(*args, **kwargs):
        original_expansion_strategy = getattr(qnode, "expansion_strategy", None)

        try:
            qnode.expansion_strategy = expansion_strategy or original_expansion_strategy
            tapes = qnode.construct(args, kwargs)
        finally:
            qnode.expansion_strategy = original_expansion_strategy

        _wire_order = wire_order or qnode.device.wires
        _wire_order = Wires(_wire_order)

        if show_all_wires and len(_wire_order) < qnode.device.num_wires:
            raise ValueError(
                "When show_all_wires is enabled, the provided wire order must contain all wires on the device."
            )

        if not qnode.device.wires.contains_wires(_wire_order):
            raise ValueError(
                f"Provided wire order {_wire_order.labels} contains wires not contained on the device: {qnode.device.wires}."
            )

        if tapes is not None:
            res = [
                t.draw(
                    charset=charset,
                    wire_order=_wire_order,
                    show_all_wires=show_all_wires,
                    max_length=max_length,
                )
                for t in tapes[0]
            ]
            return "\n".join(res)

        return qnode.qtape.draw(
            charset=charset,
            wire_order=_wire_order,
            show_all_wires=show_all_wires,
            max_length=max_length,
        )

    return wrapper


def draw(
    qnode,
    wire_order=None,
    show_all_wires=False,
    decimals=2,
    max_length=100,
    show_matrices=False,
    expansion_strategy=None,
):
    """Create a function that draws the given qnode.

    Args:
        qnode (.QNode): the input QNode that is to be drawn.
        wire_order (Sequence[Any]): the order (from top to bottom) to print the wires of the circuit
        show_all_wires (bool): If True, all wires, including empty wires, are printed.
        decimals (int): How many decimal points to include when formatting operation parameters.
            ``None`` will omit parameters from operation labels.
        max_length (int): Maximum string width (columns) when printing the circuit
        show_matrices=False (bool): show matrix valued parameters below all circuit diagrams
        expansion_strategy (str): The strategy to use when circuit expansions or decompositions
            are required.

            - ``gradient``: The QNode will attempt to decompose
              the internal circuit such that all circuit operations are supported by the gradient
              method.

            - ``device``: The QNode will attempt to decompose the internal circuit
              such that all circuit operations are natively supported by the device.


    Returns:
        A function that has the same argument signature as ``qnode``. When called,
        the function will draw the QNode.

    **Example**

    .. code-block:: python3

        @qml.qnode(qml.device('lightning.qubit', wires=2))
        def circuit(a, w):
            qml.Hadamard(0)
            qml.CRX(a, wires=[0, 1])
            qml.Rot(*w, wires=[1])
            qml.CRX(-a, wires=[0, 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    >>> print(qml.draw(circuit)(a=2.3, w=[1.2, 3.2, 0.7]))
    1: ────╭RX(2.30)──Rot(1.20,3.20,0.70)─╭RX(-2.30)─┤ ╭<Z@Z>
    0: ──H─╰C─────────────────────────────╰C─────────┤ ╰<Z@Z>

    .. UsageDetails::


    By specifying the ``decimals`` keyword, parameters are displayed to the specified precision.

    >>> print(qml.draw(circuit, decimals=4)(a=2.3, w=[1.2, 3.2, 0.7]))
    0: ──H─╭C─────────────────────────────────────╭C───────────┤ ╭<Z@Z>
    1: ────╰RX(2.3000)──Rot(1.2000,3.2000,0.7000)─╰RX(-2.3000)─┤ ╰<Z@Z>

    Parameters can be omitted by requesting ``decimals=None``:

    >>> print(qml.draw(circuit, decimals=None)(a=2.3, w=[1.2, 3.2, 0.7]))
    0: ──H─╭C───────╭C──┤ ╭<Z@Z>
    1: ────╰RX──Rot─╰RX─┤ ╰<Z@Z>

    If the parameters are not acted upon by classical processing like ``-a``, then
    ``qml.draw`` can handle string-valued parameters as well:

    >>> @qml.qnode(qml.device('lightning.qubit', wires=1))
    ... def circuit2(x):
    ...     qml.RX(x, wires=0)
    ...     return qml.expval(qml.PauliZ(0))
    >>> print(qml.draw(circuit2)("x"))
    0: ──RX(x)─┤  <Z>

    When requested with ``show_matrices=True``, matrix valued parameters are printed below the
    circuit:

    .. code-block:: python

    >>> @qml.qnode(qml.device('default.qubit', wires=2))
    ... def circuit():
    ...     qml.QubitUnitary(np.eye(2), wires=0)
    ...     qml.QubitUnitary(-np.eye(4), wires=(0,1))
    ...     return qml.expval(qml.Hermitian(np.eye(2), wires=1))
    >>> print(qml.draw(circuit, show_matrices=True)())
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

    The ``max_length`` keyword warps long circuits:

    .. code-block:: python

        rng = np.random.default_rng(seed=42)
        shape = qml.StronglyEntanglingLayers.shape(n_wires=3, n_layers=3)
        params = rng.random(shape)

        @qml.qnode(qml.device('lightning.qubit', wires=3))
        def longer_circuit(params):
            qml.StronglyEntanglingLayers(params, wires=range(3))
            return [qml.expval(qml.PauliZ(i)) for i in range(3)]

        print(qml.draw(longer_circuit, max_length=60)(params))

    .. code-block:: none

        0: ──Rot(0.77,0.44,0.86)─╭C────╭X──Rot(0.45,0.37,0.93)─╭C─╭X
        1: ──Rot(0.70,0.09,0.98)─╰X─╭C─│───Rot(0.64,0.82,0.44)─│──╰C
        2: ──Rot(0.76,0.79,0.13)────╰X─╰C──Rot(0.23,0.55,0.06)─╰X───

        ───Rot(0.83,0.63,0.76)──────────────────────╭C────╭X─┤  <Z>
        ──╭X────────────────────Rot(0.35,0.97,0.89)─╰X─╭C─│──┤  <Z>
        ──╰C────────────────────Rot(0.78,0.19,0.47)────╰X─╰C─┤  <Z>

    The ``wire_order`` keyword specifies the order of the wires from
    top to bottom:

    >>> print(qml.draw(circuit, wire_order=[1,0])(a=2.3, w=[1.2, 3.2, 0.7]))
    1: ────╭RX(2.30)──Rot(1.20,3.20,0.70)─╭RX(-2.30)─┤ ╭<Z@Z>
    0: ──H─╰C─────────────────────────────╰C─────────┤ ╰<Z@Z>

    If the device or ``wire_order`` has wires not used by operations, those wires are omitted
    unless requested with ``show_all_wires=True``

    >>> empty_qfunc = lambda : qml.expval(qml.PauliZ(0))
    >>> empty_circuit = qml.QNode(empty_qfunc, qml.device('lightning.qubit', wires=3))
    >>> print(qml.draw(empty_circuit, show_all_wires=True)())
    0: ───┤  <Z>
    1: ───┤
    2: ───┤

    Drawing also works on batch transformed circuits:

    .. code-block:: python

        @qml.gradients.param_shift(shift=0.1)
        @qml.qnode(qml.device('lightning.qubit', wires=1))
        def transformed_circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        print(qml.draw(transformed_circuit)(np.array(1.0)))

    .. code-block:: none

        0: ──RX(1.10)─┤  <Z>

        0: ──RX(0.90)─┤  <Z>

    """

    @wraps(qnode)
    def wrapper(*args, **kwargs):

        original_expansion_strategy = getattr(qnode, "expansion_strategy", None)

        try:
            qnode.expansion_strategy = expansion_strategy or original_expansion_strategy
            tapes = qnode.construct(args, kwargs)
        finally:
            qnode.expansion_strategy = original_expansion_strategy

        _wire_order = wire_order or qnode.device.wires

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
                for t in tapes[0]
            ]
            if show_matrices:
                mat_str = "\n"
                for i, mat in enumerate(cache["matrices"]):
                    mat_str += f"\nM{i} = \n{mat}"
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
    qnode, wire_order=None, show_all_wires=False, decimals=None, expansion_strategy=None, **kwargs
):
    """Draw a qnode with matplotlib

    Args:
        qnode (.QNode): the input QNode that is to be drawn.

    Keyword Args:
        wire_order (Sequence[Any]): the order (from top to bottom) to print the wires of the circuit
        show_all_wires (bool): If True, all wires, including empty wires, are printed.
        decimals (int): How many decimal points to include when formatting operation parameters.
            Default ``None`` will omit parameters from operation labels.
        fontsize (float or str): fontsize for text. Valid strings are
            ``{'xx-small', 'x-small', 'small', 'medium', large', 'x-large', 'xx-large'}``.
            Default is ``14``.
        wire_options (dict): matplotlib formatting options for the wire lines
        label_options (dict): matplotlib formatting options for the wire labels
        active_wire_notches (bool): whether or not to add notches indicating active wires.
            Defaults to ``True``.
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

    **Example**:

    .. code-block:: python

        dev = qml.device('lightning.qubit', wires=(0,1,2,3))

        @qml.qnode(dev)
        def circuit(x, z):
            qml.QFT(wires=(0,1,2,3))
            qml.IsingXX(1.234, wires=(0,2))
            qml.Toffoli(wires=(0,1,2))
            qml.CSWAP(wires=(0,2,3))
            qml.RX(x, wires=0)
            qml.CRZ(z, wires=(3,0))
            return qml.expval(qml.PauliZ(0))


        fig, ax = qml.draw_mpl(circuit)(1.2345,1.2345)
        fig.show()

    .. figure:: ../../_static/draw_mpl/main_example.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

    .. UsageDetails::

        **Decimals:**

        The keyword ``decimals`` controls how many decimal points to include when labelling the operations.
        The default value ``None`` omits parameters for brevity.

        .. code-block:: python

            @qml.qnode(dev)
            def circuit2(x, y):
                qml.RX(x, wires=0)
                qml.Rot(*y, wires=0)
                return qml.expval(qml.PauliZ(0))

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
        ``show_all_wires=True`` will empty wires be diplayed.

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

            ax.annotate("CSWAP", xy=(3, 2.5), xycoords='data', xytext=(3.8,1.5), textcoords='data',
                        arrowprops={'facecolor': 'black'}, fontsize=14)
            fig.show()

        .. figure:: ../../_static/draw_mpl/postprocessing.png
                :align: center
                :width: 60%
                :target: javascript:void(0);

        **Formatting:**

        PennyLane has inbuilt styles for controlling the appearance of the circuit drawings.
        All available styles can be determined by evaluating ``qml.drawer.available_styles()``.
        Any available string can then be passed to ``qml.drawer.use_style``.

        .. code-block:: python

            qml.drawer.use_style('black_white')
            fig, ax = qml.draw_mpl(circuit)(1.2345,1.2345)
            fig.show()


        .. figure:: ../../_static/draw_mpl/black_white_style.png
                :align: center
                :width: 60%
                :target: javascript:void(0);

        You can also control the appearance with matplotlib's provided tools, see the
        `matplotlib docs <https://matplotlib.org/stable/tutorials/introductory/customizing.html>`_ .
        For example, we can customize ``plt.rcParams``:

        .. code-block:: python

            plt.rcParams['patch.facecolor'] = 'mistyrose'
            plt.rcParams['patch.edgecolor'] = 'maroon'
            plt.rcParams['text.color'] = 'maroon'
            plt.rcParams['font.weight'] = 'bold'
            plt.rcParams['patch.linewidth'] = 4
            plt.rcParams['patch.force_edgecolor'] = True
            plt.rcParams['lines.color'] = 'indigo'
            plt.rcParams['lines.linewidth'] = 5
            plt.rcParams['figure.facecolor'] = 'ghostwhite'

            fig, ax = qml.draw_mpl(circuit)(1.2345,1.2345)
            fig.show()

        .. figure:: ../../_static/draw_mpl/rcparams.png
                :align: center
                :width: 60%
                :target: javascript:void(0);

        The wires and wire labels can be manually formatted by passing in dictionaries of
        keyword-value pairs of matplotlib options. ``wire_options`` accepts options for lines,
        and ``label_options`` accepts text options.

        .. code-block:: python

            fig, ax = qml.draw_mpl(circuit, wire_options={'color':'black', 'linewidth': 5},
                        label_options={'size': 20})(1.2345,1.2345)
            fig.show()

        .. figure:: ../../_static/draw_mpl/wires_labels.png
                :align: center
                :width: 60%
                :target: javascript:void(0);

    """

    @wraps(qnode)
    def wrapper(*args, **kwargs_qnode):
        original_expansion_strategy = getattr(qnode, "expansion_strategy", None)

        try:
            qnode.expansion_strategy = expansion_strategy or original_expansion_strategy
            qnode.construct(args, kwargs_qnode)
        finally:
            qnode.expansion_strategy = original_expansion_strategy

        _wire_order = wire_order or qnode.device.wires

        return tape_mpl(
            qnode.qtape,
            wire_order=_wire_order,
            show_all_wires=show_all_wires,
            decimals=decimals,
            **kwargs,
        )

    return wrapper
