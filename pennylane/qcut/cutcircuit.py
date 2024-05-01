# Copyright 2022 Xanadu Quantum Technologies Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Function cut_circuit for cutting a quantum circuit into smaller circuit fragments.
"""

from functools import partial
from typing import Callable, Optional, Union, Sequence

import pennylane as qml
from pennylane.measurements import ExpectationMP
from pennylane.tape import QuantumTape
from pennylane.transforms import transform
from pennylane.wires import Wires

from .cutstrategy import CutStrategy
from .kahypar import kahypar_cut
from .processing import qcut_processing_fn
from .tapes import _qcut_expand_fn, expand_fragment_tape, graph_to_tape, tape_to_graph
from .utils import find_and_place_cuts, fragment_graph, replace_wire_cut_nodes


def _cut_circuit_expand(
    tape: QuantumTape,
    use_opt_einsum: bool = False,
    device_wires: Optional[Wires] = None,
    max_depth: int = 1,
    auto_cutter: Union[bool, Callable] = False,
    **kwargs,
) -> (Sequence[QuantumTape], Callable):
    """Main entry point for expanding operations until reaching a depth that
    includes :class:`~.WireCut` operations."""
    # pylint: disable=unused-argument

    def processing_fn(res):
        return res[0]

    tapes, tapes_fn = [tape], processing_fn

    # Expand the tapes for handling Hamiltonian with two or more terms
    tape_meas_ops = tape.measurements
    if tape_meas_ops and isinstance(
        tape_meas_ops[0].obs, (qml.ops.Hamiltonian, qml.ops.LinearCombination)
    ):
        if len(tape_meas_ops) > 1:
            raise NotImplementedError(
                "Hamiltonian expansion is supported only with a single Hamiltonian"
            )

        new_meas_op = type(tape_meas_ops[0])(obs=qml.Hamiltonian(*tape_meas_ops[0].obs.terms()))
        new_tape = type(tape)(
            tape.operations, [new_meas_op], shots=tape.shots, trainable_params=tape.trainable_params
        )

        tapes, tapes_fn = qml.transforms.hamiltonian_expand(new_tape, group=False)

    return [_qcut_expand_fn(tape, max_depth, auto_cutter) for tape in tapes], tapes_fn


@partial(transform, expand_transform=_cut_circuit_expand)
def cut_circuit(
    tape: QuantumTape,
    auto_cutter: Union[bool, Callable] = False,
    use_opt_einsum: bool = False,
    device_wires: Optional[Wires] = None,
    max_depth: int = 1,
    **kwargs,
) -> (Sequence[QuantumTape], Callable):
    """
    Cut up a quantum circuit into smaller circuit fragments.

    Following the approach outlined in Theorem 2 of
    `Peng et al. <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.125.150504>`__,
    strategic placement of :class:`~.WireCut` operations can allow a quantum circuit to be split
    into disconnected circuit fragments. Each circuit fragment is then executed multiple times by
    varying the state preparations and measurements at incoming and outgoing cut locations,
    respectively, resulting in a process tensor describing the action of the fragment. The process
    tensors are then contracted to provide the result of the original uncut circuit.

    .. note::

        Only circuits that return a single expectation value are supported.

    Args:
        tape (QNode or QuantumTape): the quantum circuit to be cut
        auto_cutter (Union[bool, Callable]): Toggle for enabling automatic cutting with the default
            :func:`~.kahypar_cut` partition method. Can also pass a graph partitioning function that
            takes an input graph and returns a list of edges to be cut based on a given set of
            constraints and objective. The default :func:`~.kahypar_cut` function requires KaHyPar to
            be installed using ``pip install kahypar`` for Linux and Mac users or visiting the
            instructions `here <https://kahypar.org>`__ to compile from source for Windows users.
        use_opt_einsum (bool): Determines whether to use the
            `opt_einsum <https://dgasmith.github.io/opt_einsum/>`__ package. This package is useful
            for faster tensor contractions of large networks but must be installed separately using,
            e.g., ``pip install opt_einsum``. Both settings for ``use_opt_einsum`` result in a
            differentiable contraction.
        device_wires (Wires): Wires of the device that the cut circuits are to be run on.
            When transforming a QNode, this argument is optional and will be set to the
            QNode's device wires. Required when transforming a tape.
        max_depth (int): The maximum depth used to expand the circuit while searching for wire cuts.
            Only applicable when transforming a QNode.
        kwargs: Additional keyword arguments to be passed to a callable ``auto_cutter`` argument.
            For the default KaHyPar cutter, please refer to the docstring of functions
            :func:`~.find_and_place_cuts` and :func:`~.kahypar_cut` for the available arguments.

    Returns:
        qnode (QNode) or tuple[List[QuantumTape], function]:

        The transformed circuit as described in :func:`qml.transform <pennylane.transform>`. Executing this circuit
        will perform a process tomography of the partitioned circuit fragments and combine the results via tensor contractions.

    **Example**

    The following :math:`3`-qubit circuit contains a :class:`~.WireCut` operation. When decorated
    with ``@qml.cut_circuit``, we can cut the circuit into two :math:`2`-qubit fragments:

    .. code-block:: python

        dev = qml.device("default.qubit", wires=2)

        @qml.cut_circuit
        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.RY(0.9, wires=1)
            qml.RX(0.3, wires=2)

            qml.CZ(wires=[0, 1])
            qml.RY(-0.4, wires=0)

            qml.WireCut(wires=1)

            qml.CZ(wires=[1, 2])

            return qml.expval(qml.pauli.string_to_pauli_word("ZZZ"))

    Executing ``circuit`` will run multiple configurations of the :math:`2`-qubit fragments which
    are then postprocessed to give the result of the original circuit:

    >>> x = np.array(0.531, requires_grad=True)
    >>> circuit(x)
    0.47165198882111165

    Futhermore, the output of the cut circuit is also differentiable:

    >>> qml.grad(circuit)(x)
    tensor(-0.27698287, requires_grad=True)

    Alternatively, if the optimal wire-cut placement is unknown for an arbitrary circuit, the
    ``auto_cutter`` option can be enabled to make attempts in finding such an optimal cut. The
    following examples shows this capability on the same circuit as above but with the
    :class:`~.WireCut` removed:

    .. code-block:: python

        from functools import partial

        @partial(qml.cut_circuit, auto_cutter=True)
        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.RY(0.9, wires=1)
            qml.RX(0.3, wires=2)

            qml.CZ(wires=[0, 1])
            qml.RY(-0.4, wires=0)

            qml.CZ(wires=[1, 2])

            return qml.expval(qml.pauli.string_to_pauli_word("ZZZ"))

    >>> x = np.array(0.531, requires_grad=True)
    >>> circuit(x)
    0.47165198882111165
    >>> qml.grad(circuit)(x)
    tensor(-0.27698287, requires_grad=True)

    .. details::
        :title: Usage Details

        Manually placing :class:`~.WireCut` operations and decorating the QNode with the
        ``cut_circuit()`` batch transform is the suggested entrypoint into circuit cutting. However,
        advanced users also have the option to work directly with a :class:`~.QuantumTape` and
        manipulate the tape to perform circuit cutting using the below functionality:

        .. autosummary::
            :toctree:

            ~qcut.tape_to_graph
            ~qcut.find_and_place_cuts
            ~qcut.replace_wire_cut_nodes
            ~qcut.fragment_graph
            ~qcut.graph_to_tape
            ~qcut.expand_fragment_tape
            ~qcut.qcut_processing_fn
            ~qcut.CutStrategy

        The following shows how these elementary steps are combined as part of the
        ``cut_circuit()`` transform.

        Consider the circuit below:

        .. code-block:: python

            ops = [
                qml.RX(0.531, wires=0),
                qml.RY(0.9, wires=1),
                qml.RX(0.3, wires=2),

                qml.CZ(wires=(0,1)),
                qml.RY(-0.4, wires=0),

                qml.WireCut(wires=1),

                qml.CZ(wires=[1, 2]),
            ]
            measurements = [qml.expval(qml.pauli.string_to_pauli_word("ZZZ"))]
            tape = qml.tape.QuantumTape(ops, measurements)

        >>> print(qml.drawer.tape_text(tape))
        0: ──RX─╭●──RY────┤ ╭<Z@Z@Z>
        1: ──RY─╰Z──//─╭●─┤ ├<Z@Z@Z>
        2: ──RX────────╰Z─┤ ╰<Z@Z@Z>

        To cut the circuit, we first convert it to its graph representation:

        >>> graph = qml.qcut.tape_to_graph(tape)

        .. figure:: ../../_static/qcut_graph.svg
            :align: center
            :width: 60%
            :target: javascript:void(0);

        If, however, the optimal location of the :class:`~.WireCut` is unknown, we can use
        :func:`~.find_and_place_cuts` to make attempts in automatically finding such a cut
        given the device constraints. Using the same circuit as above but with the
        :class:`~.WireCut` removed, the same (optimal) cut can be recovered with automatic
        cutting:

        .. code-block:: python

            ops = [
                qml.RX(0.531, wires=0),
                qml.RY(0.9, wires=1),
                qml.RX(0.3, wires=2),

                qml.CZ(wires=(0,1)),
                qml.RY(-0.4, wires=0),

                qml.CZ(wires=[1, 2]),
            ]
            measurements = [qml.expval(qml.pauli.string_to_pauli_word("ZZZ"))]
            uncut_tape = qml.tape.QuantumTape(ops, measurements)

        >>> cut_graph = qml.qcut.find_and_place_cuts(
        ...     graph = qml.qcut.tape_to_graph(uncut_tape),
        ...     cut_strategy = qml.qcut.CutStrategy(max_free_wires=2),
        ... )
        >>> print(qml.qcut.graph_to_tape(cut_graph).draw())
        0: ──RX─╭●──RY────┤ ╭<Z@Z@Z>
        1: ──RY─╰Z──//─╭●─┤ ├<Z@Z@Z>
        2: ──RX────────╰Z─┤ ╰<Z@Z@Z>

        Our next step is to remove the :class:`~.WireCut` nodes in the graph and replace with
        :class:`~.MeasureNode` and :class:`~.PrepareNode` pairs.

        >>> qml.qcut.replace_wire_cut_nodes(graph)

        The :class:`~.MeasureNode` and :class:`~.PrepareNode` pairs are placeholder operations that
        allow us to cut the circuit graph and then iterate over measurement and preparation
        configurations at cut locations. First, the :func:`~.fragment_graph` function pulls apart
        the graph into disconnected components as well as returning the
        `communication_graph <https://en.wikipedia.org/wiki/Quotient_graph>`__
        detailing the connectivity between the components.

        >>> fragments, communication_graph = qml.qcut.fragment_graph(graph)

        We now convert the ``fragments`` back to :class:`~.QuantumTape` objects

        >>> fragment_tapes = [qml.qcut.graph_to_tape(f) for f in fragments]

        The circuit fragments can now be visualized:

        >>> print(fragment_tapes[0].draw(decimals=2))
        0: ──RX(0.53)─╭●──RY(-0.40)───┤  <Z>
        1: ──RY(0.90)─╰Z──MeasureNode─┤

        >>> print(fragment_tapes[1].draw(decimals=1))
        2: ──RX(0.3)─────╭Z─┤ ╭<Z@Z>
        1: ──PrepareNode─╰●─┤ ╰<Z@Z>

        Additionally, we must remap the tape wires to match those available on our device.

        >>> dev = qml.device("default.qubit", wires=2)
        >>> fragment_tapes = [qml.map_wires(t, dict(zip(t.wires, dev.wires)))[0][0] for t in fragment_tapes]

        Next, each circuit fragment is expanded over :class:`~.MeasureNode` and
        :class:`~.PrepareNode` configurations and a flat list of tapes is created:

        .. code-block::

            expanded = [qml.qcut.expand_fragment_tape(t) for t in fragment_tapes]

            configurations = []
            prepare_nodes = []
            measure_nodes = []
            for tapes, p, m in expanded:
                configurations.append(tapes)
                prepare_nodes.append(p)
                measure_nodes.append(m)

            tapes = tuple(tape for c in configurations for tape in c)

        Each configuration is drawn below:

        >>> for t in tapes:
        ...     print(qml.drawer.tape_text(t))
        ...     print()

        .. code-block::

            0: ──RX(0.53)─╭●──RY(-0.40)─┤ ╭<Z@I> ╭<Z@Z>
            1: ──RY(0.90)─╰Z────────────┤ ╰<Z@I> ╰<Z@Z>

            0: ──RX(0.53)─╭●──RY(-0.40)─┤ ╭<Z@X>
            1: ──RY(0.90)─╰Z────────────┤ ╰<Z@X>

            0: ──RX(0.53)─╭●──RY(-0.40)─┤ ╭<Z@Y>
            1: ──RY(0.90)─╰Z────────────┤ ╰<Z@Y>

            0: ──RX(0.30)─╭Z─┤ ╭<Z@Z>
            1: ──I────────╰●─┤ ╰<Z@Z>

            0: ──RX(0.30)─╭Z─┤ ╭<Z@Z>
            1: ──X────────╰●─┤ ╰<Z@Z>

            0: ──RX(0.30)─╭Z─┤ ╭<Z@Z>
            1: ──H────────╰●─┤ ╰<Z@Z>

            0: ──RX(0.30)────╭Z─┤ ╭<Z@Z>
            1: ──H─────────S─╰●─┤ ╰<Z@Z>

        The last step is to execute the tapes and postprocess the results using
        :func:`~.qcut_processing_fn`, which processes the results to the original full circuit
        output via a tensor network contraction

        >>> results = qml.execute(tapes, dev, gradient_fn=None)
        >>> qml.qcut.qcut_processing_fn(
        ...     results,
        ...     communication_graph,
        ...     prepare_nodes,
        ...     measure_nodes,
        ... )
        0.47165198882111165
    """
    # pylint: disable=unused-argument
    if len(tape.measurements) != 1:
        raise ValueError(
            "The circuit cutting workflow only supports circuits with a single output "
            "measurement"
        )

    if not all(isinstance(m, ExpectationMP) for m in tape.measurements):
        raise ValueError(
            "The circuit cutting workflow only supports circuits with expectation "
            "value measurements"
        )

    if use_opt_einsum:
        try:
            import opt_einsum  # pylint: disable=import-outside-toplevel,unused-import
        except ImportError as e:
            raise ImportError(
                "The opt_einsum package is required when use_opt_einsum is set to "
                "True in the cut_circuit function. This package can be "
                "installed using:\npip install opt_einsum"
            ) from e

    # convert the quantum tape to a DAG structure
    g = tape_to_graph(tape)

    # place WireCut(s) nodes in the DAG automatically if intended
    if auto_cutter is True or callable(auto_cutter):
        cut_strategy = kwargs.pop("cut_strategy", None) or CutStrategy(
            max_free_wires=len(device_wires)
        )

        g = find_and_place_cuts(
            graph=g,
            cut_method=auto_cutter if callable(auto_cutter) else kahypar_cut,
            cut_strategy=cut_strategy,
            **kwargs,
        )

    # replace the WireCut nodes in the DAG with Measure and Perpare nodes.
    replace_wire_cut_nodes(g)

    # decompose the DAG into subgraphs based on the replaced WireCut(s)
    # along with a quotient graph to store connections between them
    fragments, communication_graph = fragment_graph(g)

    # convert decomposed DAGs into tapes, remap their wires for device and expand them
    fragment_tapes = [graph_to_tape(f) for f in fragments]
    fragment_tapes = [
        qml.map_wires(t, dict(zip(t.wires, device_wires)))[0][0] for t in fragment_tapes
    ]
    expanded = [expand_fragment_tape(t) for t in fragment_tapes]

    # store the data necessary for classical post processing of results
    configurations = []
    prepare_nodes = []
    measure_nodes = []
    for tapes, p, m in expanded:
        configurations.append(tapes)
        prepare_nodes.append(p)
        measure_nodes.append(m)

    # flatten out the tapes to be returned
    tapes = tuple(tape for c in configurations for tape in c)

    return tapes, partial(
        qcut_processing_fn,
        communication_graph=communication_graph,
        prepare_nodes=prepare_nodes,
        measure_nodes=measure_nodes,
        use_opt_einsum=use_opt_einsum,
    )


@cut_circuit.custom_qnode_transform
def _qnode_transform(self, qnode, targs, tkwargs):
    """Here, we overwrite the QNode execution wrapper in order
    to access the device wires."""
    tkwargs.setdefault("device_wires", qnode.device.wires)
    return self.default_qnode_transform(qnode, targs, tkwargs)
