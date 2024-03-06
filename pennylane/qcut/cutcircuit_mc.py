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
Function cut_circuit_mc for cutting a quantum circuit into smaller circuit fragments using a
    Monte Carlo method, at its auxillary functions"""

import inspect
from functools import partial
from typing import Callable, List, Optional, Sequence, Tuple, Union
import numpy as np

from networkx import MultiDiGraph

import pennylane as qml
from pennylane.measurements import SampleMP
from pennylane.tape import QuantumScript, QuantumTape
from pennylane.transforms import transform
from pennylane.wires import Wires

from .cutstrategy import CutStrategy
from .kahypar import kahypar_cut
from .processing import qcut_processing_fn_mc, qcut_processing_fn_sample

from .tapes import _qcut_expand_fn, graph_to_tape, tape_to_graph
from .utils import (
    find_and_place_cuts,
    fragment_graph,
    replace_wire_cut_nodes,
    MeasureNode,
    PrepareNode,
    _prep_iminus_state,
    _prep_iplus_state,
    _prep_minus_state,
    _prep_one_state,
    _prep_plus_state,
    _prep_zero_state,
)


def _cut_circuit_mc_expand(
    tape: QuantumTape,
    classical_processing_fn: Optional[callable] = None,
    max_depth: int = 1,
    shots: Optional[int] = None,
    device_wires: Optional[Wires] = None,
    auto_cutter: Union[bool, Callable] = False,
    **kwargs,
) -> (Sequence[QuantumTape], Callable):
    """Main entry point for expanding operations in sample-based tapes until
    reaching a depth that includes :class:`~.WireCut` operations."""
    # pylint: disable=unused-argument, too-many-arguments

    def processing_fn(res):
        return res[0]

    return [_qcut_expand_fn(tape, max_depth, auto_cutter)], processing_fn


@partial(transform, expand_transform=_cut_circuit_mc_expand)
def cut_circuit_mc(
    tape: QuantumTape,
    classical_processing_fn: Optional[callable] = None,
    auto_cutter: Union[bool, Callable] = False,
    max_depth: int = 1,
    shots: Optional[int] = None,
    device_wires: Optional[Wires] = None,
    **kwargs,
) -> (Sequence[QuantumTape], Callable):
    """
    Cut up a circuit containing sample measurements into smaller fragments using a
    Monte Carlo method.

    Following the approach of `Peng et al. <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.125.150504>`__,
    strategic placement of :class:`~.WireCut` operations can allow a quantum circuit to be split
    into disconnected circuit fragments. A circuit containing sample measurements can be cut and
    processed using Monte Carlo (MC) methods. This transform employs MC methods to allow for sampled measurement
    outcomes to be recombined to full bitstrings and, if a classical processing function is supplied,
    an expectation value will be evaluated.

    Args:
        tape (QNode or QuantumTape): the quantum circuit to be cut.
        classical_processing_fn (callable): A classical postprocessing function to be applied to
            the reconstructed bitstrings. The expected input is a bitstring; a flat array of length ``wires``,
            and the output should be a single number within the interval :math:`[-1, 1]`.
            If not supplied, the transform will output samples.
        auto_cutter (Union[bool, Callable]): Toggle for enabling automatic cutting with the default
            :func:`~.kahypar_cut` partition method. Can also pass a graph partitioning function that
            takes an input graph and returns a list of edges to be cut based on a given set of
            constraints and objective. The default :func:`~.kahypar_cut` function requires KaHyPar to
            be installed using ``pip install kahypar`` for Linux and Mac users or visiting the
            instructions `here <https://kahypar.org>`__ to compile from source for Windows users.
        max_depth (int): The maximum depth used to expand the circuit while searching for wire cuts.
            Only applicable when transforming a QNode.
        shots (int): Number of shots. When transforming a QNode, this argument is
            set by the device's ``shots`` value or at QNode call time (if provided).
            Required when transforming a tape.
        device_wires (Wires): Wires of the device that the cut circuits are to be run on.
            When transforming a QNode, this argument is optional and will be set to the
            QNode's device wires. Required when transforming a tape.
        kwargs: Additional keyword arguments to be passed to a callable ``auto_cutter`` argument.
            For the default KaHyPar cutter, please refer to the docstring of functions
            :func:`~.find_and_place_cuts` and :func:`~.kahypar_cut` for the available arguments.

    Returns:
        qnode (QNode) or Tuple[List[QuantumTape], function]:

        The transformed circuit as described in :func:`qml.transform <pennylane.transform>`. Executing this circuit
        will sample from the partitioned circuit fragments and combine the results using a Monte Carlo method.

    **Example**

    The following :math:`3`-qubit circuit contains a :class:`~.WireCut` operation and a :func:`~.sample`
    measurement. When decorated with ``@qml.cut_circuit_mc``, we can cut the circuit into two
    :math:`2`-qubit fragments:

    .. code-block:: python

        dev = qml.device("default.qubit", wires=2, shots=1000)

        @qml.cut_circuit_mc
        @qml.qnode(dev)
        def circuit(x):
            qml.RX(0.89, wires=0)
            qml.RY(0.5, wires=1)
            qml.RX(1.3, wires=2)

            qml.CNOT(wires=[0, 1])
            qml.WireCut(wires=1)
            qml.CNOT(wires=[1, 2])

            qml.RX(x, wires=0)
            qml.RY(0.7, wires=1)
            qml.RX(2.3, wires=2)
            return qml.sample(wires=[0, 2])

    we can then execute the circuit as usual by calling the QNode:

    >>> x = 0.3
    >>> circuit(x)
    array([[0., 0.],
           [0., 1.],
           [1., 0.],
           ...,
           [0., 0.],
           [0., 0.],
           [0., 1.]])

    Furthermore, the number of shots can be temporarily altered when calling
    the qnode:

    >>> results = circuit(x, shots=123)
    >>> results.shape
    (123, 2)

    Alternatively, if the optimal wire-cut placement is unknown for an arbitrary circuit, the
    ``auto_cutter`` option can be enabled to make attempts in finding such a optimal cut. The
    following examples shows this capability on the same circuit as above but with the
    :class:`~.WireCut` removed:

    .. code-block:: python

        from functools import partial

        @partial(qml.cut_circuit_mc, auto_cutter=True)
        @qml.qnode(dev)
        def circuit(x):
            qml.RX(0.89, wires=0)
            qml.RY(0.5, wires=1)
            qml.RX(1.3, wires=2)

            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])

            qml.RX(x, wires=0)
            qml.RY(0.7, wires=1)
            qml.RX(2.3, wires=2)
            return qml.sample(wires=[0, 2])

    >>> results = circuit(x, shots=123)
    >>> results.shape
    (123, 2)


    .. details::
        :title: Usage Details

        Manually placing :class:`~.WireCut` operations and decorating the QNode with the
        ``cut_circuit_mc()`` batch transform is the suggested entrypoint into sampling-based
        circuit cutting using the Monte Carlo method. However,
        advanced users also have the option to work directly with a :class:`~.QuantumTape` and
        manipulate the tape to perform circuit cutting using the below functionality:

        .. autosummary::
            :toctree:

            ~qcut.tape_to_graph
            ~qcut.find_and_place_cuts
            ~qcut.replace_wire_cut_nodes
            ~qcut.fragment_graph
            ~qcut.graph_to_tape
            ~qcut.expand_fragment_tapes_mc
            ~qcut.qcut_processing_fn_sample
            ~qcut.qcut_processing_fn_mc

        The following shows how these elementary steps are combined as part of the
        ``cut_circuit_mc()`` transform.

        Consider the circuit below:

        .. code-block:: python

            np.random.seed(42)

            ops = [
                qml.Hadamard(wires=0),
                qml.CNOT(wires=[0, 1]),
                qml.X(1),
                qml.WireCut(wires=1),
                qml.CNOT(wires=[1, 2]),
            ]
            measurements = [qml.sample(wires=[0, 1, 2])]
            tape = qml.tape.QuantumTape(ops, measurements)

        >>> print(tape.draw())
        0: ──H─╭●───────────┤ ╭Sample
        1: ────╰X──X──//─╭●─┤ ├Sample
        2: ──────────────╰X─┤ ╰Sample

        To cut the circuit, we first convert it to its graph representation:

        >>> graph = qml.qcut.tape_to_graph(tape)

        If, however, the optimal location of the :class:`~.WireCut` is unknown, we can use
        :func:`~.find_and_place_cuts` to make attempts in automatically finding such a cut
        given the device constraints. Using the same circuit as above but with the
        :class:`~.WireCut` removed, a slightly different cut with identical cost can be discovered
        and placed into the circuit with automatic cutting:

        .. code-block:: python

            ops = [
                qml.Hadamard(wires=0),
                qml.CNOT(wires=[0, 1]),
                qml.X(1),
                qml.CNOT(wires=[1, 2]),
            ]
            measurements = [qml.sample(wires=[0, 1, 2])]
            uncut_tape = qml.tape.QuantumTape(ops, measurements)

        >>> cut_graph = qml.qcut.find_and_place_cuts(
        ...     graph=qml.qcut.tape_to_graph(uncut_tape),
        ...     cut_strategy=qml.qcut.CutStrategy(max_free_wires=2),
        ... )
        >>> print(qml.qcut.graph_to_tape(cut_graph).draw())
         0: ──H─╭●───────────┤  Sample[|1⟩⟨1|]
         1: ────╰X──//──X─╭●─┤  Sample[|1⟩⟨1|]
         2: ──────────────╰X─┤  Sample[|1⟩⟨1|]

        Our next step, using the original manual cut placement, is to remove the :class:`~.WireCut`
        nodes in the graph and replace with :class:`~.MeasureNode` and :class:`~.PrepareNode` pairs.

        >>> qml.qcut.replace_wire_cut_nodes(graph)

        The :class:`~.MeasureNode` and :class:`~.PrepareNode` pairs are placeholder operations that
        allow us to cut the circuit graph and then randomly select measurement and preparation
        configurations at cut locations. First, the :func:`~.fragment_graph` function pulls apart
        the graph into disconnected components as well as returning the
        `communication_graph <https://en.wikipedia.org/wiki/Quotient_graph>`__
        detailing the connectivity between the components.

        >>> fragments, communication_graph = qml.qcut.fragment_graph(graph)

        We now convert the ``fragments`` back to :class:`~.QuantumTape` objects

        >>> fragment_tapes = [qml.qcut.graph_to_tape(f) for f in fragments]

        The circuit fragments can now be visualized:

        >>> print(fragment_tapes[0].draw())
        0: ──H─╭●─────────────────┤  Sample[|1⟩⟨1|]
        1: ────╰X──X──MeasureNode─┤

        >>> print(fragment_tapes[1].draw())
        1: ──PrepareNode─╭●─┤  Sample[|1⟩⟨1|]
        2: ──────────────╰X─┤  Sample[|1⟩⟨1|]

        Additionally, we must remap the tape wires to match those available on our device.

        >>> dev = qml.device("default.qubit", wires=2, shots=1)
        >>> fragment_tapes = [qml.map_wires(t, dict(zip(t.wires, dev.wires)))[0][0] for t in fragment_tapes]

        Note that the number of shots on the device is set to :math:`1` here since we
        will only require one execution per fragment configuration. In the
        following steps we introduce a shots value that will determine the number
        of fragment configurations. When using the ``cut_circuit_mc()`` decorator
        with a QNode, this shots value is automatically inferred from the provided
        device.

        Next, each circuit fragment is randomly expanded over :class:`~.MeasureNode` and
        :class:`~.PrepareNode` configurations. For each pair, a measurement is sampled from
        the Pauli basis and a state preparation is sampled from the corresponding pair of eigenstates.

        A settings array is also given which tracks the configuration pairs. Since each of the 4
        measurements has 2 possible eigenvectors, all configurations can be uniquely identified by
        8 values. The number of rows is determined by the number of cuts and the number of columns
        is determined by the number of shots.

        >>> shots = 3
        >>> configurations, settings = qml.qcut.expand_fragment_tapes_mc(
        ...     fragment_tapes, communication_graph, shots=shots
        ... )
        >>> tapes = tuple(tape for c in configurations for tape in c)
        >>> settings
        tensor([[6, 3, 4]], requires_grad=True)

        Each configuration is drawn below:

        >>> for t in tapes:
        ...     print(qml.drawer.tape_text(t))
        ...     print("")

        .. code-block::

            0: ──H─╭●────┤  Sample[|1⟩⟨1|]
            1: ────╰X──X─┤  Sample[Z]

            0: ──H─╭●────┤  Sample[|1⟩⟨1|]
            1: ────╰X──X─┤  Sample[X]

            0: ──H─╭●────┤  Sample[|1⟩⟨1|]
            1: ────╰X──X─┤  Sample[Y]

            0: ──I─╭●─┤  Sample[|1⟩⟨1|]
            1: ────╰X─┤  Sample[|1⟩⟨1|]

            0: ──X──S─╭●─┤  Sample[|1⟩⟨1|]
            1: ───────╰X─┤  Sample[|1⟩⟨1|]

            0: ──H─╭●─┤  Sample[|1⟩⟨1|]
            1: ────╰X─┤  Sample[|1⟩⟨1|]

        The last step is to execute the tapes and postprocess the results using
        :func:`~.qcut_processing_fn_sample`, which processes the results to approximate the original full circuit
        output bitstrings.

        >>> results = qml.execute(tapes, dev, gradient_fn=None)
        >>> qml.qcut.qcut_processing_fn_sample(
        ...     results,
        ...     communication_graph,
        ...     shots=shots,
        ... )
        [array([[0., 0., 0.],
                [1., 0., 0.],
                [1., 0., 0.]])]

        Alternatively, it is possible to calculate an expectation value if a classical
        processing function is provided that will accept the reconstructed circuit bitstrings
        and return a value in the interval :math:`[-1, 1]`:

        .. code-block::

            def fn(x):
                if x[0] == 0:
                    return 1
                if x[0] == 1:
                    return -1

        >>> qml.qcut.qcut_processing_fn_mc(
        ...     results,
        ...     communication_graph,
        ...     settings,
        ...     shots,
        ...     fn
        ... )
        array(-4.)

        Using the Monte Carlo approach of `Peng et. al <https://arxiv.org/abs/1904.00102>`_, the
        ``cut_circuit_mc`` transform also supports returning sample-based expectation values of
        observables that are diagonal in the computational basis, as shown below for a `ZZ` measurement
        on wires `0` and `2`:

        .. code-block::

            from functools import partial
            dev = qml.device("default.qubit", wires=2, shots=10000)

            def observable(bitstring):
                return (-1) ** np.sum(bitstring)

            @partial(qml.cut_circuit_mc, classical_processing_fn=observable)
            @qml.qnode(dev)
            def circuit(x):
                qml.RX(0.89, wires=0)
                qml.RY(0.5, wires=1)
                qml.RX(1.3, wires=2)

                qml.CNOT(wires=[0, 1])
                qml.WireCut(wires=1)
                qml.CNOT(wires=[1, 2])

                qml.RX(x, wires=0)
                qml.RY(0.7, wires=1)
                qml.RX(2.3, wires=2)
                return qml.sample(wires=[0, 2])

        We can now approximate the expectation value of the observable using

        >>> circuit(x)
        tensor(-0.776, requires_grad=True)
    """
    # pylint: disable=unused-argument, too-many-arguments

    if len(tape.measurements) != 1:
        raise ValueError(
            "The Monte Carlo circuit cutting workflow only supports circuits "
            "with a single output measurement"
        )

    if not all(isinstance(m, SampleMP) for m in tape.measurements):
        raise ValueError(
            "The Monte Carlo circuit cutting workflow only supports circuits "
            "with sampling-based measurements"
        )

    for meas in tape.measurements:
        if meas.obs is not None:
            raise ValueError(
                "The Monte Carlo circuit cutting workflow only "
                "supports measurements in the computational basis. Please only specify "
                "wires to be sampled within qml.sample(), do not pass observables."
            )

    g = tape_to_graph(tape)

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

    replace_wire_cut_nodes(g)
    fragments, communication_graph = fragment_graph(g)
    fragment_tapes = [graph_to_tape(f) for f in fragments]
    fragment_tapes = [
        qml.map_wires(t, dict(zip(t.wires, device_wires)))[0][0] for t in fragment_tapes
    ]

    configurations, settings = expand_fragment_tapes_mc(
        fragment_tapes, communication_graph, shots=shots
    )

    tapes = tuple(tape for c in configurations for tape in c)

    if classical_processing_fn:

        def processing_fn(results):
            results = qcut_processing_fn_mc(
                results,
                communication_graph=communication_graph,
                settings=settings,
                shots=shots,
                classical_processing_fn=classical_processing_fn,
            )

            return results

    else:

        def processing_fn(results):
            results = qcut_processing_fn_sample(
                results, communication_graph=communication_graph, shots=shots
            )

            return results[0]

    return tapes, processing_fn


class CustomQNode(qml.QNode):
    """
    A subclass with a custom __call__ method. The custom QNode transform returns an instance
    of this class.
    """

    def __call__(self, *args, **kwargs):
        shots = kwargs.pop("shots", False)
        shots = shots or self.device.shots

        if not shots:
            raise ValueError(
                "A shots value must be provided in the device "
                "or when calling the QNode to be cut"
            )
        if isinstance(shots, qml.measurements.Shots):
            shots = shots.total_shots

        # find the qcut transform inside the transform program and set the shots argument
        qcut_tc = [
            tc for tc in self.transform_program if tc.transform.__name__ == "cut_circuit_mc"
        ][-1]
        qcut_tc._kwargs["shots"] = shots

        kwargs["shots"] = 1
        return super().__call__(*args, **kwargs)


@cut_circuit_mc.custom_qnode_transform
def _qnode_transform_mc(self, qnode, targs, tkwargs):
    """Here, we overwrite the QNode execution wrapper in order
    to access the device wires."""
    if tkwargs.get("shots", False):
        raise ValueError(
            "Cannot provide a 'shots' value directly to the cut_circuit_mc "
            "decorator when transforming a QNode. Please provide the number of shots in "
            "the device or when calling the QNode."
        )

    if "shots" in inspect.signature(qnode.func).parameters:
        raise ValueError(
            "Detected 'shots' as an argument of the quantum function to transform. "
            "The 'shots' argument name is reserved for overriding the number of shots "
            "taken by the device."
        )

    tkwargs.setdefault("device_wires", qnode.device.wires)

    execute_kwargs = getattr(qnode, "execute_kwargs", {}).copy()
    execute_kwargs["cache"] = False

    new_qnode = self.default_qnode_transform(qnode, targs, tkwargs)
    new_qnode.__class__ = CustomQNode
    new_qnode.execute_kwargs = execute_kwargs

    return new_qnode


MC_STATES = [
    _prep_zero_state,
    _prep_one_state,
    _prep_plus_state,
    _prep_minus_state,
    _prep_iplus_state,
    _prep_iminus_state,
    _prep_zero_state,
    _prep_one_state,
]


def _identity(wire):
    return qml.sample(qml.Identity(wires=wire))


def _pauliX(wire):
    return qml.sample(qml.X(wire))


def _pauliY(wire):
    return qml.sample(qml.Y(wire))


def _pauliZ(wire):
    return qml.sample(qml.Z(wire))


MC_MEASUREMENTS = [
    _identity,
    _identity,
    _pauliX,
    _pauliX,
    _pauliY,
    _pauliY,
    _pauliZ,
    _pauliZ,
]


def expand_fragment_tapes_mc(
    tapes: Sequence[QuantumTape], communication_graph: MultiDiGraph, shots: int
) -> Tuple[List[QuantumTape], np.ndarray]:
    """
    Expands fragment tapes into a sequence of random configurations of the contained pairs of
    :class:`MeasureNode` and :class:`PrepareNode` operations.

    For each pair, a measurement is sampled from
    the Pauli basis and a state preparation is sampled from the corresponding pair of eigenstates.
    A settings array is also given which tracks the configuration pairs. Since each of the 4
    measurements has 2 possible eigenvectors, all configurations can be uniquely identified by
    8 values. The number of rows is determined by the number of cuts and the number of columns
    is determined by the number of shots.

    .. note::

        This function is designed for use as part of the sampling-based circuit cutting workflow.
        Check out the :func:`~.cut_circuit_mc` transform for more details.

    Args:
        tapes (Sequence[QuantumTape]): the fragment tapes containing :class:`MeasureNode` and
            :class:`PrepareNode` operations to be expanded
        communication_graph (nx.MultiDiGraph): the communication (quotient) graph of the fragmented
            full graph
        shots (int): number of shots

    Returns:
        Tuple[List[QuantumTape], np.ndarray]: the tapes corresponding to each configuration and the
        settings that track each configuration pair

    **Example**

    Consider the following circuit that contains a sample measurement:

    .. code-block:: python

        ops = [
            qml.Hadamard(wires=0),
            qml.CNOT(wires=[0, 1]),
            qml.WireCut(wires=1),
            qml.CNOT(wires=[1, 2]),
        ]
        measurements = [qml.sample(wires=[0, 1, 2])]
        tape = qml.tape.QuantumTape(ops, measurements)

    We can generate the fragment tapes using the following workflow:

    >>> g = qml.qcut.tape_to_graph(tape)
    >>> qml.qcut.replace_wire_cut_nodes(g)
    >>> subgraphs, communication_graph = qml.qcut.fragment_graph(g)
    >>> tapes = [qml.qcut.graph_to_tape(sg) for sg in subgraphs]

    We can then expand over the measurement and preparation nodes to generate random
    configurations using:

    .. code-block:: python

        >>> configs, settings = qml.qcut.expand_fragment_tapes_mc(tapes, communication_graph, 3)
        >>> print(settings)
        [[1 6 2]]
        >>> for i, (c1, c2) in enumerate(zip(configs[0], configs[1])):
        ...     print(f"config {i}:")
        ...     print(c1.draw())
        ...     print("")
        ...     print(c2.draw())
        ...     print("")
        ...

        config 0:
        0: ──H─╭●─┤  Sample[|1⟩⟨1|]
        1: ────╰X─┤  Sample[Z]

        1: ──I─╭●─┤  Sample[|1⟩⟨1|]
        2: ────╰X─┤  Sample[|1⟩⟨1|]

        config 1:
        0: ──H─╭●─┤  Sample[|1⟩⟨1|]
        1: ────╰X─┤  Sample[Y]

        1: ──H──S─╭●─┤  Sample[|1⟩⟨1|]
        2: ───────╰X─┤  Sample[|1⟩⟨1|]

        config 2:
        0: ──H─╭●─┤  Sample[|1⟩⟨1|]
        1: ────╰X─┤  Sample[Y]

        1: ──X──H──S─╭●─┤  Sample[|1⟩⟨1|]
        2: ──────────╰X─┤  Sample[|1⟩⟨1|]

    """
    pairs = [e[-1] for e in communication_graph.edges.data("pair")]
    settings = np.random.choice(range(8), size=(len(pairs), shots), replace=True)

    meas_settings = {pair[0].obj.id: setting for pair, setting in zip(pairs, settings)}
    prep_settings = {pair[1].obj.id: setting for pair, setting in zip(pairs, settings)}

    all_configs = []
    for tape in tapes:
        frag_config = []
        for shot in range(shots):
            expanded_circuit_operations = []
            expanded_circuit_measurements = tape.measurements.copy()
            for op in tape.operations:
                w = op.wires[0]
                if isinstance(op, PrepareNode):
                    expanded_circuit_operations.extend(MC_STATES[prep_settings[op.id][shot]](w))
                elif not isinstance(op, MeasureNode):
                    expanded_circuit_operations.append(op)
                else:
                    expanded_circuit_measurements.append(
                        MC_MEASUREMENTS[meas_settings[op.id][shot]](w)
                    )

            frag_config.append(
                QuantumScript(
                    ops=expanded_circuit_operations,
                    measurements=expanded_circuit_measurements,
                    shots=1,
                )
            )

        all_configs.append(frag_config)

    return all_configs, settings
