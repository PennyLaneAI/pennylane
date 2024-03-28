"""
Contains the transpiler transform.
"""

from functools import partial
from typing import List, Union, Sequence, Callable

import networkx as nx

import pennylane as qml
from pennylane.transforms import transform
from pennylane.ops import Hamiltonian
from pennylane.ops import LinearCombination
from pennylane.operation import Tensor
from pennylane.ops import __all__ as all_ops
from pennylane.ops.qubit import SWAP
from pennylane.queuing import QueuingManager
from pennylane.tape import QuantumTape


def state_transposition(results, mps, new_wire_order, original_wire_order):
    """Transpose the order of any state return.

    Args:
        results (ResultBatch): the result of executing a batch of length 1

    Keyword Args:
        mps (List[MeasurementProcess]): A list of measurements processes. At least one is a ``StateMP``
        new_wire_order (Sequence[Any]): the wire order after transpile has been called
        original_wire_order (.Wires): the devices wire order

    Returns:
        Result: The result object with state dimensions transposed.

    """
    if len(mps) == 1:
        temp_mp = qml.measurements.StateMP(wires=original_wire_order)
        return temp_mp.process_state(results[0], wire_order=qml.wires.Wires(new_wire_order))
    new_results = list(results[0])
    for i, mp in enumerate(mps):
        if isinstance(mp, qml.measurements.StateMP):
            temp_mp = qml.measurements.StateMP(wires=original_wire_order)
            new_res = temp_mp.process_state(
                new_results[i], wire_order=qml.wires.Wires(new_wire_order)
            )
            new_results[i] = new_res
    return tuple(new_results)


def _process_measurements(expanded_tape, device_wires, is_default_mixed):
    measurements = expanded_tape.measurements.copy()
    if device_wires:
        for i, m in enumerate(measurements):
            if isinstance(m, qml.measurements.StateMP):
                if is_default_mixed:
                    measurements[i] = qml.density_matrix(wires=device_wires)
            elif not m.wires:
                measurements[i] = type(m)(wires=device_wires)

    return measurements


@transform
def transpile(
    tape: QuantumTape, coupling_map: Union[List, nx.Graph], device=None
) -> (Sequence[QuantumTape], Callable):
    """Transpile a circuit according to a desired coupling map

    .. warning::

        This transform does not yet support measurements of Hamiltonians or tensor products of observables. If a circuit
        is passed which contains these types of measurements, a ``NotImplementedError`` will be raised.

    Args:
        tape (QNode or QuantumTape or Callable): A quantum tape.
        coupling_map (list[tuple(int, int)] or nx.Graph): Either a list of tuples(int, int) or an instance of
            `networkx.Graph` specifying the couplings between different qubits.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[.QuantumTape], function]: The transformed circuit as described in :func:`qml.transform <pennylane.transform>`.

    **Example**

    Consider the following example circuit

    .. code-block:: python

        def circuit():
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[2, 3])
            qml.CNOT(wires=[1, 3])
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[2, 3])
            qml.CNOT(wires=[0, 3])
            return qml.probs(wires=[0, 1, 2, 3])

    which, before transpiling it looks like this:

    .. code-block:: text

        0: ──╭●──────────────╭●──╭┤ Probs
        1: ──╰X──╭●──╭●──────│───├┤ Probs
        2: ──╭●──│───╰X──╭●──│───├┤ Probs
        3: ──╰X──╰X──────╰X──╰X──╰┤ Probs

    Suppose we have a device which has connectivity constraints according to the graph:

    .. code-block:: text

        0 --- 1
        |     |
        2 --- 3

    We encode this in a coupling map as a list of the edges which are present in the graph, and then pass this, together
    with the circuit, to the transpile function to get a circuit which can be executed for the specified coupling map:

    >>> dev = qml.device('default.qubit', wires=[0, 1, 2, 3])
    >>> transpiled_circuit = qml.transforms.transpile(circuit, coupling_map=[(0, 1), (1, 3), (3, 2), (2, 0)])
    >>> transpiled_qnode = qml.QNode(transpiled_circuit, dev)
    >>> print(qml.draw(transpiled_qnode)())
    0: ─╭●────────────────╭●─┤ ╭Probs
    1: ─╰X─╭●───────╭●────│──┤ ├Probs
    2: ─╭●─│──╭SWAP─│──╭X─╰X─┤ ├Probs
    3: ─╰X─╰X─╰SWAP─╰X─╰●────┤ ╰Probs

    A swap gate has been applied to wires 2 and 3, and the remaining gates have been adapted accordingly

    """
    if device:
        device_wires = device.wires
        is_default_mixed = getattr(device, "short_name", "") == "default.mixed"
    else:
        device_wires = None
        is_default_mixed = False
    # init connectivity graph
    coupling_graph = (
        nx.Graph(coupling_map) if not isinstance(coupling_map, nx.Graph) else coupling_map
    )

    # make sure every wire is present in coupling map
    if any(wire not in coupling_graph.nodes for wire in tape.wires):
        wires = tape.wires.tolist()
        raise ValueError(
            f"Not all wires present in coupling map! wires: {wires}, coupling map: {coupling_graph.nodes}"
        )

    if any(
        isinstance(m.obs, (Hamiltonian, LinearCombination, Tensor, qml.ops.Prod))
        for m in tape.measurements
    ):
        raise NotImplementedError(
            "Measuring expectation values of tensor products, Prods, or Hamiltonians is not yet supported"
        )

    if any(len(op.wires) > 2 for op in tape.operations):
        raise NotImplementedError(
            "The transpile transform only supports gates acting on 1 or 2 qubits."
        )

    # we wrap all manipulations inside stop_recording() so that we don't queue anything due to unrolling of templates
    # or newly applied swap gates
    with QueuingManager.stop_recording():
        # this unrolls everything in the current tape (in particular templates)
        def stop_at(obj):
            return (obj.name in all_ops) and (not getattr(obj, "only_visual", False))

        expanded_tape = tape.expand(stop_at=stop_at)

        # make copy of ops
        list_op_copy = expanded_tape.operations.copy()
        wire_order = device_wires or tape.wires
        measurements = _process_measurements(expanded_tape, device_wires, is_default_mixed)

        gates = []

        while len(list_op_copy) > 0:
            op = list_op_copy[0]

            # gates which act only on one wire
            if len(op.wires) == 1:
                gates.append(op)
                list_op_copy.pop(0)
                continue

            # two-qubit gates which can be handled by the coupling map
            if (
                op.wires in coupling_graph.edges
                or tuple(reversed(op.wires)) in coupling_graph.edges
            ):
                gates.append(op)
                list_op_copy.pop(0)
                continue

            # since in each iteration, we adjust indices of each op, we reset logical -> phyiscal mapping
            wire_map = {w: w for w in wire_order}

            # to make sure two qubit gates which act on non-neighbouring qubits q1, q2 can be applied, we first look
            # for the shortest path between the two qubits in the connectivity graph. We then move the q2 into the
            # neighbourhood of q1 via swap operations.
            source_wire, dest_wire = op.wires
            # pylint:disable=too-many-function-args
            shortest_path = nx.algorithms.shortest_path(coupling_graph, source_wire, dest_wire)
            path_length = len(shortest_path) - 1
            wires_to_swap = [shortest_path[(i - 1) : (i + 1)] for i in range(path_length, 1, -1)]

            for w0, w1 in wires_to_swap:
                # swap wires
                gates.append(SWAP(wires=[w0, w1]))
                # update logical -> phyiscal mapping
                wire_map = {
                    k: (w0 if v == w1 else (w1 if v == w0 else v)) for k, v in wire_map.items()
                }

            # append op to gates with adjusted indices and remove from list
            gates.append(op.map_wires(wire_map))
            list_op_copy.pop(0)

            list_op_copy = [op.map_wires(wire_map) for op in list_op_copy]
            wire_order = [wire_map[w] for w in wire_order]
            measurements = [m.map_wires(wire_map) for m in measurements]
    new_tape = type(tape)(gates, measurements, shots=tape.shots)

    # note: no need for transposition with density matrix, so type must be `StateMP` but not `DensityMatrixMP`
    # pylint: disable=unidiomatic-typecheck
    any_state_mp = any(type(m) is qml.measurements.StateMP for m in measurements)
    if not any_state_mp or device_wires is None:

        def null_postprocessing(results):
            """A postprocesing function returned by a transform that only converts the batch of results
            into a result for a single ``QuantumTape``.
            """
            return results[0]

        return (new_tape,), null_postprocessing

    return (new_tape,), partial(
        state_transposition,
        mps=measurements,
        new_wire_order=wire_order,
        original_wire_order=device_wires,
    )


@transpile.custom_qnode_transform
def _transpile_qnode(self, qnode, targs, tkwargs):
    """Custom qnode transform for ``transpile``."""
    if tkwargs.get("device", None):
        raise ValueError(
            "Cannot provide a 'device' value directly to the defer_measurements decorator "
            "when transforming a QNode."
        )

    tkwargs.setdefault("device", qnode.device)
    return self.default_qnode_transform(qnode, targs, tkwargs)
