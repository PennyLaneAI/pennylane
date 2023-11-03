"""
Contains the transpiler transform.
"""
from typing import List, Union, Sequence, Callable

import networkx as nx

from pennylane.transforms import transform
from pennylane import Hamiltonian
from pennylane.operation import Tensor
from pennylane.ops import __all__ as all_ops
from pennylane.ops.qubit import SWAP
from pennylane.queuing import QueuingManager
from pennylane.tape import QuantumTape


@transform
def transpile(
    tape: QuantumTape, coupling_map: Union[List, nx.Graph]
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

    if any(isinstance(m.obs, (Hamiltonian, Tensor)) for m in tape.measurements):
        raise NotImplementedError(
            "Measuring expectation values of tensor products or Hamiltonians is not yet supported"
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
        measurements = expanded_tape.measurements.copy()
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
            wire_map = {w: w for w in tape.wires}

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
            measurements = [m.map_wires(wire_map) for m in measurements]
    new_tape = type(tape)(gates, measurements, shots=tape.shots)

    def null_postprocessing(results):
        """A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]

    return [new_tape], null_postprocessing
