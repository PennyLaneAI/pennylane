"""
Contains the transpiler transform.
"""
from typing import Union, List
import networkx as nx

import pennylane as qml
from pennylane import apply, Hamiltonian
from pennylane.ops.qubit import SWAP
from pennylane.operation import Tensor
from pennylane.ops import __all__ as all_ops
from pennylane.tape import QuantumTape
from pennylane.transforms import qfunc_transform
from pennylane.queuing import stop_recording
from pennylane.wires import Wires


@qfunc_transform
def transpile(tape: QuantumTape, coupling_map: Union[List, nx.Graph]):
    """Transpile a circuit according to a desired coupling map

    .. warning::

        This transform does not yet support measurements of Hamiltonians or tensor products of observables. If a circuit
        is passed which contains these types of measurements, a ``NotImplementedError`` will be raised.

    Args:
        tape (function): A quantum function.
        coupling_map (list[tuple(int, int)] or nx.Graph): Either a list of tuples(int, int) or an instance of
            `networkx.Graph` specifying the couplings between different qubits.

    Returns:
        function: the transformed quantum function

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
    >>> transpiled_circuit = qml.transforms.transpile(coupling_map=[(0, 1), (1, 3), (3, 2), (2, 0)])(circuit)
    >>> transpiled_qnode = qml.QNode(circuit, dev)
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

    gates = []

    # we wrap all manipulations inside stop_recording() so that we don't queue anything due to unrolling of templates
    # or newly applied swap gates
    with stop_recording():
        # this unrolls everything in the current tape (in particular templates)
        def stop_at(obj):
            return (obj.name in all_ops) and (not getattr(obj, "only_visual", False))

        expanded_tape = tape.expand(stop_at=stop_at)

        # make copy of ops
        list_op_copy = expanded_tape.operations.copy()
        measurements = expanded_tape.measurements.copy()

        while len(list_op_copy) > 0:
            op = list_op_copy[0]

            # gates which act only on one wire
            if op.num_wires != 2:
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
            map_wires = {w: w for w in tape.wires}

            # to make sure two qubit gates which act on non-neighbouring qubits q1, q2 can be applied, we first look
            # for the shortest path between the two qubits in the connectivity graph. We then move the q2 into the
            # neighbourhood of q1 via swap operations.
            source_wire, dest_wire = op.wires
            shortest_path = nx.algorithms.shortest_path(coupling_graph, source_wire, dest_wire)
            path_length = len(shortest_path) - 1
            wires_to_swap = [shortest_path[(i - 1) : (i + 1)] for i in range(path_length, 1, -1)]

            for w0, w1 in wires_to_swap:
                # swap wires
                gates.append(SWAP(wires=[w0, w1]))
                # update logical -> phyiscal mapping
                map_wires = {
                    k: (w0 if v == w1 else (w1 if v == w0 else v)) for k, v in map_wires.items()
                }

            # append op to gates with adjusted indices and remove from list
            gates.append(_adjust_op_indices(op, map_wires))
            list_op_copy.pop(0)

            # adjust qubit indices in remaining ops + measurements to new mapping
            list_op_copy = [_adjust_op_indices(op, map_wires) for op in list_op_copy]
            measurements = [_adjust_mmt_indices(m, map_wires) for m in measurements]

    for op in gates + measurements:
        apply(op)


def _adjust_op_indices(_op, _map_wires):
    """helper function which adjusts wires in Operation according to the map _map_wires"""
    _new_wires = Wires([_map_wires[w] for w in _op.wires])
    _params = _op.parameters
    if len(_params) == 0:
        return type(_op)(wires=_new_wires)
    return type(_op)(*_params, wires=_new_wires)


def _adjust_mmt_indices(_m, _map_wires):
    """helper function which adjusts wires in MeasurementProcess according to the map _map_wires"""
    _new_wires = Wires([_map_wires[w] for w in _m.wires])

    # change wires of observable
    if _m.obs is None:
        return type(_m)(return_type=_m.return_type, eigvals=qml.eigvals(_m), wires=_new_wires)

    _new_obs = type(_m.obs)(wires=_new_wires, id=_m.obs.id)
    return type(_m)(return_type=_m.return_type, obs=_new_obs)
