# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Transform for removing the Barrier gate from quantum circuits."""

import networkx as nx
from collections import defaultdict

import random
from itertools import product
from pennylane import numpy as np
from pennylane.measurements import MeasurementProcess
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.transforms import transform
from pennylane.typing import PostprocessingFn

def pennylane_to_networkx(tape) -> nx.DiGraph:
    """
    Converts a PennyLane QuantumTape circuit into a NetworkX directed graph,
    following a representation similar to a causal or flow graph,
    where nodes represent qubit states before/after operations and the operations themselves.

    The graph will have three types of nodes for visualization:
    1. Initial Qubit State nodes (Green): Represented as 'Qubit_{wire_idx}_Init'.
       These are the "resource" qubits (q^r) that are not outputs of any preceding gate in the graph.
    2. Operator nodes (Blue): Represented as 'Op_{op_idx}_{op_name}'.
       These are the quantum gates like Hadamard, CNOT, RX, etc.
    3. Intermediate/Final Qubit State nodes (Purple): Represented as 'Qubit_{wire_idx}_AfterOp_{op_idx}'.
       These are the "target" qubits (q^t) which are the output of an operation.
       If these qubits subsequently become inputs to another operation, they retain this type/color.

    Edges represent the flow:
    - From a qubit state node (green or purple) to an operator node (blue) if the qubit is an input to the operator.
    - From an operator node (blue) to a new qubit state node (purple) representing the qubit's state after the operation.

    Args:
        tape (qml.tape.QuantumTape): The PennyLane QuantumTape (quantum circuit) to convert.

    Returns:
        nx.DiGraph: A directed graph representing the quantum circuit.
    """
        # Create a new directed graph
    G = nx.DiGraph()

    # Get the number of wires directly from the tape's wires attribute.
    # This is more direct when dealing with a tape object.
    num_wires = len(range(min(tape.wires),max(tape.wires)+1))

    # Dictionary to keep track of the *current* qubit state node (its ID) on each wire.
    # This helps in creating sequential connections based on the flow through operations.
    current_qubit_state_node = {}

    # 1. Add Initial Qubit State Nodes (Green)
    # These represent the starting state of each qubit before any operations.
    for i in range(num_wires):
        node_name = f"Qubit_{i}_Init"
        G.add_node(node_name, type="initial_qubit_state", wire=i)
        current_qubit_state_node[i] = node_name # Initialize the current node for this wire

    # 2. Add Operator Nodes (Blue) and connect them to qubit state nodes.
    # Iterate through each operation in the PennyLane tape.
    for op_idx, op in enumerate(tape.operations):
        op_node_id = f"Op_{op_idx}_{op.name}"
        # Add the operator node to the graph with 'operator' type.
        G.add_node(op_node_id, type="operator", name=op.name)

        # Process each wire involved in the current operation.
        for wire in op.wires:
            wire_idx = wire

            # Get the current qubit state node for this wire.
            # This node represents the qubit's state *before* the current operation.
            input_qubit_node = current_qubit_state_node[wire_idx]

            # Add an edge from the input qubit state node to the operator node.
            G.add_edge(input_qubit_node, op_node_id)

            # Create a new qubit state node representing the qubit's state *after* this operation.
            # This new node will be of type 'intermediate_or_final_qubit_state' (purple).
            output_qubit_node_id = f"Qubit_{wire_idx}_AfterOp_{op_idx}"
            G.add_node(output_qubit_node_id, type="intermediate_or_final_qubit_state", wire=wire_idx, from_op=op_node_id)

            # Add an edge from the operator node to the new output qubit state node.
            G.add_edge(op_node_id, output_qubit_node_id)

            # Update the current qubit state node for this wire to the newly created output node.
            # This ensures that subsequent operations on this wire will connect from this new state.
            current_qubit_state_node[wire_idx] = output_qubit_node_id

    return G

def merge_subsets(list_of_pairs):
    """
    Merges sublists in a list of pairs such that each merged list contains all elements
    that are connected directly or indirectly through common elements, maintaining
    the original ordering of elements. This function is useful for identifying interconnected
    components or groups within a list of pairs, where a pair represents a direct connection
    between two elements.

    Parameters:
    - list_of_pairs (list of lists): A list where each sublist contains a pair of integers
      representing a direct connection.

    Returns:
    - list of lists: Merged sublists based on common elements, preserving original order.
      Each sublist represents a group of interconnected elements.
    """

    def find_merge_index(merged_list, pair):
        """
        Finds the index of the sublist within merged_list that shares a common element with the given pair.
        Returns -1 if no common element is found.

        Parameters:
        - merged_list (list of lists): The current list of merged sublists.
        - pair (list): The pair of elements to find a merge candidate for.

        Returns:
        - int: The index of the sublist in merged_list that has a common element with pair, or -1 if none.
        """
        for i, sublist in enumerate(merged_list):
            if any(elem in sublist for elem in pair):
                return i
        return -1

    def merge_and_order(sublist1, sublist2):
        """
        Merges two sublists into one, ensuring that elements from sublist2 are inserted
        into sublist1 in their relative order, maintaining the original ordering of elements.

        Parameters:
        - sublist1 (list): The first sublist to merge.
        - sublist2 (list): The second sublist to merge.

        Returns:
        - list: The merged and ordered sublist.
        """
        # Create a set for faster lookups
        sublist1_set = set(sublist1)
        for elem in sublist2:
            if elem not in sublist1_set:
                sublist1_set.add(elem)
                # Determine the position to insert the element based on relative order in sublist2
                position = next((i for i, x in enumerate(sublist1) if x in sublist2 and sublist2.index(x) > sublist2.index(elem)), len(sublist1))
                sublist1.insert(position, elem)
        return sublist1

    merged_list = []
    for pair in list_of_pairs:
        pair = list(pair)  # Convert set to list to maintain order
        merge_index = find_merge_index(merged_list, pair)
        if merge_index != -1:
            # Merge the pair into the found sublist
            merged_list[merge_index] = merge_and_order(merged_list[merge_index], pair)
        else:
            # No common element found, add the pair as a new sublist
            merged_list.append(pair)

    # Additional merging passes to ensure all interconnected sublists are fully merged
    merge_occurred = True
    while merge_occurred:
        merge_occurred = False
        for i in range(len(merged_list)):
            for j in range(i + 1, len(merged_list)):
                if any(elem in merged_list[j] for elem in merged_list[i]):
                    # Merge i-th and j-th sublists and remove the j-th sublist
                    merged_list[i] = merge_and_order(merged_list[i], merged_list[j])
                    del merged_list[j]
                    merge_occurred = True
                    break
            if merge_occurred:
                break

    return merged_list



def finalize_reuse(qubit_reuse_list, num_qubits):
    """
    Finalizes the qubit reuse list by ensuring that all qubits are included, either as part of a reuse chain
    or as individual elements if they were not involved in any reuse operation.

    Parameters:
    - qubit_reuse_list (list of lists): The current list of qubit reuse operations.
    - num_qubits (int): The total number of qubits in the quantum circuit.

    Returns:
    - list of lists: The finalized qubit reuse list with all qubits accounted for.
    """
    included_qubits = set(qubit for sublist in qubit_reuse_list for qubit in sublist)
    all_qubits = set(range(num_qubits))
    missing_qubits = all_qubits - included_qubits

    for qubit in missing_qubits:
        qubit_reuse_list.append([qubit])  # Add missing qubits as individual sets.

    return qubit_reuse_list


def update_candidate_matrix(C, output_qubit, input_qubit):
    """
    Updates the candidate matrix after the selection of specific qubits.

    Parameters:
    C (numpy.ndarray): The candidate matrix to be updated.
    output_qubit (int): The selected output qubit index.
    input_qubit (int): The selected input qubit index.

    Returns:
    numpy.ndarray: The updated candidate matrix.
    """
    n = C.shape[0]

    # Identify the sets Q_r and Q_t
    Q_r = {k for k in range(n) if C[output_qubit, k] == 0}
    Q_t = {k for k in range(n) if C[k, input_qubit] == 0}

    # Update the matrix C based on the cartesian product of Q_t and Q_r
    for k in Q_t:
        for l in Q_r:
            C[k, l] = 0

    # Set all entries in the row corresponding to the output qubit to 0
    C[output_qubit, :] = 0

    # Set all entries in the column corresponding to the input qubit to 0
    C[:, input_qubit] = 0

    return C

def best_qpath(C, output_qubit_index):
    """
    Determines the optimized reuse path for a given output qubit.

    Parameters:
    C (numpy.ndarray): The C matrix representing connections.
    output_qubit_index (int): The index of the output qubit.

    Returns:
    tuple: A tuple containing the optimized path set and the updated C matrix.
    """
    # Initialize the path and potential reuse set
    Q_p_i = [output_qubit_index]
    P_r_i = {j for j in range(C.shape[1]) if C[output_qubit_index, j] == 1}

    while P_r_i:
        D = defaultdict(set)

        # Compute the intersection sets
        for input_qubit in P_r_i:
            neighbors = set.intersection(*[set(np.where(C[k] == 1)[0])
                                           for k in (Q_p_i + [input_qubit])])
            D[input_qubit] = neighbors

        if all(len(D[input_qubit]) == 0 for input_qubit in D):
            if P_r_i:
                input_qubit = random.choice(list(P_r_i))
                Q_p_i.append(input_qubit)
                P_r_i.remove(input_qubit)

            if len(Q_p_i) > 1:
                for k in range(len(Q_p_i) - 1):
                    output_qubit = Q_p_i[k]
                    input_qubit = Q_p_i[k + 1]
                    C = update_candidate_matrix(C, output_qubit, input_qubit)

            return Q_p_i, C
        else:
            max_size = max(len(D[input_qubit]) for input_qubit in D)
            M = {input_qubit for input_qubit in D if len(D[input_qubit]) == max_size}

            if len(M) == 1:
                input_qubit = next(iter(M))
            else:
                S = {}
                for input_qubit in M:
                    neighbors_j = D[input_qubit]
                    sigma = [len(neighbors_j & D[k]) for k in M if k != input_qubit]
                    S[input_qubit] = sum(sigma)

                max_intersection = max(S.values())
                L = [input_qubit for input_qubit in S if S[input_qubit] == max_intersection]
                input_qubit = random.choice(L)

            neighbors_j = D[input_qubit]
            Q_p_i.append(input_qubit)
            P_r_i = neighbors_j

    if len(Q_p_i) > 1:
        for k in range(len(Q_p_i) - 1):
            output_qubit = Q_p_i[k]
            input_qubit = Q_p_i[k + 1]
            C = update_candidate_matrix(C, output_qubit, input_qubit)

    return Q_p_i, C


def add_qubit_reuse_edges(circuit, qubit_reuse_list):
    """
    Given the original circuit, and the a list, qubit_reuse_list, that 
    contains the ordering of qubits in the dynamic circuit, the function 
    returns a DAG that is a modified version of the original circuit DAG.
    This modified DAG is a copy of the original DAG, but with edges that 
    connects the reused qubits to the corresponding qubits that reused them
    """
    # this should be done before adding connecting edges.
    circ_dag = circuit
    # display(circ_dag.draw())
    circ_dag_copy = copy.deepcopy(circ_dag)
    total_qubits = len(circ_dag_copy.qubits)


    # Create Edges
    roots = list(circ_dag_copy.input_map.values())  # the roots or input nodes of the circuit
    terminals = list(circ_dag_copy.output_map.values()) # the output nodes of the circuit

    new_edges = []
    for qubit_list in qubit_reuse_list:
        if len(qubit_list)>0:
            # Group the qubit indices to form edges
            new_edges_indx = [(qubit_list[i], qubit_list[i + 1]) for 
                              i in range(len(qubit_list) - 1)]
            for root, terminal in new_edges_indx:
                # the root here was the terminal before.
                # The terminal is the qubit that will be 
                # reused on the root
                new_edges.append((terminals[root], roots[terminal]))

    # display(circ_dag_copy.draw())             
    for inp_node, outp_node in new_edges:
        circ_dag_copy._multi_graph.add_edge(inp_node._node_id, outp_node._node_id, inp_node.wire)
                
    return circ_dag_copy, new_edges


def static_to_dynamic_circuit(circuit, qubit_reuse_list, draw=False):
    """
    Converts a static quantum circuit into a dynamic quantum circuit by adding edges to the
    original Directed Acyclic Graph (DAG) of the circuit and reordering qubits based on 
    the qubit reuse list. This transformation is aimed at optimizing qubit reuse and improving 
    the efficiency of quantum circuit execution.

    The function first computes qubit reuse sets using the `compute_qubit_reuse_sets` function. 
    It then adds edges to the original DAG circuit based on these reuse sets and removes any 
    barrier nodes to avoid cyclical DAGs and enable topological sorting. The qubits of the 
    original circuit are reordered into a new dynamic circuit based on the qubit reuse list.

    Parameters:
    - circuit (QuantumCircuit): The static quantum circuit to be transformed into a dynamic circuit.

    Returns:
    - QuantumCircuit: The transformed dynamic quantum circuit.

    The function also handles the edge case where a cycle might be detected in the DAG after adding new edges.
    In such cases, an AssertionError is raised. For each node in the DAG, the function applies operations
    based on the remapped qubits, ensuring the dynamic circuit maintains the intended functionality of the
    original static circuit.
    """

    # Return the original circuit if no qubit reuse is found
    # if len(qubit_reuse_list) == circuit.num_qubits:
    if qubit_reuse_list is None:
        # print("This circuit is irreducible")
        return circuit

    circ_dag_copy, new_edges = add_qubit_reuse_edges(circuit, qubit_reuse_list) # add edges using the qubit_reuse_list

	# remove barrier nodes from the new dag to avoid cyclical DAG and enable topological sorting
    barrier_nodes = [node for node in list(circ_dag_copy.nodes()) if isinstance(node, DAGOpNode)  and node.op.name == "barrier"]

    for op_node in barrier_nodes:
        circ_dag_copy.remove_op_node(op_node)

    # Check if there is a cycle in the DAG
    if len(list(circ_dag_copy.nodes())) > len(list(circ_dag_copy.topological_nodes())):
        assert False, "A cycle detected in the DAG"

    # print("qubit_reuse_list length = ", len(qubit_reuse_list))
    old_to_new_qubit_remap_dict = {} # map old qubits to new qubits
    for new_qubit in range(len(qubit_reuse_list)):
        for old_qubit in qubit_reuse_list[new_qubit]:
            old_to_new_qubit_remap_dict[old_qubit] = new_qubit

    # print("qubit_reuse_list = ", qubit_reuse_list)

    # print("old_to_new_qubit_remap_dict = ", old_to_new_qubit_remap_dict)

    # loop through the nodes in the circuit DAG
    new_dag_copy = DAGCircuit()
	# Add quantum registers to new_dag_copy
    num_qubits = len(qubit_reuse_list) # number of qubits is the length of the qubit_reuse_list
    qregs = QuantumRegister(num_qubits)
    new_dag_copy.add_qreg(qregs)
    qubits_indices = {i: qubit for i, qubit in enumerate(new_dag_copy.qubits)}  # Assign index to each clbit from 0 to total number of 

    # Add classical registers to new_dag_copy
    num_clbits = circ_dag_copy.num_clbits()
    cregs = ClassicalRegister(num_clbits)
    new_dag_copy.add_creg(cregs)
    clbits_indices = {i: clbit for i, clbit in enumerate(new_dag_copy.clbits)}  # Assign index to each clbit from 0 to total number of clbits-1
    
    total_qr = len(qubit_reuse_list) # total number of qubits in dynamic circuit
    measure_reset_nodes = [edge[0].__hash__() for edge in new_edges] # nodes to replace with measure and reset
    remove_input_nodes = [edge[1].__hash__() for edge in new_edges] # nodes to replace with measure and reset
    track_qubits = []

    for node in circ_dag_copy.topological_nodes(): # it is important to organize it topologically
        if not isinstance(node, DAGInNode) and not isinstance(node, DAGOutNode):
            old_qubits = [n.index for n in node.qargs]
            clbits = [n.index for n in node.cargs]

            if len(old_qubits) > 1: # this is the two-qubit gate case
                new_qubits = [old_to_new_qubit_remap_dict[q] for q in old_qubits]
                new_wires = [qubits_indices[q] for q in new_qubits]
                new_qargs = tuple(new_wires)

                clbits_wires = [clbits_indices[c] for c in clbits]
                new_cargs = tuple(clbits_wires)

                new_node = DAGOpNode(op=node.op, qargs=new_qargs, cargs=new_cargs)
                new_dag_copy.apply_operation_back(new_node.op, qargs=new_node.qargs, cargs=new_node.cargs)

            else: # this is the one-qubit gate case
                old_qubits = old_qubits[0] # to eliminate list for 1-qubit
                new_qubits = old_to_new_qubit_remap_dict[old_qubits]
                new_wires = [qubits_indices[new_qubits]]
                new_qargs = tuple(new_wires)

                clbits_wires = [clbits_indices[c] for c in clbits]
                new_cargs = tuple(clbits_wires)
                new_node = DAGOpNode(op=node.op, qargs=new_qargs, cargs=new_cargs)
                new_dag_copy.apply_operation_back(new_node.op, qargs=new_node.qargs, cargs=new_node.cargs)
        else:
            old_qubits = node.wire.index
            new_qubits = old_to_new_qubit_remap_dict[old_qubits]
            new_wires = [qubits_indices[new_qubits]]
            new_qargs = tuple(new_wires)

            if isinstance(node, DAGInNode):
                continue # since we have accounted for the qubits
            elif isinstance(node, DAGOutNode):
                # DAGOutNode and hence, reset operation has no cargs
                if node.__hash__() in measure_reset_nodes:
                    reset_node = add_reset_op(new_qargs)
                    new_dag_copy.apply_operation_back(reset_node.op, qargs=reset_node.qargs, cargs=reset_node.cargs)
                else:
                    continue

    
    new_circ = dag_to_circuit(new_dag_copy)
    
    # if draw:
    #     # display(new_circ.draw("mpl", fold=-1))
    #     display(new_circ.draw("mpl"))
    
    return new_circ


@transform
def qubit_reuse(tape: QuantumScript) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """TODO: Update docstring

    Args:
        tape (QNode or QuantumTape or Callable): A quantum circuit.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[.QuantumTape], function]: The transformed circuit as described in :func:`qml.transform <pennylane.transform>`.

    **Example**

    The transform can be applied on :class:`QNode` directly.

    .. code-block:: python



    """
    G = pennylane_to_networkx(tape)
    in_nodes, out_nodes = [], []
    node_attrs = nx.get_node_attributes(G,"type")
    for node in node_attrs:
        data = node_attrs[node]
        if data == "initial_qubit_state":
            in_nodes.append(node)
        elif data == "intermediate_or_final_qubit_state":
            if G.out_degree(node) == 0: 
                out_nodes.append(node)

    num_wires = len(range(min(tape.wires),max(tape.wires)+1))
    B = np.eye(num_wires)
    for root_node, terminal_node in product(in_nodes, out_nodes):
        B[int(root_node.split("_")[1]) ][int(terminal_node.split("_")[1])] = nx.has_path(G, root_node, terminal_node)
    C = np.ones((num_wires, num_wires), dtype=int) - B.transpose()
    n = C.shape[0]
    R = [[i] for i in range(n)]
    
    if np.all(C == 0):
        # return R  # Irreducible circuit
        return None  # Irreducible circuit

    # TODO: this is supposed to include shots*log(n)
    for _ in range(int(np.log(n))):
        C_prime = np.copy(C)
        R_prime = []
        while np.sum(C_prime) > 0:
            r = np.sum(C_prime, axis=1)
            A_q = {i for i in range(n) if r[i] > 0}

            output_qubit = random.choice(list(A_q))
            Q_p_i, C_prime = best_qpath(C_prime, output_qubit)

            if len(Q_p_i) > 1:
                R_prime.append(Q_p_i)

        R_prime = merge_subsets(R_prime)
        R_prime = finalize_reuse(R_prime, n)

        if len(R_prime) < len(R):
            R = R_prime
    print(R)


    def null_postprocessing(results):
        """A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]  # pragma: no cover

    new_tape = tape.copy(operations=[])

    return [new_tape], null_postprocessing
