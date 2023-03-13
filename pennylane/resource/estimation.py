# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

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
This module contains the functions needed for resource estimation.
"""

from collections import defaultdict
import pennylane as qml
from pennylane import numpy as np
from pennylane.operation import AnyWires, Operation


class Resource:
    r"""Create a resource object for storing quantum resource information."""

    def __init__(self):
        self.num_wires = 0
        self.num_gates = 0
        self.gate_types = defaultdict(int)
        self.depth = 0
        self.shots = 0

    def __str__(self):
        keys = ["wires", "gates", "depth", "shots", "gate_types"]
        vals = [self.num_wires, self.num_gates, self.depth, self.shots, self.gate_types]
        items = "\n".join([str(i) for i in zip(keys, vals)])
        items = items.replace("('", "")
        items = items.replace("',", ":")
        items = items.replace(")", "")
        items = items.replace("defaultdict(<class 'int'>, ", "\n")
        return items

    def __repr__(self):
        return f"<Resource: wires={self.num_wires}, gates={self.num_gates}, depth={self.depth}, shots={self.shots}, gate_types={self.gate_types}>"

    def _ipython_display_(self):
        """Displays __str__ in ipython instead of __repr__"""
        print(str(self))


class CustomOperation(qml.DoubleExcitation):
    num_wires = 4

    def resources(self):
        resource = Resource()
        resource.num_gates = 11
        resource.gate_types = {"T": 5, "CNOT": 6}
        resource.depth = 11
        return resource


class DoubleFactorization(qml.resource.DoubleFactorization):
    def resources(self):
        resource = Resource()
        resource.num_wires = len(self.wires)
        resource.num_gates = self.gates
        resource.gate_types = {"T": resource.num_gates}
        resource.depth = resource.num_gates
        return resource


def estimate_resources_tape(tape):
    r"""Return resource estimation for a quantum tape.

    Args:
        tape (qml.QuantumTape): quantum tape

    Returns:
        qml.resource.Resource: resource information

    .. details::
        :title: Usage Details

        The followng example shows estimating resource for a quantum tape constructed with PennyLane
        gates and a custom gate:

        .. code-block:: python3

            with qml.tape.QuantumTape() as tape:
                qml.SingleExcitation(0.1, wires=[0, 1])
                qml.DoubleExcitation(0.2, wires=[0, 1, 2, 3])
                CustomOperation(wires=range(6))

        >>> resources = estimate_resources_tape(tape)()
        >>> print(resources)
        wires: 6
        gates: 42
        depth: 0
        shots: 0
        gate_types:
        {'CNOT': 22, 'CRY': 1, 'Hadamard': 6, 'RY': 8, 'T': 5}
    """

    def _estimate():
        resource = Resource()
        resource.num_wires = len(tape.wires)
        for op in tape.operations:
            if not hasattr(op, "resources"):
                # gate_types
                decomp = op.decomposition()
                for d in decomp:
                    resource.gate_types[d.name] += 1
                # num_gates
                resource.num_gates += len(decomp)
            if hasattr(op, "resources"):
                op_resource = op.resources()
                # gate_types
                for d in op_resource.gate_types:
                    resource.gate_types[d] += op_resource.gate_types[d]
                # num_gates
                resource.num_gates += sum(op_resource.gate_types.values())

                # user-defined resource - be careful with allowing this
                # for item in vars(op_resource).keys():
                #     resource_value = getattr(op_resource, item)
                #     if not hasattr(resource, item):
                #         setattr(resource, item, resource_value)
        return resource

    return _estimate


def estimate_resources_qnode(qnode):
    r"""Return resource estimation for a qnode.

    Args:
        qnode (qml.QNode): quantum node

    Returns:
        qml.resource.Resource: resource information

    .. details::
        :title: Usage Details

        The followng example shows estimating resource for a VQE workflow:

        .. code-block:: python3

            symbols  = ['Li', 'H']
            geometry = np.array([[0.0, 0.0, -0.69434785],
                                 [0.0, 0.0,  0.69434785]], requires_grad = False)
            electrons = 4
            ham, qubits = qml.qchem.molecular_hamiltonian(symbols, geometry)
            hf_state = qml.qchem.hf_state(electrons, qubits)
            singles, doubles = qml.qchem.excitations(electrons, qubits)
            excitations = singles + doubles
            wires = range(qubits)

            dev = qml.device("default.qubit", wires=qubits)

            @qml.qnode(dev)
            def circuit(params):
                qml.BasisStatePreparation(hf_state, wires=range(qubits))
                for i, excitation in enumerate(excitations):
                    if len(excitation) == 4:
                        qml.DoubleExcitation(params[i], wires=excitation)
                    else:
                        qml.SingleExcitation(params[i], wires=excitation)
                return qml.expval(ham)

            params = np.zeros(len(excitations))

            energy = circuit(params)

        >>> resources = estimate_resources_qnode(circuit)()
        >>> print(resources)
        wires: 12
        gates: 2180
        depth: 0
        shots: None
        gate_types:
        {'PauliX': 4, 'CNOT': 1096, 'CRY': 16, 'Hadamard': 456, 'RY': 608}
    """

    def _estimate():

        tape = qnode.tape
        resource = Resource()
        resource.num_wires = len(tape.wires)
        resource.shots = qnode.device.shots

        for op in tape.operations:

            if not hasattr(op, "resources"):
                # gate_types
                decomp = op.decomposition()
                for d in decomp:
                    resource.gate_types[d.name] += 1
                # num_gates
                resource.num_gates += len(decomp)

            if hasattr(op, "resources"):
                op_resource = op.resources()
                # gate_types
                for d in op_resource.gate_types:
                    resource.gate_types[d] += op_resource.gate_types[d]
                # num_gates
                resource.num_gates += sum(op_resource.gate_types.values())

        return resource

    return _estimate


def estimate_resources_op(op):
    r"""Return resource estimation for a custom operation.

    Args:
        op (Operation): custom operation

    Returns:
        qml.resource.Resource: resource information

    .. details::
        :title: Usage Details

        The followng example shows estimating resource for the DoubleFactorization custom operation:

        .. code-block:: python3

            symbols  = ['O', 'H', 'H']
            geometry = np.array([[0.00000000,  0.00000000,  0.28377432],
                                 [0.00000000,  1.45278171, -1.00662237],
                                 [0.00000000, -1.45278171, -1.00662237]], requires_grad = False)
            mol = qml.qchem.Molecule(symbols, geometry, basis_name='sto-3g')
            core, one, two = qml.qchem.electron_integrals(mol)()
            op = DoubleFactorization(one, two)

        >>> resources = estimate_resources_tape(tape)()
        >>> print(resources)
        wires: 290
        gates: 103969925
        depth: 0
        shots: 0
        gate_types:
        {'T': 103969925}
    """

    def _estimate():
        resource = Resource()
        op_resource = op.resources()
        resource.num_wires = op.resources().num_wires
        # gate_types
        for d in op_resource.gate_types:
            resource.gate_types[d] += op_resource.gate_types[d]
        # num_gates
        resource.num_gates += sum(op_resource.gate_types.values())

        return resource

    return _estimate


def estimate_resources_batch(qnode):
    r"""Return list of resource estimation objects for several batches of a qnode.

    Args:
        qnode (qml.QNode): quantum node

    Returns:
        list[qml.resource.Resource]: list of resource information objects

    .. details::
        :title: Usage Details

        The followng example shows estimating resource for a VQE workflow with grouped Hamiltonian:

        .. code-block:: python3

            symbols  = ['H', 'H']
            geometry = np.array([[0.0, 0.0, -0.69434785],
                                 [0.0, 0.0,  0.69434785]], requires_grad = False)
            electrons = 2
            ham, qubits = qml.qchem.molecular_hamiltonian(symbols, geometry, grouping_type='qwc')
            hf_state = qml.qchem.hf_state(electrons, qubits)

            dev = qml.device('default.qubit', wires = 4, shots = None)
            @qml.qnode(dev)
            def circuit(params):
                qml.BasisState(hf_state, wires = range(qubits))
                qml.SingleExcitation(params[0], wires=[0, 1])
                qml.DoubleExcitation(params[1],wires=[0, 1, 2, 3])
                return qml.expval(ham)
            params = np.array([0.1, 0.2])

        >>> resources = estimate_resources_qnode(circuit)()
        >>> print(resources)
        [<Resource: wires=4, gates=36, depth=0, shots=None, gate_types=defaultdict(<class 'int'>, {'BasisStatePreparation': 1, 'CNOT': 16, 'CRY': 1, 'Hadamard': 6, 'RY': 10, 'RX': 2})>,
         <Resource: wires=4, gates=32, depth=0, shots=None, gate_types=defaultdict(<class 'int'>, {'BasisStatePreparation': 1, 'CNOT': 16, 'CRY': 1, 'Hadamard': 6, 'RY': 8})>,
         <Resource: wires=4, gates=32, depth=0, shots=None, gate_types=defaultdict(<class 'int'>, {'BasisStatePreparation': 1, 'CNOT': 16, 'CRY': 1, 'Hadamard': 6, 'RY': 8})>,
         <Resource: wires=4, gates=32, depth=0, shots=None, gate_types=defaultdict(<class 'int'>, {'BasisStatePreparation': 1, 'CNOT': 16, 'CRY': 1, 'Hadamard': 6, 'RY': 8})>,
         <Resource: wires=4, gates=32, depth=0, shots=None, gate_types=defaultdict(<class 'int'>, {'BasisStatePreparation': 1, 'CNOT': 16, 'CRY': 1, 'Hadamard': 6, 'RY': 8})>]
    """

    def _estimate(*params):

        tape = qml.transforms.make_tape(qnode.func)(*params)

        tapes = qnode.device.batch_transform(tape)[0]

        resource_list = []

        for tape in tapes:

            resource = Resource()
            resource.num_wires = len(tape.wires)
            resource.shots = qnode.device.shots

            for op in tape.operations:

                if not hasattr(op, "resources"):
                    try:
                        decomp = op.decomposition()
                        for d in decomp:
                            # gate_types
                            resource.gate_types[d.name] += 1
                    except:
                        # gate_types
                        resource.gate_types[op.name] += 1

                if hasattr(op, "resources"):
                    op_resource = op.resources()
                    # gate_types
                    for d in op_resource.gate_types:
                        resource.gate_types[d] += op_resource.gate_types[d]

                # num_gates
                resource.num_gates = sum(resource.gate_types.values())

            resource_list.append(resource)

        return resource_list

    return _estimate
