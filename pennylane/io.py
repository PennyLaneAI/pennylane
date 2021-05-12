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
This module contains functions to load circuits from other frameworks as
PennyLane templates.
"""
from pkg_resources import iter_entry_points

# get list of installed plugin converters
plugin_converters = {entry.name: entry for entry in iter_entry_points("pennylane.io")}


def load(quantum_circuit_object, format: str):
    """Load external quantum assembly and quantum circuits from supported frameworks
    into PennyLane templates.

    .. note::

        For more details on which formats are supported
        please consult the corresponding plugin documentation:
        https://pennylane.ai/plugins.html

    **Example:**

    >>> qc = qiskit.QuantumCircuit(2)
    >>> qc.rz(0.543, [0])
    >>> qc.cx(0, 1)
    >>> my_circuit = qml.load(qc, format='qiskit')

    The ``my_circuit`` template can now be used within QNodes, as a
    two-wire quantum template.

    >>> @qml.qnode(dev)
    >>> def circuit(x):
    >>>     qml.RX(x, wires=1)
    >>>     my_circuit(wires=(1, 0))
    >>>     return qml.expval(qml.PauliZ(0))

    Args:
        quantum_circuit_object: the quantum circuit that will be converted
            to a PennyLane template
        format (str): the format of the quantum circuit object to convert from

    Returns:
        function: the PennyLane template created from the quantum circuit
        object
    """

    if format in plugin_converters:

        # loads the plugin load function
        plugin_converter = plugin_converters[format].load()

        # calls the load function of the converter on the quantum circuit object
        return plugin_converter(quantum_circuit_object)

    raise ValueError(
        "Converter does not exist. Make sure the required plugin is installed "
        "and supports conversion."
    )


def from_qiskit(quantum_circuit):
    """Loads Qiskit QuantumCircuit objects by using the converter in the
    PennyLane-Qiskit plugin.

    **Example:**

    >>> qc = qiskit.QuantumCircuit(2)
    >>> qc.rz(0.543, [0])
    >>> qc.cx(0, 1)
    >>> my_circuit = qml.from_qiskit(qc)

    The ``my_circuit`` template can now be used within QNodes, as a
    two-wire quantum template.

    >>> @qml.qnode(dev)
    >>> def circuit(x):
    >>>     qml.RX(x, wires=1)
    >>>     my_circuit(wires=(1, 0))
    >>>     return qml.expval(qml.PauliZ(0))

    Args:
        quantum_circuit (qiskit.QuantumCircuit): a quantum circuit created in qiskit

    Returns:
        function: the PennyLane template created based on the QuantumCircuit
        object
    """
    return load(quantum_circuit, format="qiskit")


def from_qasm(quantum_circuit: str):
    """Loads quantum circuits from a QASM string using the converter in the
    PennyLane-Qiskit plugin.

    **Example:**

    .. code-block:: python

        >>> hadamard_qasm = 'OPENQASM 2.0;' \\
        ...                 'include "qelib1.inc";' \\
        ...                 'qreg q[1];' \\
        ...                 'h q[0];'
        >>> my_circuit = qml.from_qasm(hadamard_qasm)

    The ``my_circuit`` template can now be used within QNodes, as a
    two-wire quantum template.

    >>> @qml.qnode(dev)
    >>> def circuit(x):
    >>>     qml.RX(x, wires=1)
    >>>     my_circuit(wires=(1, 0))
    >>>     return qml.expval(qml.PauliZ(0))

    Args:
        quantum_circuit (str): a QASM string containing a valid quantum circuit

    Returns:
        function: the PennyLane template created based on the QASM string
    """
    return load(quantum_circuit, format="qasm")


def from_qasm_file(qasm_filename: str):
    """Loads quantum circuits from a QASM file using the converter in the
    PennyLane-Qiskit plugin.

    **Example:**

    >>> my_circuit = qml.from_qasm("hadamard_circuit.qasm")

    The ``my_circuit`` template can now be used within QNodes, as a
    two-wire quantum template.

    >>> @qml.qnode(dev)
    >>> def circuit(x):
    >>>     qml.RX(x, wires=1)
    >>>     my_circuit(wires=(1, 0))
    >>>     return qml.expval(qml.PauliZ(0))

    Args:
        qasm_filename (str): path to a QASM file containing a valid quantum circuit

    Returns:
        function: the PennyLane template created based on the QASM file
    """
    return load(qasm_filename, format="qasm_file")


def from_pyquil(pyquil_program):
    """Loads pyQuil Program objects by using the converter in the
    PennyLane-Forest plugin.

    **Example:**

    >>> program = pyquil.Program()
    >>> program += pyquil.gates.H(0)
    >>> program += pyquil.gates.CNOT(0, 1)
    >>> my_circuit = qml.from_pyquil(program)

    The ``my_circuit`` template can now be used within QNodes, as a
    two-wire quantum template.

    >>> @qml.qnode(dev)
    >>> def circuit(x):
    >>>     qml.RX(x, wires=1)
    >>>     my_circuit(wires=[1, 0])
    >>>     return qml.expval(qml.PauliZ(0))

    Args:
        pyquil_program (pyquil.Program): a program created in pyQuil

    Returns:
        pennylane_forest.ProgramLoader: a ``pennylane_forest.ProgramLoader`` instance that can
        be used like a PennyLane template and that contains additional inspection properties
    """
    return load(pyquil_program, format="pyquil_program")


def from_quil(quil: str):
    """Loads quantum circuits from a Quil string using the converter in the
    PennyLane-Forest plugin.

    **Example:**

    .. code-block:: python

        >>> quil_str = 'H 0\\n'
        ...            'CNOT 0 1'
        >>> my_circuit = qml.from_quil(quil_str)

    The ``my_circuit`` template can now be used within QNodes, as a
    two-wire quantum template.

    >>> @qml.qnode(dev)
    >>> def circuit(x):
    >>>     qml.RX(x, wires=1)
    >>>     my_circuit(wires=(1, 0))
    >>>     return qml.expval(qml.PauliZ(0))

    Args:
        quil (str): a Quil string containing a valid quantum circuit

    Returns:
        pennylane_forest.ProgramLoader: a ``pennylane_forest.ProgramLoader`` instance that can
        be used like a PennyLane template and that contains additional inspection properties
    """
    return load(quil, format="quil")


def from_quil_file(quil_filename: str):
    """Loads quantum circuits from a Quil file using the converter in the
    PennyLane-Forest plugin.

    **Example:**

    >>> my_circuit = qml.from_quil_file("teleportation.quil")

    The ``my_circuit`` template can now be used within QNodes, as a
    two-wire quantum template.

    >>> @qml.qnode(dev)
    >>> def circuit(x):
    >>>     qml.RX(x, wires=1)
    >>>     my_circuit(wires=(1, 0))
    >>>     return qml.expval(qml.PauliZ(0))

    Args:
        quil_filename (str): path to a Quil file containing a valid quantum circuit

    Returns:
        pennylane_forest.ProgramLoader: a ``pennylane_forest.ProgramLoader`` instance that can
        be used like a PennyLane template and that contains additional inspection properties
    """
    return load(quil_filename, format="quil_file")
