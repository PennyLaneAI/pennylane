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
from collections import defaultdict
from importlib import metadata
from sys import version_info

# get list of installed plugin converters
__plugin_devices = (
    defaultdict(tuple, metadata.entry_points())["pennylane.io"]
    if version_info[:2] == (3, 9)
    else metadata.entry_points(group="pennylane.io")  # pylint:disable=unexpected-keyword-arg
)
plugin_converters = {entry.name: entry for entry in __plugin_devices}


def load(quantum_circuit_object, format: str, **load_kwargs):
    r"""Load external quantum assembly and quantum circuits from supported frameworks
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
    >>>     return qml.expval(qml.Z(0))

    Args:
        quantum_circuit_object: the quantum circuit that will be converted
            to a PennyLane template
        format (str): the format of the quantum circuit object to convert from
        **load_kwargs: keyword argument to pass when converting the quantum circuit
            using the plugin

    Keyword Args:
        measurements (list[MeasurementProcess]): the list of PennyLane measurements that
            overrides the terminal measurements that may be present in the imput circuit.
            Currently, only supported for Qiskit's `QuantumCircuit <https://docs.pennylane.ai/projects/qiskit>`_.

    Returns:
        function: the PennyLane template created from the quantum circuit
        object
    """

    if format in plugin_converters:
        # loads the plugin load function
        plugin_converter = plugin_converters[format].load()

        # calls the load function of the converter on the quantum circuit object
        return plugin_converter(quantum_circuit_object, **(load_kwargs or {}))

    raise ValueError(
        "Converter does not exist. Make sure the required plugin is installed "
        "and supports conversion."
    )


def from_qiskit(quantum_circuit, measurements=None):
    """Loads Qiskit QuantumCircuit objects by using the converter in the
    PennyLane-Qiskit plugin.

    The loaded object is a PennyLane quantum function that can be passed
    to a qnode to create a full PennyLane circuit.

    Args:
        quantum_circuit (qiskit.QuantumCircuit): a quantum circuit created in qiskit
        measurements (list[MeasurementProcess]): the list of PennyLane measurements that
            overrides the terminal measurements that may be present in the input circuit.

    Returns:
        function: the PennyLane template created based on the QuantumCircuit object

    **Example:**

    .. code-block:: python

        import pennylane as qml
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(2, 2)
        qc.rx(0.785, 0)
        qc.ry(1.57, 1)

        my_qfunc = qml.from_qiskit(qc)

    The ``my_qfunc`` function can now be used within QNodes, as a two-wire quantum
    template. We can also pass ``wires`` when calling the returned template to define
    which device wires it should operate on. If no wires are passed, it will default
    to sequential wire labels starting at 0.

    .. code-block:: python

        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit():
            my_qfunc(wires=["a", "b"])
            return qml.expval(qml.Z("a")), qml.var(qml.Z("b"))

    >>> circuit()
    (tensor(0.70738827, requires_grad=True),
    tensor(0.99999937, requires_grad=True))

    The measurements can also be passed directly to the function when creating the
    qfunc, making it possible to create a PennyLane circuit with ``qml.QNode`:

    >>> measurements = [qml.expval(qml.Z(0)), qml.var(qml.Z(1))]
    >>> circuit = qml.QNode(qml.from_qiskit(qc, measurements), dev)
    >>> circuit()
    (tensor(0.70738827, requires_grad=True),
    tensor(0.99999937, requires_grad=True))

    ..note::
        The ``measurement`` keyword allows one to add a list of PennyLane measurements
        that will **override** the terminal measurements present in the ``QuantumCircuit``,
        so that they are not performed before the operations specified in ``measurements``.
        Converting a QuantumCircuit with ``measurements`` set will create a quanutm function
        that does not return final or mid-circuit measurement values. See Usage Details below
        for more information on how measurements defined on the QuantumCircuit are returned if
        ``measurements=None``.

    If an existing ``QuantumCircuit`` already ends with final measurements, the QNode can return
    the expectation values of those final measurements directly:

     .. code-block:: python

        qc = QuantumCircuit(2, 2)
        qc.rx(np.pi, 0)
        qc.measure_all()

        @qml.qnode(dev)
        def circuit():
            # here measurements=None, so the measurements present in the QuantumCircuit are returned
            measurements = qml.from_qiskit(qc)()
            return [qml.expval(m) for m in measurements]

    >>> circuit()
    [tensor(1., requires_grad=True), tensor(0., requires_grad=True)]

    ..note::

        The returned measurements from the ``QuantumCircuit`` are in the measurement basis
        and are readout values between 0 (ground) and 1 (excited). Such a measurement on the
        `ith` wire does not match the result of `qml.expval(qml.PauliZ(i)`, which ranges from
        1 (ground) to -1 (excited), indicating a position on the Bloch sphere.

        I.e. we can alternatively translate the circuit and capture the same information in
        a different format as:

        >>> measurements = [qml.expval(qml.Z(0)), qml.expval(qml.Z(1))]
        >>> circuit = qml.QNode(qml.from_qiskit(qc, measurements), dev)
        >>> circuit()
        [tensor(-1., requires_grad=True), tensor(1., requires_grad=True)]

    See Usage Details below for more information regarding how to translate more complex
    circuits from Qiskit to PennyLane, including handling parameterized Qiskit circuits,
    mid-circuit measurements, and classical control flows.

    .. details::
        :title: Usage Details: Parameterized QuantumCircuits

        A Qiskit ``QuantumCircuit`` is parameterized if it contains ``Parameter`` or
        ``ParameterVector`` references that need to be given defined values to evaluate
        the circuit. These can be passed to the generated qfunc as keyword or positional
        arguments. If we define a parameterized

        .. code-block:: python

            from qiskit.circuit import QuantumCircuit, Parameter

            angle0 = Parameter("x")
            angle1 = Parameter("y")

            qc = QuantumCircuit(2, 2)
            qc.rx(angle0, 0)
            qc.ry(angle1, 1)
            qc.cx(1, 0)

        Then this circuit can be converted into a differentiable circuit in PennyLane and
        executed:

        .. code-block:: python

            import pennylane as qml
            from pennylane import numpy as np

            dev = qml.device("default.qubit")

            @qml.qnode(dev)
            def circuit(a0, a1):
                # parameters are passed from the circuit function into the template as keyword arguments
                # the keywords match the names given to the parameters in the code block above
                qml.from_qiskit(qc)(x=a0, y=a1)
                return qml.expval(qml.Z(0))

        ..note::

            The parameters can also be passed to the qfunc as positional arguments. In this case, the positions
            match those from ``qc.parameters``. Note that these are in alphabetical order by name, rather than
            in the order they are used in the circuit.

        >>> circuit(np.pi/4, 0.)
        tensor(0.70710678, requires_grad=True)

        >>> qml.grad(circuit, argnum=[0, 1])(np.pi/4, np.pi/6)
        (array(-0.61237244), array(-0.35355339))

        The ``QuantumCircuit`` may also be parameterized with a ``ParameterVector``. These can be similarly
        converted:

        .. code-block:: python

            from qiskit.circuit import ParameterVector

            angles = ParameterVector("angles", 2)

            qc = QuantumCircuit(2, 2)
            qc.rx(angles[0], 0)
            qc.ry(angles[1], 1)
            qc.cx(1, 0)

            @qml.qnode(dev)
            def circuit(angles):
                qml.from_qiskit(qc)(angles)
                return qml.expval(qml.Z(0))

        >>> angles = [3.1, 0.45]
        >>> circuit(angles)
        tensor(-0.89966835, requires_grad=True)

        #TODO: either note here that this doesn't work with gradients, or get the gradient working and add qml.grad(circuit, argnum=0)([np.pi/4, np.pi/6])

    .. details::
        :title: Usage Details: Mid-Circuit Measurements and Classical Control Flows

        Mid-circuit measurements in the ``QuantumCircuit`` will be translated into mid-circuit
        measurements in PennyLane and executed as specified. Some classical workflows in the
        QuantumCircuit can also be translated to PennyLane.

        When ``measurement=None``, all of the measurements performed in the ``QuantumCircuit`` will be included
        in the template as mid-circuit measurements, and returned for further use. For example, if we define a
        ``QuantumCircuit`` with measurements:

        .. code-block:: python

            import pennylane as qml
            from qiskit import QuantumCircuit

            qc = QuantumCircuit(2, 2)
            qc.h(0)
            qc.measure(0, 0)
            qc.rz(0.24, [0])
            qc.cx(0, 1)
            qc.measure_all()

        Then we can create a PennyLane circuit that uses this as a sub-circuit, and performs
        additional operations conditional on the results. We can also do the standard mid-circuit
        measurement statistics, like expectation value, on the returned measurements:

        .. code-block:: python

            @qml.qnode(qml.device("default.qubit"))
            def circuit():
                # apply the QuantumCircuit and retrieve the measurements
                mid_measure0, m0, m1 = qml.from_qiskit(qc)()

                # conditionally apply an additional operation based on the results
                qml.cond(mid_measure0==0, qml.RX)(np.pi/2, 0)

                # return the expectation value of one of the mid-circuit measurements, and a terminal measurement
                return qml.expval(mid_measure0), qml.expval(m1)

        >>> circuit()
        (tensor(0.5, requires_grad=True), tensor(0.5, requires_grad=True))

        The ``IfElseOp``, ``SwitchCaseOp`` and ``c_if`` classical workflows using mid-circuit
        measurements can be translated from a Qiskit ``QuantumCircuit``


        >>> print(qml.draw(circuit_loaded_qiskit_circuit)())
        0: ──H──┤↗├──RZ(0.24)─╭●─┤  <Z>
        1: ───────────────────╰X─┤  vnentropy

    """
    try:
        return load(quantum_circuit, format="qiskit", measurements=measurements)
    except ValueError as e:
        if e.args[0].split(".")[0] == "Converter does not exist":
            raise RuntimeError(
                "Conversion from Qiskit requires the PennyLane-Qiskit plugin. "
                "You can install the plugin by running: pip install pennylane-qiskit. "
                "You may need to restart your kernel or environment after installation. "
                "If you have any difficulties, you can reach out on the PennyLane forum at "
                "https://discuss.pennylane.ai/c/pennylane-plugins/pennylane-qiskit/"
            ) from e
        raise e


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
    >>>     return qml.expval(qml.Z(0))

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
    >>>     return qml.expval(qml.Z(0))

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
    >>>     return qml.expval(qml.Z(0))

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
    >>>     return qml.expval(qml.Z(0))

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
    >>>     return qml.expval(qml.Z(0))

    Args:
        quil_filename (str): path to a Quil file containing a valid quantum circuit

    Returns:
        pennylane_forest.ProgramLoader: a ``pennylane_forest.ProgramLoader`` instance that can
        be used like a PennyLane template and that contains additional inspection properties
    """
    return load(quil_filename, format="quil_file")
