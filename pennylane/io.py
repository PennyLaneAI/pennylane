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
import warnings
from collections import defaultdict
from importlib import metadata
from sys import version_info
import pennylane as qml

# Error message to show when the PennyLane-Qiskit plugin is required but missing.
_MISSING_QISKIT_PLUGIN_MESSAGE = (
    "Conversion from Qiskit requires the PennyLane-Qiskit plugin. "
    "You can install the plugin by running: pip install pennylane-qiskit. "
    "You may need to restart your kernel or environment after installation. "
    "If you have any difficulties, you can reach out on the PennyLane forum at "
    "https://discuss.pennylane.ai/c/pennylane-plugins/pennylane-qiskit/"
)

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

    .. warning::
        ``qml.load`` is deprecated. Instead, please use the functions outlined in the
        :ref:`Importing Circuits <intro_ref_importing_circuits>` quickstart guide, such as ``qml.from_qiskit``.

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
        **load_kwargs: keyword arguments to pass when converting the quantum circuit
            using the plugin. See below for details about supported keyword arguments.

    Keyword Args:
        measurements (list[MeasurementProcess]): the list of PennyLane measurements that
            overrides the terminal measurements that may be present in the imput circuit.
            Currently, only supported for Qiskit's `QuantumCircuit <https://docs.pennylane.ai/projects/qiskit>`_.

    Returns:
        function: the PennyLane template created from the quantum circuit object

    """

    _format = "pyquil" if format == "pyquil_program" else format
    warnings.warn(
        f"qml.load() is deprecated. Instead, please use the more specific qml.from_{_format}()",
        qml.PennyLaneDeprecationWarning,
    )

    if format in plugin_converters:
        # loads the plugin load function
        plugin_converter = plugin_converters[format].load()
        # calls the load function of the converter on the quantum circuit object
        return plugin_converter(quantum_circuit_object, **load_kwargs)

    raise ValueError(
        "Converter does not exist. Make sure the required plugin is installed "
        "and supports conversion."
    )


def from_qiskit(quantum_circuit, measurements=None):
    r"""Converts a Qiskit `QuantumCircuit <https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.QuantumCircuit>`_
    into a PennyLane :ref:`quantum function <intro_vcirc_qfunc>`.

    .. note::

        This function depends upon the PennyLane-Qiskit plugin. Follow the
        `installation instructions <https://docs.pennylane.ai/projects/qiskit/en/latest/installation.html>`__
        to get up and running. You may need to restart your kernel if you are running in a notebook
        environment.

    Args:
        quantum_circuit (qiskit.QuantumCircuit): a quantum circuit created in Qiskit
        measurements (None | MeasurementProcess | list[MeasurementProcess]): an optional PennyLane
            measurement or list of PennyLane measurements that overrides any terminal measurements
            that may be present in the input circuit

    Returns:
        function: The PennyLane quantum function, created based on the input Qiskit
        ``QuantumCircuit`` object.

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
    which wires it should operate on. If no wires are passed, it will default
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
    quantum function, making it possible to create a PennyLane circuit with
    :class:`qml.QNode <pennylane.QNode>`:

    >>> measurements = [qml.expval(qml.Z(0)), qml.var(qml.Z(1))]
    >>> circuit = qml.QNode(qml.from_qiskit(qc, measurements), dev)
    >>> circuit()
    (tensor(0.70738827, requires_grad=True),
    tensor(0.99999937, requires_grad=True))

    .. note::

        The ``measurements`` keyword allows one to add a list of PennyLane measurements
        that will **override** any terminal measurements present in the ``QuantumCircuit``,
        so that they are not performed before the operations specified in ``measurements``.
        ``measurements=None``.

    If an existing ``QuantumCircuit`` already contains measurements, ``from_qiskit``
    will return those measurements, provided that they are not overriden as shown above.
    These measurements can be used, e.g., for conditioning with
    :func:`qml.cond() <~.cond>`, or simply included directly within the QNode's return:

    .. code-block:: python

       qc = QuantumCircuit(2, 2)
       qc.rx(np.pi, 0)
       qc.measure_all()

       @qml.qnode(dev)
       def circuit():
           # Since measurements=None, the measurements present in the QuantumCircuit are returned.
           measurements = qml.from_qiskit(qc)()
           return [qml.expval(m) for m in measurements]

    >>> circuit()
    [tensor(1., requires_grad=True), tensor(0., requires_grad=True)]

    .. note::

        The ``measurements`` returned from a ``QuantumCircuit`` are in the computational basis
        with 0 corresponding to :math:`|0\rangle` and 1 corresponding to :math:`|1 \rangle`. This
        corresponds to the :math:`|1 \rangle \langle 1|` observable rather than the :math:`Z` Pauli
        operator.

    See below for more information regarding how to translate more complex circuits from Qiskit to
    PennyLane, including handling parameterized Qiskit circuits, mid-circuit measurements, and
    classical control flows.

    .. details::
        :title: Parameterized Quantum Circuits

        A Qiskit ``QuantumCircuit`` is parameterized if it contains
        `Parameter <https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.Parameter>`__ or
        `ParameterVector <https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.ParameterVector>`__
        references that need to be given defined values to evaluate the circuit. These can be passed
        to the generated quantum function as keyword or positional arguments. If we define a
        parameterized circuit:

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

            qfunc = qml.from_qiskit(qc, measurements=qml.expval(qml.Z(0)))
            circuit = qml.QNode(qfunc, dev)

        Now, ``circuit`` has a signature of ``(x, y)``. The parameters are ordered alphabetically.

        >>> x = np.pi / 4
        >>> y = 0
        >>> circuit(x, y)
        tensor(0.70710678, requires_grad=True)

        >>> qml.grad(circuit, argnum=[0, 1])(np.pi/4, np.pi/6)
        (array(-0.61237244), array(-0.35355339))

        The ``QuantumCircuit`` may also be parameterized with a ``ParameterVector``. These can be
        similarly converted:

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


    .. details::
        :title: Measurements and Classical Control Flows

        When ``measurement=None``, all of the measurements performed in the ``QuantumCircuit`` will
        be returned by the quantum function in the form of a :ref:`mid-circuit measurement
        <mid_circuit_measurements>`. For example, if we define a ``QuantumCircuit`` with
        measurements:

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
        additional operations conditional on the results. We can also calculate standard mid-circuit
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

        .. note::

            The order of mid-circuit measurements returned by `qml.from_qiskit()` in the example
            above is determined by the order in which measurements appear in the input Qiskit
            ``QuantumCircuit``.

        Furthermore, the Qiskit `IfElseOp <https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.IfElseOp>`__,
        `SwitchCaseOp <https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.SwitchCaseOp>`__ and
        `c_if <https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.Instruction#c_if>`__
        conditional workflows are automatically translated into their PennyLane counterparts during
        conversion. For example, if we construct a ``QuantumCircuit`` with these workflows:

        .. code-block:: python

            qc = QuantumCircuit(4, 1)
            qc.h(0)
            qc.measure(0, 0)

            # Use an `IfElseOp` operation.
            noop = QuantumCircuit(1)
            flip_x = QuantumCircuit(1)
            flip_x.x(0)
            qc.if_else((qc.clbits[0], True), flip_x, noop, [1], [])

            # Use a `SwitchCaseOp` operation.
            with qc.switch(qc.clbits[0]) as case:
                with case(0):
                    qc.y(2)

            # Use the `c_if()` function.
            qc.z(3).c_if(qc.clbits[0], True)

            qc.measure_all()

        We can convert the ``QuantumCircuit`` into a PennyLane quantum function using:

        .. code-block:: python

            dev = qml.device("default.qubit")

            measurements = [qml.expval(qml.Z(i)) for i in range(qc.num_qubits)]
            cond_circuit = qml.QNode(qml.from_qiskit(qc, measurements=measurements), dev)

        The result is:

        >>> print(qml.draw(cond_circuit)())
        0: ──H──┤↗├──────────╭||─┤  <Z>
        1: ──────║───X───────├||─┤  <Z>
        2: ──────║───║──Y────├||─┤  <Z>
        3: ──────║───║──║──Z─╰||─┤  <Z>
                 ╚═══╩══╩══╝
    """
    try:
        plugin_converter = plugin_converters["qiskit"].load()
        return plugin_converter(quantum_circuit, measurements=measurements)
    except KeyError as e:
        raise RuntimeError(_MISSING_QISKIT_PLUGIN_MESSAGE) from e


def from_qiskit_op(qiskit_op, params=None, wires=None):
    """Converts a Qiskit `SparsePauliOp <https://docs.quantum.ibm.com/api/qiskit/qiskit.quantum_info.SparsePauliOp>`__
    into a PennyLane :class:`Operator <pennylane.operation.Operator>`.

    .. note::

        This function depends upon the PennyLane-Qiskit plugin. Follow the
        `installation instructions <https://docs.pennylane.ai/projects/qiskit/en/latest/installation.html>`__
        to get up and running. You may need to restart your kernel if you are running in a notebook
        environment.

    Args:
        qiskit_op (qiskit.quantum_info.SparsePauliOp): a ``SparsePauliOp`` created in Qiskit
        params (Any): optional assignment of coefficient values for the ``SparsePauliOp``; see the
            `Qiskit documentation <https://docs.quantum.ibm.com/api/qiskit/qiskit.quantum_info.SparsePauliOp#assign_parameters>`_
            to learn more about the expected format of these parameters
        wires (Sequence | None): optional assignment of wires for the converted ``SparsePauliOp``;
            if the original ``SparsePauliOp`` acted on :math:`N` qubits, then this must be a
            sequence of length :math:`N`

    Returns:
        Operator: The PennyLane operator, created based on the input Qiskit
        ``SparsePauliOp`` object.

    .. note::

        The wire ordering convention differs between PennyLane and Qiskit: PennyLane wires are
        enumerated from left to right, while the Qiskit convention is to enumerate from right to
        left. This means a ``SparsePauliOp`` term defined by the string ``"XYZ"`` applies ``Z`` on
        wire 0, ``Y`` on wire 1, and ``X`` on wire 2. For more details, see the
        `String representation <https://docs.quantum.ibm.com/api/qiskit/qiskit.quantum_info.Pauli>`_
        section of the Qiskit documentation for the ``Pauli`` class.

    **Example**

    Consider the following script which creates a Qiskit ``SparsePauliOp``:

    .. code-block:: python

        from qiskit.quantum_info import SparsePauliOp

        qiskit_op = SparsePauliOp(["II", "XY"])

    The ``SparsePauliOp`` contains two terms and acts over two qubits:

    >>> qiskit_op
    SparsePauliOp(['II', 'XY'],
                  coeffs=[1.+0.j, 1.+0.j])

    To convert the ``SparsePauliOp`` into a PennyLane :class:`pennylane.operation.Operator`, use:

    >>> import pennylane as qml
    >>> qml.from_qiskit_op(qiskit_op)
    I(0) + X(1) @ Y(0)

    .. details::
        :title: Usage Details

        You can convert a parameterized ``SparsePauliOp`` into a PennyLane operator by assigning
        literal values to each coefficient parameter. For example, the script

        .. code-block:: python

            import numpy as np
            from qiskit.circuit import Parameter

            a, b, c = [Parameter(var) for var in "abc"]
            param_qiskit_op = SparsePauliOp(["II", "XZ", "YX"], coeffs=np.array([a, b, c]))

        defines a ``SparsePauliOp`` with three coefficients (parameters):

        >>> param_qiskit_op
        SparsePauliOp(['II', 'XZ', 'YX'],
              coeffs=[ParameterExpression(1.0*a), ParameterExpression(1.0*b),
         ParameterExpression(1.0*c)])

        The ``SparsePauliOp`` can be converted into a PennyLane operator by calling the conversion
        function and specifying the value of each parameter using the ``params`` argument:

        >>> qml.from_qiskit_op(param_qiskit_op, params={a: 2, b: 3, c: 4})
        (
            (2+0j) * I(0)
          + (3+0j) * (X(1) @ Z(0))
          + (4+0j) * (Y(1) @ X(0))
        )

        Similarly, a custom wire mapping can be applied to a ``SparsePauliOp`` as follows:

        >>> wired_qiskit_op = SparsePauliOp("XYZ")
        >>> wired_qiskit_op
        SparsePauliOp(['XYZ'],
              coeffs=[1.+0.j])
        >>> qml.from_qiskit_op(wired_qiskit_op, wires=[3, 5, 7])
        Y(5) @ Z(3) @ X(7)
    """
    try:
        plugin_converter = plugin_converters["qiskit_op"].load()
        return plugin_converter(qiskit_op, params=params, wires=wires)
    except KeyError as e:
        raise RuntimeError(_MISSING_QISKIT_PLUGIN_MESSAGE) from e


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

    You can also load the contents of a QASM file:

    .. code-block:: python

        >>> with open("hadamard_circuit.qasm", "r") as f:
        ...     my_circuit = qml.from_qasm(f.read())

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
    plugin_converter = plugin_converters["qasm"].load()
    return plugin_converter(quantum_circuit)


def from_qasm_file(qasm_filename: str):
    """Loads quantum circuits from a QASM file using the converter in the
    PennyLane-Qiskit plugin.

    **Example:**

    >>> my_circuit = qml.from_qasm_file("hadamard_circuit.qasm")

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

    .. warning::
        qml.from_qasm_file is deprecated and will be removed in a future release.
        Please use qml.from_qasm instead.

    """
    warnings.warn(
        "qml.from_qasm_file is deprecated and will be removed in a future release. "
        "Please use qml.from_qasm instead.",
        qml.PennyLaneDeprecationWarning,
    )
    plugin_converter = plugin_converters["qasm_file"].load()
    return plugin_converter(qasm_filename)


def from_pyquil(pyquil_program):
    """Loads pyQuil Program objects by using the converter in the
    PennyLane-Rigetti plugin.

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
    plugin_converter = plugin_converters["pyquil_program"].load()
    return plugin_converter(pyquil_program)


def from_quil(quil: str):
    """Loads quantum circuits from a Quil string using the converter in the
    PennyLane-Rigetti plugin.

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
    plugin_converter = plugin_converters["quil"].load()
    return plugin_converter(quil)


def from_quil_file(quil_filename: str):
    """Loads quantum circuits from a Quil file using the converter in the
    PennyLane-Rigetti plugin.

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
    plugin_converter = plugin_converters["quil_file"].load()
    return plugin_converter(quil_filename)
