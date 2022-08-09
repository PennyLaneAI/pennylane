.. role:: html(raw)
   :format: html

.. _intro_ref_meas:

Measurements
============

.. currentmodule:: pennylane.measure

PennyLane can extract different types of measurement results from quantum
devices: the expectation of an observable, its variance,
samples of a single measurement, or computational basis state probabilities.

For example, the following circuit returns the expectation value of the
:class:`~pennylane.PauliZ` observable on wire 1:

.. code-block:: python

    def my_quantum_function(x, y):
        qml.RZ(x, wires=0)
        qml.CNOT(wires=[0, 1])
        qml.RY(y, wires=1)
        return qml.expval(qml.PauliZ(1))

The available measurement functions are

:html:`<div class="summary-table">`

.. autosummary::

    ~pennylane.expval
    ~pennylane.sample
    ~pennylane.var
    ~pennylane.probs
    ~pennylane.state
    ~pennylane.density_matrix
    ~pennylane.vn_entropy
    ~pennylane.mutual_info

:html:`</div>`

.. note::

    All measurement functions support analytic differentiation, with the
    exception of :func:`~.pennylane.sample`, as it returns *stochastic*
    results.

Combined measurements
---------------------

Quantum functions can also return combined measurements of multiple observables, as long as each wire
is not measured more than once:

.. code-block:: python

    def my_quantum_function(x, y):
        qml.RZ(x, wires=0)
        qml.CNOT(wires=[0, 1])
        qml.RY(y, wires=1)
        return qml.expval(qml.PauliZ(1)), qml.var(qml.PauliX(0))

You can also use list comprehensions, and other common Python patterns:

.. code-block:: python

    def my_quantum_function(x, y):
        qml.RZ(x, wires=0)
        qml.CNOT(wires=[0, 1])
        qml.RY(y, wires=1)
        return [qml.expval(qml.PauliZ(i)) for i in range(2)]

As a full example of combined measurements, let us look at
a Bell state :math:`(|00\rangle + |11\rangle)/\sqrt{2}`, prepared
by a ``Hadamard`` and ``CNOT`` gate.

.. code-block:: python

    import pennylane as qml
    from pennylane import numpy as np

    dev = qml.device("default.qubit", wires=2, shots=1000)

    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.sample(qml.PauliZ(0)), qml.sample(qml.PauliZ(1))

The combined PauliZ-measurement of the
first and second qubit returns a list of two lists, each containing
the measurement results of the respective qubit. As a default, :func:`~.pennylane.sample`
returns 1000 samples per observable.

>>> result = circuit()
>>> result.shape
(2, 1000)

Since the two qubits are maximally entangled,
the measurement results always coincide, and the lists are therefore equal:

>>> np.all(result[0] == result[1])
True

Tensor observables
------------------

PennyLane supports measuring the tensor product of observables, by using
the ``@`` notation. For example, to measure the expectation value of
:math:`Z\otimes I \otimes X`:

.. code-block:: python3

    def my_quantum_function(x, y):
        qml.RZ(x, wires=0)
        qml.CNOT(wires=[0, 1])
        qml.RY(y, wires=1)
        qml.CNOT(wires=[0, 2])
        return qml.expval(qml.PauliZ(0) @ qml.PauliX(2))

Note that we don't need to declare the identity observable on wire 1; this is
implicitly assumed.

The tensor observable notation can be used inside all measurement functions that
accept observables as arguments,
including :func:`~.pennylane.expval`, :func:`~.pennylane.var`,
and :func:`~.pennylane.sample`.

Counts
------

To avoid dealing with long arrays for the larger numbers of shots, one can use :func:`~pennylane.counts` rather than
:func:`~pennylane.sample`. This performs the same measurement as sampling, but returns a dictionary containing the 
possible measurement outcomes and the number of occurrences for each, rather than a list of all outcomes. 

The previous example will be modified as follows:

.. code-block:: python

    dev = qml.device("default.qubit", wires=2, shots=1000)

    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.counts(qml.PauliZ(0)), qml.counts(qml.PauliZ(1))

After executing the circuit, we can directly see how many times each measurement outcome occurred:
        
>>> result = circuit()
>>> print(result)
({1: 475, -1: 525}, {1: 475, -1: 525})
 
Similarly, if the observable is not provided, the count of the observed computational basis state is returned.

.. code-block:: python

    dev = qml.device("default.qubit", wires=2, shots=1000)

    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.counts()

And the result is:
           
>>> result = circuit()
>>> print(result)
{'00': 495, '11': 505}

Per default, only observed outcomes are included in the dictionary. The kwarg ``all_outcomes=True`` can 
be used to display all possible outcomes, including those that were observed ``0`` times in sampling.

For example, we could run the previous circuit with ``all_outcomes=True``:

.. code-block:: python

dev = qml.device("default.qubit", wires=2, shots=1000)

@qml.qnode(dev)
def circuit():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.counts(all_outcomes=True)

>>> result = circuit()
>>> print(result)
{'00': 518, '01': 0, '10': 0, '11': 482}

Note: For complicated Hamiltonians, this can add considerable overhead time (due to the cost of calculating 
eigenvalues to determine possible outcomes), and as the number of qubits increases, the length of the output 
dictionary showing possible computational basis states grows rapidly. 



If counts are obtained along with a measurement function other than :func:`~.pennylane.sample`,
a tensor of tensors is returned to provide differentiability for the outputs of QNodes.

.. code-block:: python

    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0,1])
        qml.PauliX(wires=1)
        return qml.expval(qml.PauliZ(0)),qml.expval(qml.PauliZ(1)), qml.counts()

>>> result = circuit()
>>> print(result)
[tensor(0.026, requires_grad=True) tensor(0.026, requires_grad=True)
 tensor({'001': 513, '111': 487}, dtype=object, requires_grad=True)]

Probability
-----------

You can also train QNodes on computational basis probabilities, by using
the :func:`~.pennylane.probs` measurement function. The function can
accept either specified ``wires`` or an observable that rotates the
computational basis.

.. code-block:: python3

    def my_quantum_function(x, y):
        qml.RZ(x, wires=0)
        qml.CNOT(wires=[0, 1])
        qml.RY(y, wires=1)
        qml.CNOT(wires=[0, 2])
        return qml.probs(wires=[0, 1])

For example:

>>> dev = qml.device("default.qubit", wires=3)
>>> qnode = qml.QNode(my_quantum_function, dev)
>>> qnode(0.56, 0.1)
array([0.99750208, 0.00249792, 0.        , 0.        ])

The returned probability array uses lexicographical ordering,
so corresponds to a :math:`99.75\%` probability of measuring
state :math:`|00\rangle`, and a :math:`0.25\%` probability of
measuring state :math:`|01\rangle`.

Mid-circuit measurements and conditional operations
---------------------------------------------------

PennyLane allows specifying measurements in the middle of the circuit.
Quantum functions such as operations can then be conditioned on the measurement
outcome of such mid-circuit measurements:

.. code-block:: python

    def my_quantum_function(x, y):
        qml.RY(x, wires=0)
        qml.CNOT(wires=[0, 1])
        m_0 = qml.measure(1)

        qml.cond(m_0, qml.RY)(y, wires=0)
        return qml.probs(wires=[0])

A quantum function with mid-circuit measurements (defined using
:func:`~.pennylane.measure`) and conditional operations (defined using
:func:`~.pennylane.cond`) can be executed by applying the `deferred measurement
principle <https://en.wikipedia.org/wiki/Deferred_Measurement_Principle>`__. In
the example above, we apply the :class:`~.RY` rotation if the mid-circuit
measurement on qubit 1 yielded ``1`` as an outcome, otherwise doing nothing
for the ``0`` measurement outcome.

PennyLane implements the deferred measurement principle to transform
conditional operations with the :func:`~.defer_measurements` quantum
function transform.

.. code-block:: python

    transformed_qfunc = qml.transforms.defer_measurements(my_quantum_function)
    transformed_qnode = qml.QNode(transformed_qfunc, dev)
    pars = np.array([0.643, 0.246], requires_grad=True)

>>> transformed_qnode(*pars)
tensor([0.90165331, 0.09834669], requires_grad=True)

The decorator syntax applies equally well:

.. code-block:: python

    @qml.qnode(dev)
    @qml.defer_measurements
    def qnode(x, y):
        (...)

Note that we can also specify an outcome when defining a conditional operation:

.. code-block:: python

    @qml.qnode(dev)
    @qml.defer_measurements
    def qnode_conditional_op_on_zero(x, y):
        qml.RY(x, wires=0)
        qml.CNOT(wires=[0, 1])
        m_0 = qml.measure(1)

        qml.cond(m_0 == 0, qml.RY)(y, wires=0)
        return qml.probs(wires=[0])

    pars = np.array([0.643, 0.246], requires_grad=True)

>>> qnode_conditional_op_on_zero(*pars)
tensor([0.88660045, 0.11339955], requires_grad=True)

The deferred measurement principle provides a natural method to simulate the
application of mid-circuit measurements and conditional operations in a
differentiable and device-independent way. Performing true mid-circuit
measurements and conditional operations is dependent on the
quantum hardware and PennyLane device capabilities.

For more examples on applying quantum functions conditionally, refer to the
:func:`~.pennylane.cond` transform.


Changing the number of shots
----------------------------

For hardware devices where the number of shots determines the accuracy
of the expectation value and variance, as well as the number of samples returned,
it can sometimes be convenient to execute the same QNode with differing
number of shots.

For simulators like ``default.qubit``, finite shots will be simulated if
we set ``shots`` to a positive integer.

The shot number can be changed on the device itself, or temporarily altered
by the ``shots`` keyword argument when executing the QNode:


.. code-block:: python

    dev = qml.device("default.qubit", wires=1, shots=10)

    @qml.qnode(dev)
    def circuit(x, y):
        qml.RX(x, wires=0)
        qml.RY(y, wires=0)
        return qml.expval(qml.PauliZ(0))

    # execute the QNode using 10 shots
    result = circuit(0.54, 0.1)

    # execute the QNode again, now using 1 shot
    result = circuit(0.54, 0.1, shots=1)


With an increasing number of shots, the average over
measurement samples converges to the exact expectation of an observable. Consider the following
circuit:

.. code-block:: python

    # fix seed to make results reproducable
    np.random.seed(1)

    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(wires=0)
        return qml.expval(qml.PauliZ(0))

Running the simulator with ``shots=None`` returns the exact expectation.

>>> circuit(shots=None)
0.0

Now we set the device to return stochastic results, and increase the number of shots starting from ``10``.

>>> circuit(shots=10)
0.2

>>> circuit(shots=1000)
-0.062

>>> circuit(shots=100000)
0.00056

The result converges to the exact expectation.
