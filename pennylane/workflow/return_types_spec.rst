
.. _ReturnTypeSpec:

Return Type Specification
~~~~~~~~~~~~~~~~~~~~~~~~~

This section describes the shape and type of the numerical output from executing a quantum circuit
in PennyLane.

The specification applies for the entire workflow, from the device instance all the
way up to the ``QNode``.  The result object corresponding to a given circuit
will match whether the circuit is being passed to a device, processed
by a transform, having it's derivative bound to an ML interface, or returned from a ``QNode``.

While this section says ``tuple`` and includes examples using ``tuple`` throughout this document, the
return type specification allows ``tuple`` and ``list`` to be used interchangably.
When examining and postprocessing
results, you should always allow for a ``list`` to be substituted for a ``tuple``. Given their
improved performance and protection against unintended side-effects, ``tuple``'s are recommended
over ``list`` where feasible.

The nesting for dimensions from outer dimension to inner dimension is:

1. Quantum Tape in batch. This dimension will always exist for a batch of tapes.
2. Shot choice in a shot vector. This dimension will not exist of a shot vector is not present.
3. Measurement in the quantum tape. This dimension will not exist if the quantum tape only has one measurement.
4. Parameter broadcasting.  Does not exist if no parameter broadcasting. Adds to array shape instead of adding tuple nesting.
5. Fundamental measurement shape.

Individual measurements
-----------------------

Each individual measurement corresponds to its own type of result. This result can be
a Tensor-like (Python number, numpy array, ML array), but may also be any other type of object.
For example, :class:`~.CountsMP` corresponds to a dictionary. We can also imagine a scenario where
a measurement corresponds to some other type of custom data structure.

>>> def example_value(m):
...     tape = qml.tape.QuantumScript((), (m,), shots=50)
...     return qml.device('default.qubit').execute(tape)
>>> example_value(qml.probs(wires=0))
array([1., 0.])
>>> example_value(qml.expval(qml.Z(0)))
1.0
>>> example_value(qml.counts(wires=0))
{'0': 50}
>>> example_value(qml.sample(wires=0))
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0])


Empty Wires
^^^^^^^^^^^

Some measurments allow broadcasting over all available wires, like ``qml.probs()``, ``qml.sample()``,
or ``qml.state()``. In such a case, the measurement process instance should have empty wires.
The shape of the result object may be dictated either by the device or the other operations present in the circuit.

>>> qml.probs().wires
<Wires = []>
>>> tape = qml.tape.QuantumScript([qml.S(0)], (qml.probs(),))
>>> qml.device('default.qubit').execute(tape)
array([1., 0.])
>>> qml.device('default.mixed', wires=(0,1,2)).execute(tape)
array([1., 0., 0., 0., 0., 0., 0., 0.])

Broadcasting
^^^^^^^^^^^^

Parameter broadcasting adds a leading dimension to the numeric array itself.

If the corresponding tape has a ``batch_size`` and the result object is numeric, then the numeric object should
gain a leading dimension.

>>> op = qml.RX((0, np.pi/4, np.pi/2), wires=0)
>>> tape = qml.tape.QuantumScript((op,), [qml.probs(wires=0)])
>>> result = qml.device('default.qubit').execute(tape)
>>> result
array([[1.        , 0.        ],
       [0.85355339, 0.14644661],
       [0.5       , 0.5       ]])
>>> result.shape
(3, 2)
>>> tape = qml.tape.QuantumScript((op,), [qml.expval(qml.Z(0))])
>>> result = qml.device('default.qubit').execute(tape)
>>> result
array([1.00000000e+00, 7.07106781e-01, 2.22044605e-16])
>>> result.shape
(3,)

Non-tensorlike arrays may handle broadcasting in different ways. The ``'default.qubit'`` output
for :class:`~.CountsMP` is a list of dictionaries, but when used in conjunction with
:func:`~.transforms.broadcast_expand`, the result object becomes a ``numpy.ndarray`` of dtype ``object``.

>>> tape = qml.tape.QuantumScript((op,), (qml.counts(),), shots=50)
>>> result = qml.device('default.qubit').execute(tape)
>>> result
[{'0': 50}, {'0': 46, '1': 4}, {'0': 32, '1': 18}]
>>> batch, fn = qml.transforms.broadcast_expand(tape)
>>> fn(qml.device('default.qubit').execute(batch))
array([{'0': 50}, {'0': 39, '1': 11}, {'0': 28, '1': 22}], dtype=object)


Single Tape
-----------

If the tape has a single measurement, then the result corresponding to that tape simply obeys the specification
above.  Otherwise, the result for a single tape is a ``tuple`` where each entry corresponds to each
of the corresponding measurements. In the below example, the first entry corresponds to the first
measurement process ``qml.expval(qml.Z(0))``, the second entry corresponds to the second measurement process
``qml.probs(wires=0)``, and the third result corresponds to the third measurement process ``qml.state()``.

>>> tape = qml.tape.QuantumScript((), (qml.expval(qml.Z(0)), qml.probs(wires=0), qml.state()))
>>> qml.device('default.qubit').execute(tape)
(1.0, array([1., 0.]), array([1.+0.j, 0.+0.j]))

Shot vectors
^^^^^^^^^^^^

When a shot vector is present ``shots.has_partitioned_shot``, the measurement instead becomes a
tuple where each entry corresponds to a different shot value.

>>> measurements = (qml.expval(qml.Z(0)), qml.probs(wires=0))
>>> tape = qml.tape.QuantumScript((), measurements, shots=(50,50,50))
>>> result = qml.device('default.qubit').execute(tape)
>>> result
((1.0, array([1., 0.])), (1.0, array([1., 0.])), (1.0, array([1., 0.])))
>>> result[0]
(1.0, array([1., 0.]))
>>> tape = qml.tape.QuantumScript((), [qml.counts(wires=0)], shots=(1, 10, 100))
>>> qml.device('default.qubit').execute(tape)
({'0': 1}, {'0': 10}, {'0': 100})

Let's look at an example with all forms of nesting.  Here, we have a tape with a batch size of ``3``, three
diferent measurements with different fundamental shapes, and a shot vector with three different values.

>>> op = qml.RX((1.2, 2.3, 3.4), 0)
>>> ms = (qml.expval(qml.Z(0)), qml.probs(wires=0), qml.counts())
>>> tape = qml.tape.QuantumScript((op,), ms, shots=(1, 100, 1000))
>>> result = qml.device('default.qubit').execute(tape)
>>> result
((array([ 1., -1., -1.]),
array([[1., 0.],
       [0., 1.],
       [0., 1.]]),
[{'0': 1}, {'1': 1}, {'1': 1}]),
(array([ 0.3 , -0.66, -0.98]),
array([[0.61, 0.39],
       [0.13, 0.87],
       [0.03, 0.97]]),
[{'0': 61, '1': 39}, {'0': 13, '1': 87}, {'0': 3, '1': 97}]),
(array([ 0.364, -0.648, -0.962]),
array([[0.669, 0.331],
       [0.165, 0.835],
       [0.012, 0.988]]),
[{'0': 669, '1': 331}, {'0': 165, '1': 835}, {'0': 12, '1': 988}]))


>>> result[0][0] # first shot value, first measurement
array([ 1., -1., -1.])
>>> result[0][0][0] # first shot value, first measurement, and parameter of 1.2
1.0
>>> result[1][2] # second shot value, third measurement, all three parameter values
[{'0': 74, '1': 26}, {'0': 23, '1': 77}, {'1': 100}]


Batches
-------

A batch is a tuple or list of multiple tapes.  In this case, the result should always be a tuple
where each entry corresponds to the result for the corresponding tape.

>>> tape1 = qml.tape.QuantumScript([qml.X(0)], [qml.state()])
>>> tape2 = qml.tape.QuantumScript([qml.Hadamard(0)], [qml.counts()], shots=100)
>>> tape3 = qml.tape.QuantumScript([], [qml.expval(qml.Z(0)), qml.expval(qml.X(0))])
>>> batch = (tape1, tape2, tape3)
>>> qml.device('default.qubit').execute(batch)
(array([0.+0.j, 1.+0.j]), {'0': 50, '1': 50}, (1.0, 0.0))