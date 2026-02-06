
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

>>> import pennylane as qp
>>> import numpy as np
>>> def example_value(m):
...     tape = qp.tape.QuantumScript((), (m,), shots=10)
...     return qp.device('default.qubit').execute(tape)
>>> example_value(qp.probs(wires=0))
array([1., 0.])
>>> example_value(qp.expval(qp.Z(0)))
np.float64(1.0)
>>> example_value(qp.counts(wires=0))
{np.str_('0'): np.int64(10)}
>>> example_value(qp.sample(wires=0))
array([[0],
       [0],
       [0],
       [0],
       [0],
       [0],
       [0],
       [0],
       [0],
       [0]])

Empty Wires
^^^^^^^^^^^

Some measurements allow broadcasting over all available wires, like ``qp.probs()``, ``qp.sample()``,
or ``qp.state()``. In such a case, the measurement process instance should have empty wires.
The shape of the result object may be dictated either by the device or the other operations present in the circuit.

>>> qp.probs().wires
Wires([])
>>> tape = qp.tape.QuantumScript([qp.S(0)], (qp.probs(),))
>>> qp.device('default.qubit').execute(tape)
array([1., 0.])
>>> qp.device('default.qubit', wires=(0,1,2)).execute(tape)
array([1., 0.])
>>> new_tape = qp.tape.QuantumScript([qp.S(0), qp.S(1)], (qp.probs(),))
>>> qp.device('default.qubit', wires=(0,1,2)).execute(new_tape)
array([1., 0., 0., 0.])

Broadcasting
^^^^^^^^^^^^

Parameter broadcasting adds a leading dimension to the numeric array itself.

If the corresponding tape has a ``batch_size`` and the result object is numeric, then the numeric object should
gain a leading dimension.  Note that a batch size of ``1`` is still a batch size,
and still should correspond to a leading dimension.

>>> op = qp.RX((0, np.pi/4, np.pi/2), wires=0)
>>> tape = qp.tape.QuantumScript((op,), [qp.probs(wires=0)])
>>> result = qp.device('default.qubit').execute(tape)
>>> result
array([[1.        , 0.        ],
       [0.853..., 0.1464...],
       [0.5       , 0.5       ]])
>>> result.shape
(3, 2)
>>> tape = qp.tape.QuantumScript((op,), [qp.expval(qp.Z(0))])
>>> result = qp.device('default.qubit').execute(tape)
>>> result
array([1.    , 0.7071, 0.    ])
>>> result.shape
(3,)

Non-tensorlike arrays may handle broadcasting in different ways. The ``'default.qubit'`` output
for :class:`~.CountsMP` is a list of dictionaries, but when used in conjunction with
:func:`~.transforms.broadcast_expand`, the result object becomes a ``numpy.ndarray`` of dtype ``object``.

>>> tape = qp.tape.QuantumScript((op,), (qp.counts(),), shots=50)
>>> result = qp.device('default.qubit', seed=42).execute(tape)
>>> print(result)
[{np.str_('0'): np.int64(50)}, {np.str_('0'): np.int64(49), np.str_('1'): np.int64(1)}, {np.str_('0'): np.int64(28), np.str_('1'): np.int64(22)}]
>>> batch, fn = qp.transforms.broadcast_expand(tape)
>>> print(fn(qp.device('default.qubit', seed=42).execute(batch)))
[{np.str_('0'): np.int64(50)}
 {np.str_('0'): np.int64(49), np.str_('1'): np.int64(1)}
 {np.str_('0'): np.int64(28), np.str_('1'): np.int64(22)}]

Single Tape
-----------

If the tape has a single measurement, then the result corresponding to that tape simply obeys the specification
above.  Otherwise, the result for a single tape is a ``tuple`` where each entry corresponds to each
of the corresponding measurements. In the below example, the first entry corresponds to the first
measurement process ``qp.expval(qp.Z(0))``, the second entry corresponds to the second measurement process
``qp.probs(wires=0)``, and the third result corresponds to the third measurement process ``qp.state()``.

>>> tape = qp.tape.QuantumScript((), (qp.expval(qp.Z(0)), qp.probs(wires=0), qp.state()))
>>> qp.device('default.qubit').execute(tape)
(np.float64(1.0), array([1., 0.]), array([1.+0.j, 0.+0.j]))

Shot vectors
^^^^^^^^^^^^

When a shot vector is present ``shots.has_partitioned_shot``, the measurement instead becomes a
tuple where each entry corresponds to a different shot value.

>>> measurements = (qp.expval(qp.Z(0)), qp.probs(wires=0))
>>> tape = qp.tape.QuantumScript((), measurements, shots=(50,50,50))
>>> result = qp.device('default.qubit').execute(tape)
>>> result
((np.float64(1.0), array([1., 0.])), (np.float64(1.0), array([1., 0.])), (np.float64(1.0), array([1., 0.])))
>>> result[0]
(np.float64(1.0), array([1., 0.]))
>>> tape = qp.tape.QuantumScript((), [qp.counts(wires=0)], shots=(1, 10, 100))
>>> qp.device('default.qubit').execute(tape)
({np.str_('0'): np.int64(1)}, {np.str_('0'): np.int64(10)}, {np.str_('0'): np.int64(100)})

Let's look at an example with all forms of nesting.  Here, we have a tape with a batch size of ``3``, three
different measurements with different fundamental shapes, and a shot vector with three different values.

>>> op = qp.RX((1.2, 2.3, 3.4), 0)
>>> ms = (qp.expval(qp.Z(0)), qp.probs(wires=0), qp.counts())
>>> tape = qp.tape.QuantumScript((op,), ms, shots=(1, 100, 1000))
>>> result = qp.device('default.qubit', seed=42).execute(tape)
>>> from pprint import pprint
>>> pprint(result)  # for better readability
((array([-1., -1., -1.]),
  array([[0., 1.],
       [0., 1.],
       [0., 1.]]),
  [{np.str_('1'): np.int64(1)},
   {np.str_('1'): np.int64(1)},
   {np.str_('1'): np.int64(1)}]),
 (array([ 0.38, -0.6 , -0.98]),
  array([[0.71, 0.29],
       [0.19, 0.81],
       [0.02, 0.98]]),
  [{np.str_('0'): np.int64(71), np.str_('1'): np.int64(29)},
   {np.str_('0'): np.int64(19), np.str_('1'): np.int64(81)},
   {np.str_('0'): np.int64(2), np.str_('1'): np.int64(98)}]),
 (array([ 0.362, -0.688, -0.964]),
  array([[0.678, 0.322],
       [0.164, 0.836],
       [0.014, 0.986]]),
  [{np.str_('0'): np.int64(678), np.str_('1'): np.int64(322)},
   {np.str_('0'): np.int64(164), np.str_('1'): np.int64(836)},
   {np.str_('0'): np.int64(14), np.str_('1'): np.int64(986)}]))
>>> result[0][0] # first shot value, first measurement
array([-1., -1., -1.])
>>> result[0][0][0] # first shot value, first measurement, and parameter of 1.2
np.float64(-1.0)
>>> result[1][2] # second shot value, third measurement, all three parameter values
[{np.str_('0'): np.int64(71), np.str_('1'): np.int64(29)}, {np.str_('0'): np.int64(19), np.str_('1'): np.int64(81)}, {np.str_('0'): np.int64(2), np.str_('1'): np.int64(98)}]

Batches
-------

A batch is a tuple or list of multiple tapes.  In this case, the result should always be a tuple
where each entry corresponds to the result for the corresponding tape.

>>> tape1 = qp.tape.QuantumScript([qp.X(0)], [qp.state()])
>>> tape2 = qp.tape.QuantumScript([qp.Hadamard(0)], [qp.counts()], shots=100)
>>> tape3 = qp.tape.QuantumScript([], [qp.expval(qp.Z(0)), qp.expval(qp.X(0))])
>>> batch = (tape1, tape2, tape3)
>>> qp.device('default.qubit', seed=42).execute(batch)
(array([0.+0.j, 1.+0.j]), {np.str_('0'): np.int64(53), np.str_('1'): np.int64(47)}, (np.float64(1.0), np.float64(0.0)))
