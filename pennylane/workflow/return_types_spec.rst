Return Type Specification
=========================

Individual measurements
-----------------------

Each individual measurement corresponds to its own type of result. This result can be
a Tensor-like (Python number, numpy array, ML array), but may also be any other type of object.
For example, :class:`~.CountsMP` corresponds to a dictionary. We can also imagine a scenario where
a measurement corresponds to some other type of custom data structure. The important part is that
the measurement process dictates the type, shape, and dtype of the result object.

The result corresponding to a given measurement process should also be independent of any other
measurements present at the same time.  For example, requesting the probability ``qml.probs(wires=0)``
as well should not affect the result corresponding to an expectation value ``qml.expval(qml.Z(0))``.

>>> def example_value(m):
...     tape = qml.tape.QuantumScript([], [m], shots=50)
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


**Empty Wires:**

Some measurments allow broadcasting over all available wires, like ``qml.probs()`` or ``qml.sample()``. In such
a case, the measurement process instance should have empty wires. In such a case, the shape of the result
object may be dictated either by the device or the other operations present in the circuit.

>>> qml.probs().wires
<Wires = []>
>>> tape = qml.tape.QuantumScript([qml.S(0)], [qml.probs()])
>>> qml.device('default.qubit').execute(tape)
array([1., 0.])
>>> qml.device('lightning.qubit', wires=(0,1,2)).execute(tape)
array([1., 0., 0., 0., 0., 0., 0., 0.])

**Broadcasting:**

If the corresponding tape has a ``batch_size`` and the result object is numeric, then the numeric object should
gain a leading dimension. Non-tensorlike arrays may handle broadcasting 

Single Tape
-----------

(WIP) Mid-circuit measurements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Note that this specification is currently under refinement!**

If the tape has mid circuit measurements and one single shot, then the result object
should instead be a tuple of the above specification followed by a dictionary mapping the
circuits mid-circuit measurements to their measured values.

>>> m0 = qml.measure(0)
>>> meaurements = [qml.expval(qml.PauliZ(0)), qml.probs(wires=(0,1))]
>>> tape = qml.tape.QuantumScript(m0.measurements, measurements, shots=1)
>>> qml.device('default.qubit').execute(tape)
((1.0, array([1., 0., 0., 0.])), {measure(wires=[0]): 0})


Batches
-------


Jacobians
---------

