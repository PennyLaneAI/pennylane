qp.tape
========

.. currentmodule:: pennylane.tape

This module is called ``tape`` to a former class ``QuantumTape`` that has now been removed in favor of :class:`~.QuantumScript`.
:class:`~.QuantumScript` should be accessed from the ``core`` module now, but this module as left as a stub for convenience.


Quantum scripts are a datastructure that can represent quantum circuits and measurement statistics in PennyLane.

In addition to being created internally by QNodes, quantum scripts can also be created  and executed manually.

Finally, quantum tapes are fully compatible with autodifferentiating via Autograd, JAX, 
and PyTorch.

.. warning::

    Unless you are a PennyLane or plugin developer, you likely do not need
    to use these classes directly.

    See the :doc:`quantum circuits <../introduction/circuits>` page for more
    details on creating QNodes, as well as the :func:`~pennylane.qnode` decorator
    and :func:`~pennylane.QNode` constructor.

.. _tape-vs-script:

QuantumTape versus QuantumScript
--------------------------------

A ``QuantumScript`` is purely a representation of a quantum circuit, and can only be constructed
via initialization. Once it is initialized, the contents should then remain immutable throughout its lifetime.

>>> from pennylane.core.qscript import QuantumScript
>>> ops = [qp.PauliX(0)]
>>> measurements = [qp.expval(qp.PauliZ(0))]
>>> QuantumScript(ops, measurements, shots=10)
<QuantumScript: wires=[0], params=0>

To capture a quantum function, we recommend queuing a quantum function into an :class:`~.AnnotatedQueue`,
and then processing that into a ``QuantumScript``.

>>> with qp.core.AnnotatedQueue() as q:
...     qfunc(*args, **kwargs)
>>> QuantumScript.from_queue(q)
<QuantumScript: wires=[0], params=0> 


Quantum Script
~~~~~~~~~~~~~~


.. currentmodule:: pennylane.core.qscript

.. autosummary::
    :toctree: api

    ~QuantumScript
    ~QuantumScriptBatch
    ~QuantumScriptOrBatch
    ~make_qscript
    ~process_queue