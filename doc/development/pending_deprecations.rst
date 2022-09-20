.. _pending_deprecations:

Pending Deprecations
====================

* ```QueuingContext`` is renamed ``QueuingManager``. 
   - Still accessible via ``QueuingContext`` in v0.27
   - Only accessible via ``QueuingManager`` in v0.28

* ``QueuingManager.safe_update_info`` and ``AnnotateQueue.safe_update_info`` are removed
   - Still accessible in v0.27
   - Removed in v0.28

* ``qml.tape.stop_recording`` and ``QuantumTape.stop_recording`` are moved to 
  ``qml.QueuingManager.stop_recording``
   - Still accessible in v0.27
   - Removed in v0.28


* In-place inversion is now deprecated. This includes ``op.inv()`` and ``op.inverse=value``. Please
  use ``qml.adjoint`` or ``qml.pow`` instead. 
  - Still accessible in v0.27 and v0.28
  - Removed in v0.29

  Don't use:

  >>> v1 = qml.PauliX(0).inv()
  >>> v2 = qml.PauliX(0)
  >>> v2.inverse = True

  Instead use:

  >>> qml.adjoint(qml.PauliX(0))
  Adjoint(PauliX(wires=[0]))
  >>> qml.pow(qml.PauliX(0), -1)
  PauliX(wires=[0])**-1
  >>> qml.pow(qml.PauliX(0), -1, lazy=False)
  PauliX(wires=[0])
  >>> qml.PauliX(0) ** -1
  PauliX(wires=[0])**-1

* ``qml.ExpvalCost`` has been deprecated, and usage will now raise a warning.
  - Deprecated in v0.24
  - Will be removed in v0.28

  Instead, it is recommended to simply
  pass Hamiltonians to the ``qml.expval`` function inside QNodes:

  ```python
  @qml.qnode(dev)
  def ansatz(params):
      some_qfunc(params)
      return qml.expval(Hamiltonian)
  ```

Already Removed
---------------

* The ``ObservableReturnTypes`` ``Sample``, ``Variance``, ``Expectation``, ``Probability``, ``State``, and ``MidMeasure``
  have been moved to ``m`easurements`` from ``operation``.
  - Deprecated in v0.23
  - To be removed in v0.27

* The ``qml.utils.expand`` function is now removed; ``qml.operation.expand_matrix`` should be used
  instead.
  - Deprecated in v0.24
  - To be removed in v0.27


