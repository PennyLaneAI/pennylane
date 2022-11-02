.. _deprecations:

Deprecations
============

Pending deprecations
--------------------

* In-place inversion — ``op.inv()`` and ``op.inverse=value`` — is deprecated. Please
  use ``qml.adjoint`` or ``qml.pow`` instead. 

  - Still accessible in v0.27 and v0.28
  - Will be removed in v0.29

  Don't use:

  >>> v1 = qml.PauliX(0).inv()
  >>> v2 = qml.PauliX(0)
  >>> v2.inverse = True

  Instead, use:

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

  .. code-block:: python

    @qml.qnode(dev)
    def ansatz(params):
        some_qfunc(params)
        return qml.expval(Hamiltonian)

* ``QueuingContext`` is renamed ``QueuingManager``. 

  - Still accessible via ``QueuingContext`` in v0.27
  - Will only be accessible via ``QueuingManager`` in v0.28

* ``QueuingManager.safe_update_info`` and ``AnnotateQueue.safe_update_info`` are removed.

  - Still accessible in v0.27
  -  Will be removed in v0.28

* ``qml.tape.stop_recording`` and ``QuantumTape.stop_recording`` are moved to ``qml.QueuingManager.stop_recording``

  - Still accessible in v0.27
  -  Will be removed in v0.28

* ``qml.tape.get_active_tape`` is deprecated. Please use ``qml.QueuingManager.active_context()`` instead.

  - Still accessible in v0.27
  -  Will be removed in v0.28

* ``qml.transforms.qcut.remap_tape_wires`` is deprecated. Please use ``qml.map_wires`` instead.

  - Still accessible in v0.27
  - Will be removed in v0.28

Completed deprecation cycles
----------------------------

* ``ObservableReturnTypes`` ``Sample``, ``Variance``, ``Expectation``, ``Probability``, ``State``, and ``MidMeasure``
  are moved to ``measurements`` from ``operation``.

  - Deprecated in v0.23
  - Removed in v0.27

* The ``qml.utils.expand`` function is deprecated. ``qml.math.expand_matrix`` should be used
  instead.

  - Deprecated in v0.24
  - Removed in v0.27