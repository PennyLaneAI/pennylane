.. _deprecations:

Deprecations
============

Pending deprecations
--------------------

* The ``get_operation`` tape method is updated to return the operation index as well, changing its signature.

  - The new signature is available by changing the arg ``return_op_index`` to ``True`` in v0.29
  - The old signature is replaced with the new one in v0.30

* The ``observables`` argument in ``QubitDevice.statistics`` is deprecated. Please use ``circuit``
  instead. Using a list of observables in ``QubitDevice.statistics`` is deprecated. Please use a
  ``QuantumTape`` instead.

  - Still accessible in v0.28
  - Will be removed in v0.29

* The ``seed_recipes`` argument in ``qml.classical_shadow`` and ``qml.shadow_expval`` is deprecated.
  A new argument ``seed`` has been added, which defaults to ``None`` and can contain an integer with the 
  wanted seed.

  - Still accessible in v0.28
  - Will be removed in v0.29

* The ``grouping`` module is deprecated. The functionality has been moved and
  reorganized in the new ``pauli`` module under ``pauli/utils.py`` or ``pauli/grouping/``.

  - Still accessible in v0.27, v0.28
  - Will be removed in v0.29

  The functions from ``grouping/pauli.py``, ``grouping/transformations.py`` and
  ``grouping/utils.py`` have been moved to ``pauli/utils.py``. The remaining functions
  have been consolidated in the ``pauli/grouping/`` directory.

* ``qml.ExpvalCost`` has been deprecated, and usage will now raise a warning.
  
  - Deprecated in v0.24
  - Will be removed in v0.30

  Instead, it is recommended to simply
  pass Hamiltonians to the ``qml.expval`` function inside QNodes:

  .. code-block:: python

    @qml.qnode(dev)
    def ansatz(params):
        some_qfunc(params)
        return qml.expval(Hamiltonian)

* ``qml.transforms.make_tape`` has been deprecated, and usage will now raise a warning.
  Instead, use ``qml.tape.make_qscript``.

  - Deprecated in v0.28
  - Will be removed in v0.29

* The ``collections`` module has been deprecated.

  - Deprecated in v0.29
  - Will be removed in v0.31

* ``qml.op_sum``` is deprecated. Users should use ``qml.sum`` instead.

  - Deprecated in v0.29.
  - Will be removed in v0.31.

* A warning has been added in ``Evolution`` redirecting users to ``qml.evolve``. This was added
  because we want to change the behaviour of ``Evolution``, adding a ``-1`` to the given parameter.

  - Deprecated in v0.29.
  - Will be removed in v0.30.

Completed deprecation cycles
----------------------------


* In-place inversion — ``op.inv()`` and ``op.inverse=value`` — is deprecated. Please
  use ``qml.adjoint`` or ``qml.pow`` instead. 

  - Still accessible in v0.27 and v0.28
  - Removed in v0.29

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

* The ``qml.utils.decompose_hamiltonian()`` method is removed. Please
  use ``qml.pauli_decompose()``.

  - Still accessible in v0.27
  - Removed in v0.28

* ``qml.tape.get_active_tape`` is deprecated. Please use ``qml.QueuingManager.active_context()`` instead.

  - Deprecated in v0.27
  - Removed in v0.28

* ``qml.transforms.qcut.remap_tape_wires`` is deprecated. Please use ``qml.map_wires`` instead.

  - Deprecated in v0.27
  - Removed in v0.28

* ``QuantumTape.inv()`` is deprecated. Please use ``QuantumTape.adjoint()`` instead. This method
  returns a new tape instead of modifying itself in-place.

  - Deprecated in v0.27
  - Removed in v0.28

* ``qml.tape.stop_recording`` and ``QuantumTape.stop_recording`` are moved to ``qml.QueuingManager.stop_recording``

  - Deprecated in v0.27
  - Removed in v0.28

* ``QueuingContext`` is renamed ``QueuingManager``. 

  - Deprecated name ``QueuingContext`` in v0.27
  - Removed in v0.28

* ``QueuingManager.safe_update_info`` and ``AnnotateQueue.safe_update_info`` are removed.

  - Deprecated in v0.27
  - Removed in v0.28

* ``ObservableReturnTypes`` ``Sample``, ``Variance``, ``Expectation``, ``Probability``, ``State``, and ``MidMeasure``
  are moved to ``measurements`` from ``operation``.

  - Deprecated in v0.23
  - Removed in v0.27

* The ``qml.utils.expand`` function is deprecated. ``qml.math.expand_matrix`` should be used
  instead.

  - Deprecated in v0.24
  - Removed in v0.27

* The ``qml.Operation.get_parameter_shift`` method is removed. Use the methods of the ``gradients`` module
  for general parameter-shift rules instead.

  - Deprecated in v0.22
  - Removed in v0.28

* ``qml.transforms.measurement_grouping`` has been removed. Please use ``qml.transforms.hamiltonian_expand``
  instead. 

  - Deprecated in v0.28
  - Removed in v0.29

