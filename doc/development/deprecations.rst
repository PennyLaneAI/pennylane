.. _deprecations:

Deprecations
============

Pending deprecations
--------------------

* `Observable.return_type` is deprecated. Instead, you should inspect the type
  of the surrounding measurement process.

  - Deprecated in v0.34
  - Will be removed in v0.35

* ``single_tape_transform``, ``batch_transform``, ``qfunc_transform``, and ``op_transform`` are
  deprecated. Instead switch to using the new ``qml.transform`` function. Please refer to
  `the transform docs <https://docs.pennylane.ai/en/stable/code/qml_transforms.html#custom-transforms>`_
  to see how this can be done.

  - Deprecated in v0.34
  - Will be removed in v0.36

* ``QuantumScript.is_sampled`` and ``QuantumScript.all_sampled`` are deprecated. Users should now validate
  these properties manually.

  .. code-block:: python

    from pennylane.measurements import *
    sample_types = (SampleMP, CountsMP, ClassicalShadowMP, ShadowExpvalMP)
    is_sample_type = [isinstance(m, sample_types) for m in tape.measurements]
    is_sampled = any(is_sample_type)
    all_sampled = all(is_sample_type)

  - Deprecated in v0.34
  - Will be removed in v0.35

* Passing additional arguments to a transform that decorates a QNode should now be done through use
  of ``functools.partial``. For example, the :func:`~pennylane.metric_tensor` transform has an
  optional ``approx`` argument which should now be set using:

  .. code-block:: python

    from functools import partial

    @partial(qml.metric_tensor, approx="block-diag")
    @qml.qnode(dev)
    def circuit(weights):
        ...

  The previously-recommended approach is now deprecated:

  .. code-block:: python

    @qml.metric_tensor(approx="block-diag")
    @qml.qnode(dev)
    def circuit(weights):
        ...

  Alternatively, consider calling the transform directly:

  .. code-block:: python

    @qml.qnode(dev)
    def circuit(weights):
        ...

    transformed_circuit = qml.metric_tensor(circuit, approx="block-diag")

  - Deprecated in v0.33
  - Will be removed in v0.35

* ``qml.ExpvalCost`` has been deprecated, and usage will now raise a warning.

  - Deprecated in v0.24
  - Will be removed in v0.35

  Instead, it is recommended to simply
  pass Hamiltonians to the ``qml.expval`` function inside QNodes:

  .. code-block:: python

    @qml.qnode(dev)
    def ansatz(params):
        some_qfunc(params)
        return qml.expval(Hamiltonian)

Completed deprecation cycles
----------------------------

* Specifying ``control_values`` passed to ``qml.ctrl`` as a string is no longer supported.

  - Deprecated in v0.25
  - Removed in v0.34

* ``qml.gradients.pulse_generator`` has become ``qml.gradients.pulse_odegen`` to adhere to paper naming conventions.

  - Deprecated in v0.33
  - Removed in v0.34

* The ``prep`` keyword argument in ``QuantumScript`` has been removed.
  ``StatePrepBase`` operations should be placed at the beginning of the ``ops`` list instead.

  - Deprecated in v0.33
  - Removed in v0.34

* The public methods of ``DefaultQubit`` are pending changes to
  follow the new device API.

  We will be switching to the new device interface in a coming release.
  In this new interface, simulation implementation details
  will be abstracted away from the device class itself and provided by composition, rather than inheritance.
  Therefore, some public and private methods from ``DefaultQubit`` will no longer exist, though its behaviour
  in a workflow will remain the same.

  If you directly interact with device methods, please consult
  :class:`pennylane.devices.Device` and
  :class:`pennylane.devices.DefaultQubit`
  for more information on what the new interface will look like and be prepared
  to make updates in a coming release. If you have any feedback on these
  changes, please create an
  `issue <https://github.com/PennyLaneAI/pennylane/issues>`_ or post in our
  `discussion forum <https://discuss.pennylane.ai/>`_.

  - Deprecated in v0.31
  - Changed in v0.33

* The behaviour of ``Operator.__eq__`` and ``Operator.__hash__`` has been updated. Their documentation
  has been updated to reflect the incoming changes.

  The changes to operator equality allow users to use operator equality the same way as
  with ``qml.equal``. With the changes to hashing, unique operators that are equal now have the same
  hash. These changes now allow behaviour such as the following:

  >>> qml.RX(0.1, wires=0) == qml.RX(0.1, wires=0)
  True
  >>> {qml.PauliZ(0), qml.PauliZ(0)}
  {PauliZ(wires=[0])}

  Meanwhile, the previous behaviour is shown below:

  >>> qml.RX(0.1, wires=0) == qml.RX(0.1, wires=0)
  False
  >>> {qml.PauliZ(0), qml.PauliZ(0)}
  {PauliZ(wires=[0]), PauliZ(wires=[0])}

  - Added in v0.32
  - Behaviour changed in v0.33

* ``qml.qchem.jordan_wigner`` had been removed.
  Use ``qml.jordan_wigner`` instead. List input to define the fermionic operator
  is no longer accepted; the fermionic operators ``qml.FermiA``, ``qml.FermiC``,
  ``qml.FermiWord`` and ``qml.FermiSentence`` should be used instead. See the
  :mod:`pennylane.fermi` module documentation and the
  `Fermionic Operator <https://pennylane.ai/qml/demos/tutorial_fermionic_operators>`_
  tutorial for more details.

  - Deprecated in v0.32
  - Removed in v0.33

* The ``tuple`` input type in ``qubit_observable`` has been removed. Please use a fermionic
  operator object. The ``tuple`` return type in ``fermionic_hamiltonian`` and
  ``fermionic_observable`` has been removed and these functions will return a fermionic operator
  by default.

  - Deprecated in v0.32
  - Removed in v0.33

* The ``sampler_seed`` argument of ``qml.gradients.spsa_grad`` has been removed.
  Instead, the ``sampler_rng`` argument should be set, either to an integer value, which will be used
  to create a PRNG internally, or to a NumPy pseudo-random number generator (PRNG) created via
  ``np.random.default_rng(seed)``.
  The advantage of passing a PRNG is that one can reuse that PRNG when calling ``spsa_grad``
  multiple times, for instance during an optimization procedure.

  - Deprecated in v0.32
  - Removed in v0.33

* The ``RandomLayers.compute_decomposition`` keyword argument ``ratio_imprivitive`` has been changed to
  ``ratio_imprim`` to match the call signature of the operation.

  - Deprecated in v0.32
  - Removed in v0.33

* The ``QuantumScript.set_parameters`` method and the ``QuantumScript.data`` setter have
  been removed. Please use ``QuantumScript.bind_new_parameters`` instead.

  - Deprecated in v0.32
  - Removed in v0.33

* The ``observables`` argument in ``QubitDevice.statistics`` is removed. Please use ``circuit``
  instead. Using a list of observables in ``QubitDevice.statistics`` is removed. Please use a
  ``QuantumTape`` instead.

  - Still accessible in v0.28-v0.31
  - Removed in v0.32


* The CV observables ``qml.X`` and ``qml.P`` have been removed. Use ``qml.QuadX`` and ``qml.QuadP`` instead.

  - Deprecated in v0.32
  - Removed in v0.33


* The method ``tape.unwrap()`` and corresponding ``UnwrapTape`` and ``Unwrap`` classes are
  removed.

  - Deprecated in v0.32
  - Removed in v0.33

  Instead of ``tape.unwrap()``, use :func:`~.transforms.convert_to_numpy_parameters`:

  .. code-block:: python

    from pennylane.transforms import convert_to_numpy_parameters

    qscript = qml.tape.QuantumTape([qml.RX(torch.tensor(0.1234), 0)],
                                     [qml.expval(qml.Hermitian(torch.eye(2), 0))] )
    unwrapped_qscript = convert_to_numpy_parameters(qscript)

    torch_params = qscript.get_parameters()
    numpy_params = unwrapped_qscript.get_parameters()

* ``qml.enable_return`` and ``qml.disable_return`` have been removed. The old return types are no longer available.

  - Deprecated in v0.32
  - Removed in v0.33

* The ``mode`` keyword argument in ``QNode`` has been removed, as it was only used in the old return
  system (which has also been removed). Please use ``grad_on_execution`` instead.

  - Deprecated in v0.32
  - Removed in v0.33

* ``qml.math.purity``, ``qml.math.vn_entropy``, ``qml.math.mutual_info``, ``qml.math.fidelity``,
  ``qml.math.relative_entropy``, and ``qml.math.max_entropy`` no longer support state vectors as
  input. Please call ``qml.math.dm_from_state_vector`` on the input before passing to any of these functions.

  - Still accepted in v0.31
  - Removed in v0.32

* The ``do_queue`` keyword argument in ``qml.operation.Operator`` has been removed. This affects
  all child classes, such as ``Operation``, ``Observable``, ``SymbolicOp`` and more. Instead of
  setting ``do_queue=False``, use the ``qml.QueuingManager.stop_recording()`` context.

  - Deprecated in v0.31
  - Removed in v0.32

* The ``qml.specs`` dictionary longer supports direct key access to certain keys. Instead
  these quantities can be accessed as fields of the new ``Resources`` object saved under
  ``specs_dict["resources"]``:

  - ``num_operations`` is no longer supported, use ``specs_dict["resources"].num_gates``
  - ``num_used_wires`` is no longer supported, use ``specs_dict["resources"].num_wires``
  - ``gate_types`` is no longer supported, use ``specs_dict["resources"].gate_types``
  - ``gate_sizes`` is no longer supported, use ``specs_dict["resources"].gate_sizes``
  - ``depth`` is no longer supported, use ``specs_dict["resources"].depth``

  These keys were still accessible in v0.31 and removed in v0.32.

* ``qml.math.reduced_dm`` has been removed. Please use ``qml.math.reduce_dm`` or ``qml.math.reduce_statevector`` instead.

  - Still accessible in v0.31
  - Removed in v0.32

* ``QuantumScript``'s ``name`` keyword argument and property are removed.
  This also affects ``QuantumTape`` and ``OperationRecorder``.

  - Deprecated in v0.31
  - Removed in v0.32

* The ``Operation.base_name`` property is removed. Please use ``Operator.name`` or ``type(obj).__name__`` instead.

  - Still accessible in v0.31
  - Removed in v0.32

* ``LieAlgebraOptimizer`` has been renamed. Please use ``RiemannianGradientOptimizer`` instead.

  - Deprecated in v0.31
  - Removed in v0.32


* The ``grouping_type`` and ``grouping_method`` arguments of ``qchem.molecular_hamiltonian()`` are removed.

  - Deprecated in v0.31
  - Removed in v0.32

  Instead, simply construct a new instance of ``Hamiltonian`` with the grouping specified:

  .. code-block:: python

    H, qubits = molecular_hamiltonian(symbols, coordinates)
    grouped_h = qml.Hamiltonian(
        H.coeffs,
        H.ops,
        grouping_type=grouping_type,
        groupingmethod=grouping_method,
    )

* ``zyz_decomposition`` and ``xyx_decomposition`` are removed, use ``one_qubit_decomposition`` with a rotations
  keyword instead.

  - Deprecated in v0.31
  - Removed in v0.32

* The ``qml.utils.sparse_hamiltonian`` function has been removed. ``~.Hamiltonian.sparse_matrix`` should be used instead.

  - Deprecated in v0.29
  - Removed in v0.31

* The ``collections`` module has been removed.

  - Deprecated in v0.29
  - Removed in v0.31

* ``qml.op_sum`` has been removed. Users should use ``qml.sum`` instead.

  - Deprecated in v0.29.
  - Removed in v0.31.

* The argument ``argnum`` for gradient transforms using the Jax interface is replaced by ``argnums``.

  - ``argnum`` is automatically changed to ``argnums`` for gradient transforms using JAX and a warning is raised in v0.30
  - ``argnums`` is the only option for gradient transforms using JAX in v0.31

* ``Evolution`` now adds a ``-1`` to the input parameter. Beforehand, the minus sign was not included.

  - Transition warning added in v0.29.
  - Updated to current behaviour in v0.30.

* The ``seed_recipes`` argument in ``qml.classical_shadow`` and ``qml.shadow_expval`` has been removed.
  An argument ``seed`` which defaults to ``None`` can contain an integer with the wanted seed.

  - Still accessible in v0.28, v0.29
  - Removed in v0.30

* The ``get_operation`` tape method is updated to return the operation index as well, changing its signature.

  - The new signature is available by changing the arg ``return_op_index`` to ``True`` in v0.29
  - The old signature is replaced with the new one in v0.30


* The ``grouping`` module has been removed. The functionality has been moved and
  reorganized in the new ``pauli`` module under ``pauli/utils.py`` or ``pauli/grouping/``.

  - Still accessible in v0.27, v0.28, v0.29, v0.30
  - Removed in v0.31

  The functions from ``grouping/pauli.py``, ``grouping/transformations.py`` and
  ``grouping/utils.py`` have been moved to ``pauli/utils.py``. The remaining functions
  have been consolidated in the ``pauli/grouping/`` directory.

* ``qml.VQECost`` is removed.

   - Deprecated in 0.13
   - Removed in 0.29

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

* ``qml.transforms.make_tape`` was previously deprecated, but there is no longer a plan to remove it.
  It no longer raises a warning, and the functionality is unchanged.

  - Deprecated in v0.28
  - Un-deprecated in v0.29
