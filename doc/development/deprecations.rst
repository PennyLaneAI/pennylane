.. _deprecations:

Deprecations
============

All PennyLane deprecations will raise a ``qml.exceptions.PennyLaneDeprecationWarning``. Pending and completed
deprecations are listed below.

Pending deprecations
--------------------

* The ``transform_program`` property of ``QNode`` has been renamed to ``compile_pipeline``.
  The deprecated access through ``transform_program`` will be removed in PennyLane v0.46.

  - Deprecated in v0.45
  - Will be removed in v0.46

* The ``qml.transforms.create_expand_fn`` has been deprecated and will be removed in v0.46.
  Please use the ``qml.transforms.decompose`` function for decomposing circuits.

  - Deprecated in v0.45
  - Will be removed in v0.46

* Setting shots on a device through the ``shots`` keyword argument is deprecated. Instead,
  please specify shots using the ``shots`` keyword argument of :class:`~.QNode`, or use the
  :func:`pennylane.set_shots` transform on the :class:`~.QNode`.

  .. code-block:: python

    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev, shots=1000)
    def circuit(x):
        qml.RX(x, wires=0)
        return qml.expval(qml.Z(0))

    circuit_analytic = qml.set_shots(circuit, None)

  - Deprecated in v0.43
  - Will be removed in a future version

* Maintenance support for the ``tensorflow`` interface has been deprecated and will be dropped in PennyLane v0.44.
  Future versions of PennyLane are not guaranteed to work with TensorFlow.
  Instead, we recommend using the :doc:`jax </introduction/interfaces/jax>` or :doc:`torch </introduction/interfaces/torch>` interface for
  machine learning applications to benefit from enhanced support and features.

  - Deprecated in v0.43
  - Will be removed in v0.44

* Specifying ``shots`` as a keyword argument when executing a :class:`~.QNode` is deprecated and will be removed in v0.44.
  Instead, please set shots on ``QNode`` initialization, or use the :func:`~.workflow.set_shots` transform to set the number of shots.

  - Deprecated in v0.43
  - Will be removed in v0.44

Completed removal of legacy operator arithmetic
-----------------------------------------------

In PennyLane v0.40, the legacy operator arithmetic system has been removed, and is fully replaced by the new
operator arithmetic functionality that was introduced in v0.36. Check out the :ref:`Updated operators <new_opmath>` page
for details on how to port your legacy code to the new system. The following functionality has been removed:

* In PennyLane v0.40, legacy operator arithmetic has been removed. This includes :func:`pennylane.operation.enable_new_opmath`,
  :func:`pennylane.operation.disable_new_opmath`, :class:`pennylane.ops.Hamiltonian`, and :class:`pennylane.operation.Tensor`. Note
  that ``qml.Hamiltonian`` will continue to dispatch to :class:`pennylane.ops.LinearCombination`.

  - Deprecated in v0.39
  - Removed in v0.40

* :meth:`pennylane.pauli.PauliSentence.hamiltonian` and :meth:`pennylane.pauli.PauliWord.hamiltonian` have been removed. Instead, please use
  :meth:`pennylane.pauli.PauliSentence.operation` and :meth:`pennylane.pauli.PauliWord.operation` respectively.

  - Deprecated in v0.39
  - Removed in v0.40

* :func:`pennylane.pauli.simplify` has been removed. Instead, please use :func:`pennylane.simplify` or :meth:`pennylane.operation.Operator.simplify`.

  - Deprecated in v0.39
  - Removed in v0.40

Completed deprecation cycles
----------------------------

* Maintenance support of NumPy<2.0 has been removed. PennyLane v0.45 and beyond are not guaranteed to work with NumPy<2.0.
  We recommend upgrading your version of NumPy to benefit from enhanced support and features.
  
  - Deprecated in v0.44
  - Removed in v0.45

* ``compute_qfunc_decomposition`` and ``has_qfunc_decomposition`` have been removed from  :class:`~.Operator`
  and all subclasses that implemented them. The graph decomposition system should be used to enable capture instead.

  - Removed in v0.45

* The :func:`pennylane.devices.preprocess.mid_circuit_measurements` transform is deprecated. Instead,
  the device should determine which mcm method to use, and explicitly include :func:`~pennylane.transforms.dynamic_one_shot`
  or :func:`~pennylane.transforms.defer_measurements` in its preprocess transforms if necessary. See
  :func:`DefaultQubit.setup_execution_config <pennylane.devices.DefaultQubit.setup_execution_config>` and
  :func:`DefaultQubit.preprocess_transforms <pennylane.devices.DefaultQubit.preprocess_transforms>` for an example.

  - Deprecated in v0.44
  - Removed in v0.45

* The ``custom_decomps`` keyword argument to ``qml.device`` has been deprecated and will be removed
  in 0.45. Instead, with ``qml.decomposition.enable_graph()``, new decomposition rules can be defined as
  quantum functions with registered resources. See :mod:`pennylane.decomposition` for more details.

  - Deprecated in v0.44
  - Removed in v0.45

* ``argnum`` has been renamed ``argnums`` in ``qml.grad``, ``qml.jacobian``, ``qml.jvp``, and ``qml.vjp``
  to better match Catalyst and JAX.

  - Deprecated in v0.44
  - Removed in v0.45

* Access to the following functions and classes from the `~pennylane.resources` module have 
  been removed. Instead, these functions must be imported from the `~pennylane.estimator` module.

    - ``qml.estimator.estimate_shots`` in favor of ``qml.resources.estimate_shots``
    - ``qml.estimator.estimate_error`` in favor of ``qml.resources.estimate_error``
    - ``qml.estimator.FirstQuantization`` in favor of ``qml.resources.FirstQuantization``
    - ``qml.estimator.DoubleFactorization`` in favor of ``qml.resources.DoubleFactorization``

  - Deprecated in v0.44
  - Removed in v0.45

* Passing a function to the ``gate_set`` argument in the :func:`~pennylane.transforms.decompose` transform
  is removed. The ``gate_set`` argument expects a static iterable of operator type and/or operator names,
  and the function should be passed to the ``stopping_condition`` argument instead.

  - Deprecated in v0.44
  - Removed in v0.45

  The example below illustrates how you can provide a function as the ``stopping_condition`` in addition to providing a
  ``gate_set``. The decomposition of each operator will then stop once it reaches the gates in the ``gate_set`` or the
  ``stopping_condition`` is satisfied.

  .. code-block:: python

    import pennylane as qml
    from functools import partial

    @partial(qml.transforms.decompose, gate_set={"H", "T", "CNOT"}, stopping_condition=lambda op: len(op.wires) <= 2)
    @qml.qnode(qml.device("default.qubit"))
    def circuit():
        qml.Hadamard(wires=[0])
        qml.Toffoli(wires=[0,1,2])
        return qml.expval(qml.Z(0))

  >>> print(qml.draw(circuit)())
  0: ──H────────╭●───────────╭●────╭●──T──╭●─┤  <Z>
  1: ────╭●─────│─────╭●─────│───T─╰X──T†─╰X─┤
  2: ──H─╰X──T†─╰X──T─╰X──T†─╰X──T──H────────┤

* The :attr:`pennylane.operation.Operator.is_hermitian` property has been removed and replaced 
  with :attr:`pennylane.operation.Operator.is_verified_hermitian` as it better reflects the functionality of this property.
  Alternatively, consider using the :func:`pennylane.is_hermitian` function instead as it provides a more reliable check for hermiticity.
  Please be aware that it comes with a higher computational cost.

  - Deprecated in v0.44
  - Removed in v0.45

* The ``QuantumScript.to_openqasm`` method is deprecated and will be removed in version v0.44.
  Instead, the ``qml.to_openqasm`` function should be used.

  - Deprecated in v0.43
  - Removed in v0.44

* Providing ``num_steps`` to :func:`pennylane.evolve`, :func:`pennylane.exp`, :class:`pennylane.ops.Evolution`,
  and :class:`pennylane.ops.Exp` has been disallowed. Instead, use :class:`~.TrotterProduct` for approximate
  methods, providing the ``n`` parameter to perform the Suzuki-Trotter product approximation of a Hamiltonian
  with the specified number of Trotter steps.

  - Deprecated in v0.43
  - Removed in v0.44

  As a concrete example, consider the following case:

  .. code-block:: python

    coeffs = [0.5, -0.6]
    ops = [qml.X(0), qml.X(0) @ qml.Y(1)]
    H_flat = qml.dot(coeffs, ops)

  Instead of computing the Suzuki-Trotter product approximation as:

  >>> qml.evolve(H_flat, num_steps=2).decomposition()
  [RX(0.5, wires=[0]),
  PauliRot(-0.6, XY, wires=[0, 1]),
  RX(0.5, wires=[0]),
  PauliRot(-0.6, XY, wires=[0, 1])]

  The same result can be obtained using :class:`~.TrotterProduct` as follows:

  >>> decomp_ops = qml.adjoint(qml.TrotterProduct(H_flat, time=1.0, n=2)).decomposition()
  >>> [simp_op for op in decomp_ops for simp_op in map(qml.simplify, op.decomposition())]
  [RX(0.5, wires=[0]),
  PauliRot(-0.6, XY, wires=[0, 1]),
  RX(0.5, wires=[0]),
  PauliRot(-0.6, XY, wires=[0, 1])]

* ``qml.qnn.cost.SquaredErrorLoss`` has been removed. Instead, this hybrid workflow can be accomplished 
  with a function like ``loss = lambda *args: (circuit(*args) - target)**2``.

  - Deprecated in v0.43
  - Removed in v0.44

* Access to ``add_noise``, ``insert`` and noise mitigation transforms from the ``pennylane.transforms`` module has been removed.
  Instead, these functions should be imported from the ``pennylane.noise`` module.	

  - Deprecated in v0.43	
  - Removed in v0.44

* Specifying the ``work_wire_type`` argument in ``qml.ctrl`` and other controlled operators as ``"clean"`` or 
  ``"dirty"`` is disallowed. Use ``"zeroed"`` to indicate that the work wires are initially in the :math:`|0\rangle`
  state, and ``"borrowed"`` to indicate that the work wires can be in any arbitrary state. In both cases, the
  work wires are restored to their original state upon completing the decomposition.

  - Deprecated in v0.43
  - Removed in v0.44

* Some unnecessary methods of the ``qml.CircuitGraph`` class have been removed:

    - ``print_contents`` in favor of ``print(obj)``
    - ``observables_in_order`` in favor of ``observables``
    - ``operations_in_order`` in favor of ``operations``
    - ``ancestors_in_order(obj)`` in favor of ``ancestors(obj, sort=True)``
    - ``descendants_in_order(obj)`` in favor of ``descendants(obj, sort=True)``

  - Deprecated in v0.43
  - Removed in v0.44

* The ``qml.QNode.add_transform`` method is removed.
  Instead, please use ``QNode.transform_program.push_back(transform_container=transform_container)``.

  - Deprecated in v0.43
  - Removed in v0.44

* The value ``None`` has been removed as a valid argument to the ``level`` parameter in the
  :func:`pennylane.workflow.get_transform_program`, :func:`pennylane.workflow.construct_batch`,
  :func:`pennylane.draw`, :func:`pennylane.draw_mpl`, and :func:`pennylane.specs` transforms.
  Please use ``level='device'`` instead to apply the transform at the device level.

  - Deprecated in v0.43
  - Removed in v0.44

* ``pennylane.devices.DefaultExecutionConfig`` has been removed. Instead, use ``qml.devices.ExecutionConfig()``
  to create a default execution configuration.

  - Deprecated in v0.43
  - Removed in v0.44

* ``MeasurementProcess.expand`` is removed. The relevant method can be replaced with 
  ``qml.tape.QuantumScript(mp.obs.diagonalizing_gates(), [type(mp)(eigvals=mp.obs.eigvals(), wires=mp.obs.wires)])``.
  
  - Deprecated in v0.43
  - Removed in v0.44

* The ``QuantumScript.shape`` and ``QuantumScript.numeric_type`` properties are removed.
  Instead, the corresponding ``.shape`` or ``.numeric_type`` of the ``MeasurementProcess`` class should be used.

  - Deprecated in v0.43
  - Removed in v0.44

* The boolean functions provided by ``pennylane.operation`` have been removed. See below for an example of alternative code to use.
  These include ``not_tape``, ``has_gen``, ``has_grad_method``,  ``has_multipar``, ``has_nopar``, ``has_unitary_gen``,
  ``is_measurement``, ``defines_diagonalizing_gates``, and ``gen_is_multi_term_hamiltonian``.

  - Deprecated in v0.42
  - Removed in v0.43

.. code-block:: python

    from pennylane.operation import TermsUndefinedError, Operator

    def not_tape(obj):
        return not isinstance(obj, qml.tape.QuantumScript)

    def has_gen(obj):
        return obj.has_generator

    def has_grad_method(obj):
        return obj.grad_method is not None

    def has_multipar(obj):
        return obj.num_params > 1

    def has_nopar(obj):
        return obj.num_params == 0

    def has_unitary_gen(obj):
        return obj in qml.ops.qubit.attributes.has_unitary_generator

    def is_measurement(obj):
        return isinstance(obj, qml.measurements.MeasurementProcess)

    def defines_diagonalizing_gates(obj):
        return obj.has_diagonalizing_gates

    def gen_is_multi_term_hamiltonian(obj):
        if not isinstance(obj, Operator) or not obj.has_generator:
            return False
        try:
            generator = obj.generator()
            _, ops = generator.terms()
            return len(ops) > 1
        except TermsUndefinedError:
            return False

* ``lie_closure``, ``structure_constants``, and ``center`` can no longer be accessed via ``qml.pauli``. Top level import and usage is advised. They now live in
  the ``liealg`` module.

  .. code-block:: python

    import pennylane.liealg
    from pennylane.liealg import lie_closure, structure_constants, center

  - Deprecated in v0.40
  - Removed in v0.43

* ``qml.operation.Observable`` has been removed. To indicate that an operator is an observable, operator developers should manually set
  ``is_hermitian = True`` and update the ``queue`` function to stop it from being
  processed into the circuit.

  .. code-block:: python

      class MyObs(Operator):
          is_hermitian = True

          def queue(self, context=qml.QueuingManager):
              return self

  To check if an operator is likely to be hermitian, the ``op.is_hermitian`` property can be checked.

  Instead of ``Observable.compare``, ``qml.equal`` and ``op1 == op2`` should be used to compare instances.

  - Deprecated in v0.42
  - Removed in v0.43

* ``qml.operation.WiresEnum``, ``qml.operation.AllWires``, and ``qml.operation.AnyWires`` have been removed. If an operation can act
  on any number of wires, ``Operator.num_wires = None`` should be used instead. This is the default, and does not need
  to be overridden unless the operator developer wants to validate that the correct number of wires is passed.

  - Deprecated in v0.42
  - Removed in v0.43

* The ``qml.QNode.get_gradient_fn`` method has been removed. Instead, use :func:`~.workflow.get_best_diff_method` to obtain the differentiation method.

  - Deprecated in v0.42
  - Removed in v0.43

* Top-level access to ``DeviceError``, ``PennyLaneDeprecationWarning``, ``QuantumFunctionError`` and ``ExperimentalWarning`` 
  is now removed in v0.43. Please import these objects from the new ``pennylane.exceptions`` module.

  - Deprecated in v0.42
  - Removed in v0.43

* Specifying gradient keyword arguments as any additional keyword argument to the qnode is now removed in v0.42.
  The gradient keyword arguments should be passed to the new keyword argument ``gradient_kwargs`` via an explicit 
  dictionary, like ``gradient_kwargs={"h": 1e-4}``.

  - Deprecated in v0.41
  - Removed in v0.42

* The ``return_type`` property of ``MeasurementProcess`` has been removed.
  If observable type checking is needed, please use ``isinstance`` instead.

  - Deprecated in v0.41
  - Removed in v0.42

* The ``KerasLayer`` class in ``qml.qnn.keras`` has been removed because Keras 2 is no longer actively maintained.
  Please consider using a different machine learning framework, like `PyTorch <demos/tutorial_qnn_module_torch>`_ 
  or `JAX <demos/tutorial_How_to_optimize_QML_model_using_JAX_and_Optax>`_.

  - Deprecated in v0.41
  - Removed in v0.42

* The ``qml.gradients.hamiltonian_grad`` function has been removed because this gradient recipe is no
  longer required with the :doc:`new operator arithmetic system </news/new_opmath>`.

  - Deprecated in v0.41
  - Removed in v0.42

* Accessing terms of a tensor product (e.g., ``op = X(0) @ X(1)``) via ``op.obs`` has been removed.
  A user should use :class:`op.operands <~.CompositeOp>` instead.

  - Deprecated in v0.36
  - Removed in v0.42

* The ``mcm_config`` keyword argument to ``qml.execute`` has been removed.
  Instead, use the ``mcm_method`` and ``postselect_mode`` arguments.

  - Deprecated in v0.41
  - Removed in v0.42

* The ``inner_transform`` and ``config`` keyword arguments in ``qml.execute`` have been removed.
  If more detailed control over the execution is required, use ``qml.workflow.run`` with these arguments instead.

  - Deprecated in v0.41
  - Removed in v0.42

* ``op.ops`` and ``op.coeffs`` for ``Sum`` and ``Prod`` have been removed. Instead, please use
  :meth:`~.Operator.terms`.

  - Deprecated in v0.35
  - Removed in v0.42

* Specifying ``pipeline=None`` with ``qml.compile`` has been removed. 
  A sequence of transforms should now always be specified.

  - Deprecated in v0.41
  - Removed in v0.42

* The ``control_wires`` argument in the ``qml.ControlledQubitUnitary`` class has been removed. 
  Instead, please use the ``wires`` argument.

  - Deprecated in v0.41
  - Removed in v0.42

* The ``ControlledQubitUnitary`` no longer accepts `QubitUnitary` objects as arguments as its ``base``. 
  Instead, use ``qml.ctrl`` to construct a controlled `QubitUnitary`.

  - Deprecated in v0.41
  - Removed in v0.42  

* ``MultiControlledX`` no longer accepts strings as control values.

  - Deprecated in v0.36
  - Removed in v0.41

* The input argument ``control_wires`` of ``MultiControlledX`` has been removed.

  - Deprecated in v0.22
  - Removed in v0.41

* The ``decomp_depth`` argument in :func:`~pennylane.transforms.set_decomposition` has been removed. 

  - Deprecated in v0.40
  - Removed in v0.41

* The ``max_expansion`` argument in :func:`~pennylane.devices.preprocess.decompose` has been removed. 

  - Deprecated in v0.40
  - Removed in v0.41

* The ``tape`` and ``qtape`` properties of ``QNode`` have been removed. 
  Instead, use the ``qml.workflow.construct_tape`` function.

  - Deprecated in v0.40
  - Removed in v0.41

* The ``gradient_fn`` keyword argument to ``qml.execute`` has been removed. Instead, it has been replaced with ``diff_method``.

  - Deprecated in v0.40
  - Removed in v0.41

* The ``QNode.get_best_method`` and ``QNode.best_method_str`` methods have been removed. 
  Instead, use the ``qml.workflow.get_best_diff_method`` function. 

  - Deprecated in v0.40
  - Removed in v0.41

* The ``output_dim`` property of ``qml.tape.QuantumScript`` has been removed. Instead, use method ``shape`` of ``QuantumScript`` or ``MeasurementProcess`` to get the same information.

  - Deprecated in v0.40
  - Removed in v0.41

* The ``qml.qsvt_legacy`` function has been removed.
  Instead, use ``qml.qsvt``. The new functionality takes an input polynomial instead of angles.

  - Deprecated in v0.40
  - Removed in v0.41

* The ``qml.qinfo`` module has been removed. Please see the respective functions in the ``qml.math`` and ``qml.measurements``
  modules instead.

  - Deprecated in v0.39
  - Removed in v0.40

* Top level access to ``Device``, ``QubitDevice``, and ``QutritDevice`` have been removed. Instead, they
  are available as ``qml.devices.LegacyDevice``, ``qml.devices.QubitDevice``, and ``qml.devices.QutritDevice``
  respectively.

  - Deprecated in v0.39
  - Removed in v0.40

* The :class:`~pennylane.BasisStatePreparation` template has been removed.
  Instead, use :class:`~pennylane.BasisState`.

  - Deprecated in v0.39
  - Removed in v0.40


* The ``qml.QubitStateVector`` template has been removed. Instead, use :class:`~pennylane.StatePrep`.

  - Deprecated in v0.39
  - Removed in v0.40

* ``qml.broadcast`` has been removed. Users should use ``for`` loops instead.

  - Deprecated in v0.39
  - Removed in v0.40

* The ``max_expansion`` argument for :func:`~pennylane.transforms.decompositions.clifford_t_decomposition`
  has been removed.

  - Deprecated in v0.39
  - Removed in v0.40

* The ``'ancilla'`` argument for :func:`~pennylane.iterative_qpe` has been removed. Instead, use the ``'aux_wire'``
  argument.

  - Deprecated in v0.39
  - Removed in v0.40

* The ``expand_depth`` argument for :func:`~pennylane.transforms.compile` has been removed.

  - Deprecated in v0.39
  - Removed in v0.40

* The ``qml.workflow.set_shots`` helper function has been removed. We no longer interact with the legacy device interface in our code.
  Instead, shots should be specified on the tape, and the device should use these shots.

  - Deprecated in v0.38
  - Removed in v0.40

* ``QNode.gradient_fn`` is removed. Please use ``QNode.diff_method`` instead. ``QNode.get_gradient_fn`` can also be used to
  process the diff method.

  - Deprecated in v0.39
  - Removed in v0.40

* The ``qml.shadows.shadow_expval`` transform has been removed. Instead, please use the
  ``qml.shadow_expval`` measurement process.

  - Deprecated in v0.39
  - Removed in v0.40

* PennyLane Lightning and Catalyst will no longer support ``manylinux2014`` (GLIBC 2.17) compatibile Linux operating systems, and will be migrated to ``manylinux_2_28`` (GLIBC 2.28). See `pypa/manylinux <https://github.com/pypa/manylinux>`_ for additional details.

  - Last supported version of ``manylinux2014`` with v0.36
  - Fully migrated to ``manylinux_2_28`` with v0.37

* The ``simplify`` argument in ``qml.Hamiltonian`` and ``qml.ops.LinearCombination`` has been removed.
  Instead, ``qml.simplify()`` can be called on the constructed operator.

  - Deprecated in v0.37
  - Removed in v0.39

* The ``decomp_depth`` argument in ``qml.device`` is removed.

  - Deprecated in v0.38
  - Removed in v0.39

* The functions ``qml.qinfo.classical_fisher`` and ``qml.qinfo.quantum_fisher`` have been removed and migrated to the ``qml.gradients``
  module. Therefore, ``qml.gradients.classical_fisher`` and ``qml.gradients.quantum_fisher`` should be used instead.

  - Deprecated in v0.38
  - Removed in v0.39

* All of the legacy devices (any with the name ``default.qubit.{autograd,torch,tf,jax,legacy}``) are removed. Use ``default.qubit`` instead,
  as it supports backpropagation for the many backends the legacy devices support.

  - Deprecated in v0.38
  - Removed in v0.39

* The logic for internally switching a device for a different backpropagation
  compatible device is removed, as it was in place for removed ``default.qubit.legacy``.

  - Deprecated in v0.38
  - Removed in v0.39

* `Operator.expand` is now removed. Use `qml.tape.QuantumScript(op.decomposition())` instead.

  - Deprecated in v0.38
  - Removed in v0.39

* The ``expansion_strategy`` attribute of ``qml.QNode`` is removed.
  Users should make use of ``qml.workflow.construct_batch``, should they require fine control over the output tape(s).

  - Deprecated in v0.38
  - Removed in v0.39

* The ``expansion_strategy`` argument in ``qml.specs``, ``qml.draw``, and ``qml.draw_mpl`` is removed. 
  Instead, use the ``level`` argument which provides a superset of options.

  - Deprecated in v0.38
  - Removed in v0.39

* The ``max_expansion`` argument in ``qml.QNode`` is removed.

  - Deprecated in v0.38
  - Removed in v0.39

* The ``expand_fn`` argument in ``qml.execute`` is removed.
  Instead, please create a ``qml.transforms.core.TransformProgram`` with the desired preprocessing and pass it to the ``transform_program`` argument of ``qml.execute``.

  - Deprecated in v0.38
  - Removed in v0.39

* The ``max_expansion`` argument in ``qml.execute`` is removed.
  Instead, please use ``qml.devices.preprocess.decompose`` with the desired expansion level, add it to a ``TransformProgram``, and pass it to the ``transform_program`` argument of ``qml.execute``.

  - Deprecated in v0.38
  - Removed in v0.39

* The ``override_shots`` argument in ``qml.execute`` is removed.
  Instead, please add the shots to the ``QuantumTape``\ s to be executed.

  - Deprecated in v0.38
  - Removed in v0.39

* The ``device_batch_transform`` argument in ``qml.execute`` is removed.
  Instead, please create a ``qml.transforms.core.TransformProgram`` with the desired preprocessing and pass it to the ``transform_program`` argument of ``qml.execute``.

  - Deprecated in v0.38
  - Removed in v0.39

* The functions ``qml.transforms.sum_expand`` and ``qml.transforms.hamiltonian_expand`` are removed.
  Instead, ``qml.transforms.split_non_commuting`` can be used for equivalent behaviour.

  - Deprecated in v0.38
  - Removed in v0.39

* ``queue_idx`` attribute has been removed from the ``Operator``, ``CompositeOp``, and ``SymboliOp`` classes. Instead, the index is now stored as the label of the ``CircuitGraph.graph`` nodes.

  - Deprecated in v0.38
  - Removed in v0.38

* ``qml.from_qasm`` no longer removes measurements from the QASM code. Use 
  ``measurements=[]`` to remove measurements from the original circuit.

  - Deprecated in v0.37
  - Default behaviour changed in v0.38

* ``qml.transforms.map_batch_transform`` has been removed, since transforms can be applied directly to a batch of tapes.
  See :func:`~.pennylane.transform` for more information.

  - Deprecated in v0.37
  - Removed in v0.38

* ``qml.from_qasm_file`` has been removed. Instead, the user can open the file and then load its content using ``qml.from_qasm``.

  >>> with open("test.qasm", "r") as f:
  ...     circuit = qml.from_qasm(f.read())

  - Deprecated in v0.36
  - Removed in v0.37

* The ``qml.load`` function is a general-purpose way to convert circuits into PennyLane from other
  libraries. It has been removed in favour of the more specific functions ``from_qiskit``, ``from_qasm``, etc.

  - Deprecated in v0.36
  - Removed in v0.37

* ``single_tape_transform``, ``batch_transform``, ``qfunc_transform``, ``op_transform``,
  ``gradient_transform`` and ``hessian_transform`` are deprecated. Instead switch to using the new
  ``qml.transform`` function. Please refer to
  `the transform docs <https://docs.pennylane.ai/en/stable/code/qml_transforms.html#custom-transforms>`_
  to see how this can be done.

  - Deprecated in v0.34
  - Removed in v0.36

* ``PauliWord`` and ``PauliSentence`` no longer use ``*`` for matrix and tensor products,
  but instead use ``@`` to conform with the PennyLane convention.

  - Deprecated in v0.35
  - Removed in v0.36

* The private functions ``_pauli_mult``, ``_binary_matrix`` and ``_get_pauli_map`` from the
  ``pauli`` module have been removed, as they are no longer used anywhere and the same
  functionality can be achieved using newer features in the ``pauli`` module.

  - Deprecated in v0.35
  - Removed in v0.36

* Calling ``qml.matrix`` without providing a ``wire_order`` on objects where the wire order could be
  ambiguous now raises an error. This includes tapes with multiple wires, QNodes with a device that
  does not provide wires, or quantum functions.

  - Deprecated in v0.35
  - Raises an error in v0.36

* ``qml.pauli.pauli_mult`` and ``qml.pauli.pauli_mult_with_phase`` are now removed. Instead, you
  should use ``qml.simplify(qml.prod(pauli_1, pauli_2))`` to get the reduced operator.

  >>> op = qml.simplify(qml.prod(qml.PauliX(0), qml.PauliZ(0)))
  >>> op
  -1j*(PauliY(wires=[0]))
  >>> [phase], [base] = op.terms()
  >>> phase, base
  (-1j, PauliY(wires=[0]))

  - Deprecated in v0.35
  - Removed in v0.36

* ``MeasurementProcess.name`` and ``MeasurementProcess.data`` have been removed, as they contain
  dummy values that are no longer needed.

  - Deprecated in v0.35
  - Removed in v0.36

* The contents of ``qml.interfaces`` is moved inside ``qml.workflow``.

  - Contents moved in v0.35
  - Old import path removed in v0.36

* The method ``Operator.validate_subspace(subspace)``, only employed under a specific set of qutrit
  operators, has been relocated to the ``qml.ops.qutrit.parametric_ops`` module and has been removed
  from the ``Operator`` class.

  - Deprecated in v0.35
  - Removed in v0.36

* ``qml.transforms.one_qubit_decomposition`` and ``qml.transforms.two_qubit_decomposition`` are removed. Instead,
  you should use ``qml.ops.one_qubit_decomposition`` and ``qml.ops.two_qubit_decomposition``.

  - Deprecated in v0.34
  - Removed in v0.35

* Passing additional arguments to a transform that decorates a QNode should now be done through use
  of ``functools.partial``. For example, the :func:`~pennylane.metric_tensor` transform has an
  optional ``approx`` argument which should now be set using:

  .. code-block:: python

    from functools import partial

    @partial(qml.metric_tensor, approx="block-diag")
    @qml.qnode(dev)
    def circuit(weights):
        ...

  The previously-recommended approach is now removed:

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
  - Removed in v0.35

* ``Observable.return_type`` has been removed. Instead, you should inspect the type
  of the surrounding measurement process.

  - Deprecated in v0.34
  - Removed in v0.35

* ``ClassicalShadow.entropy()`` no longer needs an ``atol`` keyword as a better
  method to estimate entropies from approximate density matrix reconstructions
  (with potentially negative eigenvalues) has been implemented.

  - Deprecated in v0.34
  - Removed in v0.35

* ``QuantumScript.is_sampled`` and ``QuantumScript.all_sampled`` have been removed.
  Users should now validate these properties manually.

  .. code-block:: python

    from pennylane.measurements import *
    sample_types = (SampleMP, CountsMP, ClassicalShadowMP, ShadowExpvalMP)
    is_sample_type = [isinstance(m, sample_types) for m in tape.measurements]
    is_sampled = any(is_sample_type)
    all_sampled = all(is_sample_type)

  - Deprecated in v0.34
  - Removed in v0.35

* ``qml.ExpvalCost`` has been removed. Users should use ``qml.expval()`` instead.

  .. code-block:: python

    @qml.qnode(dev)
    def cost_function(params):
        some_qfunc(params)
        return qml.expval(Hamiltonian)

  - Deprecated in v0.24
  - Removed in v0.35

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
