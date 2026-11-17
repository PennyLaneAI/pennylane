.. role:: html(raw)
   :format: html

.. _intro_coming_from_classic:

Coming from PennyLane Classic
=============================

The architecture underpinning the singular, unified PennyLane2 pipeline is
responsible for performance improvements and new features, and aligns with
necessities for scalable fault-tolerant quantum computation on real hardware.
While PennyLane2 has a very similar user-interface compared to PennyLane Classic
(v0.45.1), some deviations are required. This page will go over those details.

.. note::

    PennyLane Classic means ``pennylane==0.45.1``.

Differences at a glance
-----------------------

.. list-table::
    :header-rows: 1
    :widths: 20 40 40
    :stub-columns: 1
    :class: comparison-table

    * -
      - PennyLane Classic (v0.45.1)
      - PennyLane2
    * - Running a circuit
      - Call a QNode directly.
      - Decorate the QNode, or a function that calls QNodes and other classical
        processing within it, with :func:`@qp.qjit <pennylane.qjit>`.
    * - Inspecting circuits
      - Drawing with ``draw/draw_mpl``, resource analysis with ``specs``, and
        resource estimation in ``labs.estimator_beta``.
      - Resource estimates with :func:`~pennylane.estimate`, analysis with
        :func:`~pennylane.analyze`, and runtime tracking with
        :func:`~pennylane.track`. Drawing with ``draw/draw_mpl`` and resource
        estimation with ``labs.estimator_beta`` is not supported.
    * - Numerical arrays
      - ``pennylane.numpy`` (Autograd), PyTorch, and JAX (``jax.numpy``).
      - JAX (``jax.numpy``) and NumPy.
    * - Compilation passes
      - Based on the "tape" representation (a straight line / list of
        instructions describing the circuit), including conversions to a
        directed acyclical graph, the ZX calculus representation, and more.
      - :doc:`Catalyst compiler passes <catalyst:dev/transforms>` written in
        MLIR. See :doc:`compiling_circuits` for more information.
    * - Circuit decomposition
      - Each operator defines one decomposition with ``compute_decomposition``,
        integrating with the ``decompose`` transform. The ``decompose``
        transform also supports stopping conditions.
      - Each operator can define many decomposition rules, integrating with a
        :func:`graph-based decomposition framework <pennylane.transforms.decompose>`
        that solves for the most resource efficient decomposition pathway. No
        support for stopping conditions.
    * - Custom operators
      - Subclass :class:`~pennylane.core.operator.Operation`, define a matrix and/or
        ``compute_decomposition``.
      - Use :func:`~pennylane.subcircuit`, or subclass
        :class:`~pennylane.core.operator.Operator2`, define many decomposition
        rules via :func:`~.add_decomps`.
    * - Supported devices
      - Python devices such as ``default.qubit``, plugins such as
        PennyLane-Qiskit, and more.
      - Only :doc:`devices that Catalyst supports <catalyst:dev/devices>`, such
        as ``lightning.qubit``, ``lightning.kokkos`` and ``lightning.gpu``.
        Devices must be given ``wires``. New plugin devices must follow
        :doc:`/development/plugins`.
    * - Control flow
      - Python ``for``, ``while`` and ``if/elif/else``.
      - :func:`~pennylane.for_loop`, :func:`~pennylane.while_loop` and
        :func:`~pennylane.cond`, or Python control flow with
        ``@qp.qjit(autograph=True)``. Note that in both of those cases, program
        structure is preserved.
    * - Gradients
      - Gradient methods and optimizers that interface with JAX and PyTorch.
      - Differentiation with :func:`~pennylane.grad` and
        :func:`~pennylane.jacobian` must happen inside :func:`~.qjit`,
        parameter-shift, finite-difference, and adjoint gradient methods are
        supported on the quantum parts of the program, and optimization *must*
        occur via `Optax <https://optax.readthedocs.io>`__.

Defining circuits
-----------------

Here is a typical PennyLane Classic circuit:

.. code-block:: python

    import pennylane as qp
    import numpy as np

    dev = qp.device("default.qubit")

    @qp.set_shots(1000)
    @qp.qnode(dev)
    def circuit(params, n_layers):

        for i in range(n_layers):
            qp.RX(params[0], wires=0)
            qp.CNOT(wires=[0, 1])

        if n_layers > 2:
            qp.RY(params[1], wires=1)

        return qp.expval(qp.Z(1))

    params = np.array([0.1, 0.2])
    result = circuit(params, 3)

And here is the same workflow in PennyLane2:

.. code-block:: python

    import pennylane as qp
    from jax import numpy as jnp

    dev = qp.device("lightning.qubit", wires=2)

    @qp.qjit(autograph=True)
    @qp.qnode(dev, shots=1000)
    def circuit(params, n_layers):

        for i in range(n_layers):
            qp.RX(params[0], wires=0)
            qp.CNOT(wires=[0, 1])

        if n_layers > 2:
            qp.RY(params[1], wires=1)

        return qp.expval(qp.Z(1))

    params = jnp.array([0.1, 0.2])
    result = circuit(params, 3)

The changes are:

#. Decorate the QNode with :func:`pennylane.qjit`.

#. Add ``autograph=True`` so that Python control flow can be written (``for``,
   ``while``, ``if/elif/else``) and program structure is still preserved (loops
   are not unrolled).

#. Choose a :doc:`supported device <catalyst:dev/devices>` and give it
   ``wires``.

#. Use ``jax.numpy`` or regular NumPy for arrays (for more information on which
   data type to use and when, see :doc:`qjit`).

Taking gradients
----------------

Here is a simple PennyLane Classic circuit that we take the gradient of:

.. code-block:: python

  import pennylane as qp
  import pennylane.numpy as np

  @qp.qnode(qp.device("default.qubit"))
  def circuit(angles):
      qp.RX(angles[0], wires=0)
      qp.RY(angles[1], wires=1)
      qp.RZ(angles[2], wires=0)
      return qp.expval(qp.Z(0) + qp.X(1))

.. code-block:: pycon

  >>> angles = np.array([0.2, 0.9, -1.4], requires_grad=True)
  >>> qp.grad(circuit)(angles)
  [-1.98669331e-01  6.21609968e-01 -2.71082344e-17]

In PennyLane2, calls to PennyLane2 differentiation methods must happen *within*
``qjit``.

.. code-block:: python

  import pennylane as qp
  import jax.numpy as jnp

  @qp.qnode(qp.device("lightning.qubit", wires=2))
  def circuit(angles):
      qp.RX(angles[0], wires=0)
      qp.RY(angles[1], wires=1)
      qp.RZ(angles[2], wires=0)
      return qp.expval(qp.Z(0) + qp.X(1))

  @qp.qjit
  def f(angles):
      grad = qp.grad(circuit)
      return grad(angles)

.. code-block:: pycon

  >>> angles = jnp.array([0.2, 0.9, -1.4])
  >>> f(angles)
  [-1.98669331e-01  6.21609968e-01  1.16928756e-16]

Alternatively, JAX differentiation methods can be called *outside* of ``qjit``.

.. code-block:: pycon

  >>> import jax
  >>> jax.grad(circuit)(angles)
  [-1.98669331e-01  6.21609968e-01  1.16928756e-16]

Optimizing circuits
-------------------

Here is a simple circuit that is optimized with ``qp.AdamOptimizer`` from
PennyLane Classic:

.. code-block:: python

  import pennylane as qp
  import pennylane.numpy as np

  @qp.qnode(qp.device("default.qubit"))
  def circuit(angles):
      qp.RX(angles[0], wires=0)
      qp.RY(angles[1], wires=1)
      qp.RZ(angles[2], wires=0)
      return qp.expval(qp.Z(0) + qp.X(1))

  angles = np.array([0.2, 0.9, -1.4], requires_grad=True)
  opt = qp.AdamOptimizer(stepsize=0.1)

  for _ in range(10):
      angles = opt.step(circuit, angles)

.. code-block:: pycon

  >>> circuit(angles)
  tensor(1.67218462, requires_grad=True)

With PennyLane2, optimization must happen with optimizers provided by the
`Optax <https://optax.readthedocs.io>`__ package, where it is often more
performant if the entire optimization loop is placed *within* ``qjit``.

.. code-block:: python

  import pennylane as qp
  import jax.numpy as jnp
  import jax
  import optax

  opt = optax.adam(0.01)
  angles = jnp.array([0.2, 0.9, -1.4])

  def update_step(args):
      angles, opt_state = args
      grads = qp.grad(circuit)(angles)
      updates, opt_state = opt.update(grads, opt_state)
      angles = optax.apply_updates(angles, updates)
      return (angles, opt_state)

  @qp.qjit
  def optimize(angles):
      opt_state = opt.init(angles)

      for i in range(10):
          angles, opt_state = update_step((angles, opt_state))

      return angles

.. code-block:: pycon

  >>> angles = optimize(angles)
  >>> circuit(angles)
  1.672125640350707

For more information on optimization with Optax, check out our demo titled
:doc:`demo:demos/tutorial_How_to_optimize_QML_model_using_JAX_catalyst_and_Optax`.

Transforms and compilation passes
---------------------------------

A complete list of supported transforms in PennyLane2 can be found in
:doc:`compiling_circuits`. The following transforms from PennyLane Classic are
*not* compatible with PennyLane2, including custom transforms implemented with
``pennylane.transform``.

.. details::
    :title: PennyLane Classic transforms and compilation passes not compatible with PennyLane2

    .. hlist::
        :columns: 3

        * `add_noise <https://docs.pennylane.ai/en/stable/code/api/pennylane.add_noise.html>`__
        * `adjoint_metric_tensor <https://docs.pennylane.ai/en/stable/code/api/pennylane.adjoint_metric_tensor.html>`__
        * `apply_controlled_Q <https://docs.pennylane.ai/en/stable/code/api/pennylane.apply_controlled_Q.html>`__
        * `batch_input <https://docs.pennylane.ai/en/stable/code/api/pennylane.batch_input.html>`__
        * `batch_params <https://docs.pennylane.ai/en/stable/code/api/pennylane.batch_params.html>`__
        * `broadcast_expand <https://docs.pennylane.ai/en/stable/code/api/pennylane.transforms.broadcast_expand.html>`__
        * `circuit_spectrum <https://docs.pennylane.ai/en/stable/code/api/pennylane.fourier.circuit_spectrum.html>`__
        * `clifford_t_decomposition <https://docs.pennylane.ai/en/stable/code/api/pennylane.clifford_t_decomposition.html>`__
        * `commutation_dag <https://docs.pennylane.ai/en/stable/code/api/pennylane.commutation_dag.html>`__
        * `commute_controlled <https://docs.pennylane.ai/en/stable/code/api/pennylane.transforms.commute_controlled.html>`__
        * `compile <https://docs.pennylane.ai/en/stable/code/api/pennylane.compile.html>`__
        * `convert_to_mbqc_formalism <https://docs.pennylane.ai/en/stable/code/api/pennylane.ftqc.convert_to_mbqc_formalism.html>`__
        * `convert_to_mbqc_gateset <https://docs.pennylane.ai/en/stable/code/api/pennylane.ftqc.convert_to_mbqc_gateset.html>`__
        * `convert_to_numpy_parameters <https://docs.pennylane.ai/en/stable/code/api/pennylane.transforms.convert_to_numpy_parameters.html>`__
        * `cut_circuit <https://docs.pennylane.ai/en/stable/code/api/pennylane.cut_circuit.html>`__
        * `cut_circuit_mc <https://docs.pennylane.ai/en/stable/code/api/pennylane.cut_circuit_mc.html>`__
        * `decomp_inspector <https://docs.pennylane.ai/en/stable/code/api/pennylane.transforms.decomp_inspector.html>`__
        * `defer_measurements <https://docs.pennylane.ai/en/stable/code/api/pennylane.defer_measurements.html>`__
        * `diagonalize_mcms <https://docs.pennylane.ai/en/stable/code/api/pennylane.ftqc.diagonalize_mcms.html>`__
        * `diagonalize_measurements <https://docs.pennylane.ai/en/stable/code/api/pennylane.transforms.diagonalize_measurements.html>`__
        * `finite_diff <https://docs.pennylane.ai/en/stable/code/api/pennylane.gradients.finite_diff.html>`__
        * `fold_global <https://docs.pennylane.ai/en/stable/code/api/pennylane.noise.fold_global.html>`__
        * `hadamard_grad <https://docs.pennylane.ai/en/stable/code/api/pennylane.gradients.hadamard_grad.html>`__
        * `insert <https://docs.pennylane.ai/en/stable/code/api/pennylane.noise.insert.html>`__
        * `map_wires <https://docs.pennylane.ai/en/stable/code/api/pennylane.map_wires.html>`__
        * `match_controlled_iX_gate <https://docs.pennylane.ai/en/stable/code/api/pennylane.transforms.match_controlled_iX_gate.html>`__
        * `match_relative_phase_toffoli <https://docs.pennylane.ai/en/stable/code/api/pennylane.transforms.match_relative_phase_toffoli.html>`__
        * `merge_amplitude_embedding <https://docs.pennylane.ai/en/stable/code/api/pennylane.transforms.merge_amplitude_embedding.html>`__
        * `metric_tensor <https://docs.pennylane.ai/en/stable/code/api/pennylane.metric_tensor.html>`__
        * `mitigate_with_zne <https://docs.pennylane.ai/en/stable/code/api/pennylane.noise.mitigate_with_zne.html>`__
        * `optimize_t_count <https://docs.pennylane.ai/en/stable/code/api/pennylane.transforms.zx.optimize_t_count.html>`__
        * `param_shift <https://docs.pennylane.ai/en/stable/code/api/pennylane.gradients.param_shift.html>`__
        * `param_shift_cv <https://docs.pennylane.ai/en/stable/code/api/pennylane.gradients.param_shift_cv.html>`__
        * `param_shift_hessian <https://docs.pennylane.ai/en/stable/code/api/pennylane.gradients.param_shift_hessian.html>`__
        * `parity_matrix <https://docs.pennylane.ai/en/stable/code/api/pennylane.transforms.parity_matrix.html>`__
        * `pattern_matching_optimization <https://docs.pennylane.ai/en/stable/code/api/pennylane.pattern_matching_optimization.html>`__
        * `phase_polynomial <https://docs.pennylane.ai/en/stable/code/api/pennylane.transforms.phase_polynomial.html>`__
        * `pulse_odegen <https://docs.pennylane.ai/en/stable/code/api/pennylane.gradients.pulse_odegen.html>`__
        * `push_hadamards <https://docs.pennylane.ai/en/stable/code/api/pennylane.transforms.zx.push_hadamards.html>`__
        * `quantum_fisher <https://docs.pennylane.ai/en/stable/code/api/pennylane.gradients.quantum_fisher.html>`__
        * `quantum_monte_carlo <https://docs.pennylane.ai/en/stable/code/api/pennylane.quantum_monte_carlo.html>`__
        * `reduce_non_clifford <https://docs.pennylane.ai/en/stable/code/api/pennylane.transforms.zx.reduce_non_clifford.html>`__
        * `remove_barrier <https://docs.pennylane.ai/en/stable/code/api/pennylane.transforms.remove_barrier.html>`__
        * `resolve_dynamic_wires <https://docs.pennylane.ai/en/stable/code/api/pennylane.transforms.resolve_dynamic_wires.html>`__
        * `rowcol <https://docs.pennylane.ai/en/stable/code/api/pennylane.transforms.rowcol.html>`__
        * `rz_phase_gradient <https://docs.pennylane.ai/en/stable/code/api/pennylane.transforms.rz_phase_gradient.html>`__
        * `shadow_state <https://docs.pennylane.ai/en/stable/code/api/pennylane.shadows.shadow_state.html>`__
        * `sign_expand <https://docs.pennylane.ai/en/stable/code/api/pennylane.transforms.sign_expand.html>`__
        * `single_qubit_fusion <https://docs.pennylane.ai/en/stable/code/api/pennylane.transforms.single_qubit_fusion.html>`__
        * `snapshots <https://docs.pennylane.ai/en/stable/code/api/pennylane.snapshots.html>`__
        * `split_non_commuting <https://docs.pennylane.ai/en/stable/code/api/pennylane.transforms.split_non_commuting.html>`__
        * `spsa_grad <https://docs.pennylane.ai/en/stable/code/api/pennylane.gradients.spsa_grad.html>`__
        * `stoch_pulse_grad <https://docs.pennylane.ai/en/stable/code/api/pennylane.gradients.stoch_pulse_grad.html>`__
        * `to_zx <https://docs.pennylane.ai/en/stable/code/api/pennylane.transforms.to_zx.html>`__
        * `todd <https://docs.pennylane.ai/en/stable/code/api/pennylane.transforms.zx.todd.html>`__
        * `transpile <https://docs.pennylane.ai/en/stable/code/api/pennylane.transforms.transpile.html>`__
        * `undo_swaps <https://docs.pennylane.ai/en/stable/code/api/pennylane.transforms.undo_swaps.html>`__
        * `unitary_to_rot <https://docs.pennylane.ai/en/stable/code/api/pennylane.transforms.unitary_to_rot.html>`__

Resource profiling
------------------

TOOD

.. _classic_plugin_devices:

Creating devices
----------------

Devices from plugins built on the PennyLane Classic device API, such as
third-party hardware and simulator plugins, are not supported with PennyLane2.
If you intend on developing one or migrating an existing device to be compatible
with PennyLane2, see :doc:`/development/plugins`.

.. _classic_custom_operators:

Migrating custom operators
--------------------------

In PennyLane Classic, a custom operator is made by subclassing
``pennylane.core.operator.Operation`` and overriding methods such as
``compute_decomposition``. These operators are not compatible with PennyLane2.
In PennyLane2, operators are implemented in a way the ``qjit`` compiler
understands.

For more information on making an operator compatible with PennyLane2, consult
:doc:`/development/adding_operators`.

Module removals
---------------

The following modules of PennyLane Classic are not supported in PennyLane2.

.. list-table::
    :header-rows: 1
    :widths: 35 65
    :class: comparison-table

    * - Removed
      - What to use instead
    * - ``qp.qnn`` module, including ``TorchLayer``
      - ``qp.qnn.iqp_expval`` is now :func:`qp.math.iqp_expval
        <pennylane.math.iqp_expval>`
    * - TensorFlow or PyTorch interface
      - JAX
    * - ``qp.io`` module, including conversions to Qiskit, OpenQASM, Qualtran,
        and more
      - None
    * - ``qp.qaoa`` module: mixer and cost Hamiltonians, ``cost_layer``,
        ``mixer_layer`` and the ``cycle`` helpers
      - Build the Hamiltonians and layers yourself
        :class:`~pennylane.QAOAEmbedding` is unaffected.
    * - ``qp.pulse`` module, including ``ParametrizedHamiltonian``,
        ``ParametrizedEvolution``, ``stoch_pulse_grad`` and ``pulse_odegen``
      - None
    * - ``qp.fourier`` module
      - ``qp.fourier.reconstruct`` is now ``qp.optimize.reconstruct``
    * - ``qp.qcut`` module, including ``cut_circuit``, ``cut_circuit_mc`` and
        ``WireCut``
      - None
    * - Continuous-variable (CV) operators, observables, templates and
        devices, including ``default.gaussian`` and ``param_shift_cv``
      - None
    * - Qutrit and qudit operators and devices
      - None
    * - ``qp.resource.Resources``, ``ResourceOperator``, ``ErrorOperator`` and
        the ``qp.resource.error`` module
      - :func:`~pennylane.estimate`, :func:`~pennylane.analyze` and
        :func:`~pennylane.track`. See :doc:`resource_profiling`

.. _unsupported_classic_subroutines:

Unsupported subroutines
-----------------------

Relative to PennyLane Classic, the implementation of quantum operators in
PennyLane2 is quite different. This means that for an operator in PennyLane
Classic to be compatible with PennyLane2, a non-trivial migration is required.

The following PennyLane Classic operators have not been migrated to PennyLane2.
As operators are migrated, this list will be updated.

.. details::
    :title: PennyLane Classic operators not migrated to PennyLane2

    .. hlist::
        :columns: 3

        * `Adder <https://docs.pennylane.ai/en/stable/code/api/pennylane.Adder.html>`__
        * `AllSinglesDoubles <https://docs.pennylane.ai/en/stable/code/api/pennylane.AllSinglesDoubles.html>`__
        * `AmplitudeAmplification <https://docs.pennylane.ai/en/stable/code/api/pennylane.AmplitudeAmplification.html>`__
        * `AmplitudeDamping <https://docs.pennylane.ai/en/stable/code/api/pennylane.AmplitudeDamping.html>`__
        * `AmplitudeEmbedding <https://docs.pennylane.ai/en/stable/code/api/pennylane.AmplitudeEmbedding.html>`__
        * `AngleEmbedding <https://docs.pennylane.ai/en/stable/code/api/pennylane.AngleEmbedding.html>`__
        * `ApproxTimeEvolution <https://docs.pennylane.ai/en/stable/code/api/pennylane.ApproxTimeEvolution.html>`__
        * `ArbitraryStatePreparation <https://docs.pennylane.ai/en/stable/code/api/pennylane.ArbitraryStatePreparation.html>`__
        * `ArbitraryUnitary <https://docs.pennylane.ai/en/stable/code/api/pennylane.ArbitraryUnitary.html>`__
        * `BasicEntanglerLayers <https://docs.pennylane.ai/en/stable/code/api/pennylane.BasicEntanglerLayers.html>`__
        * `BBQRAM <https://docs.pennylane.ai/en/stable/code/api/pennylane.BBQRAM.html>`__
        * `BitFlip <https://docs.pennylane.ai/en/stable/code/api/pennylane.BitFlip.html>`__
        * `Channel <https://docs.pennylane.ai/en/stable/code/api/pennylane.operation.Channel.html>`__
        * `CommutingEvolution <https://docs.pennylane.ai/en/stable/code/api/pennylane.CommutingEvolution.html>`__
        * `ControlledSequence <https://docs.pennylane.ai/en/stable/code/api/pennylane.ControlledSequence.html>`__
        * `CosineWindow <https://docs.pennylane.ai/en/stable/code/api/pennylane.CosineWindow.html>`__
        * `CPhaseShift00 <https://docs.pennylane.ai/en/stable/code/api/pennylane.CPhaseShift00.html>`__
        * `CPhaseShift01 <https://docs.pennylane.ai/en/stable/code/api/pennylane.CPhaseShift01.html>`__
        * `CPhaseShift10 <https://docs.pennylane.ai/en/stable/code/api/pennylane.CPhaseShift10.html>`__
        * `DepolarizingChannel <https://docs.pennylane.ai/en/stable/code/api/pennylane.DepolarizingChannel.html>`__
        * `DoubleExcitation <https://docs.pennylane.ai/en/stable/code/api/pennylane.DoubleExcitation.html>`__
        * `DoubleExcitationMinus <https://docs.pennylane.ai/en/stable/code/api/pennylane.DoubleExcitationMinus.html>`__
        * `DoubleExcitationPlus <https://docs.pennylane.ai/en/stable/code/api/pennylane.DoubleExcitationPlus.html>`__
        * `DoubleFactorization <https://docs.pennylane.ai/en/stable/code/api/pennylane.estimator.qpe_resources.DoubleFactorization.html>`__
        * `Evolution <https://docs.pennylane.ai/en/stable/code/api/pennylane.ops.op_math.Evolution.html>`__
        * `Exp <https://docs.pennylane.ai/en/stable/code/api/pennylane.ops.op_math.Exp.html>`__
        * `FABLE <https://docs.pennylane.ai/en/stable/code/api/pennylane.FABLE.html>`__
        * `FermionicDoubleExcitation <https://docs.pennylane.ai/en/stable/code/api/pennylane.FermionicDoubleExcitation.html>`__
        * `FermionicSingleExcitation <https://docs.pennylane.ai/en/stable/code/api/pennylane.FermionicSingleExcitation.html>`__
        * `FermionicSWAP <https://docs.pennylane.ai/en/stable/code/api/pennylane.FermionicSWAP.html>`__
        * `FirstQuantization <https://docs.pennylane.ai/en/stable/code/api/pennylane.estimator.qpe_resources.FirstQuantization.html>`__
        * `FromBloq <https://docs.pennylane.ai/en/stable/code/api/pennylane.FromBloq.html>`__
        * `GateFabric <https://docs.pennylane.ai/en/stable/code/api/pennylane.GateFabric.html>`__
        * `GeneralizedAmplitudeDamping <https://docs.pennylane.ai/en/stable/code/api/pennylane.GeneralizedAmplitudeDamping.html>`__
        * `GraphStatePrep <https://docs.pennylane.ai/en/stable/code/api/pennylane.ftqc.GraphStatePrep.html>`__
        * `GroverOperator <https://docs.pennylane.ai/en/stable/code/api/pennylane.GroverOperator.html>`__
        * `Hermitian <https://docs.pennylane.ai/en/stable/code/api/pennylane.Hermitian.html>`__
        * `HilbertSchmidt <https://docs.pennylane.ai/en/stable/code/api/pennylane.HilbertSchmidt.html>`__
        * `HybridQRAM <https://docs.pennylane.ai/en/stable/code/api/pennylane.HybridQRAM.html>`__
        * `IntegerComparator <https://docs.pennylane.ai/en/stable/code/api/pennylane.IntegerComparator.html>`__
        * `IQPEmbedding <https://docs.pennylane.ai/en/stable/code/api/pennylane.IQPEmbedding.html>`__
        * `kUpCCGSD <https://docs.pennylane.ai/en/stable/code/api/pennylane.kUpCCGSD.html>`__
        * `LinearCombination <https://docs.pennylane.ai/en/stable/code/api/pennylane.ops.op_math.LinearCombination.html>`__
        * `LocalHilbertSchmidt <https://docs.pennylane.ai/en/stable/code/api/pennylane.LocalHilbertSchmidt.html>`__
        * `MERA <https://docs.pennylane.ai/en/stable/code/api/pennylane.MERA.html>`__
        * `ModExp <https://docs.pennylane.ai/en/stable/code/api/pennylane.ModExp.html>`__
        * `MottonenStatePreparation <https://docs.pennylane.ai/en/stable/code/api/pennylane.MottonenStatePreparation.html>`__
        * `MPS <https://docs.pennylane.ai/en/stable/code/api/pennylane.MPS.html>`__
        * `MPSPrep <https://docs.pennylane.ai/en/stable/code/api/pennylane.MPSPrep.html>`__
        * `Multiplier <https://docs.pennylane.ai/en/stable/code/api/pennylane.Multiplier.html>`__
        * `Operation <https://docs.pennylane.ai/en/stable/code/api/pennylane.operation.Operation.html>`__
        * `OrbitalRotation <https://docs.pennylane.ai/en/stable/code/api/pennylane.OrbitalRotation.html>`__
        * `OutAdder <https://docs.pennylane.ai/en/stable/code/api/pennylane.OutAdder.html>`__
        * `OutPoly <https://docs.pennylane.ai/en/stable/code/api/pennylane.OutPoly.html>`__
        * `ParticleConservingU1 <https://docs.pennylane.ai/en/stable/code/api/pennylane.ParticleConservingU1.html>`__
        * `ParticleConservingU2 <https://docs.pennylane.ai/en/stable/code/api/pennylane.ParticleConservingU2.html>`__
        * `PauliError <https://docs.pennylane.ai/en/stable/code/api/pennylane.PauliError.html>`__
        * `Permute <https://docs.pennylane.ai/en/stable/code/api/pennylane.Permute.html>`__
        * `PhaseAdder <https://docs.pennylane.ai/en/stable/code/api/pennylane.PhaseAdder.html>`__
        * `PhaseDamping <https://docs.pennylane.ai/en/stable/code/api/pennylane.PhaseDamping.html>`__
        * `PhaseFlip <https://docs.pennylane.ai/en/stable/code/api/pennylane.PhaseFlip.html>`__
        * `PrepSelPrep <https://docs.pennylane.ai/en/stable/code/api/pennylane.PrepSelPrep.html>`__
        * `Projector <https://docs.pennylane.ai/en/stable/code/api/pennylane.Projector.html>`__
        * `PSWAP <https://docs.pennylane.ai/en/stable/code/api/pennylane.PSWAP.html>`__
        * `QAOAEmbedding <https://docs.pennylane.ai/en/stable/code/api/pennylane.QAOAEmbedding.html>`__
        * `QDrift <https://docs.pennylane.ai/en/stable/code/api/pennylane.QDrift.html>`__
        * `QROMStatePreparation <https://docs.pennylane.ai/en/stable/code/api/pennylane.QROMStatePreparation.html>`__
        * `QuantumMonteCarlo <https://docs.pennylane.ai/en/stable/code/api/pennylane.QuantumMonteCarlo.html>`__
        * `QubitCarry <https://docs.pennylane.ai/en/stable/code/api/pennylane.QubitCarry.html>`__
        * `QubitChannel <https://docs.pennylane.ai/en/stable/code/api/pennylane.QubitChannel.html>`__
        * `QubitDensityMatrix <https://docs.pennylane.ai/en/stable/code/api/pennylane.QubitDensityMatrix.html>`__
        * `Qubitization <https://docs.pennylane.ai/en/stable/code/api/pennylane.Qubitization.html>`__
        * `QubitSum <https://docs.pennylane.ai/en/stable/code/api/pennylane.QubitSum.html>`__
        * `RandomLayers <https://docs.pennylane.ai/en/stable/code/api/pennylane.RandomLayers.html>`__
        * `Reflection <https://docs.pennylane.ai/en/stable/code/api/pennylane.Reflection.html>`__
        * `ResetError <https://docs.pennylane.ai/en/stable/code/api/pennylane.ResetError.html>`__
        * `RotXZX <https://docs.pennylane.ai/en/stable/code/api/pennylane.ftqc.RotXZX.html>`__
        * `ScalarSymbolicOp <https://docs.pennylane.ai/en/stable/code/api/pennylane.ops.op_math.ScalarSymbolicOp.html>`__
        * `SelectOnlyQRAM <https://docs.pennylane.ai/en/stable/code/api/pennylane.SelectOnlyQRAM.html>`__
        * `SimplifiedTwoDesign <https://docs.pennylane.ai/en/stable/code/api/pennylane.SimplifiedTwoDesign.html>`__
        * `SingleExcitationMinus <https://docs.pennylane.ai/en/stable/code/api/pennylane.SingleExcitationMinus.html>`__
        * `SingleExcitationPlus <https://docs.pennylane.ai/en/stable/code/api/pennylane.SingleExcitationPlus.html>`__
        * `Snapshot <https://docs.pennylane.ai/en/stable/code/api/pennylane.Snapshot.html>`__
        * `SparseHamiltonian <https://docs.pennylane.ai/en/stable/code/api/pennylane.SparseHamiltonian.html>`__
        * `SpecialUnitary <https://docs.pennylane.ai/en/stable/code/api/pennylane.SpecialUnitary.html>`__
        * `StatePrep <https://docs.pennylane.ai/en/stable/code/api/pennylane.StatePrep.html>`__
        * `StronglyEntanglingLayers <https://docs.pennylane.ai/en/stable/code/api/pennylane.StronglyEntanglingLayers.html>`__
        * `SubroutineOp <https://docs.pennylane.ai/en/stable/code/api/pennylane.templates.core.SubroutineOp.html>`__
        * `Superposition <https://docs.pennylane.ai/en/stable/code/api/pennylane.Superposition.html>`__
        * `ThermalRelaxationError <https://docs.pennylane.ai/en/stable/code/api/pennylane.ThermalRelaxationError.html>`__
        * `TrotterizedQfunc <https://docs.pennylane.ai/en/stable/code/api/pennylane.TrotterizedQfunc.html>`__
        * `TrotterProduct <https://docs.pennylane.ai/en/stable/code/api/pennylane.TrotterProduct.html>`__
        * `TTN <https://docs.pennylane.ai/en/stable/code/api/pennylane.TTN.html>`__
        * `TwoLocalSwapNetwork <https://docs.pennylane.ai/en/stable/code/api/pennylane.TwoLocalSwapNetwork.html>`__
        * `UCCSD <https://docs.pennylane.ai/en/stable/code/api/pennylane.UCCSD.html>`__

Further reading
---------------

* :doc:`qjit`: how ``qjit`` compiles and runs your workflow.
* :doc:`sharp_bits`: things to watch out for in compiled workflows.
* :doc:`/development/adding_operators`: custom operators in detail.
* :doc:`/development/deprecations`: the full list of deprecations and removals.
* :doc:`/development/release_notes`: release notes for every version.
