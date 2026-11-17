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
        rules via :func:`~.add_decomps`. See :ref:`classic_custom_operators`.
    * - Supported devices
      - Any PennyLane device, including Python devices such as
        ``default.qubit``, plugins such as PennyLane-Qiskit, and more.
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

Migrating workflows
-------------------

Simple circuits
^^^^^^^^^^^^^^^

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

Gradients
^^^^^^^^^

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

Optimization loops
^^^^^^^^^^^^^^^^^^

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

Transforms
^^^^^^^^^^

Built-in compilation passes, such as :func:`~pennylane.decompose`,
:func:`~pennylane.transforms.cancel_inverses` and
:func:`~pennylane.transforms.merge_rotations`, work in PennyLane2. A transform
that you wrote yourself with :func:`~pennylane.transform` acts on tapes, so it
can't be applied to a compiled program: ``qjit`` raises an error saying the
transform doesn't have a ``pass_name``. Check whether a built-in pass does what
you need (see :doc:`compiling_circuits`). If not, please
`tell us <https://discuss.pennylane.ai>`__ what your transform does.

Resource profiling
^^^^^^^^^^^^^^^^^^

TOOD

.. _classic_plugin_devices:

Creating devices
----------------

Devices from plugins built on the PennyLane Classic device API, such as
third-party hardware and simulator plugins, don't work with PennyLane2 as they
are. If you use one, keep using PennyLane Classic
(``pip install pennylane==0.45.1``) until the plugin supports PennyLane2. If you
develop one, see :doc:`/development/plugins` to make it work with PennyLane2.

.. _classic_custom_operators:

Migrating custom operators
--------------------------

In PennyLane Classic, you define a custom operator by subclassing
:class:`~pennylane.core.operator.Operation` and overriding methods such as
``compute_decomposition``. These operators can't be compiled by PennyLane2. In
PennyLane2, operators are :class:`~pennylane.core.operator.Operator2` instances,
which describe their arguments in a way the compiler understands.

The quickest way to migrate a custom operator is :func:`~pennylane.subcircuit`.
It turns a quantum function into an operator, and registers the function body
as the operator's decomposition.

Take this PennyLane Classic operator, which flips one wire and rotates
another:

.. code-block:: python

    import pennylane as qp

    class FlipAndRotate(qp.operation.Operation):
        grad_method = "A"

        def __init__(self, angle, wire_rot, wire_flip=None, do_flip=False, id=None):
            self._hyperparameters = {"do_flip": do_flip}
            all_wires = qp.wires.Wires(wire_rot) + qp.wires.Wires(wire_flip)
            super().__init__(angle, wires=all_wires, id=id)

        @property
        def num_params(self):
            return 1

        @staticmethod
        def compute_decomposition(angle, wires, do_flip):
            op_list = []
            if do_flip:
                op_list.append(qp.X(wires[1]))
            op_list.append(qp.RX(angle, wires=wires[0]))
            return op_list

        def adjoint(self):
            return FlipAndRotate(
                -self.parameters[0], self.wires[0], self.wires[1],
                do_flip=self.hyperparameters["do_flip"],
            )

        @classmethod
        def _unflatten(cls, data, metadata):
            wires = metadata[0]
            hyperparams = dict(metadata[1])
            return cls(data[0], wire_rot=wires[0], wire_flip=wires[1], do_flip=hyperparams["do_flip"])

In PennyLane2, the same operator is a decorated quantum function:

.. code-block:: python

    import pennylane as qp

    def flip_and_rotate_resources(angle, do_flip, wires):
        if do_flip:
            return {qp.X: 1, qp.RX: 1}
        return {qp.RX: 1}

    @qp.subcircuit(
        dynamic_argnames=("angle",),
        compilable_argnames=("do_flip",),
        wire_argnames=("wires",),
    )
    @qp.register_resources(flip_and_rotate_resources)
    def FlipAndRotate(angle, do_flip, wires):
        if do_flip:
            qp.X(wires[1])
        qp.RX(angle, wires=wires[0])

>>> FlipAndRotate(0.3, True, wires=[0, 1])
FlipAndRotate(angle=0.3, do_flip=True, wires=[0, 1])

To use it in a compiled circuit, decompose it into gates the device supports
with :func:`~pennylane.decompose`:

.. code-block:: python

    dev = qp.device("lightning.qubit", wires=2)

    @qp.qjit
    @qp.decompose(gate_set={"RX", "PauliX"})
    @qp.qnode(dev)
    def circuit(angle):
        FlipAndRotate(angle, True, wires=[0, 1])
        return qp.expval(qp.Z(0))

Each part of the Classic operator maps to PennyLane2 as follows:

.. list-table::
    :header-rows: 1
    :widths: 40 60

    * - PennyLane Classic
      - PennyLane2
    * - Trainable parameters passed positionally, and ``num_params``
      - List the arguments in ``dynamic_argnames``.
    * - Wires passed with ``wires=``, combined in ``__init__``
      - List each wire argument in ``wire_argnames``. The operator's wires
        are collected from them in order.
    * - ``_hyperparameters``
      - List the arguments in ``compilable_argnames`` if the compiler should
        see their values, or in ``static_argnames`` if it shouldn't. Pass them
        as Python values, not arrays or traced values.
    * - ``compute_decomposition``
      - The body of the quantum function. Its resources, the gates it uses,
        are declared with :func:`~pennylane.register_resources`.
    * - ``_flatten`` and ``_unflatten``
      - Not needed. The argument categories tell PennyLane2 how to rebuild the
        operator.
    * - ``grad_method``
      - Not needed. Gradients are computed through the decomposition.
    * - ``id``
      - Removed. Operators no longer accept ``id``.
    * - Validation in ``__init__``
      - Subclass :class:`~pennylane.core.operator.Operator2` (below).
    * - :func:`qp.ops.functions.assert_valid <pennylane.testing.assert_valid>`
      - :func:`qp.testing.assert_valid <pennylane.testing.assert_valid>`.

Every argument of the operator must be in exactly one of the argument
categories. They're described in
:ref:`Operator argument types <operator_argument_types>`.

If your operator needs more than a decomposition, such as input validation, a
matrix (``compute_matrix``) or its own ``adjoint`` or ``pow`` method, subclass
:class:`~pennylane.core.operator.Operator2` instead, and add decompositions with
:func:`~pennylane.add_decomps`. :doc:`/development/adding_operators` walks
through both approaches, including operators with mid-circuit measurements and
conditional branches.

Notable breaking changes
------------------------

Removed modules
^^^^^^^^^^^^^^^

The following features of PennyLane Classic are not available in PennyLane2.

.. list-table::
    :header-rows: 1
    :widths: 35 65

    * - Removed
      - What to use instead
    * - ``qp.qnn`` module, including ``TorchLayer``
      - ``qp.qnn.iqp_expval`` is now :func:`qp.math.iqp_expval
        <pennylane.math.iqp_expval>`.
    * - ``qp.qaoa`` module: mixer and cost Hamiltonians, ``cost_layer``,
        ``mixer_layer`` and the ``cycle`` helpers
      - Build the Hamiltonians and layers yourself.
        :class:`~pennylane.QAOAEmbedding` is unaffected.
    * - ``qp.pulse`` module, including ``ParametrizedHamiltonian``,
        ``ParametrizedEvolution``, ``stoch_pulse_grad`` and ``pulse_odegen``
      - None.
    * - ``qp.fourier`` module
      - ``qp.fourier.reconstruct`` is now ``qp.optimize.reconstruct``.
    * - ``qp.qcut`` module, including ``cut_circuit``, ``cut_circuit_mc`` and
        ``WireCut``
      - None.
    * - Continuous-variable (CV) operators, observables, templates and
        devices, including ``default.gaussian`` and ``param_shift_cv``
      - None.
    * - Qutrit and qudit operators and devices
      - None. Qudit functionality remains in ``pennylane.labs``.
    * - TensorFlow interface
      - JAX.
    * - Compiling with CUDA Quantum through ``qjit``
      - None.
    * - ``qp.Configuration``, and passing ``config`` to
        :func:`~pennylane.device`
      - None.
    * - ``qp.resource.Resources``, ``ResourceOperator``, ``ErrorOperator`` and
        the ``qp.resource.error`` module
      - :func:`~pennylane.estimate`, :func:`~pennylane.analyze` and
        :func:`~pennylane.track`. See :doc:`resource_profiling`.
    * - Resource information of tapes
      - :func:`~pennylane.analyze` on a ``qjit``-compiled QNode.
    * - Plxpr transforms, and running plxpr without ``qjit``
      - ``qp.qjit``.
    * - ``qp.decomposition.reconstruct``
      - None.
    * - Python 3.11
      - Python 3.12 or later.

APIs that changed
^^^^^^^^^^^^^^^^^

If your code uses any of the following, update it as shown.

.. list-table::
    :header-rows: 1
    :widths: 45 55

    * - PennyLane Classic
      - PennyLane2
    * - ``circuit(x, shots=100)``
      - ``qp.set_shots(circuit, shots=100)(x)``, or ``shots=`` in
        :func:`~pennylane.qnode`.
    * - ``op.data = new_data``
      - ``qp.ops.functions.bind_new_parameters(op, new_data)``, which returns
        a new operator.
    * - ``id=`` on operators and measurements
      - Removed. On ``MidMeasure``, use ``meas_uid``.
    * - A ``generator`` property on a custom operator
      - A ``generator()`` method.
    * - ``Operator._queue_category = None``
      - Override ``Operator.queue``.
    * - ``qnode.transform_program`` and ``qp.workflow.get_transform_program``
      - ``qnode.compile_pipeline`` and
        :func:`qp.workflow.get_compile_pipeline
        <pennylane.workflow.get_compile_pipeline>`.
    * - ``BoundTransform.transform``
      - ``BoundTransform.tape_transform``.
    * - ``QuantumScript.expand``, ``qp.tape.expand_tape`` and
        ``qp.transforms.create_expand_fn``
      - :func:`~pennylane.decompose`.
    * - :func:`~pennylane.decompose` on tapes
      - :func:`~pennylane.decompose` is a compilation pass that runs under
        ``qjit``.
    * - ``qp.QuantumPhaseEstimation(matrix, target_wires=..., estimation_wires=...)``
      - ``qp.QuantumPhaseEstimation(qp.QubitUnitary(matrix, wires=...),
        estimation_wires=...)``
    * - ``data=`` in :class:`~pennylane.QROM`, :class:`~pennylane.BBQRAM`,
        :class:`~pennylane.HybridQRAM` and :class:`~pennylane.SelectOnlyQRAM`
      - ``bitstrings=``
    * - ``qp.IQP(..., num_wires=n)``
      - ``qp.IQP(..., wires=range(n))``
    * - ``pattern=`` in :class:`~pennylane.IQPEmbedding`, as pairs of wire
        labels
      - Pairs of indices into ``wires``. For ``wires=["z", "a", "k"]``,
        ``[["z", "k"]]`` becomes ``[[0, 2]]``.
    * - An integer as the state in :class:`~pennylane.BasisState`
      - A binary array, for example from ``qp.math.int_to_binary``.
    * - ``wires=`` in :class:`~pennylane.GlobalPhase`
      - Removed.
    * - ``qp.ctrl(qp.GlobalPhase(phi), control=...)``, as a ``Controlled``
        operator
      - Returns a :class:`~pennylane.PhaseShift` or a controlled
        ``PhaseShift`` when all control values are ``1``.
    * - ``return_global_phase=`` in ``qp.math`` decomposition helpers such as
        ``zyz_rotation_angles``
      - Removed. The global phase is always returned as the last value.
    * - ``aux_wire=None`` in ``qp.gradients.hadamard_grad`` with
        ``mode="reversed"`` or ``"standard"``
      - Pass an ``aux_wire``.
    * - ``qp.math.is_abstract``
      - ``qp.math.is_tracer``
    * - ``qp.ops.functions.assert_valid``
      - :func:`qp.testing.assert_valid <pennylane.testing.assert_valid>`
    * - ``qp.tape.plxpr_to_tape``
      - ``qp.testing.plxpr_to_tape``
    * - ``qp.queuing.process_queue``
      - ``from pennylane.tape.qscript import process_queue``
    * - ``qp.Backline.backline``
      - ``qp.Backline.placement``

PennyLane2 also requires JAX 0.7.1, and ZX transforms require
``pyzx>=0.10``.

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
