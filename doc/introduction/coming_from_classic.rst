.. role:: html(raw)
   :format: html

.. _intro_coming_from_classic:

Coming from PennyLane Classic
=============================

PennyLane Classic is PennyLane up to and including v0.45. If you have PennyLane
Classic code, this page tells you what carries over to PennyLane2, what you need
to change, and what is no longer available.

.. note::

    PennyLane Classic remains available. If you need a feature that PennyLane2
    doesn't support, install it with ``pip install pennylane==0.45.1`` and use
    the `PennyLane Classic documentation <https://docs.pennylane.ai/en/v0.45.1/>`__.

What's different
----------------

In PennyLane Classic, a QNode builds a *tape* (a list of operators and
measurements) by running your Python code, and devices, transforms, gradients
and drawing all work on that tape.

PennyLane2 compiles your program to a single program representation, MLIR, with
the :doc:`Catalyst compiler <qjit>`. Every workflow is decorated with
:func:`@qp.qjit(capture=True) <pennylane.qjit>`, and compilation, analysis and
execution all work on the compiled program rather than on a tape. This keeps
control flow and dynamic values intact, so programs scale to hundreds of qubits
and every tool sees the same program.

The table below summarizes the most common changes.

.. list-table::
    :header-rows: 1
    :widths: 22 39 39

    * -
      - PennyLane Classic
      - PennyLane2
    * - Running a circuit
      - Call the QNode directly.
      - Decorate the QNode, or a function that calls it, with
        ``@qp.qjit(capture=True)``.
    * - Devices
      - Any PennyLane device, including Python devices such as
        ``default.qubit``.
      - Devices that Catalyst supports, such as ``lightning.qubit``,
        ``lightning.kokkos`` and ``lightning.gpu``. Devices must be given
        ``wires``.
    * - Arrays
      - ``pennylane.numpy`` (Autograd), PyTorch, JAX or TensorFlow.
      - ``jax.numpy`` or NumPy. See :doc:`sharp_bits`.
    * - Control flow
      - Python ``for``, ``while`` and ``if``, unrolled into the tape.
      - :func:`~pennylane.for_loop`, :func:`~pennylane.while_loop` and
        :func:`~pennylane.cond`, or Python control flow with
        ``qp.qjit(capture=True, autograph=True)``. Loops stay loops in the
        compiled program.
    * - Transforms
      - Tape transforms, including your own written with
        :func:`~pennylane.transform`.
      - Compilation passes, such as :func:`~pennylane.decompose` and
        :func:`~pennylane.transforms.merge_rotations`. See
        :doc:`compiling_circuits`.
    * - Gradients
      - Many gradient methods and optimizers.
      - :func:`~pennylane.grad` and :func:`~pennylane.jacobian` inside
        ``qjit``, with the parameter-shift rule or the adjoint method.
        Optimize with `Optax <https://optax.readthedocs.io>`__.
    * - Custom operators
      - Subclass :class:`~pennylane.operation.Operation`.
      - Use :func:`~pennylane.subcircuit`, or subclass
        :class:`~pennylane.core.operator.Operator2`. See
        :ref:`classic_custom_operators`.
    * - Inspecting circuits
      - :func:`~pennylane.draw`, :func:`~pennylane.specs` and others, on
        tapes.
      - :func:`~pennylane.draw`, :func:`~pennylane.specs`,
        :func:`~pennylane.analyze` and :func:`~pennylane.estimate`, on the
        compiled program.

Migrating a workflow
--------------------

Here is a typical PennyLane Classic workflow:

.. code-block:: python

    import pennylane as qp
    from pennylane import numpy as np

    dev = qp.device("default.qubit")

    @qp.qnode(dev, shots=1000)
    def circuit(params, n_layers):
        for i in range(n_layers):
            qp.RX(params[0], wires=0)
            qp.CNOT(wires=[0, 1])
        if n_layers > 2:
            qp.RY(params[1], wires=1)
        return qp.expval(qp.Z(1))

    params = np.array([0.1, 0.2], requires_grad=True)
    result = circuit(params, 3)

And here is the same workflow in PennyLane2:

.. code-block:: python

    import pennylane as qp
    from jax import numpy as jnp

    dev = qp.device("lightning.qubit", wires=2)

    @qp.qjit(capture=True, autograph=True)
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

#. **Decorate with qjit.** ``@qp.qjit(capture=True)`` compiles the QNode.
   You can also decorate a function that calls one or more QNodes, which
   compiles your classical processing too.

#. **Choose a supported device, and give it wires.** ``default.qubit`` and
   other devices written in Python can't run compiled programs. Use
   ``lightning.qubit`` or another Catalyst-compatible device, and set
   ``wires`` when you create it.

#. **Use jax.numpy for arrays.** Replace ``pennylane.numpy`` with
   ``jax.numpy``. Arrays no longer need ``requires_grad``: say which arguments
   to differentiate with ``argnums`` in :func:`~pennylane.grad`.

#. **Make control flow compilable.** Python ``for`` loops and ``if`` statements
   that depend on a function argument need AutoGraph
   (``autograph=True``), which converts them to compiled control flow.
   Without it, you'll see an error asking you to enable AutoGraph.
   Alternatively, write the loop with :func:`~pennylane.for_loop` and the
   condition with :func:`~pennylane.cond`.

#. **Set shots on the QNode.** Shots are set with the ``shots`` argument of
   :func:`~pennylane.qnode` or with :func:`~pennylane.set_shots`. Passing
   ``shots`` when you call the QNode is no longer supported.

Optimization loops
^^^^^^^^^^^^^^^^^^

PennyLane Classic optimizers like ``qp.GradientDescentOptimizer`` step through
an optimization in Python. In PennyLane2, compile the whole optimization loop
and use `Optax <https://optax.readthedocs.io>`__ for the optimizer. The
:doc:`qjit` page has a full example.

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

.. _classic_custom_operators:

Migrating custom operators
--------------------------

In PennyLane Classic, you define a custom operator by subclassing
:class:`~pennylane.operation.Operation` and overriding methods such as
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

    @qp.qjit(capture=True)
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

Breaking changes
----------------

Removed features
^^^^^^^^^^^^^^^^

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
      - :func:`~pennylane.specs`, :func:`~pennylane.analyze` and
        :func:`~pennylane.estimate`. See :doc:`resource_estimation`.
    * - Tapes in :func:`~pennylane.specs`, its ``level="gradient"`` and
        ``level="all-mlir"`` options, and ``QuantumScript.specs``
      - :func:`~pennylane.specs` on a ``qjit``-compiled QNode.
    * - Plxpr transforms, and running plxpr without ``qjit``
      - ``qp.qjit(capture=True)``.
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
      - Removed. On :class:`~pennylane.ops.MidMeasure`, use ``meas_uid``.
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

PennyLane Classic features still included
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some PennyLane Classic features are still in the package, but they work on
tapes and don't compile with ``qjit``: running QNodes without ``qjit``,
devices written in Python such as ``default.qubit``, ``pennylane.numpy`` and
PyTorch arrays, tape transforms, the gradient transforms in
:mod:`~pennylane.gradients`, and the PennyLane optimizers. They are no longer
developed, and may be removed in a future release. Don't build new work on
them.

If PennyLane2 is missing something you rely on, please let us know on the
`PennyLane Discussion Forum <https://discuss.pennylane.ai>`__ or by
`opening an issue <https://github.com/PennyLaneAI/pennylane/issues>`__.

Further reading
---------------

* :doc:`qjit`: how ``qjit`` compiles and runs your workflow.
* :doc:`sharp_bits`: things to watch out for in compiled workflows.
* :doc:`/development/adding_operators`: custom operators in detail.
* :doc:`/development/deprecations`: the full list of deprecations and removals.
* :doc:`/development/release_notes`: release notes for every version.
