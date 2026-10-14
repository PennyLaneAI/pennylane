.. _contributing_operators:

Adding new quantum operators
============================

This page will outline the steps to help create custom quantum operators that
can either be formally added to PennyLane2 or just used locally.

.. note::

    The term "quantum operators" encompasses anything that can be programmed
    onto a quantum device. This includes
    :doc:`gates </introduction/operations>`, elaborate
    :doc:`subroutines </introduction/subroutines>`, and mid-circuit
    measurements.

Important classes and functions
###############################

Decomposition and resource estimation
-------------------------------------

Adding a custom operator is often convenient because it represents some
self-contained block that not only will be used repetitively, but also because
it could be *implemented* (or decomposed into elementary gates) differently
ways.

PennyLane2 offers a robust and modularized decomposition algorithm, wherein any
quantum operator can define different decompositions in terms of other
operators. Creating new decomposition rules involves the following functions and
modules.

.. autosummary::

    ~pennylane.register_resources
    ~pennylane.add_decomps
    ~pennylane.typing

The decomposition algorithm in PennyLane2 bases its output on the resources
required to compile a circuit into a target gate set. The
:func:`~pennylane.register_resources` function is the avenue to specifying
an individual decomposition rule's quantum resources. An example will be
provided further down this page that uses :func:`~pennylane.register_resources`,
:func:`~pennylane.add_decomps`, and the :mod:`~pennylane.typing` module.

Operator creation
-----------------

The two most important functionalities in PennyLane2 for creating new operators
are:

.. autosummary::

    ~pennylane.core.operator.Operator2
    ~pennylane.subcircuit

The :class:`~pennylane.core.operator.Operator2` class and the
:func:`~pennylane.subcircuit` function only differ syntactically; using either
will result in a valid quantum operator. For most use cases,
:func:`~pennylane.subcircuit` is recommended for its ease of use. Utilizing the
:class:`~pennylane.core.operator.Operator2` class is recommended for the utmost
customization.

Operator argument types
#######################

When creating a new quantum operator in PennyLane2, the input arguments to the
operator must be accurately binned into the following categories for
compatibility with the :func:`~pennylane.qjit` compiler.

* ``wire_argnames`` : The names of arguments corresponding to wires. Values for
  these arguments are automatically wrapped in :class:`~.Wires` objects by the
  :class:`~pennylane.core.operator.Operator2` constructor. For example,
  :class:`~pennylane.MultiControlledX`'s ``wires`` and ``work_wires`` arguments
  belong in ``wire_argnames``.

* ``dynamic_argnames`` : The names of arguments that are scalars, arrays, or
  castable to arrays. These arguments can be treated by the
  :func:`~pennylane.qjit` compiler as dynamic. For example, the ``bitstrings``
  argument in :class:`~pennylane.QROM` is an array of integer values (``0`` and
  ``1``), and can therefore be in ``dynamic_argnames``.

* ``static_argnames`` : The names of arguments whose concrete values are known
  at compile time, but that do not need to be accessible to the
  :func:`~pennylane.qjit` compiler. For example, the ``partial`` argument in
  :class:`~pennylane.Select` is a Boolean flag that only changes how the
  operator is decomposed, and therefore belongs in ``static_argnames``.

* ``compilable_argnames`` : The names of arguments whose concrete values are
  known at compile time and *are* visible to the :func:`~pennylane.qjit`
  compiler. These can be numbers, strings, and lists, tuples, or dictionaries of
  them. For example, the ``pauli_word`` argument in :class:`~pennylane.PauliRot`
  is a string (e.g., ``"XYZ"``), and can therefore be in
  ``compilable_argnames``. An operator can specify either
  ``compilable_argnames`` or ``static_argnames`` and ``hybrid_argnames``, but
  not both.

* ``hybrid_argnames`` : The names of arguments that hold dynamic data wrapped in
  static structures (Pytrees), such as other operators or collections of wires
  or arrays. These names cannot appear in ``dynamic_argnames``,
  ``static_argnames``, or ``compilable_argnames``, but they can appear in
  ``wire_argnames`` when the argument contains nested structures of wires. For
  example, the ``ops`` argument in :class:`~pennylane.Select` is a list of
  operators, and therefore belongs in ``hybrid_argnames``.

An example
##########

Creating a custom operator using both :func:`~pennylane.subcircuit` and
:class:`~pennylane.core.operator.Operator2` will be shown in this section.

The following is an example for a two-wire gate called ``FlipAndEntangle``. This
operator flips a wire, measures another wire (a mid-circuit measurement) and
conditionally applies a different :class:`~pennylane.CNOT` gate given the
mid-circuit measurement result.

First, its input arguments need to be binned into the proper categories. Its
call signature is ``FlipAndRotate(angle, rot_axes, wires)``.

* ``angle`` : the angle of rotation (a ``float``). This can go in
  ``dynamic_argnames``.
* ``rot_axes`` : a length-2 list of some combination of ``"X"``, ``"Y"``, and
  ``"Z"``, representing an axis of rotation. Depending on the measurement result
  of ``wires[0]``, the wire that gets flipped, a rotation gate corresponding to
  the axis of rotation is applied to ``wires[1]``. This can go in
  ``compilable_argnames``.
* ``wires`` : the wires the operator acts on. The first wire gets flipped, the
  other is rotated. This can go in ``wire_argnames``.

We can define ``FlipAndRotate`` as a quantum function and turn it into a proper
quantum operator by decorating with :func:`~pennylane.subcircuit` :

.. code-block:: python

    import pennylane as qp

    def flip_rotate_resources(rot_axes):
        return {
            qp.PauliRot(qp.typing.Float, rot_axes[0], wires=qp.typing.Wire): 1,
            qp.PauliRot(qp.typing.Float, rot_axes[1], wires=qp.typing.Wire): 1,
            qp.X: 1,
            qp.ops.mid_measure.MidMeasure(qp.typing.Wire): 1
        }

    @qp.subcircuit(
        dynamic_argnams={"angle"},
        compilable_argnames={"rot_axes"},
        wire_argnames={"wires"},
    )
    @qp.register_resources(flip_rotate_resources)
    def FlipAndRotate(angle, rot_axes, wires):
        qp.X(wires[0])
        m = qp.measure(1)

        def true_fn():
            qp.PauliRot(2 * angle, rot_axes[0], wires=wires[1])

        def false_fn():
            qp.PauliRot(2 * angle, rot_axes[1], wires=wires[1])

        qp.cond(m, true_fn, false_fn)()

>>> type(FlipAndRotate)



Abstraction
###########

Operators in quantum mechanics are maps that act on vector spaces, and in differentiable quantum computing, these
maps can depend on a set of trainable parameters. The :class:`~.Operator` class
serves as the main abstraction of such objects, and all operators (such as gates, channels, observables)
inherit from it.

>>> from jax import numpy as jnp
>>> op = qp.Rot(jnp.array(0.1), jnp.array(0.2), jnp.array(0.3), wires=["a"])
>>> isinstance(op, qp.operation.Operator)
True

The basic components of operators are the following:

#. **The name of the operator** (:attr:`.Operator.name`), which may have a canonical, universally known interpretation (such as a "Hadamard" gate),
   or could be a name specific to PennyLane.

   >>> op.name
   Rot

#. **The subsystems that the operator addresses** (:attr:`.Operator.wires`), which mathematically speaking defines the subspace that it acts on.

   >>> op.wires
   Wires(['a'])

#. **Trainable parameters** (:attr:`.Operator.parameters`) that the map depends on, such as a rotation angle,
   which can be fed to the operator as tensor-like objects. For example, since we used jax arrays to
   specify the three rotation angles of ``op``, the parameters are jax ``Arrays``.

   >>> op.parameters
   [Array(0.1, dtype=float32, weak_type=True),
    Array(0.2, dtype=float32, weak_type=True),
    Array(0.3, dtype=float32, weak_type=True)]

#. **Non-trainable hyperparameters** (:attr:`.Operator.hyperparameters`) that influence the action of the operator.
   Not every operator has hyperparameters.

   >>> op.hyperparameters
   {}

#. Possible **symbolic or numerical representations** of the operator, which can be used by PennyLane's
   devices to interpret the map. Examples are:

   * Representation as a **product of operators** (:meth:`.Operator.decomposition`):

     >>> op = qp.Rot(0.1, 0.2, 0.3, wires=["a"])
     >>> op.decomposition()
     [RZ(0.1, wires=['a']), RY(0.2, wires=['a']), RZ(0.3, wires=['a'])]

   * Representation as a **linear combination of operators** (:meth:`.Operator.terms`):

     >>> op = qp.Hamiltonian([1., 2.], [qp.PauliX(0), qp.PauliZ(0)])
     >>> op.terms()
     ((1.0, 2.0), [PauliX(wires=[0]), PauliZ(wires=[0])])

   * Representation via the **eigenvalue decomposition** specified by eigenvalues (for the diagonal matrix, :meth:`.Operator.eigvals`)
     and diagonalizing gates (for the unitaries :meth:`.Operator.diagonalizing_gates`):

     >>> op = qp.PauliX(0)
     >>> op.diagonalizing_gates()
     [H(0)]
     >>> op.eigvals()
     [ 1 -1]

   * Representation as a **matrix** (:meth:`.Operator.matrix`), as specified by a global wire order that tells us where the
     wires are found on a register:

     >>> op = qp.PauliRot(0.2, "X", wires=["b"])
     >>> op.matrix(wire_order=["a", "b"])
     [[9.95e-01-2.26e-18j 2.72e-17-9.98e-02j, 0+0j, 0+0j]
      [2.72e-17-9.98e-02j 9.95e-01-2.26e-18j, 0+0j, 0+0j]
      [0+0j, 0+0j, 9.95e-01-2.26e-18j 2.72e-17-9.98e-02j]
      [0+0j, 0+0j, 2.72e-17-9.98e-02j 9.95e-01-2.26e-18j]]

   * Representation as a **sparse matrix** (:meth:`.Operator.sparse_matrix`):

     >>> from scipy.sparse.coo import coo_matrix
     >>> row = np.array([0, 1])
     >>> col = np.array([1, 0])
     >>> data = np.array([1, -1])
     >>> mat = coo_matrix((data, (row, col)), shape=(4, 4))
     >>> op = qp.SparseHamiltonian(mat, wires=["a"])
     >>> op.sparse_matrix(wire_order=["a"])
     (0, 1)   1
     (1, 0) - 1

New operators can be created by applying arithmetic functions to operators, such as addition, scalar multiplication,
multiplication, taking the adjoint, or controlling an operator. At the moment, such arithmetic is only implemented for
specific subclasses.

* Operators inheriting from :class:`~.Operator` support addition and scalar multiplication:

  >>> op = qp.PauliX(0) + 0.1 * qp.PauliZ(0)
  >>> op.name
  Hamiltonian
  >>> op
    (0.1) [Z0]
  + (1.0) [X0]

* Operators may define a hermitian conjugate:

  >>> qp.RX(1., wires=0).adjoint()
  RX(-1.0, wires=[0])

Creating custom operators
#########################

A custom operator can be created by inheriting from :class:`~.Operator` or one of its subclasses.

The following is an example for a custom gate that possibly flips a qubit and then rotates another qubit.
The custom operator defines a decomposition, which the devices can use (since it is unlikely that a device
knows a native implementation for ``FlipAndRotate``). It also defines an adjoint operator.

.. code-block:: python

    import pennylane as qp


    class FlipAndRotate(qp.operation.Operation):

        # This attribute tells PennyLane what differentiation method to use. Here
        # we request parameter-shift (or "analytic") differentiation.
        grad_method = "A"

        def __init__(self, angle, wire_rot, wire_flip=None, do_flip=False):

            # checking the inputs --------------

            if do_flip and wire_flip is None:
                raise ValueError("Expected a wire to flip; got None.")

            # note: we use the framework-agnostic math library since
            # trainable inputs could be tensors of different types
            shape = qp.math.shape(angle)
            if len(shape) > 1:
                raise ValueError(f"Expected a scalar angle; got angle of shape {shape}.")

            #------------------------------------

            # do_flip is not trainable but influences the action of the operator,
            # which is why we define it to be a hyperparameter
            self._hyperparameters = {
                "do_flip": do_flip
            }

            # we extract all wires that the operator acts on,
            # relying on the Wire class arithmetic
            all_wires = qp.wires.Wires(wire_rot) + qp.wires.Wires(wire_flip)

            # The parent class expects all trainable parameters to be fed as positional
            # arguments, and all wires acted on fed as a keyword argument.
            super().__init__(angle, wires=all_wires)

        @property
        def num_params(self):
            # if it is known before creation, define the number of parameters to expect here,
            # which makes sure an error is raised if the wrong number was passed
            return 1

        @staticmethod
        def compute_decomposition(angle, wires, do_flip):  # pylint: disable=arguments-differ
            # Overwriting this method defines the decomposition of the new gate, as it is
            # called by Operator.decomposition().
            # The general signature of this function is (*parameters, wires, **hyperparameters).
            op_list = []
            if do_flip:
                op_list.append(qp.PauliX(wires=wires[1]))
            op_list.append(qp.RX(angle, wires=wires[0]))
            return op_list

        def adjoint(self):
            # the adjoint operator of this gate simply negates the angle
            return FlipAndRotate(-self.parameters[0], self.wires[0], self.wires[1], do_flip=self.hyperparameters["do_flip"])

        @classmethod
        def _unflatten(cls, data, metadata):
            # as the class differs from the standard `__init__` call signature of
            # (*data, wires=wires, **hyperparameters), the _unflatten method that
            # must be defined as well
            # _unflatten recreates an operation from the serialized data and metadata of ``Operator._flatten``
            # copied_op = type(op)._unflatten(*op._flatten())
            wires = metadata[0]
            hyperparams = dict(metadata[1])
            return cls(data[0], wire_rot=wires[0], wire_flip=wires[1], do_flip=hyperparams['do_flip'])


The new gate can now be created as follows:

>>> op = FlipAndRotate(0.1, wire_rot="q3", wire_flip="q1", do_flip=True)
>>> op
FlipAndRotate(0.1, wires=['q3', 'q1'])
>>> op.decomposition()
[PauliX(wires=['q1']), RX(0.1, wires=['q3'])]
>>> op.adjoint()
FlipAndRotate(-0.1, wires=['q3', 'q1'])

Once the class has been created, you can run a suite of validation checks using :func:`.ops.functions.assert_valid`.
This function will warn you of some common errors in custom operators.

>>> qp.ops.functions.assert_valid(op)

If the above operator omitted the ``_unflatten`` custom definition, it would raise:

.. code-block::


    TypeError: FlipAndRotate.__init__() got an unexpected keyword argument 'wires'


    The above exception was the direct cause of the following exception:

    AssertionError: FlipAndRotate._unflatten must be able to reproduce the original operation
    from (0.1,) and (Wires(['q3', 'q1']), (('do_flip', True),)). You may need to override
    either the _unflatten or _flatten method.
    For local testing, try type(op)._unflatten(*op._flatten())


The new gate can be used with PennyLane devices.

``DefaultQubit`` first checks if the operator has a matrix using the :attr:`~.Operator.has_matrix` property.

- If the device registers support for an operation with the same name,
  PennyLane leaves the gate implementation up to the device. The device
  might have a hardcoded implementation, *or* it may refer to one of the
  numerical representations of the operator (such as :meth:`.Operator.matrix`).

- If the device does not support an operation, PennyLane will automatically
  decompose the gate using :meth:`.Operator.decomposition`.

.. code-block:: python

    from pennylane import numpy as np

    dev = qp.device("default.qubit", wires=["q1", "q2", "q3"])

    @qp.qnode(dev)
    def circuit(angle):
        FlipAndRotate(angle, wire_rot="q1", wire_flip="q1")
        return qp.expval(qp.PauliZ("q1"))

>>> a = np.array(3.14)
>>> circuit(a)
-0.9999987318946099

If all gates used in the decomposition have gradient recipes defined,
we can even compute gradients of circuits that use the new gate without any extra effort.

>>> qp.grad(circuit)(a)
-0.0015926529164868282

.. note::

    The example of ``FlipAndRotate`` is simple enough that one could write a function

    .. code-block:: python

        def FlipAndRotate(angle, wire_rot, wire_flip=None, do_flip=False):
            if do_flip:
                qp.PauliX(wires=wire_flip)
            qp.RX(angle, wires=wire_rot)

    and call it in the quantum function *as if it was a gate*.
    However, classes allow much more functionality, such as defining the adjoint gate above,
    defining the shape expected for the trainable parameter(s), or specifying gradient rules.

Defining special properties of an operator
##########################################

Apart from the main :class:`~.Operator` class, operators with special methods or representations
are implemented as subclasses :class:`~.Operation` and :class:`~.Channel`.

However, unlike many other frameworks, PennyLane does not use class
inheritance to define fine-grained properties of operators,
such as whether it is its own self-inverse, if it is diagonal,
or whether it can be decomposed into Pauli rotations. This avoids changing the inheritance structure
every time an application needs to query a new property.

Instead, PennyLane uses "attributes", which are bookkeeping classes that list operators
which fulfill a specific property.

For example, we can create a new attribute, ``pauli_ops``, like so:

>>> from pennylane.ops.qubits.attributes import Attribute
>>> pauli_ops = Attribute(["PauliX", "PauliY", "PauliZ"])

We can check either a string or an Operation for inclusion in this set:

>>> qp.PauliX(0) in pauli_ops
True
>>> "Hadamard" in pauli_ops
False

We can also dynamically add operators to the sets at runtime. This is useful
for adding custom operations to the attributes such as ``composable_rotations``
and ``self_inverses`` that are used in compilation transforms. For example,
suppose you have created a new operation ``MyGate``, which you know to be its
own inverse. Adding it to the set, like so

>>> from pennylane.ops.qubits.attributes import self_inverses
>>> self_inverses.add("MyGate")

Attributes can also be queried by devices to use special tricks that allow more efficient
implementations. The onus is on the contributors of new operators to add them to the right attributes.

.. note::

    The attributes for qubit gates are currently found in ``pennylane/ops/qubit/attributes.py``.

    Included attributes are listed in the ``Operation``
    `documentation <https://pennylane.readthedocs.io/en/latest/code/qp_operation.html#operation-attributes>`__.

Adding your new operator to PennyLane
#####################################

If you want PennyLane to natively support your new operator, you have to make a Pull Request that adds it
to the appropriate folder in ``pennylane/ops/``. The
tests are added to a file of a similar name and location in ``tests/ops/``. If your operator defines an
ansatz, add it to the appropriate subfolder in ``pennylane/templates/``.

The new operation may have to be imported in the module's ``__init__.py`` file in order to be imported correctly.

Make sure that all hyperparameters and errors are tested, and that the parameters can be passed as
tensors from all supported autodifferentiation frameworks.

Don't forget to also add the new operator to the documentation in the ``doc/introduction/operations.rst`` file, or to
the list of subroutines if it is an ansatz. The latter is done by adding an entry, consisting of the
subroutine and the path to a thumbnail image, to the ``subroutine-table`` in the correct section of
``doc/introduction/subroutines.rst``:

.. code-block:: rst

  .. subroutine-table::
      :nosignatures:

      ~pennylane.MyNewSubroutine    /_static/templates/<subroutine_type>/my_new_subroutine.png

.. note::

  The table displays the first line of the subroutine's docstring, so make sure that it is a concise
  summary of what the subroutine does. The thumbnail should have the same style as the other
  images in ``doc/_static/templates/``.

Here are a few more tips for adding operators:

* *Choose the name carefully.* Good names tell the user what the operator is used for,
  or what architecture it implements. Ask yourself if a gate of a similar name could
  be added soon in a different context.

* *Write good docstrings.* Explain what your operator does in a clear docstring with ample examples.
  You find more about Pennylane standards in the guidelines on :doc:`/development/guide/documentation`.

* *Efficient representations.* Try to implement representations as efficiently as possible, since they may
  be constructed several times.

* *Input checks.* Checking the inputs of the operation introduces an overhead and clashes with tools like
  just-in-time compilation. Find a balance of adding meaningful sanity checks (such as for the shape of tensors),
  but keeping them to a minimum.


Custom subroutines
------------------

Creating a custom subroutine can be as simple as defining a function that creates operations and does not have a return
statement:

.. code-block:: python

    import pennylane as qp

    def MySubroutine(a, b, wires):
        c = qp.math.sin(a) + b
        qp.RX(c, wires=wires[0])

    n_wires = 3
    dev = qp.device("lightning.qubit", wires=n_wires)

    @qp.qjit(capture=True)
    @qp.qnode(dev)
    def circuit(a, b):
        MySubroutine(a, b, wires=range(n_wires))
        return qp.expval(qp.PauliZ(0))

>>> circuit(2, 3)
Array(-0.71950657, dtype=float64)

.. note::

    Classical processing inside a subroutine must be compatible with JIT compilation. PennyLane's
    :mod:`math <pennylane.math>` library provides framework-agnostic functions, such as the
    ``qp.math.sin`` used above, that can be used for this purpose.

To turn a quantum function into a subroutine class like the built-in subroutines above, decorate it with
:func:`~pennylane.subcircuit`. The quantum function must not return anything, and its resources must be
registered with :func:`~pennylane.register_resources`. Its arguments are classified via keyword arguments
such as ``dynamic_argnames``; arguments named ``wires`` are treated as wires by default:

.. code-block:: python

    import numpy as np

    @qp.subcircuit(dynamic_argnames=("weights",))
    @qp.register_resources({qp.RY: 2, qp.CNOT: 1})
    def EntangledRotations(weights, wires):
        qp.RY(weights[0], wires=wires[0])
        qp.RY(weights[1], wires=wires[1])
        qp.CNOT(wires=wires)

    dev = qp.device("lightning.qubit", wires=2)

    @qp.qjit(capture=True)
    @qp.decompose(gate_set={qp.RY, qp.CNOT})
    @qp.qnode(dev)
    def circuit(weights):
        EntangledRotations(weights, wires=[0, 1])
        return qp.expval(qp.PauliZ(1))

    weights = np.array([0.1, 0.2])

``EntangledRotations`` is now an :class:`~.Operator2` subclass, and appears as a single operation in
the circuit. Its quantum function body is registered as its decomposition, which
:func:`~pennylane.decompose` uses to lower it into gates supported by the device:

>>> print(qp.draw(circuit, level="top")(weights))
0: ─╭EntangledRotations(M0)─┤
1: ─╰EntangledRotations(M0)─┤  <Z>
<BLANKLINE>
M0 =
[0.1 0.2]
>>> print(qp.draw(circuit)(weights))
0: ──RY(0.10)─╭●─┤
1: ──RY(0.20)─╰X─┤  <Z>
>>> circuit(weights)
Array(0.97517033, dtype=float64)

As suggested by the camel-case naming, built-in subroutines in PennyLane are classes. Classes are more complex
data structures than functions, since they can define properties and methods of subroutines (such as gradient
recipes or matrix representations). Consult the :ref:`Contributing operators <contributing_operators>`
page to learn how to code up your own subroutine class, and how to add it to the PennyLane subroutine library.

Layering Function
-----------------

The layer function creates a new subroutine by repeatedly applying a sequence of quantum
gates to a set of wires. You can import this function both via
``qp.layer`` and ``qp.templates.layer``.

.. autosummary::

    pennylane.layer
