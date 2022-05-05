.. _contributing_operators:

Adding new operators
====================

The following steps will help you to create custom operators, and to
potentially add them to PennyLane.

Note that in PennyLane, a circuit ansatz consisting of multiple gates is also an operator --- one whose
action is defined by specifying a representation as a combination of other operators.
For historical reasons, you find circuit ansaetze in the ``pennylane/template/`` folder,
while all other operations are found in ``pennylane/ops/``.

The base classes to construct new operators, :class:`~.Operator` and
corresponding subclasses, are found in ``pennylane/operations.py``.

Abstraction
###########

Operators in quantum mechanics are maps that act on vector spaces, and in differentiable quantum computing, these
maps can depend on a set of trainable parameters. The :class:`~.Operator` class
serves as the main abstraction of such objects, and all operators (such as gates, channels, observables)
inherit from it.

>>> from jax import numpy as jnp
>>> op = qml.Rot(jnp.array(0.1), jnp.array(0.2), jnp.array(0.3), wires=["a"])
>>> isinstance(op, qml.operation.Operator)
True

The basic components of operators are the following:

#. **The name of the operator** (:attr:`.Operator.name`), which may have a canonical, universally known interpretation (such as a "Hadamard" gate),
   or could be a name specific to PennyLane.

   >>> op.name
   Rot

#. **The subsystems that the operator addresses** (:attr:`.Operator.wires`), which mathematically speaking defines the subspace that it acts on.

   >>> op.wires
   <Wires = ['a']>

#. **Trainable parameters** (:attr:`.Operator.parameters`) that the map depends on, such as a rotation angle,
   which can be fed to the operator as tensor-like objects. For example, since we used jax arrays to
   specify the three rotation angles of ``op``, the parameters are jax ``DeviceArrays``.

   >>> op.parameters
   [DeviceArray(0.1, dtype=float32, weak_type=True),
    DeviceArray(0.2, dtype=float32, weak_type=True),
    DeviceArray(0.3, dtype=float32, weak_type=True)]

#. **Non-trainable hyperparameters** (:attr:`.Operator.hyperparameters`) that influence the action of the operator.
   Not every operator has hyperparameters.

   >>> op.hyperparameters
   {}

#. Possible **symbolic or numerical representations** of the operator, which can be used by PennyLane's
   devices to interpret the map. Examples are:

   * Representation as a **product of operators** (:meth:`.Operator.decomposition`):

     >>> op = qml.Rot(0.1, 0.2, 0.3, wires=["a"])
     >>> op.decomposition()
     [RZ(0.1, wires=['a']), RY(0.2, wires=['a']), RZ(0.3, wires=['a'])]

   * Representation as a **linear combination of operators** (:meth:`.Operator.terms`):

     >>> op = qml.Hamiltonian([1., 2.], [qml.PauliX(0), qml.PauliZ(0)])
     >>> op.terms()
     ((1.0, 2.0), [PauliX(wires=[0]), PauliZ(wires=[0])])

   * Representation via the **eigenvalue decomposition** specified by eigenvalues (for the diagonal matrix, :meth:`.Operator.eigvals`)
     and diagonalizing gates (for the unitaries :meth:`.Operator.diagonalizing_gates`):

     >>> op = qml.PauliX(0)
     >>> op.diagonalizing_gates()
     [Hadamard(wires=[0])]
     >>> op.eigvals()
     [ 1 -1]

   * Representation as a **matrix** (:meth:`.Operator.matrix`), as specified by a global wire order that tells us where the
     wires are found on a register:

     >>> op = qml.PauliRot(0.2, "X", wires=["b"])
     >>> op.matrix(wire_order=["a", "b"])
     [[9.95e-01-2.26e-18j 2.72e-17-9.98e-02j, 0+0j, 0+0j]
      [2.72e-17-9.98e-02j 9.95e-01-2.26e-18j, 0+0j, 0+0j]
      [0+0j, 0+0j, 9.95e-01-2.26e-18j 2.72e-17-9.98e-02j]
      [0+0j, 0+0j, 2.72e-17-9.98e-02j 9.95e-01-2.26e-18j]]

    .. note::

        The :meth:`.Operator.matrix` method is temporary and will be renamed to :meth:`.Operator.matrix` in an
        upcoming release. It is recommended to use the higher-level :func:`~.matrix` function where possible.

   * Representation as a **sparse matrix** (:meth:`.Operator.sparse_matrix`):

     >>> from scipy.sparse.coo import coo_matrix
     >>> row = np.array([0, 1])
     >>> col = np.array([1, 0])
     >>> data = np.array([1, -1])
     >>> mat = coo_matrix((data, (row, col)), shape=(4, 4))
     >>> op = qml.SparseHamiltonian(mat, wires=["a"])
     >>> op.sparse_matrix(wire_order=["a"])
     (0, 1)   1
     (1, 0) - 1

New operators can be created by applying arithmetic functions to operators, such as addition, scalar multiplication,
multiplication, taking the adjoint, or controlling an operator. At the moment, such arithmetic is only implemented for
specific subclasses.

* Operators inheriting from :class:`~.Observable` support addition and scalar multiplication:

  >>> op = qml.PauliX(0) + 0.1 * qml.PauliZ(0)
  >>> op.name
  Hamiltonian
  >>> op
    (0.1) [Z0]
  + (1.0) [X0]

* Operators may define a hermitian conjugate:

  >>> qml.RX(1., wires=0).adjoint()
  RX(-1.0, wires=[0])

Creating custom operators
#########################

A custom operator can be created by inheriting from :class:`~.Operator` or one of its subclasses.

The following is an example for a custom gate that possibly flips a qubit and then rotates another qubit.
The custom operator defines a decomposition, which the devices can use (since it is unlikely that a device
knows a native implementation for ``FlipAndRotate``). It also defines an adjoint operator.

.. code-block:: python

    import pennylane as qml


    class FlipAndRotate(qml.operation.Operation):

        # Define how many wires the operator acts on in total.
        # In our case this may be one or two, which is why we
        # use the AnyWires Enumeration to indicate a variable number.
        num_wires = qml.operation.AnyWires

        # This attribute tells PennyLane what differentiation method to use. Here
        # we request parameter-shift (or "analytic") differentiation.
        grad_method = "A"

        def __init__(self, angle, wire_rot, wire_flip=None, do_flip=False,
                           do_queue=True, id=None):

            # checking the inputs --------------

            if do_flip and wire_flip is None:
                raise ValueError("Expected a wire to flip; got None.")

            # note: we use the framework-agnostic math library since
            # trainable inputs could be tensors of different types
            shape = qml.math.shape(angle)
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
            all_wires = qml.wires.Wires(wire_rot) + qml.wires.Wires(wire_flip)

            # The parent class expects all trainable parameters to be fed as positional
            # arguments, and all wires acted on fed as a keyword argument.
            # The id keyword argument allows users to give their instance a custom name.
            # The do_queue keyword argument specifies whether or not
            # the operator is queued when created in a tape context.
            super().__init__(angle, wires=all_wires, do_queue=do_queue, id=id)

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
                op_list.append(qml.PauliX(wires=wires[1]))
            op_list.append(qml.RX(angle, wires=wires[0]))
            return op_list

        def adjoint(self):
            # the adjoint operator of this gate simply negates the angle
            return FlipAndRotate(-self.parameters[0], self.wires[0], self.wires[1], do_flip=self.hyperparameters["do_flip"])

The new gate can now be created as follows:

>>> op = FlipAndRotate(0.1, wire_rot="q3", wire_flip="q1", do_flip=True)
>>> op
FlipAndRotate(0.1, wires=['q3', 'q1'])
>>> op.decomposition()
[PauliX(wires=['q1']), RX(0.1, wires=['q3'])]
>>> op.adjoint()
FlipAndRotate(-0.1, wires=['q3', 'q1'])

The new gate can be used with PennyLane devices. PennyLane checks with the device
whether it supports operations using the operation name.

- If the device registers support for an operation with the same name,
  PennyLane leaves the gate implementation up to the device. The device
  might have a hardcoded implementation, *or* it may refer to one of the
  numerical representations of the operator (such as :meth:`.Operator.matrix`).
  
- If the device does not register support for an operation with the same
  name, PennyLane will automatically decompose the gate using :meth:`.Operator.decomposition`.

.. code-block:: python

    from pennylane import numpy as np

    dev = qml.device("default.qubit", wires=["q1", "q2", "q3"])

    @qml.qnode(dev)
    def circuit(angle):
        FlipAndRotate(angle, wire_rot="q1", wire_flip="q1")
        return qml.expval(qml.PauliZ("q1"))

>>> a = np.array(3.14)
>>> circuit(a)
-0.9999987318946099

If all gates used in the decomposition have gradient recipes defined,
we can even compute gradients of circuits that use the new gate without any extra effort.

>>> qml.grad(circuit)(a)
-0.0015926529164868282

.. note::

    The example of ``FlipAndRotate`` is simple enough that one could write a function

    .. code-block:: python

        def FlipAndRotate(angle, wire_rot, wire_flip=None, do_flip=False):
            if do_flip:
                qml.PauliX(wires=wire_flip)
            qml.RX(angle, wires=wire_rot)

    and call it in the quantum function *as if it was a gate*.
    However, classes allow much more functionality, such as defining the adjoint gate above,
    defining the shape expected for the trainable parameter(s), or specifying gradient rules.

Defining special properties of an operator
##########################################

Apart from the main :class:`~.Operator` class, operators with special methods or representations
are implemented as subclasses :class:`~.Operation`, :class:`~.Observable`, :class:`~.Channel`,
:class:`~.CVOperation` and :class:`~.CVObservable`.

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

>>> qml.PauliX(0) in pauli_ops
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
    `documentation <https://pennylane.readthedocs.io/en/latest/code/qml_operation.html#operation-attributes>`__.

Adding your new operator to PennyLane
#####################################

If you want PennyLane to natively support your new operator, you have to make a Pull Request that adds it
to the appropriate folder in ``pennylane/ops/``. The
tests are added to a file of a similar name and location in ``tests/ops/``. If your operator defines an
ansatz, add it to the appropriate subfolder in ``pennylane/templates/``.

The new operation may have to be imported in the module's ``__init__.py`` file in order to be imported correctly.

Make sure that all hyperparameters and errors are tested, and that the parameters can be passed as
tensors from all supported autodifferentiation frameworks.

Don't forget to also add the new operator to the documentation in the ``docs/introduction/operations.rst`` file, or to
the template gallery if it is an ansatz. The latter is done by adding a ``gallery-item``
to the correct section in ``doc/introduction/templates.rst``:

.. code-block::

  .. gallery-item::
    :link: ../code/api/pennylane.templates.<templ_type>.MyNewTemplate.html
    :description: MyNewTemplate
    :figure: ../_static/templates/<templ_type>/my_new_template.png

.. note::

  This loads the image of the template added to ``doc/_static/templates/test_<templ_type>/``. Make sure that
  this image has the same dimensions and style as other template icons in the folder.

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
