.. _contributing_operators:

Custom operators
================

The following steps will help you to create custom operators, and to
potentially add them to PennyLane.

Note that in PennyLane, a circuit ansatz consisting of multiple gates is also an operator - one whose
action is defined by specifying a representation as a combination of other operators.
For historical reasons, you find circuit ansaetze in the ``pennylane/template`` folder,
while all other operations are found in ``pennylane/ops``.

The base classes to construct new operators are found in ``pennylane/operations.py``.

Abstraction
###########

Operators in quantum mechanics are maps that act on vector spaces. The highest-level operator class
(the :class:`~.Operator` class which we will introduce in the next section) serves as the main abstraction of such objects.

.. code-block:: python

   >>> from jax import numpy as jnp
   >>> op = qml.Rot(jnp.array(0.1), jnp.array(0.2), jnp.array(0.3), wires=["a"])
   >>> isinstance(op, qml.operation.Operator)
   True

The basic components of operators are the following:

#. **The name of the operator**, which may have a canonical, universally known interpretation (such as a "Hadamard" gate),
   or could be a name specific to PennyLane.

   .. code-block:: python

       >>> op.name
       Rot

#. **The subsystems that the operator addresses**, which mathematically speaking defines the subspace that it acts on.

   .. code-block:: python

       >>> op.wires
       <Wires = ['a']>

#. **Trainable parameters** that the map depends on, such as a rotation angle,
   which can be fed to the operator as tensor-like objects. For example, since we used jax arrays to
   specify the three rotation angles of ``op``, the parameters are jax ``DeviceArrays``.

   .. code-block:: python

      >>> op.parameters
      [DeviceArray(0.1, dtype=float32, weak_type=True),
       DeviceArray(0.2, dtype=float32, weak_type=True),
       DeviceArray(0.3, dtype=float32, weak_type=True)]

#. **Non-trainable hyperparameters** that influence the action of the operator. Not every operator has hyperparameters.

   .. code-block:: python

       >>> op.hyperparameters
       {}

#. Possible **symbolic or numerical representations** of the operator, which can be used by PennyLane's
   devices to interpret the map. Examples are:

   * Representation as a **product of operators**:

     .. code-block:: python

         >>> op = qml.Rot(0.1, 0.2, 0.3, wires=["a"])
         >>> op.decomposition()
         [RZ(0.1, wires=['a']), RY(0.2, wires=['a']), RZ(0.3, wires=['a'])]

   * Representation as a **linear combination of operators**:

     .. code-block:: python

         >>> op = qml.Hamiltonian([1., 2.], [qml.PauliX(0), qml.PauliZ(0)])
         >>> op.terms()
         ((1.0, 2.0), [PauliX(wires=[0]), PauliZ(wires=[0])])

   * Representation via the **eigenvalue decomposition** specified by eigenvalues (for the diagonal matrix)
     and diagonalizing gates (for the unitaries):

     .. code-block:: python

         >>> op = qml.PauliX(0)
         >>> op.diagonalizing_gates()
         [Hadamard(wires=[0])]
         >>> op.eigvals()
         [ 1 -1]

   * Representation as a **matrix**, as specified by a global wire order that tells us where the
     wires are found on a register:

     .. code-block:: python

         >>> op = qml.PauliRot(0.2, "X", wires=["b"])
         >>> op.matrix(wire_order=["a", "b"])
         [[9.95e-01-2.26e-18j 2.72e-17-9.98e-02j, 0+0j, 0+0j]
          [2.72e-17-9.98e-02j 9.95e-01-2.26e-18j, 0+0j, 0+0j]
          [0+0j, 0+0j, 9.95e-01-2.26e-18j 2.72e-17-9.98e-02j]
          [0+0j, 0+0j, 2.72e-17-9.98e-02j 9.95e-01-2.26e-18j]]

   * Representation as a **sparse matrix**:

     .. code-block:: python

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

* Observables support addition and scalar multiplication:

  .. code-block:: pycon

      >>> op = qml.PauliX(0) + 0.1 * qml.PauliZ(0)
      >>> op.name
      Hamiltonian
      >>> op
        + (1.0) [X0]

* Operations define a hermitian conjugate:

  .. code-block:: pycon

      >>> qml.RX(1., wires=0).adjoint()
      RX(-1.0, wires=[0])

Creating custom operators
#########################

A custom operator can be created by inheriting from :class:`~.Operator` or one of its subclasses.

The following is an example for a custom gate that rotates a qubit and possibly flips another qubit.
The custom operator defines a decomposition, which the devices will use (since it is unlikely that a device
knows a native implementation for ``FlipAndRotate``). It also defines an adjoint operator.

.. code-block:: python

    import pennylane as qml


    class FlipAndRotate(qml.operation.Operation):
        """One-sentence description of the operator.

        More explanation. How is the operator defined, what are typical usage contexts?
        What is the meaning of the different inputs? What options does a user have?

        Args:
            Inputs are described here

        **Example**

        Various usage examples explain how the operator is employed in practice.
        """

        # define how many wires to expect; here we cannot define a fixed number of wires,
        # and use the AnyWires Enumeration instead
        num_wires = qml.operation.AnyWires

        # this attribute tells PennyLane what differentiation method to use; here
        # we request parameter-shift (or "automatic") differentiation
        grad_method = "A"

        def __init__(self, angle, wire_rot,
                           wire_flip=None, do_flip=False,
                           do_queue=True, id=None):

            # checking the inputs --------------

            if do_flip and wire_flip is None:
                raise ValueError("Expected a wire to flip; got None.")

            # note: we use the framework-agnostic math library since
            # trainable inputs could be tensors if different types
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

            # The parent class expects trainable parameters to be fed as positional
            # arguments, and all wires acted on fed as a keyword argument.
            # The id allows users to give their instance a custom name.
            # The do_queue keyword argument specifies whether or not
            # the operator is queued in a tape context or not.
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

.. code-block:: python

    >>> op = FlipAndRotate(0.1, wire_rot="q3", wire_flip="q1", do_flip=True)
    >>> op
    FlipAndRotate(0.1, wires=['q3', 'q1'])
    >>> op.decomposition()
    [PauliX(wires=['q1']), RX(0.1, wires=['q3'])]
    >>> op.adjoint()
    FlipAndRotate(-0.1, wires=['q3', 'q1'])

The new gate can be used in devices, which access the decomposition to interpret it:

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

We can even compute gradients of circuits that use the new gate.

>>> qml.grad(circuit)(a)
-0.0015926529164868282

Defining special properties of an operator
##########################################

Apart from the main ``Operator`` class, operators with special methods or representations
are implemented as subclasses ``Operation``, ``Observable``, ``Channel``,
``CVOperation`` and ``CVOperation``.

However, unlike many other frameworks, PennyLane does not use class
inheritance to define properties of operators, such as whether it is its own self-inverse, if it is diagonal,
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
for adding custom operations to the attributes such as `composable_rotations`
and ``self_inverses`` that are used in compilation transforms. For example,
suppose you have created a new Operation, `MyGate`, which you know to be its
own inverse. Adding it to the set, like so

>>> from pennylane.ops.qubits.attributes import self_inverses
>>> self_inverses.add("MyGate")

These attributes can be queried by devices and compilation pipelines to use special tricks that speed
up computation. The onus is on the contributors of new operators to add them to the right attributes.

.. note::

    The attributes for qubit gates are currently found in ``pennylane/ops/qubit/attributes.py``.

Adding your new operator to PennyLane
#####################################

Once the new operator is coded up, it is added to the appropriate folder in ``pennylane/ops/``. The
tests are added to a file of a similar name and location in ``tests/ops/``. If your operator defines an
ansatz, add it to the appropriate subfolder in ``pennylane/templates``.

The new operation may have to be imported in the module's ``__init__.py`` file in order to be imported correctly.

Make sure that all hyperparameters and errors are tested, and that the parameters can be passed as
tensors from all supported autodifferentiation frameworks.

Don't forget to also add the new operator to documentation in the ``docs/introduction/operations.rst`` file, or to
the template gallery if it is an ansatz. The latter is done by adding a ``customgalleryitem``
to the correct section in ``doc/introduction/templates.rst``:

.. code-block::

  .. customgalleryitem::
    :link: ../code/api/pennylane.templates.<templ_type>.MyNewTemplate.html
    :description: MyNewTemplate
    :figure: ../_static/templates/<templ_type>/my_new_template.png

.. note::

  This loads the image of the template added to ``doc/_static/templates/test_<templ_type>/``. Make sure that
  this image has the same dimensions and style as other template icons in the folder.

Here are a few more tipps for adding operators:

* *Choose the name carefully.* Good names tell the user what the operator is used for,
  or what architecture it implements. Ask yourself if a gate of a similar name could
  be added soon in a different context.

* *Write good docstrings.* Explain what your operator does in a clear docstring with ample examples.

* *Efficient representations.* Try to implement representations as efficiently as possible, since they may
  be constructed several times.

* *Input checks.* Checking the inputs of the operation introduces an overhead and clashes with tools like
  just-in-time compilation. Find a balance of adding meaningful sanity checks (such as for the shape of tensors),
  but keeping them to a minimum.
