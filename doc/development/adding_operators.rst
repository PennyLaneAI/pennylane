.. _contributing_operators:

Adding custom operators
=======================

The following steps will help you to add your favourite operator to PennyLane.

Note that in PennyLane, a circuit ansatz consisting of multiple gates is also an operator - one whose
map is defined as a combination of other operators. For historical reasons, you find circuit ansaetze
in the ``pennylane/template`` folder, while all other operations are found in ``pennylane/ops``.

The base classes to construct new operators are found in ``pennylane/operations.py``.

Abstraction
###########

Operators in quantum mechanics are maps that act on vector spaces. The highest-level operator class
(the ``Operator`` class which we will introduce in the next section) serves as the main abstraction of such objects.
Its basic components are the following:

#. **The name of the operator**, which may have a canonical, universally known interpretation (such as a "Hadamard" gate),
   or could be a name specific to PennyLane.

   .. code-block:: python

       >>> from jax import numpy as jnp
       >>> op = qml.Rot(jnp.array(0.1), jnp.array(0.2), jnp.array(0.3), wires=["a"])
       >>> op.name
       Rot

#. **The subsystems that the operator addresses**, which mathematically speaking defines the subspace that it acts on.

   .. code-block:: python

       >>> op.wires
       <Wires = ['a']>

#. **Trainable parameters** that the map depends on, such as a rotation angle,
   which can be fed to the operator as tensor-like objects.

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

   * Representation as a **product of operators**

     .. code-block:: python

         >>> op = qml.Rot(0.1, 0.2, 0.3, wires=["a"])
         >>> op.decomposition()
         [RZ(0.1, wires=['a']), RY(0.2, wires=['a']), RZ(0.3, wires=['a'])]

   * Representation as a **linear combination of operators**

     .. code-block:: python

         >>> op = qml.Hamiltonian([1., 2.], [qml.PauliX(0), qml.PauliZ(0)])
         >>> op.terms()
         ((1.0, 2.0), [PauliX(wires=[0]), PauliZ(wires=[0])])

   * Representation by the **eigenvalue decomposition**

     .. code-block:: python

         >>> op = qml.PauliX(0)
         >>> op.diagonalizing_gates()
         [Hadamard(wires=[0])]
         >>> op.eigvals()
         [ 1 -1]

   * Representation as a **matrix**

     .. code-block:: python

         >>> op = qml.PauliRot(0.2, "X", wires=["b"])
         >>> op.matrix()
         [[9.95004177e-01-2.25761781e-18j 2.72169462e-17-9.98334214e-02j]
          [2.72169462e-17-9.98334214e-02j 9.95004177e-01-2.25761781e-18j]]

   * Representation as a **sparse matrix**

     .. code-block:: python

         >>> from scipy.sparse.coo import coo_matrix
         >>> row = np.array([0, 1])
         >>> col = np.array([1, 0])
         >>> data = np.array([1, -1])
         >>> mat = coo_matrix((data, (row, col)), shape=(4, 4))
         >>> op = qml.SparseHamiltonian(mat, wires=["a"])
         >>> op.sparse_matrix()
         (0, 1)   1
         (1, 0) - 1

New operators can be created by applying arithmetic functions to operators, such as addition, scalar multiplication,
multiplication, taking the adjoint, or controlling an operator. At the moment, such arithmetic is only implemented for
specific subclasses.

.. code-block:: pycon

    >>> # ``Observable`` defines addition and scalar multiplication
    >>> op = qml.PauliX(0) + 0.1 * qml.PauliZ(0)
    >>> op.name
    Hamiltonian
    >>> op
      (0.1) [Z0]
    + (1.0) [X0]

    >>> # ``Operation`` defindes the hermitian conjugate
    >>> qml.RX(1., wires=0).adjoint()
    RX(-1.0, wires=[0])

Operator base class
###################

The ``Operator`` base class provides default functionality to store name, wires, parameters, hyperparameters
and representations. In addition, it defines a few methods that connect operators to other building blocks
in PennyLane, such as expansion used by tapes or queueing functionality.

Roughly speaking, the architecture of the ``Operator`` base class is this:

.. code-block:: python

    class Operator(abc.ABC):

        def __init__(self, *params, wires=None):
            # the default name is inferred from the class
            self._name = self.__class__.__name__
            # turn wires into a PennyLane Wires object and store
            self._wires = Wires(wires)
            # store the parameters in an internal representation
            self.data = list(params)

        @property
        def name(self):
            return self._name

        @property
        def wires(self):
            return self._wires

        @property
        def parameters(self):
            return self.data.copy()

        @property
        def hyperparameters(self):
            # check for hyperparameters added by a child class
            if hasattr(self, "_hyperparameters"):
                return self._hyperparameters
            # else create and return empty hyperparameters as default
            self._hyperparameters = {}
            return self._hyperparameters

    # decomposition representation (instance method)
    def decomposition(self):
        return self.compute_decomposition(*self.parameters, self.wires, **self.hyperparameters)

    # decomposition representation (static method)
    @staticmethod
    def compute_decomposition(*params, wires=None, **hyperparameters):
        raise DecompositionUndefinedError

    # other representations
    ...

The representations, for which we see the ``decomposition`` as one example above, are accessible by
instance methods such as ``decomposition()``. These instance methods call a static method that uses the prefix
``compute_``, and in which the actual representation is computed. Sometimes, such a computation simply returns a
fixed object, but at other times a time-consuming calculation is performed. The idea of static methods is that
they can be cached across all instances of the same operator class, which can speed up computations drastically.

Defining special properties of an operator
##########################################

Apart from the main ``Operator`` class, operators with special properties (such as those with a Kraus matrix
representation) are implemented as general subclasses ``Operation``, ``Observable``, ``Channel``,
``CVOperation`` and ``CVOperation``. However, unlike many other frameworks, PennyLane does not use class
inheritance to define properties of operators such as whether it is its own self-inverse, if it is diagonal,
or whether it can be decomposed into Pauli rotations. The reason is that we want to avoid changing the inheritance structure
every time an application needs to query a new property.

Instead, properties are recorded in "attributes", which are bookkeeping classes listing those operators
that fulfill a specific property.

For example, we can create a new Attribute, ``pauli_ops``, like so:

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

The attributes for qubit gates are currently found in ``pennylane/ops/qubit/attributes.py``.

Creating new Operators
######################

The first job of adding a new Operator is to create a subclasse that overwrites as many of these default properties
as possible. First decide which general class you want to subclass --- if your operator is used as a unitary gate,
you may want to inherit from ``Operation`` which provides functionality to control a gate, while an observable
may best inherit from ``Observable``.

The following is an example for a custom gate that rotates a qubit and possibly flips another qubit.
The custom operator defines a decomposition, which the devices will use (since it is unlikely that a device
knows a native implementation for ``FlipAndRotate``), as well as an adjoint operator.

.. note::
    You will see a few bits and pieces that weren't explained above, such as the class attribute ``num_wires``,
    ``grad_method``, or the keyword argument ``do_queue``, which are currently undergoing a refactor - more
    to follow soon.

.. code-block:: python

    import pennylane as qml


    class FlipAndRotate(qml.operation.Operation):
        """One-sentence description of the operator.

        More explanation about the operator.

        Args:
            Inputs are described here

        **Example**

        Usage examples to be added here.
        """
        # if wire_rot and wire_flip are the same we have 1 wire, else 2,
        # which is why we cannot define a fixed number of wires, and use the AnyWires Enumeration instead
        num_wires = qml.operation.AnyWires
        grad_method = "A"  # supports parameter-shift differentiation

        def __init__(self, angle, wire_rot, wire_flip=None, do_flip=False, do_queue=True, id=None):

            # checking the inputs --------------
            if do_flip and wire_flip is None:
                raise ValueError("Need to specify a wire to flip")

            # note: we use the framework-agnostic math library for inputs that could be tensors
            if len(qml.math.shape(angle)) > 1:
                raise ValueError("Expected a scalar angle.")
            #------------------------------------

            # do_flip is not trainable but influences the map,
            # which is why we define it to be a hyperparameter
            self._hyperparameters = {
                "do_flip": do_flip
            }

            # We can turn into Wires objects here to use addition
            # Alternatively, we can work with the raw input and rely on ``super``
            # to turn the wires into a Wires object.
            all_wires = qml.wires.Wires(wire_rot) + qml.wires.Wires(wire_flip)

            super().__init__(angle, wires=all_wires, do_queue=do_queue, id=id)

        @property
        def num_params(self):
            # if it is a fixed value, define the number of parameters to expect here,
            # which makes sure an error is raised if the wrong number was passed
            return 1

        @staticmethod
        def compute_decomposition(angle, wires, do_flip):  # pylint: disable=arguments-differ
            """Overwriting this method defines the decomposition of the new gate.

            This method has to have the general signature (*parameters, wires, **hyperparameters).
            In our case, tha parameters consist of the angle, and the hyperparameters of do_flip.
            Using concrete argument names makes it easier to interpret the decomposition.

            .. note::
                If the gate defined other hyperparameters that we do not use in this method, a signature of the form
                (angle, wires, do_flip, **kwargs) could be used.
            """
            op_list = []
            if do_flip:
                op_list.append(qml.PauliX(wires=wires[1]))
            op_list.append(qml.RX(angle, wires=wires[0]))
            return op_list

        def adjoint(self):
            # the adjoint of this gate simply negates the angle
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

Here are a few more tipps:

* *Choose the name carefully.* Good names tell the user what a template is used for,
  or what architecture it implements. Ask yourself if a gate of a similar name could
  be added soon in a different context.

* *Write good docstrings.* Explain what your operator does in a clear docstring with ample examples.

* *Efficient representations.* Try implement representations as efficiently as possible, since they may
  be constructed several times.

* *Input checks.* Checking the inputs of the operation introduces an overhead and clashes with tools like
  just-in-time compilation. Find a balance of adding meaningful sanity checks (such as for the shape of tensors),
  but keeping them to a minimum.
