.. _contributing_operators:

Contributing new operators
==========================

The following steps will help you to add your favourite operator to PennyLane.

Note that in PennyLane, a circuit ansatz consisting of multiple gates is also an operator - one whose
map is defined as a combination of other operators. For historical reasons, you find circuit ansaetze
in the ``pennylane/template`` folder, while all other operations are found in ``pennylane/ops``.

The base classes to construct new operators are found in ``pennylane/operations.py``.

Basic idea
##########

Operators in quantum mechanics are maps that act on vector spaces. The highest-level operator class
serves as the main abstraction of such objects. Its basic components are the following:

#. The name of the operator, which may have a canonical, universally known interpretation (such as a "Hadamard" gate),
   or could be a name specific to PennyLane.

   .. code-block:: python

       >>> from jax import numpy as jnp
       >>> op = qml.PauliRot(jnp.array(0.2), "XY", wires=["a", "b"])
       >>> op.name
       PauliRot

#. The subsystems that the operator addresses, which mathematically speaking defines the subspace that it acts on.

   .. code-block:: python

       >>> op.wires
       <Wires = ['a', 'b']>

#. Trainable parameters that the map depends on, such as a rotation angle,
   which can be fed to the operator as tensor-like objects.

   .. code-block:: python

      >>> op.parameters
      [DeviceArray(0.2, dtype=float32, weak_type=True)]

#. Non-trainable hyperparameters that influence the action of the operator, such as the Pauli word that
   a Pauli rotation rotates around.

   .. code-block:: python

       >>> op.hyperparameters
       {'pauli_word': 'XY'}

#. Possible symbolic or numerical representations of the operator, which can be used by PennyLane's
   devices to interpret the map. Examples are:

   * Representation as a product of operators

     .. code-block:: python

         >>> op = qml.Rot(0.1, 0.2, 0.3, wires=["a"])
         >>> op.decomposition()
         [RZ(0.1, wires=['a']), RY(0.2, wires=['a']), RZ(0.3, wires=['a'])]

   * Representation as a linear combination of operators

    .. code-block:: python

        >>> op = qml.Hamiltonian([1., 2.], [qml.PauliX(0), qml.PauliZ(0)])
        >>> op.terms()
        ((1.0, 2.0), [PauliX(wires=[0]), PauliZ(wires=[0])])

   * Representation by the eigenvalue decomposition

     .. code-block:: python

         >>> op = qml.PauliX(0)
         >>> op.diagonalizing_gates()
         [Hadamard(wires=[0])]
         >>> op.eigvals()
         [ 1 -1]

   * Representation as a matrix

     .. code-block:: python

         >>> op = qml.PauliRot(0.2, "X", wires=["b"])
         >>> op.matrix()
         [[9.95004177e-01-2.25761781e-18j 2.72169462e-17-9.98334214e-02j]
          [2.72169462e-17-9.98334214e-02j 9.95004177e-01-2.25761781e-18j]]

   * Representation as a sparse matrix

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

.. code-block:: python

    >>> op = qml.PauliX(0) + 0.1*qml.PauliZ(0)
    >>> op.name
    Hamiltonian

    >>> qml.RX(1., wires=0).adjoint()
    RX(-1.0, wires=[0])

Operator base class
###################

The operator base class provides default functionality to store name, wires, parameters, hyperparameters
and representations. In addition, it defines a few methods that connect Operators to other building blocks
in PennyLane, such as expansion used by tapes or queueing functionality.

Roughly speaking, the architecture of the base class is this:

.. code-block:: python

    class Operator(abc.ABC):

        def __init__(self, *params, wires=None):
            # the default name is inferred from the class
            self._name = self.__class__.__name__
            # turn wires into a Wires object and store
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

Apart from the main ``Operator`` class, operators with special properties (such as those with a Kraus matrix
representation) are implemented as general subclasses ``Operation``, ``Observable``, ``Channel`` or ``CVOperation``.
However, unlike many other frameworks, PennyLane does not

Creating new Operators
#######################

The main job of adding a new Operator is to create a subclasses that overwrites as many of these default properties
as possible. First decide which general class you want to subclass - if your operator is used as a unitary gate,
you may want to inherit from ``Operation`` which provides functionality to control a gate, while an observable
may best inherit from ``Observable``.

The following is an example for a custom gate that rotates a qubit and possibly flips another qubit.
The custom operator defines a decomposition, which the devices will use (since it is unlikely that a device
knows a native implementation for ``FlipAndRotate``), as well as an adjoint method.

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
        num_wires = qml.operation.AnyWires  # if wire_rot and wire_flip are the same we have 1 wire, else 2
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

            # can turn into Wires objects here, or use other Iterables
            all_wires = qml.wires.Wires(wire_rot) + qml.wires.Wires(wire_flip)

            super().__init__(angle, wires=all_wires, do_queue=do_queue, id=id)

        @property
        def num_params(self):
            # if it is a fixed value, define the number of parameters to expect here,
            # which makes sure an error is raised if the wrong number was passed
            return 1

        @staticmethod
        def compute_decomposition(angle, wires, do_flip):  # pylint: disable=arguments-differ
            """Overwriting the static ``compute_`` methods defines a representation,
            for example here a representation as a sequence of a flip and a rotation gate.

            The ``compute_`` methods expect the signature ``(*parameters, wires, **hyperparameters)``
            or (for numerical representations) ``(*parameters, **hyperparameters)``. Defining the
            parameters and hyperparameters by name makes the representation easier to read.

            If a representation does not make use of all hyperparameters, a signature of the form
            ``(param1, wires, hyperparam1, **kwargs)`` can be used.
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

The new gate can be used in devices, which access the decomposition to implement it:

.. code-block:: python

    from pennylane import numpy as pnp

    dev = qml.device("default.qubit", wires=["q1", "q2", "q3"])

    @qml.qnode(dev)
    def circuit(angle):
        FlipAndRotate(angle, wire_rot="q1", wire_flip="q1")
        return qml.expval(qml.PauliZ("q1"))

    >>> a = pnp.array(3.14)
    >>> circuit(a)
    -0.9999987318946099

We can even compute gradients of circuits that use the new gate.

.. code-block:: python

    >>> qml.grad(circuit)(a)
    -0.0015926529164868282


Adding your new operator to PennyLane
#####################################

Once the new operator is coded up, it is added to the appropriate folder in ``pennylane/ops/``. The
tests are added to a file of a similar name and location in ``tests/ops/``. Make sure that all hyperparameters
are tested, and that the parameters can be passed as tensors from all supported autodifferentiation frameworks.

The new operation may have to be imported in the module's ``__init__.py`` file in order to be imported correctly.

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

Overall, it is recommended to consider the following:

* *Choose the name carefully.* Good names tell the user what a template is used for,
  or what architecture it implements. The class name (i.e., ``MyNewTemplate``) is written in camel case.

* *Explicit decompositions.* Try to implement the decomposition in the ``decomposition()`` function
  without the use of convenient methods like the :func:`~.broadcast` function - this avoids
  unnecessary overhead.

* *Write an extensive docstring that explains how to use the template.* Include a sketch of the template (add the
  file to the ``doc/_static/templates/<templ_type>/`` directory). You should also display a small usage example
  at the beginning of the docstring. If you want to explain the behaviour in more detail, add a section starting
  with the ``.. UsageDetails::`` directive at the end of the docstring.
  Use the docstring of one of the existing templates for inspiration, such as
  :func:`AmplitudeEmbedding <pennylane.templates.embeddings.AmplitudeEmbedding>`.

* *Input checks.* While checking the inputs of the template for consistency introduces an overhead and should be
  kept to a minimum, it is still advised to do some basic sanity checks, for example making sure that the shape of the
  parameters is correct.
