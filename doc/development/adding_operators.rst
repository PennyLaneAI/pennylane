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

Creating new operators in PennyLane2 can be done using one of the two items
below.

.. autosummary::

    ~pennylane.subcircuit
    ~pennylane.core.operator.Operator2

The :func:`~pennylane.subcircuit` function and the
:class:`~pennylane.core.operator.Operator2` only differ syntactically; using
either will result in a valid quantum operator that is agnostic to how it
originated.

For most cases, :func:`~pennylane.subcircuit` is recommended for its ease of
use. Utilizing the :class:`~pennylane.core.operator.Operator2` class is
recommended for the utmost customization.

.. _operator_argument_types:

Operator argument types
#######################

When creating a new quantum operator in PennyLane2, the input arguments to the
operator must be accurately binned into the following categories for
compatibility with the :func:`~pennylane.qjit` compiler.

* ``wire_argnames`` : The names of arguments corresponding to wires. For
  example, :class:`~pennylane.MultiControlledX`'s ``wires`` and ``work_wires``
  arguments belong in ``wire_argnames``.

* ``dynamic_argnames`` : The names of arguments that are scalars, arrays, or
  castable to arrays. These arguments can be treated by the
  :func:`~pennylane.qjit` compiler as *dynamic*. For example, the ``bitstrings``
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

The following is an example for a two-wire gate called ``FlipAndRotate``. This
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

Using the qp.subcircuit function
--------------------------------

We can define ``FlipAndRotate`` as a quantum function and turn it into a proper
quantum operator by decorating with :func:`~pennylane.subcircuit`. Before
decorating with :func:`~pennylane.subcircuit`, however, the quantum resources
associated with the implementation of ``FlipAndRotate`` need to be declared
with :func:`~pennylane.register_resources`.

.. code-block:: python

    import pennylane as qp

    # The resource function must have the same signature as the operator
    def flip_rotate_resources(angle, rot_axes, wires):
        return {
            # Declare the resources of BOTH conditional branches
            qp.PauliRot(qp.typing.Float, rot_axes[0], wires=qp.typing.Wire): 1,
            qp.PauliRot(qp.typing.Float, rot_axes[1], wires=qp.typing.Wire): 1,
            qp.X: 1,
            qp.ops.mid_measure.MidMeasure(qp.typing.Wire): 1
        }

    @qp.subcircuit(
        dynamic_argnames=("angle",),
        compilable_argnames=("rot_axes",),
        wire_argnames=("wires",),
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

.. note::

    - Resource functions and decomposition rule definitions must have the same
      signature as the operator.

    - Resources declared in the resource function must cover *all* operators
      contained within every conditional branch, not just one branch.

    - Some operators have arguments that affect their resource usage, like the
      ``pauli_word`` of :class:`~pennylane.PauliRot` and the number of wires
      that it acts on. Every unique instance of such an operator must be
      declared in the resources function, where abstract argument types in the
      :mod:`~pennylane.typing` module can be used as placeholder values.


In defining ``FlipAndRotate`` this way, it is an ``Operator2`` subclass

>>> from pennylane.core.operator import Operator2
>>> op = FlipAndRotate(0.1, ("X", "Y"), wires=(0, 1))
>>> isinstance(op, Operator2)
True

and the implementation provided in the quantum function body is registered as a
decomposition rule that integrates as expected with :func:`~pennylane.decompose`
and resource estimation.

.. code-block:: python

    dev = qp.device("lightning.qubit", wires=2)

    @qp.qjit(capture=True)
    @qp.decompose(gate_set={"PauliX", "MidMeasureMP", "PauliRot"})
    @qp.qnode(dev)
    def circuit(a):
        FlipAndRotate(a, ("X", "Y"), wires=[0, 1])
        return qp.expval(qp.Z(1)), qp.expval(qp.X(1))

>>> qp.list_decomps(FlipAndRotate)
Available Decomposition Rules:
0: FlipAndRotate_decomp
>>> qp.analyze(circuit, level=1)(0.1).resources.quantum_operations
{'MidCircuitMeasure': 1, 'PauliRot': 1, 'PauliX': 1}
>>> qp.estimate(circuit, target={"PauliX", "MidMeasureMP", "PauliRot"})(0.1).resources.quantum_operations
{'MidCircuitMeasure': 1, 'PauliRot': 1, 'PauliX': 1}
>>> qp.track(circuit)(0.1)[1].resources.quantum_operations
{'MidCircuitMeasure': 1, 'PauliRot-Phi-w1': 1, 'PauliX': 1}

Note that the decomposition rule name given by default is
``qfunc.name + "_decomp"`` (i.e., the name of the quantum function plus
"_decomp").

Subclassing Operator2
---------------------

The exact same result as what is shown above with using
:func:`~pennylane.subcircuit` can be achieved by creating a new class that
inherits from :class:`~pennylane.core.operator.Operator2`. The benefits to
doing so are mostly related to testing, like being able to easily validate
input types and shapes.

.. code-block:: python

    from pennylane.core.operator import Operator2

    class FlipAndRotate(Operator2):

        dynamic_argnames = ("angle",)
        compilable_argnames = ("rot_axes",)
        wire_argnames = ("wires",)

        def __init__(self, angle, rot_axes, wires):
            rot_axes = tuple(rot_axes)
            if len(rot_axes) != 2 or any(ax not in ("I", "X", "Y", "Z") for ax in rot_axes):
                raise ValueError(f"rot_axes must be two of 'I', 'X', 'Y', 'Z'; got {rot_axes}.")
            super().__init__(angle, rot_axes, wires)


    # The resource function and decomposition rule must have the same signature as the operator
    def _flip_and_rotate_resources(angle, rot_axes, wires):
        # Declare the resources of BOTH conditional branches
        return {
            qp.X: 1,
            qp.ops.MidMeasure(Wire[1]): 1,
            qp.PauliRot(qp.typing.Float, pauli_word=rot_axes[0], wires=qp.typing.Wire[1]): 1,
            qp.PauliRot(qp.typing.Float, pauli_word=rot_axes[1], wires=qp.typing.Wire[1]): 1,
        }


    @qp.register_resources(_flip_and_rotate_resources)
    def FlipAndRotate_decomp(angle, rot_axes, wires):
        qp.X(wires[0])
        m = qp.measure(wires[0])

        def true_fn():
            qp.PauliRot(2 * angle, rot_axes[0], wires=wires[1])

        def false_fn():
            qp.PauliRot(2 * angle, rot_axes[1], wires=wires[1])

        qp.cond(m, true_fn, false_fn)()


    qp.add_decomps(FlipAndRotate, FlipAndRotate_decomp)

Adding your new operator to PennyLane
#####################################

If you want PennyLane to natively support your new operator, you have to make a
Pull Request that adds it to the appropriate folder in ``pennylane/``. The tests
are added to a file of a similar name and location in ``tests/``. Generally, the
new operator should also be imported in the module's ``__init__.py`` file in
order for users to import the operator directly from ``pennylane``.

Don't forget to also add the new operator to the documentation in the
``doc/introduction/operations.rst`` file, or to
``doc/introduction/subroutines.rst`` if it falls under that category. The latter
is done by adding an entry, consisting of the subroutine and the path to a
thumbnail image, to the ``subroutine-table`` in the correct section of
``doc/introduction/subroutines.rst``:

.. code-block:: rst

  .. subroutine-table::
      :nosignatures:

      ~pennylane.MyNewSubroutine    /_static/templates/<subroutine_type>/my_new_subroutine.png

.. note::

  The table displays the first line of the subroutine's docstring, so make sure
  that it is a concise summary of what the subroutine does.

Here are a few more tips for adding operators:

* *Choose the name carefully.* Good names tell the user what the operator is used for,
  or what architecture it implements. Ask yourself if a gate of a similar name could
  be added soon in a different context.

* *Write good docstrings.* Explain what your operator does in a clear docstring with ample examples.
  You find more about our standards in the guidelines on :doc:`/development/guide/documentation`.

* *Efficient representations.* Try to implement representations as efficiently as possible, since they may
  be constructed several times.

* *Input checks.* Checking the inputs of the operation introduces an overhead and clashes with tools like
  just-in-time compilation. Find a balance of adding meaningful sanity checks (such as for the shape of tensors),
  but keeping them to a minimum.

Technical notes
###############

The :class:`~pennylane.core.operator.Operator2` class serves as the main
abstraction of such objects in PennyLane2, and is designed for compatibility
with the :func:`~pennylane.qjit` compiler. Gates and subroutines must inherit
from it.

>>> from pennylane.core.operator import Operator2
>>> op = qp.Rot(0.1, 0.2, 0.3, wires=0)
>>> isinstance(op, Operator2)
True

The basic components of operators are the following:

#. **The name of the operator** (:attr:`~.Operator2.name`), which is the name of
   its class.

   >>> op.name
   'Rot'

#. **The wires that the operator acts on** (:attr:`~.Operator2.wires`), which
   mathematically speaking define the subspace that it acts on. These are
   collected from the arguments in ``wire_argnames``, in the order in which they
   are listed. Work wires (``work_wires``) are *not* included.

   >>> op.wires
   Wires(0)

#. **The arguments of the operator** (:attr:`~.Operator2.arguments`), a
   dictionary mapping each argument name to its value. Every argument is also
   accessible as an attribute.

   >>> op.arguments
   {'phi': 0.1, 'theta': 0.2, 'omega': 0.3, 'wires': Wires(0)}
   >>> op.phi
   0.1

   Each argument belongs to one of the categories described in
   :ref:`Operator argument types <operator_argument_types>`, which are declared
   as class variables. The arguments of each category can be accessed with
   :attr:`~.Operator2.wire_args`, :attr:`~.Operator2.dynamic_args`,
   :attr:`~.Operator2.static_args`, :attr:`~.Operator2.compilable_args`, and
   :attr:`~.Operator2.hybrid_args`:

   >>> qp.Rot.dynamic_argnames
   ('phi', 'theta', 'omega')
   >>> op = qp.Select([qp.X(0), qp.Z(0)], control=[1], partial=True)
   >>> op.hybrid_args
   {'ops': [X(0), Z(0)]}
   >>> op.static_args
   {'partial': True}
   >>> op.wire_args
   {'control': Wires([1]), 'work_wires': Wires([])}

#. Possible **symbolic or numerical representations** of the operator, which can
   be used by PennyLane2's devices and compiler to interpret the map. Examples
   are:

   * Representation via the **eigenvalue decomposition** specified by
     eigenvalues (for the diagonal matrix, :meth:`~.Operator2.eigvals`) and
     diagonalizing gates (for the unitaries,
     :meth:`~.Operator2.diagonalizing_gates`):

     >>> op = qp.X(0)
     >>> op.diagonalizing_gates()
     [H(0)]
     >>> op.eigvals()
     array([ 1., -1.])

   * Representation as a **matrix** (:meth:`~.Operator2.matrix`), as specified
     by a global wire order that tells us where the wires are found on a
     register:

     >>> op = qp.PauliRot(0.2, "X", wires=["b"])
     >>> qp.math.round(op.matrix(wire_order=["a", "b"]), 3)
     array([[0.995+0.j , 0.   -0.1j, 0.   +0.j , 0.   +0.j ],
            [0.   -0.1j, 0.995+0.j , 0.   +0.j , 0.   +0.j ],
            [0.   +0.j , 0.   +0.j , 0.995+0.j , 0.   -0.1j],
            [0.   +0.j , 0.   +0.j , 0.   -0.1j, 0.995+0.j ]])

   * Representation via a **generator** (:meth:`~.Operator2.generator`):

     >>> qp.RX(0.5, wires=0).generator()
     -0.5 * X(0)

   Whether a representation is defined can be checked with the class properties
   :attr:`~.Operator2.has_decomposition`, :attr:`~.Operator2.has_matrix`, and so
   on.
