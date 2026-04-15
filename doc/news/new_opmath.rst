.. _new_opmath:

Updated Operators
=================

In version ``0.36`` of PennyLane we changed some things behind the scenes on how operators and arithmetic operations between them are handled.
This realizes a few objectives:

1. To make it as easy to work with PennyLane operators as it would be with pen and paper.
2. To improve the efficiency of operator arithmetic.

In many cases, these changes should not break code and the difference to previous
versions may not be noticeable.

This page provides additional details about operator arithmetic updates and can be
used to troubleshoot issues for those affected users.

.. note::

    If you are looking for a quick fix, jump to the :ref:`Troubleshooting_opmath` section!

    After visiting the :ref:`Troubleshooting_opmath` section, if you are still stuck then you can:

    - Post on the PennyLane `discussion forum <https://discuss.pennylane.ai>`_.

    - If you suspect that your issue is due to a bug in PennyLane itself, please open a
      `bug report <https://github.com/PennyLaneAI/pennylane/issues/new?labels=bug+%3Abug%3A&template=bug_report.yml&title=[BUG]>`_
      on the PennyLane GitHub page.

Summary of the update
---------------------

.. rst-class:: admonition tip

    After a period of deprecation, the legacy behaviour for operators was removed in PennyLane version 0.40.
    Anyone using the latest PennyLane will automatically use the updated operator arithmetic. Ideally, your code
    should not break when making this update. If it still does, it likely only requires some minor changes.
    For that, see the :ref:`Troubleshooting_opmath` section. If you were using any of the functions explictly
    provided to continue using the deprecated behaviour, like ``qp.operation.disable_new_opmath()``, that
    code will need to be removed.

* The underlying system for performing arithmetic with operators has been changed. Arithmetic can be carried out using
  standard Python operations like ``+``, ``*`` and ``@`` or via arithmetic functions located in :mod:`~.op_math`.

* You can now easily access Pauli operators via :obj:`~.pennylane.I`, :obj:`~.pennylane.X`, :obj:`~.pennylane.Y`, and :obj:`~.pennylane.Z`.

  >>> from pennylane import I, X, Y, Z
  >>> X(0)
  X(0)

  The original long-form names :class:`~.Identity`, :class:`~.PauliX`, :class:`~.PauliY`, and :class:`~.PauliZ` remain available and are functionally equivalent to :obj:`~.pennylane.I`, :obj:`~.pennylane.X`, :obj:`~.pennylane.Y`, and :obj:`~.pennylane.Z`, but
  use of the short-form names is now recommended.

* Operators in PennyLane can have a backend Pauli representation, which can be used to perform faster operator arithmetic. Now, the Pauli
  representation will be automatically used for calculations when available.
  You can access the ``pauli_rep`` attribute of any operator whenever it is available.

  >>> op = X(0) + Y(0)
  >>> op.pauli_rep
  1.0 * X(0)
  + 1.0 * Y(0)
  >>> type(op.pauli_rep)
  pennylane.pauli.pauli_arithmetic.PauliSentence

  You can transform the :class:`~.pennylane.pauli.PauliSentence` back to a suitable :class:`~.pennylane.operation.Operator` via the :meth:`~pennylane.pauli.PauliSentence.operation` method.

  >>> op.pauli_rep.operation()
  X(0) + Y(0)

* Extensive improvements have been made to the string representations of PennyLane operators,
  making them shorter and possible to copy-paste as valid PennyLane code.

  >>> 0.5 * X(0)
  0.5 * X(0)
  >>> 0.5 * (X(0) + Y(1))
  0.5 * (X(0) + Y(1))

  Sums with many terms are broken up into multiple lines, but can still be copied back as valid
  code:

  >>> 0.5 * (X(0) @ X(1)) + 0.7 * (X(1) @ X(2)) + 0.8 * (X(2) @ X(3))
  (
      0.5 * (X(0) @ X(1))
    + 0.7 * (X(1) @ X(2))
    + 0.8 * (X(2) @ X(3))
  )

.. details::
    :title: Technical details
    :href: technical-details

    The changes between the old and new system mainly concern Python operators ``+ - * / @``,
    that now create the following ``Operator`` subclass instances.


    +--------------------------------------------+----------------------+---------------------------+
    |                                            | Legacy               | Updated Operators         |
    +============================================+======================+===========================+
    | tensor products                            | ``operation.Tensor`` | ``ops.Prod``              |
    | ``X(0) @ X(1)``                            |                      |                           |
    +--------------------------------------------+----------------------+---------------------------+
    | sums                                       | ``ops.Hamiltonian``  | ``ops.Sum``               |
    | ``X(0) + X(1)``                            |                      |                           |
    +--------------------------------------------+----------------------+---------------------------+
    | scalar products                            | ``ops.Hamiltonian``  | ``ops.SProd``             |
    | ``1.5 * X(1)``                             |                      |                           |
    +--------------------------------------------+----------------------+---------------------------+
    | ``qp.dot(coeffs,ops)``                    | ``ops.Sum``          | ``ops.Sum``               |
    +--------------------------------------------+----------------------+---------------------------+
    | ``qp.Hamiltonian(coeffs, ops)``           | ``ops.Hamiltonian``  | ``ops.LinearCombination`` |
    +--------------------------------------------+----------------------+---------------------------+
    | ``qp.ops.LinearCombination(coeffs, ops)`` | n/a                  | ``ops.LinearCombination`` |
    +--------------------------------------------+----------------------+---------------------------+


    The three main new opmath classes :class:`~.pennylane.ops.op_math.SProd`, :class:`~.pennylane.ops.op_math.Prod`, and :class:`~.pennylane.ops.op_math.Sum` have already been around for a while.
    E.g., :func:`~.pennylane.dot` has always returned a :class:`~.pennylane.ops.op_math.Sum` instance.

    **Usage**

    Besides the python operators, you can also use the constructors :func:`~.pennylane.s_prod`, :func:`~.pennylane.prod`, and :func:`~.pennylane.sum`.
    For composite operators, we can access their constituents via the ``op.operands`` attribute.

    >>> op = qp.sum(X(0), X(1), X(2))
    >>> op.operands
    (X(0), X(1), X(2))

    In case all terms are composed of operators with a valid ``pauli_rep``, then the composite
    operator also has a valid ``pauli_rep`` in terms of a :class:`~.pennylane.pauli.PauliSentence` instance. This is often handy for fast
    arithmetic processing.

    >>> op.pauli_rep
    1.0 * X(0)
    + 1.0 * X(1)
    + 1.0 * X(2)

    Further, composite operators can be simplified using :func:`~.pennylane.simplify` or the ``op.simplify()`` method.

    >>> op = 0.5 * X(0) + 0.5 * Y(0) - 1.5 * X(0) - 0.5 * Y(0) # no simplification by default
    >>> op.simplify()
    -1.0 * X(0)
    >>> qp.simplify(op)
    -1.0 * X(0)

    Note that the simplification never happens in-place, such that the original operator is left unaltered.

    >>> op
    (
        0.5 * X(0)
      + 0.5 * Y(0)
      + -1 * 1.5 * X(0)
      + -1 * 0.5 * Y(0)
    )

    We are often interested in obtaining a list of coefficients and `pure` operators.
    We can do so by using the ``op.terms()`` method.

    >>> op = 0.5 * (X(0) @ X(1) + Y(0) @ Y(1) + 2 * Z(0) @ Z(1)) - 1.5 * I() + 0.5 * I()
    >>> op.terms()
    ([0.5, 0.5, 1.0, -1.0], [X(1) @ X(0), Y(1) @ Y(0), Z(1) @ Z(0), I()])

    As seen by this example, this method already takes care of arithmetic simplifications.

    **qp.Hamiltonian**

    The classes :class:`~.pennylane.operation.Tensor` and :class:`~.pennylane.ops.Hamiltonian` have been removed.
    The familiar ``qp.Hamiltonian`` can still be used, which dispatches to ``LinearCombination`` and offers the same
    usage and functionality but with different implementation details.

    >>> import pennylane as qp
    >>> from pennylane import X
    >>> H = qp.Hamiltonian([0.5, 0.5], [X(0), X(1)])
    >>> type(H)
    pennylane.ops.op_math.linear_combination.LinearCombination

.. _Troubleshooting_opmath:

Troubleshooting
---------------

You may experience issues with PennyLane's updated operator arithmetic in version ``v0.36`` and above if you have existing code that works with an earlier version of PennyLane.
To help identify a fix, select the option below that describes your situation.

.. details::
    :title: My old PennyLane script does not run anymore
    :href: old-script-broken

    We recommend to do the following checks:

    * Check explicit use of the legacy :class:`~Tensor` class. If you find it in your script it can just be changed from ``Tensor(*terms)`` to ``qp.prod(*terms)`` with the same call signature.

    * Check explicit use of the ``op.obs`` attribute, where ``op`` is some operator. This is how the terms of a tensor product are accessed in :class:`~.pennylane.operation.Tensor` instances. Use ``op.operands`` instead.

      .. code-block:: python

          op = X(0) @ X(1)
          assert op.operands == (X(0), X(1))
    
    * Check explicit use of ``qp.ops.Hamiltonian``. In that case, simply change to ``qp.Hamiltonian``.
      This will dispatch to the ``LinearCombination`` class, which offers the same API and functionality
      with different implementation details.

    If for some unexpected reason your script still breaks, please post on the PennyLane `discussion forum <https://discuss.pennylane.ai>`_ or open a
    `bug report <https://github.com/PennyLaneAI/pennylane/issues/new?labels=bug+%3Abug%3A&template=bug_report.yml&title=[BUG]>`_
    on the PennyLane GitHub page.

.. details::
    :title: Sharp bits about the qp.Hamiltonian dispatch
    :href: sharp-bits-hamiltonian

    The API of :class:`~.ops.op_math.LinearCombination` is mostly identical to that of the removed ``qp.ops.Hamiltonian``.

    One small difference is that ``Hamiltonian.simplify()`` no longer alters the instance in-place. Instead, you must do the

    following:

    >>> H1 = qp.Hamiltonian([0.5, 0.5], [X(0) @ X(1), X(0) @ X(1)])
    >>> H1 = H1.simplify()

.. details::
    :title: Sharp bits about the nesting structure of new opmath instances
    :href: sharp-bits-nesting

    The type of the final operator is determined by the outermost operation. The resulting object is a nested
    structure (sums of s/prods or s/prods of sums).

    >>> op = 0.5 * (X(0) @ X(1)) + 0.5 * (Y(0) @ Y(1))
    >>> type(op)
    pennylane.ops.op_math.sum.Sum

    >>> op.operands
    (0.5 * (X(0) @ X(1)), 0.5 * (Y(0) @ Y(1)))

    >>> type(op.operands[0]), type(op.operands[1])
    (pennylane.ops.op_math.sprod.SProd, pennylane.ops.op_math.sprod.SProd)

    >>> op.operands[0].scalar, op.operands[0].base, type(op.operands[0].base)
    (0.5, X(0) @ X(1), pennylane.ops.op_math.prod.Prod)

    We could construct an equivalent operator with a different nesting structure.

    >>> op = (0.5 * X(0)) @ X(1) + (0.5 * Y(0)) @ Y(1)
    >>> op.operands
    ((0.5 * X(0)) @ X(1), (0.5 * Y(0)) @ Y(1))

    >>> type(op.operands[0]), type(op.operands[1])
    (pennylane.ops.op_math.prod.Prod, pennylane.ops.op_math.prod.Prod)

    >>> op.operands[0].operands
    (0.5 * X(0), X(1))

    >>> type(op.operands[0].operands[0]), type(op.operands[0].operands[1])
    (pennylane.ops.op_math.sprod.SProd,
     pennylane.ops.qubit.non_parametric_ops.PauliX)
    
    There is yet another way to construct the same, equivalent, operator.
    We can bring all of them to the same format by using ``op.simplify()``, which brings the operator down to 
    the form :math:`\sum_i c_i \hat{O}_i`, where :math:`c_i` is a scalar coefficient and :math:`\hat{O}_i` is a
    pure operator or tensor product of operators.

    >>> op1 = 0.5 * (X(0) @ X(1)) + 0.5 * (Y(0) @ Y(1))
    >>> op2 = (0.5 * X(0)) @ X(1) + (0.5 * Y(0)) @ Y(1)
    >>> op3 = 0.5 * (X(0) @ X(1) + Y(0) @ Y(1))
    >>> qp.equal(op1, op2), qp.equal(op2, op3), qp.equal(op3, op1)
    (True, False, False)

    >>> op1 = op1.simplify()
    >>> op2 = op2.simplify()
    >>> op3 = op3.simplify()
    >>> qp.equal(op1, op2), qp.equal(op2, op3), qp.equal(op3, op1)
    (True, True, True)

    >>> op1, op2, op3
    (0.5 * (X(1) @ X(0)) + 0.5 * (Y(1) @ Y(0)),
     0.5 * (X(1) @ X(0)) + 0.5 * (Y(1) @ Y(0)),
     0.5 * (X(1) @ X(0)) + 0.5 * (Y(1) @ Y(0)))
    
    We can also obtain those scalar coefficients and tensor product operators via the ``op.terms()`` method.

    >>> coeffs, ops = op1.terms()
    >>> coeffs, ops
    ([0.5, 0.5], [X(1) @ X(0), Y(1) @ Y(0)])

.. details::
    :title: I am unsure what to do
    :href: unsure

    Please carefully read through the options above. If you are still stuck, you can:

    - Post on the PennyLane `discussion forum <https://discuss.pennylane.ai>`_. Please include
      a complete block of code demonstrating your issue so that we can quickly troubleshoot.

    - If you suspect that your issue is due to a bug in PennyLane itself, please open a
      `bug report <https://github.com/PennyLaneAI/pennylane/issues/new?labels=bug+%3Abug%3A&template=bug_report.yml&title=[BUG]>`_
      on the PennyLane GitHub page.
