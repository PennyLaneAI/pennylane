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

    The opt-in feature ``qml.operation.enable_new_opmath()`` is now the default. Ideally, your code should not break.
    If it still does, it likely only requires some minor changes. For that, see the :ref:`Troubleshooting_opmath` section.
    You can still opt-out and run legacy code via ``qml.operation.disable_new_opmath()``.


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

  You can transform the :class:`~.pennylane.pauli.PauliSentence` back to a suitable :class:`~.pennylane.operation.Operator` via the :meth:`~pennylane.pauli.PauliSentence.operation` or :meth:`~pennylane.pauli.PauliSentence.hamiltonian`   method.

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
    | ``qml.dot(coeffs,ops)``                    | ``ops.Sum``          | ``ops.Sum``               |
    +--------------------------------------------+----------------------+---------------------------+
    | ``qml.Hamiltonian(coeffs, ops)``           | ``ops.Hamiltonian``  | ``ops.LinearCombination`` |
    +--------------------------------------------+----------------------+---------------------------+
    | ``qml.ops.LinearCombination(coeffs, ops)`` | n/a                  | ``ops.LinearCombination`` |
    +--------------------------------------------+----------------------+---------------------------+


    The three main new opmath classes :class:`~.pennylane.ops.op_math.SProd`, :class:`~.pennylane.ops.op_math.Prod`, and :class:`~.pennylane.ops.op_math.Sum` have already been around for a while.
    E.g., :func:`~.pennylane.dot` has always returned a :class:`~.pennylane.ops.op_math.Sum` instance.

    **Usage**

    Besides the python operators, you can also use the constructors :func:`~.pennylane.s_prod`, :func:`~.pennylane.prod`, and :func:`~.pennylane.sum`.
    For composite operators, we can access their constituents via the ``op.operands`` attribute.

    >>> op = qml.sum(X(0), X(1), X(2))
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
    >>> qml.simplify(op)
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

    **qml.Hamiltonian**

    The legacy classes :class:`~.pennylane.operation.Tensor` and :class:`~.pennylane.Hamiltonian` will soon be deprecated.
    :class:`~.ops.op_math.LinearCombination` offers the same API as :class:`~.pennylane.Hamiltonian` but works well with new opmath classes.

    Depending on whether or not new opmath is active, ``qml.Hamiltonian`` will return either of the two classes.

    >>> import pennylane as qml
    >>> from pennylane import X
    >>> qml.operation.active_new_opmath()
    True
    >>> H = qml.Hamiltonian([0.5, 0.5], [X(0), X(1)])
    >>> type(H)
    pennylane.ops.op_math.linear_combination.LinearCombination

    >>> qml.operation.disable_new_opmath()
    >>> qml.operation.active_new_opmath()
    False
    >>> H = qml.Hamiltonian([0.5, 0.5], [X(0), X(1)])
    >>> type(H)
    pennylane.ops.qubit.hamiltonian.Hamiltonian

    Both classes offer the same API and functionality, so a user does not have to worry about those implementation details.

.. _Troubleshooting_opmath:

Troubleshooting
---------------

You may experience issues with PennyLane's updated operator arithmetic in version ``v0.36`` and above if you have existing code that works with an earlier version of PennyLane.
To help identify a fix, select the option below that describes your situation.

.. details::
    :title: My old PennyLane script does not run anymore
    :href: old-script-broken

    A quick-and-dirty fix for this issue is to deactivate new opmath at the beginning of the script via ``qml.operation.disable_new_opmath()``.
    We recommend to do the following checks instead

    * Check explicit use of the legacy :class:`~Tensor` class. If you find it in your script it can just be changed from ``Tensor(*terms)`` to ``qml.prod(*terms)`` with the same call signature.

    * Check explicit use of the ``op.obs`` attribute, where ``op`` is some operator. This is how the terms of a tensor product are accessed in :class:`~.pennylane.operation.Tensor` instances. Use ``op.operands`` instead.

      .. code-block:: python

          # new opmath enabled (default)
          op = X(0) @ X(1)
          assert op.operands == (X(0), X(1))

          with qml.operation.disable_new_opmath_cm():
              # context manager that disables new opmath temporarilly
              op = X(0) @ X(1)
              assert op.obs == [X(0), X(1)]
    
    * Check explicit use of ``qml.ops.Hamiltonian``. In that case, simply change to ``qml.Hamiltonian``.

      >>> op = qml.ops.Hamiltonian([0.5, 0.5], [X(0) @ X(1), X(1) @ X(2)])
      ValueError: Could not create circuits. Some or all observables are not valid.
      >>> op = qml.Hamiltonian([0.5, 0.5], [X(0) @ X(1), X(1) @ X(2)])
      >>> isinstance(op, qml.ops.LinearCombination)
      True
    
    * Check if you are explicitly enabling and disabling new opmath somewhere in your script. Mixing both systems is not supported.

    If for some unexpected reason your script still breaks, please post on the PennyLane `discussion forum <https://discuss.pennylane.ai>`_ or open a
    `bug report <https://github.com/PennyLaneAI/pennylane/issues/new?labels=bug+%3Abug%3A&template=bug_report.yml&title=[BUG]>`_
    on the PennyLane GitHub page.

.. details::
    :title: Sharp bits about the qml.Hamiltonian dispatch
    :href: sharp-bits-hamiltonian

    One of the reasons that :class:`~.ops.op_math.LinearCombination` exists is that the old Hamiltonian class is not compatible with new opmath tensor products.
    We can try to instantiate an old ``qml.ops.Hamiltonian`` class with a ``X(0) @ X(1)`` tensor product, which returns a :class:`~.pennylane.ops.Prod` instance with new opmath enabled.

    >>> qml.operation.active_new_opmath() # confirm opmath is active (by default)
    True
    >>> qml.ops.Hamiltonian([0.5], [X(0) @ X(1)])
    PennyLaneDeprecationWarning: Using 'qml.ops.Hamiltonian' with new operator arithmetic is deprecated. Instead, use 'qml.Hamiltonian', or use 'qml.operation.disable_new_opmath()' to continue to access the legacy functionality. See https://docs.pennylane.ai/en/stable/development/deprecations.html for more details.
    ValueError: Could not create circuits. Some or all observables are not valid.

    However, using ``qml.Hamiltonian`` works as expected.

    >>> qml.Hamiltonian([0.5], [X(0) @ X(1)])
    0.5 * (X(0) @ X(1))

    The API of :class:`~.ops.op_math.LinearCombination` is identical to that of :class:`~.Hamiltonian`. We can group observables or simplify upon initialization.

    >>> H1 = qml.Hamiltonian([0.5, 0.5, 0.5], [X(0) @ X(1), X(0), Y(0)], grouping_type="qwc", simplify=True)
    >>> H2 = qml.ops.LinearCombination([0.5, 0.5, 0.5], [X(0) @ X(1), X(0), Y(0)], grouping_type="qwc", simplify=True)
    >>> H1 == H2
    True

    One small difference is that ``ham.simplify()`` no longer alters the instance in-place. In either case (legacy/new opmath), the following works.

    >>> H1 = qml.Hamiltonian([0.5, 0.5], [X(0) @ X(1), X(0) @ X(1)])
    >>> H1 = H1.simplify() # work for new and legacy opmath

.. details::
    :title: I want to contribute to PennyLane and need to provide legacy support in tests
    :href: PL-developer

    If you want to contribute a new feature to PennyLane or update an existing one, you likely also need to update the tests.

    .. note::
        Please refrain from explicitly using ``qml.operation.disable_new_opmath()`` and ``qml.operation.enable_new_opmath()`` anywhere in tests as that globally
        changes the status of new opmath and thereby can affect other tests.

        .. code-block:: python3

            def test_some_legacy_opmath_behavior():
                qml.operation.disable_new_opmath() # dont do this
                # testing some legacy behavior things

            def test_some_new_opmath_behavior():
                assert qml.operation.active_new_opmath()
                # will fail because the previous test globally disabled new opmath

        Instead, please use the fixtures below, or the context managers ``qml.operation.disable_new_opmath_cm()`` and ``qml.operation.enable_new_opmath_cm()``.

        >>> with qml.operation.disable_new_opmath_cm():
        ...     op = qml.Hamiltonian([0.5], [X(0) @ X(1)])
        >>> assert isinstance(op, qml.ops.Hamiltonian)

    Our continuous integration (CI) test suite is running all tests with the new opmath enabled by default.
    We also periodically run the CI test suite with new opmath disabled, as we support both the new and legacy systems for a limited time.
    In case a test needs to be adopted for either case, you can use the following fixtures.

    * Use ``@pytest.mark.usefixtures("use_legacy_opmath")`` to test functionality that is explicitly only supported by legacy opmath (e.g., for backward compatibility).
      
      .. code-block:: python3

        @pytest.mark.usefixtures("use_legacy_opmath")
        def test_qml_hamiltonian_legacy_opmath():
            assert qml.Hamiltonian == qml.ops.Hamiltonian

        def test_qml_hamiltonian()
            assert qml.Hamiltonian == qml.ops.LinearCombination

    * Use ``@pytest.mark.usefixtures("use_new_opmath")`` to test functionality that `only` works with new opmath. That is because for the intermittent period 
      of supporting both systems, we periodically run the test suite with new opmath disabled.

      .. code-block:: python3

        @pytest.mark.usefixtures("use_new_opmath")
        def test_qml_hamiltonian_new_opmath():
            assert qml.Hamiltonian == qml.ops.LinearCombination
    
    * Use ``@pytest.mark.usefixtures("use_legacy_and_new_opmath")`` if you want to test support for both systems in one single test. You can use ``qml.operation.active_new_opmath``
      inside the test to account for minor differences between both systems.

      .. code-block:: python3

        @pytest.mark.usefixtures("use_legacy_and_new_opmath")
        def test_qml_hamiltonian_new_opmath():
            if qml.operation.active_new_opmath():
                assert qml.Hamiltonian == qml.ops.LinearCombination
            
            if not qml.operation.active_new_opmath():
                assert qml.Hamiltonian == qml.ops.Hamiltonian
    
    One sharp bit about testing is that ``pytest`` runs collection and test execution separately. That means that instances generated outside the test, e.g., for parametrization, have been created
    using the respective system. So you may need to also put that creation in the appropriate context manager.

    .. code-block:: python3

        # in some test file
        with qml.operation.disable_new_opmath_cm():
            legacy_ham_example = qml.Hamiltonian(coeffs, ops) # creates a Hamiltonian instance

        @pytest.mark.usefixtures("use_legacy_opmath")
        @pytest.mark.parametrize("ham", [legacy_ham_example])
        def test_qml_hamiltonian_legacy_opmath(ham):
            assert isinstance(ham, qml.Hamiltonian) # True
            assert isinstance(ham, qml.ops.Hamiltonian) # True

    Alternatively, you can convert them back to legacy Hamiltonian instances using ``qml.operation.convert_to_legacy_H()``. 

    .. code-block:: python3

        ham_example = qml.Hamiltonian(coeffs, ops) # creates a LinearCombination instance

        @pytest.mark.usefixtures("use_new_opmath")
        @pytest.mark.parametrize("ham", [ham_example])
        def test_qml_hamiltonian_new_opmath(ham):
            assert isinstance(ham, qml.Hamiltonian) # True
            assert not isinstance(ham, qml.ops.Hamiltonian) # True

        @pytest.mark.usefixtures("use_legacy_opmath")
        @pytest.mark.parametrize("ham", [ham_example])
        def test_qml_hamiltonian_legacy_opmath(ham):
            # Most likely you wanted to test things with a Hamiltonian instance
            legacy_ham_example = convert_to_legacy_H(ham)
            assert isinstance(legacy_ham_example, qml.ops.Hamiltonian) # True
            assert isinstance(legacy_ham_example, qml.Hamiltonian) # True because we are in legacy opmath context
            assert not isinstance(legacy_ham_example, qml.ops.LinearCombination) # True
    
    For all that, keep in mind that ``qml.Hamiltonian`` points to :class:`~pennylane.Hamiltonian` and :class:`LinearCombination` depending on the status of ``qml.operation.active_new_opmath()``.
    So if you want to test something specifically for the old :class:`~pennylane.Hamiltonian`` class, use ``qml.ops.Hamiltonian`` instead.

.. details::
    :title: Sharp bits about the nesting structure of new opmath instances
    :href: sharp-bits-nesting

    The type of the final operator is determined by the outermost operation. The resulting object is a nested structure (sums of s/prods or s/prods of sums).

    >>> qml.operation.enable_new_opmath() # default soon
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
    the form :math:`\sum_i c_i \hat{O}_i`, where :math:`c_i` is a scalar coefficient and :math:`\hat{O}_i` is a pure operator or tensor product of operators.

    >>> op1 = 0.5 * (X(0) @ X(1)) + 0.5 * (Y(0) @ Y(1))
    >>> op2 = (0.5 * X(0)) @ X(1) + (0.5 * Y(0)) @ Y(1)
    >>> op3 = 0.5 * (X(0) @ X(1) + Y(0) @ Y(1))
    >>> qml.equal(op1, op2), qml.equal(op2, op3), qml.equal(op3, op1)
    (True, False, False)

    >>> op1 = op1.simplify()
    >>> op2 = op2.simplify()
    >>> op3 = op3.simplify()
    >>> qml.equal(op1, op2), qml.equal(op2, op3), qml.equal(op3, op1)
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
