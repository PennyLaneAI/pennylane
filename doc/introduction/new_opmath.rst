.. _new_opmath:

New opmath
==========

Version ``0.36`` of PennyLane updated the default ``Operator`` subclasses, changing the operator arithmetic.
An end-user should not notice any (breaking) changes.
This pages should help developers with troubleshooting and guide in the process of legacy support while both systems are supported.

.. note::

    If you are looking for a quick fix, jump to the :ref:`Troubleshooting` section!

    After visiting the :ref:`Troubleshooting` section, if you are still stuck then you can:

    - Post on the PennyLane `discussion forum <https://discuss.pennylane.ai>`_.

    - If you suspect that your issue is due to a bug in PennyLane itself, please open a
      `bug report <https://github.com/PennyLaneAI/pennylane/issues/new?labels=bug+%3Abug%3A&template=bug_report.yml&title=[BUG]>`_
      on the PennyLane GitHub page.

.. note::

    The old return system has been completely removed from PennyLane as of version 0.33.0, along with the
    ``enable_return()`` and ``disable_return()`` toggle functions.

Summary of the update
---------------------

The opt-in feature ``qml.operation.enable_new_opmath()`` is now the default. You can still opt-out via
``qml.operation.disable_new_opmath()``.

The changes between the old and new system mainly concern Python operators ``+ - * / @``,
that now create the following ``Operator`` subclass instances.

+----------------------------------+------------------+------------------------+
|                                  | Legacy           | New opmath             |
+==================================+==================+========================+
| tensor products                  | ``operation.Tensor`` | ``ops.Prod``       |
| ``X(0) @ X(1)``                  |                  |                        |
+----------------------------------+------------------+------------------------+
| sums                             | ``ops.Hamiltonian`` | ``ops.Sum``         |
| ``X(0) + X(1)``                  |                  |                        |
+----------------------------------+------------------+------------------------+
| scalar products                  | ``ops.Hamiltonian`` | ``ops.SProd``       |
| ``1.5 * X(1)``                   |                  |                        |
+----------------------------------+------------------+------------------------+
| ``qml.dot(coeffs,ops)``          | ``ops.Sum``      | ``ops.Sum``            |
+----------------------------------+------------------+------------------------+
| ``qml.Hamiltonian(coeffs, ops)`` | ``ops.Hamiltonian`` | ``ops.LinearCombination``|
+----------------------------------+------------------+------------------------+

The three main new opmath classes ``SProd``, ``Prod``, and ``Sum`` have already been around for a while.
E.g. ``qml.dot()`` has always returned a ``Sum`` instance.
The legacy classes :class:`~Tensor` and :class:`~Hamiltonian` will soon be deprecated.
:class:`~LinearCombination` offers the same API as :class:`~Hamiltonian` but works well with new opmath classes.

Depending on whether or not new opmath is active, ``qml.Hamiltonian`` will return either of the two classes.

>>> import pennylane as qml
>>> from pennylane import X
>>> qml.operation.active_new_opmath()
True
>>> H = qml.Hamiltonian([0.5, 0.5], [X(0), X(1)])
>>> type(H)
pennylane.ops.op_math.linear_combination.LinearCombination

>>> qml.operation.disable_new_opmath_()
>>> qml.operation.active_new_opmath()
False
>>> H = qml.Hamiltonian([0.5, 0.5], [X(0), X(1)])
>>> type(H)
pennylane.ops.qubit.hamiltonian.Hamiltonian


.. _Troubleshooting:

Troubleshooting
---------------

If you are a developer or power-user that explicitly uses ``qml.operation.Tensor`` or ``qml.ops.Hamiltonian``, you
may run into one of the following common issues.

.. details::
    :title: My old PennyLane script does not run anymore
    :href: old-script-broken

    A quick-and-dirty fix for this issue is to deactivate new opmath at the beginning of the script via ``qml.operation.disable_new_opmath()``.
    We recommend to do the following checks instead

    * Check explicit use of the legacy :class:`~Tensor` class. If you find it in your script it can just be changed from ``Tensor(*terms)`` to ``qml.prod(*terms)`` with the same signature.

    * Check explicit use of ``op.obs`` attribute, where ``op`` is some operator. This is how the terms of a tensor product is accessed in :class:`~Tensor` instances. Use ``op.operands`` instead.

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

    If for some unexpected reason your script still breaks, please see the :ref:`I am unsure what to do <unsure>` section below.

.. details::
    :title: I want to contribute to PennyLane
    :href: PL-developer

    If you want to contribute a new feature to PennyLane or update an existing one, you likely also need to update the tests.
    Please refrain from explicitly using ``qml.operation.disable_new_opmath()`` and ``qml.operation.enable_new_opmath()`` anywhere in tests and code as that globally
    changes the status of new opmath and thereby can affect other parts of your code or other tests.

    Instead, please use the context managers ``qml.operation.disable_new_opmath_cm()`` and `qml.operation.enable_new_opmath_cm()``.

    >>> with qml.operation.disable_new_opmath_cm():
    ...     op = qml.Hamiltonian([0.5], [X(0) @ X(1)])
    >>> assert isinstance(op, qml.ops.Hamiltonian)

    Our continuous integration (CI) test suite is running all tests with the default of new opmath being enabled.
    We also periodically run the CI test suite with new opmath disabled, as we support both new and legacy systems for some limited time.
    In case a test needs to be adopted for either case, you can use the following fixtures.

    * Use ``@pytest.mark.usefixtures("use_legacy_opmath")`` to test functionality that is explicitly only supported by legacy opmath, e.g. for backward compatibility.

    * Use ``@pytest.mark.usefixtures("use_new_opmath")`` to test functionality that `only` works with new opmath. That is because for the intermittent period 
      of supporting both systems, we periodically run the test suite with new opmath disabled.
    
    * Use ``@pytest.mark.usefixtures("use_legacy_and_new_opmath")`` if you want to test support for both systems in one single test. You can use ``qml.operation.active_new_opmath``
      inside the test to account for minor differences between both systems.
    
    One sharp bit about testing is that ``pytest`` runs collection and test execution separately. That means that instances generated outside the test, e.g. for parametrization, have been created
    using the respective system. So you may need to also put that creation in the appropriate context manager.


.. details::
    :title: I am unsure what to do
    :href: unsure

    Please carefully read through the options above. If you are still stuck, you can:

    - Post on the PennyLane `discussion forum <https://discuss.pennylane.ai>`_. Please include
      a complete block of code demonstrating your issue so that we can quickly troubleshoot.

    - If you suspect that your issue is due to a bug in PennyLane itself, please open a
      `bug report <https://github.com/PennyLaneAI/pennylane/issues/new?labels=bug+%3Abug%3A&template=bug_report.yml&title=[BUG]>`_
      on the PennyLane GitHub page.
