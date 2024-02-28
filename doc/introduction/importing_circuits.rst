.. role:: html(raw)
   :format: html

.. _intro_ref_importing_circuits:

Importing Circuits
==================

PennyLane offers the :mod:`~.io` module to import quantum circuits and operations that were
constructed outside of PennyLane. This includes circuits defined using `Qiskit <https://www.ibm.com/quantum/qiskit>`__,
`OpenQASM <https://openqasm.com/>`_, and `Quil <https://docs.rigetti.com/qcs/guides/quil>`_.

.. note::

    To import a quantum circuit defined using a particular framework, you will need to install the
    corresponding PennyLane plugin for that framework. More information about PennyLane plugins is
    available on the `plugins <https://pennylane.ai/plugins.html>`_ page.

Importing Quantum Circuits in PennyLane
---------------------------------------

TODO


Importing Quantum Operations in PennyLane
-----------------------------------------

TODO

Import Functions
----------------

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.from_pyquil
    ~pennylane.from_qasm
    ~pennylane.from_qasm_file
    ~pennylane.from_qiskit
    ~pennylane.from_qiskit_op
    ~pennylane.from_quil
    ~pennylane.from_quil_file

:html:`</div>`
