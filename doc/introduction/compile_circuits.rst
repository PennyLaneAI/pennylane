.. role:: html(raw)
   :format: html

.. _intro_ref_opt:

Compiling circuits
==================

"how do I change/compile my quantum algorithms?"
compile transforms (single qubit fusion, pattern matching, SWAP transpile, etc.)
Compilation transforms for specific environments (error mitigation)
general manipulation (insert, merge amplitude gates, hamiltonian expansion)
decompositions (custom decompositions, expansion etc.)
circuit cutting (qcut)
small section at the bottom linking to functions for 'how do I write my own compilation transform?'

 Pauli words
^^^^^^^^^^^^^^^^^^^^

Grouping Pauli words can be used for the optimizing the measurement of qubit
Hamiltonians. Along with groups of observables, post-measurement rotations can
also be obtained using :func:`~.optimize_measurements`:

.. code-block:: python

    >>> obs = [qml.PauliY(0), qml.PauliX(0) @ qml.PauliX(1), qml.PauliZ(1)]
    >>> coeffs = [1.43, 4.21, 0.97]
    >>> post_rotations, diagonalized_groupings, grouped_coeffs = optimize_measurements(obs, coeffs)
    >>> post_rotations
    [[RY(-1.5707963267948966, wires=[0]), RY(-1.5707963267948966, wires=[1])],
     [RX(1.5707963267948966, wires=[0])]]

The post-measurement rotations can be used to diagonalize the partitions of
observables found.

For further details on measurement optimization, grouping observables through
solving the minimum clique cover problem, and auxiliary functions, refer to the
:doc:`/code/qml_grouping` subpackage.
