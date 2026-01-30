qml.math
========

Overview
--------

.. currentmodule:: pennylane.math

.. automodapi:: pennylane.math
    :no-heading:
    :no-inherited-members:
    :skip: binary_finite_reduced_row_echelon, binary_is_independent, binary_matrix_rank, binary_select_basis, binary_solve_linear_system, int_to_binary

Functions for binary arithmetic and binary linear algebra
---------------------------------------------------------

There are some dedicated functions for handling binary numerics, such as linear algebraic
functions over :math:`\mathbb{Z}_2` or traceable integer-to-binary conversion.

.. autosummary::
    :toctree: api

    ~binary_finite_reduced_row_echelon
    ~binary_is_independent
    ~binary_matrix_rank
    ~binary_select_basis
    ~binary_solve_linear_system
    ~int_to_binary

Utility functions for decompositions
------------------------------------

.. autosummary::
    :toctree: api

    ~decomposition.decomp_int_to_powers_of_two
    ~decomposition.givens_decomposition
    ~decomposition.su2su2_to_tensor_products
    ~decomposition.xyx_rotation_angles
    ~decomposition.xzx_rotation_angles
    ~decomposition.zxz_rotation_angles
    ~decomposition.zyz_rotation_angles
