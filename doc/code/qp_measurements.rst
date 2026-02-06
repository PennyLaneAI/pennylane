qp.measurements
================

.. currentmodule:: pennylane.measurements

.. warning::

    Unless you are a PennyLane or plugin developer, you likely do not need
    to use these classes directly.

    See the :doc:`main measurements page <../introduction/measurements>` for
    details on available measurements.


.. automodule:: pennylane.measurements

Overview
--------

Top-level measurement functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :mod:`~pennylane.measurements` functions for creating standard PennyLane measurement
processes are top-level imports:

.. autosummary::

    ~pennylane.classical_shadow
    ~pennylane.counts
    ~pennylane.density_matrix
    ~pennylane.expval
    ~pennylane.measure
    ~pennylane.mutual_info
    ~pennylane.probs
    ~pennylane.purity
    ~pennylane.sample
    ~pennylane.shadow_expval
    ~pennylane.state
    ~pennylane.var
    ~pennylane.vn_entropy


.. automodapi:: pennylane.measurements
    :no-heading:
    :no-main-docstr:
    :include-all-objects:
    :inheritance-diagram:
    :skip: MeasurementShapeError, classical_shadow, counts, density_matrix, expval, mutual_info, probs, purity, sample, shadow_expval, state, var, vn_entropy
