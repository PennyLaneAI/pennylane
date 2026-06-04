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


Core Infrastructure
^^^^^^^^^^^^^^^^^^^

.. autosummary::

    ~pennylane.core.measurements.MeasurementProcess
    ~pennylane.core.measurements.StateMeasurement
    ~pennylane.core.measurements.SampleMeasurement
    ~pennylane.core.measurements.MeasurementTransform


Measurement Classes
^^^^^^^^^^^^^^^^^^^

.. autosummary::

    ~pennylane.measurements.ClassicalShadowMP
    ~pennylane.measurements.CountsMP
    ~pennylane.measurements.DensityMatrixMP
    ~pennylane.measurements.ExpectationMP
    ~pennylane.measurements.MutualInfoMP
    ~pennylane.measurements.NullMeasurement
    ~pennylane.measurements.ProbabilityMP
    ~pennylane.measurements.PurityMP
    ~pennylane.measurements.SampleMP
    ~pennylane.measurements.ShadowExpvalMP
    ~pennylane.measurements.StateMP
    ~pennylane.measurements.VarianceMP
    ~pennylane.measurements.VnEntropyMP