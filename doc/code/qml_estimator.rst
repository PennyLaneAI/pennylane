qml.estimator
=============

.. currentmodule:: pennylane.estimator

.. automodule:: pennylane.estimator

The :mod:`~.estimator` module is the centre for quantum resource estimation in PennyLane.  
Functionality within :mod:`~.estimator` is intended purely for resource estimation, and is *not* intended for true program compilation and execution.

For a quick introduction to :mod:`~.estimator`, take a look at our demo on
`How to use PennyLane for Resource Estimation <https://pennylane.ai/qml/demos/re_how_to_use_pennylane_for_resource_estimation>`_!

.. seealso::
    The :mod:`~.resource` module for resource tracking of compiled programs.

Resource Estimation
-------------------

The :func:`~.estimator.estimate.estimate` function is the standard entry point for estimating the
resources required by a quantum program. Submit a program written with either standard PennyLane
operators (:class:`~.Operation`) or resource operators (:class:`~.estimator.resource_operator.ResourceOperator`),
and instantly receive the resource estimate in the form of :class:`~.estimator.resources_base.Resources`.

.. currentmodule:: pennylane.estimator

.. autosummary::
    :toctree: api

    ~estimate.estimate
    ~resources_base.Resources
    ~resource_config.ResourceConfig


Resource Operators
------------------

.. currentmodule:: pennylane.estimator

.. autosummary::
    :toctree: api

    ~resource_operator.ResourceOperator
    ~resource_operator.CompressedResourceOp
    ~resource_operator.GateCount


Qubit Management
----------------

.. automodsumm:: pennylane.estimator.wires_manager
    :toctree: api
    :no-inherited-members:
    :classes-only:
    :skip: QueuingManager


Resource Operators
------------------

.. automodsumm:: pennylane.estimator.ops
    :toctree: api
    :no-inherited-members:
    :classes-only:
    :noindex:


Resource Templates
------------------

.. automodsumm:: pennylane.estimator.templates
    :toctree: api
    :no-inherited-members:
    :classes-only:
    :noindex:


Resource Hamiltonians
---------------------

.. automodsumm:: pennylane.estimator.compact_hamiltonian
    :toctree: api
    :no-inherited-members:
    :classes-only:


QPE Resource Classes
--------------------

.. currentmodule:: pennylane.estimator

.. autosummary::
    :toctree: api

    ~qpe_resources.FirstQuantization
    ~qpe_resources.DoubleFactorization


Measurement Functions
---------------------

.. currentmodule:: pennylane.estimator

.. autosummary::
    :toctree: api

    ~measurement.estimate_shots
    ~measurement.estimate_error
