qml.estimator
=============

.. currentmodule:: pennylane.estimator

.. automodule:: pennylane.estimator

The :mod:`~.estimator` module is the centre for quantum resource estimation in PennyLane.  
Functionality within :mod:`~.estimator` is intended purely for resource estimation, and is *not* intended for true program compilation and execution.

.. seealso::
    The :mod:`~.resource` module for resource tracking of compiled programs.

Resource Estimation Functions:
------------------------------

:func:`~.estimator.estimate.estimate` is the standard entry point for estimating the resources required by a quantum program.
Submit a program written with either standard PennyLane operators (:class:`~.Operation`)
or resource operators (:class:`~.estimator.resource_operator.ResourceOperator`),
and instantly receive the resource estimate in the form of :class:`~.estimator.resources_base.Resources`.

.. currentmodule:: pennylane.estimator

.. autosummary::
    :toctree: api

    ~estimate.estimate
    ~resource_operator.resource_rep


Resource Estimation Base Classes:
---------------------------------

.. currentmodule:: pennylane.estimator

.. autosummary::
    :toctree: api

    ~resources_base.Resources
    ~resource_operator.ResourceOperator
    ~resource_operator.CompressedResourceOp
    ~resource_operator.GateCount
    ~resource_config.ResourceConfig


Qubit Management Classes:
-------------------------

.. automodsumm:: pennylane.estimator.wires_manager
    :toctree: api
    :no-inherited-members:
    :classes-only:
    :skip: QueuingManager


Resource Operators:
-------------------

.. automodsumm:: pennylane.estimator.ops
    :toctree: api
    :no-inherited-members:
    :classes-only:
    :noindex:


Resource Templates:
-------------------

.. automodsumm:: pennylane.estimator.templates
    :toctree: api
    :no-inherited-members:
    :classes-only:
    :noindex:


Resource Hamiltonians:
----------------------

.. automodsumm:: pennylane.estimator.compact_hamiltonian
    :toctree: api
    :no-inherited-members:
    :classes-only: