qml.estimator
=============

.. currentmodule:: pennylane.estimator

.. automodule:: pennylane.estimator


Resource Estimation Functions:
------------------------------

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
    :skip: OutOfPlaceSquare, PhaseGradient, OutMultiplier, SemiAdder, QFT, AQFT, BasisRotation, Select, QROM, SelectPauliRot, ControlledSequence, QPE, IterativeQPE

Resource Templates:
-------------------

.. automodsumm:: pennylane.estimator.ops.templates
    :toctree: api
    :no-inherited-members:
    :classes-only:
    :noindex:
