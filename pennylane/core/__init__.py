# Copyright 2018-2026 Xanadu Quantum Technologies Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The core abstractions of PennyLane.

Wires
~~~~~

.. currentmodule:: pennylane.core.wires
.. autosummary::
    :toctree: api

    Wires
    WiresLike
    DynamicWire
    is_abstract_wire
    AbstractQubit

Queuing
~~~~~~~

.. currentmodule:: pennylane.core.queuing
.. autosummary::
    :toctree: api

    ~QueuingManager
    ~AnnotatedQueue
    ~apply

Operator Types
~~~~~~~~~~~~~~

.. currentmodule:: pennylane.core.operator
.. autosummary::
    :toctree: api

    ~Operator
    ~Operator1
    ~Operator2
    ~Operation
    ~CV
    ~CVObservable
    ~CVOperation
    ~Channel
    ~StatePrepBase

.. currentmodule:: pennylane.core.operator

.. inheritance-diagram:: Operator Operation Channel CV CVObservable CVOperation StatePrepBase
    :parts: 1


Measurements
~~~~~~~~~~~~

.. currentmodule:: pennylane.core.measurements
.. autosummary::
    :toctree: api

    ~MeasurementProcess
    ~StateMeasurement
    ~SampleMeasurement
    ~MeasurementTransform

Shots
~~~~~

.. currentmodule:: pennylane.core.shots
.. autosummary::
    :toctree: api

    ~Shots
    ~ShotCopies
    ~ShotsLike

Quantum Script
~~~~~~~~~~~~~~

.. currentmodule:: pennylane.core.qscript
.. autosummary::
    :toctree: api

    ~QuantumScript
    ~QuantumScriptBatch
    ~QuantumScriptOrBatch
    ~make_qscript
    ~process_queue


Transforms
~~~~~~~~~~

.. currentmodule:: pennylane.core.transforms
.. autosummary::
    :toctree: api

    ~Transform
    ~transform
    ~BoundTransform
    ~CompilePipeline


"""

from .wires import Wires, DynamicWire, is_abstract_qubit, WiresLike, AbstractQubit

from .queuing import QueuingManager, AnnotatedQueue, apply

from .operator import (
    CV,
    Channel,
    CVObservable,
    CVOperation,
    Operation,
    Operator,
    Operator1,
    Operator2,
    StatePrepBase,
    StatePrepBase2,
)
from .measurements import (
    MeasurementProcess,
    StateMeasurement,
    SampleMeasurement,
    MeasurementTransform,
)
from .shots import Shots, ShotCopies, ShotsLike
from .qscript import (
    QuantumScript,
    QuantumScriptBatch,
    QuantumScriptOrBatch,
    make_qscript,
    process_queue,
)
from .transforms import transform, Transform, BoundTransform, CompilePipeline

__all__ = [
    "Wires",
    "DynamicWire",
    "is_abstract_qubit",
    "WiresLike",
    "AbstractQubit",
    "QueuingManager",
    "AnnotatedQueue",
    "apply",
    "Operator",
    "Operator1",
    "Operator2",
    "Operation",
    "Channel",
    "CV",
    "CVOperation",
    "CVObservable",
    "StatePrepBase",
    "StatePrepBase2",
    "MeasurementProcess",
    "StateMeasurement",
    "SampleMeasurement",
    "MeasurementTransform",
    "Shots",
    "ShotCopies",
    "ShotsLike",
    "QuantumScript",
    "QuantumScriptBatch",
    "QuantumScriptOrBatch",
    "make_qscript",
    "process_queue",
    "transform",
    "Transform",
    "BoundTransform",
    "CompilePipeline",
]
