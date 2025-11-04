# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
This submodule contains the API for the integration of PennyLane and Catalyst with xDSL.

.. currentmodule:: pennylane.compiler.python_compiler

.. warning::

    This module is currently experimental and will not maintain API stability between releases.

To get started with the unified compiler and xDSL, check out the
:doc:`quickstart guide </development/unified_compiler/unified_compiler_cookbook>`

.. automodapi:: pennylane.compiler.python_compiler
    :no-heading:
    :include-all-objects:

Dialects
--------

.. currentmodule:: pennylane.compiler.python_compiler.dialects
.. autosummary::
    :toctree: api

    Catalyst
    MBQC
    Quantum
    QEC
    StableHLO
    Transform


Available Transforms
--------------------

Quantum
~~~~~~~

.. currentmodule:: pennylane.compiler.python_compiler.transforms.quantum
.. autosummary::
    :toctree: api

    combine_global_phases_pass
    CombineGlobalPhasesPass
    diagonalize_final_measurements_pass
    DiagonalizeFinalMeasurementsPass
    iterative_cancel_inverses_pass
    IterativeCancelInversesPass
    measurements_from_samples_pass
    MeasurementsFromSamplesPass
    merge_rotations_pass
    MergeRotationsPass
    split_non_commuting_pass
    SplitNonCommutingPass


MBQC
~~~~

.. currentmodule:: pennylane.compiler.python_compiler.transforms.mbqc
.. autosummary::
    :toctree: api

    convert_to_mbqc_formalism_pass
    ConvertToMBQCFormalismPass
    decompose_graph_state_pass
    DecomposeGraphStatePass
    OutlineStateEvolutionPass
    outline_state_evolution_pass
    null_decompose_graph_state_pass
    NullDecomposeGraphStatePass


Transforms Core API
-------------------

.. currentmodule:: pennylane.compiler.python_compiler.pass_api
.. autosummary::
    :toctree: api

    ApplyTransformSequence
    available_passes
    PassDispatcher
    register_pass
    TransformFunctionsExt
    TransformInterpreterPass
    compiler_transform


Inspection and Visualization
----------------------------

.. currentmodule:: pennylane.compiler.python_compiler.pass_api
.. autosummary::
    :toctree: api

    draw
    generate_mlir_graph

"""

from .compiler import Compiler
from .parser import QuantumParser
from .pass_api import compiler_transform
from .visualization import QMLCollector


__all__ = [
    "Compiler",
    "compiler_transform",
    "QuantumParser",
    "QMLCollector",
]
