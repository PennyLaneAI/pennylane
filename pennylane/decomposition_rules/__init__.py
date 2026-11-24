# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Built-in decomposition rules, organized like their operator and template modules.

The modules directly beneath this package mirror :mod:`pennylane.ops`, and the
``templates`` branch mirrors :mod:`pennylane.templates`.
The drawing-specific ``LabelledOp`` rule remains in :mod:`pennylane.drawer.label`.
Rules are loaded automatically when PennyLane is imported.
Use :func:`pennylane.list_decomps` to retrieve the registered rules for an operator.
Unambiguous rules, helpers, and factories can also be imported from this package.

.. currentmodule:: pennylane.decomposition_rules

.. autosummary::

    qubit
    common
    null_decomp
    identity
    meta
    templates
"""

from .common import null_decomp
from .identity import controlled_g_phase_standard
from . import meta
from .qubit import arithmetic_ops
from .qubit import matrix_ops
from .qubit.non_parametric_ops import (
    _PPRBasis,
    controlled_hadamard_standard,
    controlled_z_standard,
    controlled_z_to_ccz,
    ctrl_x_to_mcx,
)
from .qubit.parametric_ops_single_qubit import (
    _YToZBasis,
    _ZToYBasis,
    controlled_phase_shift_standard,
    controlled_rot_standard,
    controlled_rx_standard,
)
from .qubit.parametric_ops_multi_qubit import (
    _IsingXYBasis,
    _IsingXYRotations,
    _MultiRZLadder,
    _PauliRotBasis,
)
from .qubit.qchem_ops import (
    _SingleExcitationBasis,
    _SingleExcitationCRZBasis,
    _SingleExcitationRZRotations,
    _SingleExcitationRotations,
)
from .qubit import special_unitary
from .qubit import state_preparation
from .templates.embeddings import amplitude

__all__ = [
    "_IsingXYBasis",
    "_IsingXYRotations",
    "_MultiRZLadder",
    "_PPRBasis",
    "_PauliRotBasis",
    "_SingleExcitationBasis",
    "_SingleExcitationCRZBasis",
    "_SingleExcitationRZRotations",
    "_SingleExcitationRotations",
    "_YToZBasis",
    "_ZToYBasis",
    "controlled_g_phase_standard",
    "controlled_hadamard_standard",
    "controlled_phase_shift_standard",
    "controlled_rot_standard",
    "controlled_rx_standard",
    "controlled_z_standard",
    "controlled_z_to_ccz",
    "ctrl_x_to_mcx",
    "null_decomp",
]
