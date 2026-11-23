# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""This module contains decompositions for quantum circuits."""

from importlib import import_module as _import_module

from .clifford_t_transform import clifford_t_decomposition
from .gridsynth import gridsynth

_FACTORY_MODULES = {
    "make_phase_gradient_distillation_decomp": "pennylane.decomposition_rules.transforms.decompositions.phase_gradient_state_prep_decomp",
    "make_rz_to_phase_gradient_decomp": "pennylane.decomposition_rules.transforms.decompositions.rz_phase_gradient",
    "validate_phase_gradient_wires": "pennylane.decomposition_rules.transforms.decompositions.rz_phase_gradient",
    "phase_gradient_precision_stats": "pennylane.decomposition_rules.transforms.decompositions.rz_phase_gradient",
    "make_selectpaulirot_to_phase_gradient_decomp": "pennylane.decomposition_rules.transforms.decompositions.select_pauli_rot_phase_gradient",
    "make_crz_to_phase_gradient_decomp": "pennylane.decomposition_rules.transforms.decompositions.crz_phase_gradient",
}


__all__ = ["clifford_t_decomposition", "gridsynth", *_FACTORY_MODULES]


def __getattr__(name):
    """Resolve public rule factories after built-in initialization."""
    if name in _FACTORY_MODULES:
        return getattr(_import_module(_FACTORY_MODULES[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
