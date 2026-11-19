# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This is the top level module from which all basic functions and classes of
PennyLane can be directly imported.
"""

# pylint: disable=wrong-import-order


import warnings

from pennylane2 import _compatibility

_compatibility.check_classic_import()

from pennylane2 import exceptions
from pennylane2.boolean_fn import BooleanFn
from pennylane2 import numpy
from pennylane2 import queuing
from pennylane2.core.apply import apply
from pennylane2.core.queuing import QueuingManager

from pennylane2 import compiler
from pennylane2.compiler import qjit
from pennylane2 import capture
from pennylane2.capture import hint
from pennylane2 import core
from pennylane2 import control_flow
from pennylane2.control_flow import for_loop, while_loop
from pennylane2 import math
from pennylane2 import operation
from pennylane2.core import operator
from pennylane2 import allocation
from pennylane2.allocation import allocate, deallocate
from pennylane2 import decomposition
from pennylane2.decomposition import (
    register_resources,
    register_condition,
    add_decomps,
    list_decomps,
    exclude_decomps,
    inspect_decomps,
    resource_rep,
)
from pennylane2 import templates
from pennylane2 import pauli
from pennylane2.pauli import pauli_decompose
from pennylane2 import qchem
from pennylane2.fermi import (
    FermiC,
    FermiA,
    FermiWord,
    FermiSentence,
    jordan_wigner,
    parity_transform,
    bravyi_kitaev,
)
from pennylane2.bose import (
    BoseSentence,
    BoseWord,
    binary_mapping,
    unary_mapping,
    christiansen_mapping,
)
from pennylane2.qchem import (
    taper,
    symmetry_generators,
    paulix_ops,
    taper_operation,
    import_operator,
    from_openfermion,
    to_openfermion,
)
from pennylane2.numeric_hamiltonians import (
    CDFHamiltonian,
    CGFHamiltonian,
    VibronicHamiltonian,
)
from pennylane2._grad import grad, jacobian, vjp, jvp, value_and_grad
from pennylane2._version import __version__
from pennylane2.about import about
from pennylane2.circuit_graph import CircuitGraph
from pennylane2.registers import registers
from pennylane2.measurements import (
    counts,
    density_matrix,
    expval,
    probs,
    sample,
    state,
    var,
    vn_entropy,
    purity,
    mutual_info,
    classical_shadow,
    shadow_expval,
)
from pennylane2.ops import *
from pennylane2.ops import (
    adjoint,
    ctrl,
    cond,
    change_op_basis,
    exp,
    sum,
    pow,
    prod,
    s_prod,
    measure,
)
from pennylane2.ops import LinearCombination as Hamiltonian
from pennylane2.decomposition import gate_sets
from pennylane2.templates import layer
from pennylane2.templates.embeddings import *
from pennylane2.templates.layers import *
from pennylane2.templates.tensornetworks import *
from pennylane2.templates.swapnetworks import *
from pennylane2.templates.state_preparations import *
from pennylane2.templates.subroutines import *
from pennylane2.workflow import QNode, qnode, execute, set_shots, marker
from pennylane2 import workflow
from pennylane2.resource import analyze, estimate, specs, track
from pennylane2 import resource

from pennylane2.transforms import (
    transform,
    batch_params,
    batch_input,
    batch_partial,
    compile,
    decompose,
    decomp_inspector,
    defer_measurements,
    dynamic_one_shot,
    quantum_monte_carlo,
    apply_controlled_Q,
    commutation_dag,
    pattern_matching,
    pattern_matching_optimization,
    clifford_t_decomposition,
    gridsynth,
    CompilePipeline,
)
from pennylane2.noise import (
    add_noise,
    insert,
    mitigate_with_zne,
    fold_global,
    poly_extrapolate,
    richardson_extrapolate,
    exponential_extrapolate,
)
from pennylane2.ops.functions import (
    dot,
    eigvals,
    equal,
    assert_equal,
    evolve,
    generator,
    is_commuting,
    is_hermitian,
    is_unitary,
    map_wires,
    matrix,
    simplify,
    iterative_qpe,
    commutator,
    comm,
    single_qubit_zyz_angles,
    subcircuit,
)
from pennylane2.ops.identity import I
from pennylane2.optimize import *
from pennylane2.debugging import (
    snapshots,
    breakpoint,
    debug_expval,
    debug_state,
    debug_probs,
    debug_tape,
)
from pennylane2.shadows import ClassicalShadow
from pennylane2.gradients import metric_tensor, adjoint_metric_tensor
from pennylane2 import gradients
from pennylane2.drawer import draw, draw_mpl

from pennylane2.io import (
    from_qasm,
    to_openqasm,
    from_qiskit,
    from_qiskit_noise,
    from_qiskit_op,
    from_qasm3,
)

from pennylane2 import logging

from pennylane2 import data

from pennylane2 import noise
from pennylane2.noise import NoiseModel

from pennylane2.devices import Tracker
from pennylane2.devices.device_constructor import device, refresh_devices

from pennylane2.backline import (
    Backline,
    Controller,
    Coprocessor,
    CoprocessorFunction,
    Endpoint,
)
from pennylane2.backline.runtime import declare as runtime_declare
from pennylane2.backline.runtime import runtime_call

from pennylane2 import spin

from pennylane2 import liealg
from pennylane2.liealg import lie_closure, structure_constants, center

from pennylane2 import estimator

from pennylane2 import testing

from importlib.metadata import version as _metadata_version

from importlib.util import find_spec as _find_spec
from packaging.version import Version as _Version

if _find_spec("numpy") is not None:
    if (numpy_version := _Version(_metadata_version("numpy"))) < _Version(
        "2.0.0"
    ):  # pragma: no cover
        warnings.warn(
            "PennyLane v0.44 has dropped maintainence support for NumPy < 2.0.0. "
            f"You have version {numpy_version} installed. "
            "Future versions of PennyLane will not work with NumPy<2.0. "
            "Please consider upgrading NumPy using "
            "`python -m pip install numpy --upgrade`. ",
            exceptions.PennyLaneDeprecationWarning,
        )

from ._entry_points_utils import _setup_entry_points

# add to _entry_point_groups as new groups are added and desired to be accessed top-level
_entry_point_groups = ["pennylane2.drawer"]

__all__, __getattr__, __dir__ = _setup_entry_points(__name__, _entry_point_groups)


def version():
    """Returns the PennyLane version number."""
    return __version__
