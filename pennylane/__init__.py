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
import warnings

from pennylane import exceptions
from pennylane.boolean_fn import BooleanFn
from pennylane import numpy
from pennylane.queuing import QueuingManager, apply

from pennylane import compiler
from pennylane.compiler import qjit
from pennylane import capture
from pennylane import control_flow
from pennylane.control_flow import for_loop, while_loop
from pennylane import kernels
from pennylane import math
from pennylane import operation
from pennylane import allocation
from pennylane.allocation import allocate, deallocate
from pennylane import decomposition
from pennylane.decomposition import (
    register_resources,
    register_condition,
    add_decomps,
    list_decomps,
    resource_rep,
)
from pennylane import templates
from pennylane import pauli
from pennylane.pauli import pauli_decompose
from pennylane.resource import specs
from pennylane import resource
from pennylane import qchem
from pennylane.fermi import (
    FermiC,
    FermiA,
    FermiWord,
    FermiSentence,
    jordan_wigner,
    parity_transform,
    bravyi_kitaev,
)
from pennylane.bose import (
    BoseSentence,
    BoseWord,
    binary_mapping,
    unary_mapping,
    christiansen_mapping,
)
from pennylane.qchem import (
    taper,
    symmetry_generators,
    paulix_ops,
    taper_operation,
    import_operator,
    from_openfermion,
    to_openfermion,
)
from pennylane._grad import grad, jacobian, vjp, jvp
from pennylane._version import __version__
from pennylane.about import about
from pennylane.circuit_graph import CircuitGraph
from pennylane.configuration import Configuration
from pennylane.registers import registers
from pennylane.measurements import (
    counts,
    density_matrix,
    measure,
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
from pennylane.ops import *
from pennylane.ops import adjoint, ctrl, cond, change_op_basis, exp, sum, pow, prod, s_prod
from pennylane.ops import LinearCombination as Hamiltonian
from pennylane.templates import layer
from pennylane.templates.embeddings import *
from pennylane.templates.layers import *
from pennylane.templates.tensornetworks import *
from pennylane.templates.swapnetworks import *
from pennylane.templates.state_preparations import *
from pennylane.templates.subroutines import *
from pennylane import qaoa
from pennylane.workflow import QNode, qnode, execute, set_shots
from pennylane import workflow

from pennylane.transforms import (
    transform,
    batch_params,
    batch_input,
    batch_partial,
    compile,
    defer_measurements,
    dynamic_one_shot,
    quantum_monte_carlo,
    apply_controlled_Q,
    commutation_dag,
    pattern_matching,
    pattern_matching_optimization,
    clifford_t_decomposition,
)
from pennylane.noise import (
    add_noise,
    insert,
    mitigate_with_zne,
    fold_global,
    poly_extrapolate,
    richardson_extrapolate,
    exponential_extrapolate,
)
from pennylane.ops.functions import (
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
)
from pennylane.ops.identity import I
from pennylane.optimize import *
from pennylane.debugging import (
    snapshots,
    breakpoint,
    debug_expval,
    debug_state,
    debug_probs,
    debug_tape,
)
from pennylane.shadows import ClassicalShadow
from pennylane.qcut import cut_circuit, cut_circuit_mc
from pennylane import pulse

from pennylane import fourier
from pennylane.gradients import metric_tensor, adjoint_metric_tensor
from pennylane import gradients  # pylint:disable=wrong-import-order
from pennylane.drawer import draw, draw_mpl

from pennylane.io import (
    from_pyquil,
    from_qasm,
    to_openqasm,
    from_qiskit,
    from_qiskit_noise,
    from_qiskit_op,
    from_quil,
    from_quil_file,
    FromBloq,
    bloq_registers,
    from_qasm3,
    to_bloq,
)

# pylint:disable=wrong-import-order
from pennylane import logging  # pylint:disable=wrong-import-order

from pennylane import data

from pennylane import noise
from pennylane.noise import NoiseModel

from pennylane.devices import Tracker
from pennylane.devices.device_constructor import device, refresh_devices

from pennylane import spin

from pennylane import liealg
from pennylane.liealg import lie_closure, structure_constants, center
from pennylane import qnn

from pennylane import estimator

from importlib.metadata import version as _metadata_version
from importlib.util import find_spec as _find_spec
from packaging.version import Version as _Version

if _find_spec("jax") is not None:
    if (jax_version := _Version(_metadata_version("jax"))) > _Version("0.6.2"):  # pragma: no cover
        warnings.warn(
            "PennyLane is not yet compatible with JAX versions > 0.6.2. "
            f"You have version {jax_version} installed. "
            "Please downgrade JAX to 0.6.2 to avoid runtime errors using "
            "python -m pip install jax~=0.6.0 jaxlib~=0.6.0",
            RuntimeWarning,
        )

# Look for an existing configuration file
default_config = Configuration("config.toml")


def __getattr__(name):

    if name == "plugin_devices":
        # pylint: disable=import-outside-toplevel
        from pennylane.devices.device_constructor import plugin_devices

        return plugin_devices

    raise AttributeError(f"module 'pennylane' has no attribute '{name}'")


def version():
    """Returns the PennyLane version number."""
    return __version__
