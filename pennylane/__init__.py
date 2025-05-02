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


from pennylane.boolean_fn import BooleanFn
import pennylane.numpy
from pennylane.queuing import QueuingManager, apply

import pennylane.compiler
from pennylane.compiler import qjit
import pennylane.capture
import pennylane.control_flow
from pennylane.control_flow import for_loop, while_loop
import pennylane.kernels
import pennylane.math
import pennylane.operation
import pennylane.decomposition
from pennylane.decomposition import (
    register_resources,
    add_decomps,
    list_decomps,
    resource_rep,
)
import pennylane.qnn
import pennylane.templates
import pennylane.pauli
from pennylane.pauli import pauli_decompose
from pennylane.resource import specs
import pennylane.resource
import pennylane.qchem
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
from pennylane.tracker import Tracker
from pennylane.registers import registers
from pennylane.io import (
    from_pyquil,
    from_qasm,
    from_qiskit,
    from_qiskit_noise,
    from_qiskit_op,
    from_quil,
    from_quil_file,
    FromBloq,
    bloq_registers,
)
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
from pennylane.ops import adjoint, ctrl, cond, exp, sum, pow, prod, s_prod
from pennylane.ops import LinearCombination as Hamiltonian
from pennylane.templates import layer
from pennylane.templates.embeddings import *
from pennylane.templates.layers import *
from pennylane.templates.tensornetworks import *
from pennylane.templates.swapnetworks import *
from pennylane.templates.state_preparations import *
from pennylane.templates.subroutines import *
from pennylane import qaoa
from pennylane.workflow import QNode, qnode, execute
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
    add_noise,
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
import pennylane.pulse

import pennylane.fourier
from pennylane.gradients import metric_tensor, adjoint_metric_tensor
import pennylane.gradients  # pylint:disable=wrong-import-order
from pennylane.drawer import draw, draw_mpl

# pylint:disable=wrong-import-order
import pennylane.logging  # pylint:disable=wrong-import-order

import pennylane.data

import pennylane.noise
from pennylane.noise import NoiseModel

from pennylane.devices.device_constructor import device, refresh_devices

import pennylane.spin

import pennylane.liealg
from pennylane.liealg import lie_closure, structure_constants, center

# Look for an existing configuration file
default_config = Configuration("config.toml")


class DeviceError(Exception):
    """Exception raised when it encounters an illegal operation in the quantum circuit."""


class QuantumFunctionError(Exception):
    """Exception raised when an illegal operation is defined in a quantum function."""


class PennyLaneDeprecationWarning(UserWarning):
    """Warning raised when a PennyLane feature is being deprecated."""


class ExperimentalWarning(UserWarning):
    """Warning raised to indicate experimental/non-stable feature or support."""


def __getattr__(name):

    if name == "plugin_devices":
        return pennylane.devices.device_constructor.plugin_devices

    raise AttributeError(f"module 'pennylane' has no attribute '{name}'")


def version():
    """Returns the PennyLane version number."""
    return __version__
