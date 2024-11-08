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


import pennylane.capture
import pennylane.compiler
import pennylane.data
import pennylane.fourier
import pennylane.gradients  # pylint:disable=wrong-import-order
import pennylane.kernels

# pylint:disable=wrong-import-order
import pennylane.logging  # pylint:disable=wrong-import-order
import pennylane.math
import pennylane.noise
import pennylane.numpy
import pennylane.operation
import pennylane.pauli
import pennylane.pulse
import pennylane.qchem
import pennylane.qinfo
import pennylane.qnn
import pennylane.resource
import pennylane.spin
import pennylane.templates
from pennylane import qaoa
from pennylane._grad import grad, jacobian, jvp, vjp
from pennylane._version import __version__
from pennylane.about import about
from pennylane.boolean_fn import BooleanFn
from pennylane.circuit_graph import CircuitGraph
from pennylane.compiler import for_loop, qjit, while_loop
from pennylane.configuration import Configuration
from pennylane.debugging import (
    breakpoint,
    debug_expval,
    debug_probs,
    debug_state,
    debug_tape,
    snapshots,
)
from pennylane.devices.device_constructor import device, refresh_devices
from pennylane.drawer import draw, draw_mpl
from pennylane.fermi import FermiA, FermiC, bravyi_kitaev, jordan_wigner, parity_transform
from pennylane.gradients import adjoint_metric_tensor, metric_tensor
from pennylane.io import *
from pennylane.measurements import (
    classical_shadow,
    counts,
    density_matrix,
    expval,
    measure,
    mutual_info,
    probs,
    purity,
    sample,
    shadow_expval,
    state,
    var,
    vn_entropy,
)
from pennylane.noise import NoiseModel
from pennylane.ops import *
from pennylane.ops import adjoint, cond, ctrl, exp, pow, prod, s_prod, sum
from pennylane.ops.functions import (
    assert_equal,
    comm,
    commutator,
    dot,
    eigvals,
    equal,
    evolve,
    generator,
    is_commuting,
    is_hermitian,
    is_unitary,
    iterative_qpe,
    map_wires,
    matrix,
    simplify,
)
from pennylane.ops.identity import I
from pennylane.optimize import *
from pennylane.pauli import center, lie_closure, pauli_decompose, structure_constants
from pennylane.qchem import (
    from_openfermion,
    import_operator,
    paulix_ops,
    symmetry_generators,
    taper,
    taper_operation,
    to_openfermion,
)
from pennylane.qcut import cut_circuit, cut_circuit_mc
from pennylane.queuing import QueuingManager, apply
from pennylane.registers import registers
from pennylane.resource import specs
from pennylane.shadows import ClassicalShadow
from pennylane.templates import broadcast, layer
from pennylane.templates.embeddings import *
from pennylane.templates.layers import *
from pennylane.templates.state_preparations import *
from pennylane.templates.subroutines import *
from pennylane.templates.swapnetworks import *
from pennylane.templates.tensornetworks import *
from pennylane.tracker import Tracker
from pennylane.transforms import (
    add_noise,
    apply_controlled_Q,
    batch_input,
    batch_params,
    batch_partial,
    clifford_t_decomposition,
    commutation_dag,
    compile,
    defer_measurements,
    dynamic_one_shot,
    pattern_matching,
    pattern_matching_optimization,
    quantum_monte_carlo,
    transform,
)
from pennylane.workflow import QNode, execute, qnode

# Look for an existing configuration file
default_config = Configuration("config.toml")


class DeviceError(Exception):
    """Exception raised when it encounters an illegal operation in the quantum circuit."""


class QuantumFunctionError(Exception):
    """Exception raised when an illegal operation is defined in a quantum function."""


class PennyLaneDeprecationWarning(UserWarning):
    """Warning raised when a PennyLane feature is being deprecated."""


del globals()["Hamiltonian"]


def __getattr__(name):
    if name == "Hamiltonian":
        if pennylane.operation.active_new_opmath():
            return pennylane.ops.LinearCombination
        return pennylane.ops.Hamiltonian

    if name == "plugin_devices":
        return pennylane.devices.device_constructor.plugin_devices

    raise AttributeError(f"module 'pennylane' has no attribute '{name}'")


def version():
    """Returns the PennyLane version number."""
    return __version__
