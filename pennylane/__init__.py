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

# pylint: disable=wrong-import-position

import importlib as _importlib

from pennylane._version import __version__
from pennylane.configuration import Configuration

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


# Does this need to be top-level?
from .boolean_fn import BooleanFn

# Not necessary to import these, but purely for "slight" convenience
from .decomposition import register_resources, add_decomps, list_decomps, resource_rep
from .pauli import pauli_decompose
from .fermi import (
    FermiC,
    FermiA,
    FermiWord,
    FermiSentence,
    jordan_wigner,
    parity_transform,
    bravyi_kitaev,
)
from .bose import (
    BoseSentence,
    BoseWord,
    binary_mapping,
    unary_mapping,
    christiansen_mapping,
)
from .qchem import (
    taper,
    symmetry_generators,
    paulix_ops,
    taper_operation,
    import_operator,
    from_openfermion,
    to_openfermion,
)
from .shadows import ClassicalShadow
from .qcut import cut_circuit, cut_circuit_mc
from .gradients import metric_tensor, adjoint_metric_tensor
from .noise import NoiseModel
from .drawer import draw, draw_mpl
from .liealg import lie_closure, structure_constants, center

# Used often so top-level import is crucial
from .queuing import QueuingManager, apply
from .resource import specs
from .compiler import qjit
from .control_flow import for_loop, while_loop
from ._grad import grad, jacobian, vjp, jvp
from ._version import __version__
from .about import about
from .circuit_graph import CircuitGraph
from .configuration import Configuration
from .tracker import Tracker
from .registers import registers
from .io import (
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
from .measurements import (
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
from .ops import *
from .ops import LinearCombination as Hamiltonian
from .templates import *
from .workflow import QNode, qnode, execute
from .transforms import (
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
from .ops.functions import (
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
from .ops.identity import I
from .optimize import *
from .debugging import (
    snapshots,
    breakpoint,
    debug_expval,
    debug_state,
    debug_probs,
    debug_tape,
)
from .devices.device_constructor import device, refresh_devices

submodules = [
    "numpy",
    "compiler",
    "capture",
    "control_flow",
    "kernels",
    "math",
    "operation",
    "decomposition",
    "qnn",
    "templates",
    "pauli",
    "resource",
    "qchem",
    "qaoa",
    "pulse",
    "fourier",
    "gradients",
    "logging",
    "data",
    "noise",
    "liealg",
    "spin",
]


# pylint: disable=no-else-return
def __getattr__(name):

    # pylint: disable=import-outside-toplevel
    if name == "plugin_devices":
        from pennylane.devices.device_constructor import plugin_devices

        return plugin_devices

    if name in submodules:
        return _importlib.import_module(f"pennylane.{name}")
    else:
        try:
            return globals()[name]
        except KeyError as exc:
            raise AttributeError(f"module 'pennylane' has no attribute '{name}'") from exc


def version():
    """Returns the PennyLane version number."""
    return __version__
