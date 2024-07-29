"""Transform to diagonalize measurements on a tape, assuming all measurements are commuting."""

from copy import copy
from functools import singledispatch

import pennylane as qml
from pennylane.devices.preprocess import null_postprocessing
from pennylane.operation import Tensor
from pennylane.ops import CompositeOp, LinearCombination, SymbolicOp
from pennylane.transforms.core import transform

# pylint: disable=protected-access

_default_supported_obs = frozenset(["PauliZ"])


@transform
def diagonalize_tape_measurements(tape, supported_base_obs=_default_supported_obs):
    """Diagonalize the measurements on a tape if they are not supported. Can diagonalize the
    tape if all observables commute."""

    bad_obs_input = [
        o for o in supported_base_obs if not o in {"PauliX", "PauliY", "PauliZ", "Hadamard"}
    ]

    if bad_obs_input:
        raise ValueError(
            "Supported base observables must be a subset of [PauliX, PauliY, PauliZ, Hadamard] "
            f"but received {list(bad_obs_input)}"
        )

    _visited_obs = ([], [])  # tracks which observables and wires have been diagonalized
    diagonalizing_gates = []
    new_measurements = []

    for m in tape.measurements:
        if m.obs:
            gates, new_obs, _visited_obs = _diagonalize_observable(
                m.obs, _visited_obs, supported_base_obs
            )
            diagonalizing_gates.extend(gates)

            meas = copy(m)
            meas.obs = new_obs
            new_measurements.append(meas)
        else:
            new_measurements.append(m)

    new_operations = tape.operations.copy()
    new_operations.extend(diagonalizing_gates)

    new_tape = type(tape)(
        ops=new_operations,
        measurements=new_measurements,
        shots=tape.shots,
        trainable_params=tape.trainable_params,
    )

    return (new_tape,), null_postprocessing


def _check_if_diagonalizing(obs, _visited_obs, switch_basis):

    # its already been diagonalized - we will not be applying any gates
    if obs in _visited_obs[0]:
        return False, _visited_obs

    # a different observable has been diagonalized on the same wire - error
    if obs.wires in _visited_obs[1]:
        raise RuntimeError(
            f"Expected only a single observable per wire, but {obs} "
            f"overlaps with other obserables on the tape."
        )

    # we diagonalize if it's an operator we are switching the basis for
    _visited_obs[0].append(obs)
    _visited_obs[1].append(obs.wires[0])
    return switch_basis, _visited_obs


def _diagonalize_observable(
    observable,
    _visited_obs=None,
    supported_base_obs=_default_supported_obs,
):
    """takes an observable and changes all unsupported obs to the measurement
    basis. Applies diagonalizing gates if the observable being diagonalized is
    on one of the diagonalization wires."""

    if _visited_obs is None:
        _visited_obs = ([], [])

    if not isinstance(observable, (qml.X, qml.Y, qml.Z, qml.Hadamard)):
        return _diagonalize_compound_observable(
            observable, _visited_obs, supported_base_obs=supported_base_obs
        )

    # also validates that the wire hasn't already been diagonalized and updated _visited_obs
    switch_basis = observable.name not in supported_base_obs
    diagonalize, _visited_obs = _check_if_diagonalizing(observable, _visited_obs, switch_basis)

    if isinstance(observable, qml.Z):  # maybe kind of redundant
        return [], observable, _visited_obs

    new_obs = qml.Z(wires=observable.wires) if switch_basis else observable
    diagonalizing_gates = observable.diagonalizing_gates() if diagonalize else []

    return diagonalizing_gates, new_obs, _visited_obs


def _get_obs_and_gates(obs_list, _visited_obs, supported_base_obs=_default_supported_obs):
    """Calls _diagonalize_observable on each observable in a list. After each observable is diagonalized,
    if diagonalizing gates were added, its wire is removed from the list of wires still needing
    diagonalization. This prevents diagonalizing twice on the same wire for overlapping measurements, like
    [qml.X(0), qml.X(0)@qml.Y(1)]"""

    new_obs = []
    diagonalizing_gates = []

    for o in obs_list:
        gates, obs, _visited_obs = _diagonalize_observable(o, _visited_obs, supported_base_obs)
        if gates:
            diagonalizing_gates.extend(gates)
        new_obs.append(obs)

    return diagonalizing_gates, new_obs


@singledispatch
def _diagonalize_compound_observable(
    observable, _visited_obs, supported_base_obs=_default_supported_obs
):
    """takes an observable consisting of multiple other observables, and changes all
    unsupported obs to the measurement basis. Applies diagonalizing gates unless the
    observable being diagonalized isn't on one of the diagonalization wires."""

    raise NotImplementedError(
        f"Unable to convert observable of type {type(observable)} to the measurement basis"
    )


@_diagonalize_compound_observable.register
def _diagonalize_symbolic_op(
    observable: SymbolicOp, _visited_obs, supported_base_obs=_default_supported_obs
):
    diagonalizing_gates, new_base, _visited_obs = _diagonalize_observable(
        observable.base, _visited_obs, supported_base_obs
    )

    new_observable = copy(observable)
    new_observable._hyperparameters["base"] = new_base

    return diagonalizing_gates, new_observable, _visited_obs


@_diagonalize_compound_observable.register
def _diagonalize_tensor(
    observable: Tensor, _visited_obs, supported_base_obs=_default_supported_obs
):
    diagonalizing_gates, new_obs = _get_obs_and_gates(
        observable.obs, _visited_obs, supported_base_obs
    )

    new_observable = copy(observable)
    new_observable.obs = new_obs

    return diagonalizing_gates, new_observable, _visited_obs


@_diagonalize_compound_observable.register
def _diagonalize_hamiltonian(
    observable: qml.ops.Hamiltonian,
    _visited_obs,
    supported_base_obs=_default_supported_obs,
):
    diagonalizing_gates, new_ops = _get_obs_and_gates(
        observable.ops, _visited_obs, supported_base_obs
    )

    new_observable = copy(observable)
    new_observable._ops = new_ops

    return diagonalizing_gates, new_observable, _visited_obs


@_diagonalize_compound_observable.register
def _diagonalize_composite_op(
    observable: LinearCombination, _visited_obs, supported_base_obs=_default_supported_obs
):
    diagonalizing_gates, new_operands = _get_obs_and_gates(
        observable.operands, _visited_obs, supported_base_obs
    )

    new_observable = LinearCombination(observable.coeffs, new_operands)

    return diagonalizing_gates, new_observable, _visited_obs


@_diagonalize_compound_observable.register
def _diagonalize_composite_op(
    observable: CompositeOp, _visited_obs, supported_base_obs=_default_supported_obs
):
    diagonalizing_gates, new_operands = _get_obs_and_gates(
        observable.operands, _visited_obs, supported_base_obs
    )

    print(new_operands)
    new_observable = observable.__class__(*new_operands)

    return diagonalizing_gates, new_observable, _visited_obs
