"""Transform to diagonalize measurements on a tape, assuming all measurements are commuting."""

from copy import copy
from functools import singledispatch

import pennylane as qml
from pennylane.devices.preprocess import null_postprocessing
from pennylane.operation import Tensor
from pennylane.ops import CompositeOp, SymbolicOp
from pennylane.transforms.core import transform

# pylint: disable=protected-access

_default_unsupported_obs = frozenset(["PauliX", "PauliY", "Hadamard"])


@transform
def diagonalize_tape_measurements(tape, unsupported_obs=_default_unsupported_obs):
    """Diagonalize the measurements on a tape if they are not supported. Assumes all observables commute."""

    bad_obs_input = set(unsupported_obs) - {"PauliX", "PauliY", "Hadamard"}

    if bad_obs_input:
        raise ValueError(
            "Unsupported observables to diagonalize must be a subset of "
            f"[PauliX, PauliY, Hadamard] but received {list(bad_obs_input)}"
        )

    diagonalization_wires = list(tape.wires)
    diagonalizing_gates = []
    new_measurements = []

    for m in tape.measurements:
        if m.obs:
            wires = [w for w in m.obs.wires if w in diagonalization_wires]
            gates, new_obs = diagonalize_observable(m.obs, wires, unsupported_obs)
            if gates:
                diagonalizing_gates.extend(gates)
                for w in wires:
                    diagonalization_wires.remove(w)

            meas = copy(m)
            meas.obs = new_obs
            new_measurements.append(meas)
        else:
            new_measurements.append(m)

    new_operations = tape.operations.copy()

    if diagonalizing_gates:
        new_operations.extend(diagonalizing_gates)

    new_tape = type(tape)(
        ops=new_operations,
        measurements=new_measurements,
        shots=tape.shots,
        trainable_params=tape.trainable_params,
    )

    return new_tape, null_postprocessing


def diagonalize_observable(
    observable, diagonalization_wires, unsupported_obs=_default_unsupported_obs
):
    """takes an observable and changes all unsupported obs to the measurement
    basis. Applies diagonalizing gates if the observable being diagonalized is
    on one of the diagonalization wires."""

    if isinstance(observable, qml.Z):
        return [], observable

    if isinstance(observable, (qml.X, qml.Y, qml.Hadamard)):
        switch_basis = observable.name in unsupported_obs
        diagonalize = switch_basis and observable.wires[0] in diagonalization_wires

        new_obs = qml.Z(wires=observable.wires) if switch_basis else observable
        diagonalizing_gates = observable.diagonalizing_gates() if diagonalize else []

        return diagonalizing_gates, new_obs

    return _diagonalize_complex_observable(
        observable, diagonalization_wires, unsupported_obs=unsupported_obs
    )


def get_obs_and_gates(obs_list, diagonalization_wires, unsupported_obs=_default_unsupported_obs):
    """Calls diagonalize_observables on each observable in a list. After each observable is diagonalized,
    if diagonalizing gates were added, its wire is removed from the list of wires still needing
    diagonalization. This prevents diagonalizing twice on the same wire for overlapping measurements, like
    [qml.X(0), qml.X(0)@qml.Y(1)]"""

    new_obs = []
    diagonalizing_gates = []

    for o in obs_list:
        gates, obs = diagonalize_observable(o, diagonalization_wires, unsupported_obs)
        if gates:
            diagonalizing_gates.extend(gates)
            for w in obs.wires:
                diagonalization_wires.remove(w)
        new_obs.append(obs)

    return diagonalizing_gates, new_obs


@singledispatch
def _diagonalize_complex_observable(
    observable, diagonalization_wires, unsupported_obs=_default_unsupported_obs
):
    """takes an observable consisting of multiple other observables, and changes all
    unsupported obs to the measurement basis. Applies diagonalizing gates unless the
    observable being diagonalized isn't on one of the diagonalization wires."""

    raise NotImplementedError(
        f"Unable to convert observable of type {type(observable)} to the measurement basis"
    )


@_diagonalize_complex_observable.register
def _diagonalize_symbolic_op(
    observable: SymbolicOp, diagonalization_wires, unsupported_obs=_default_unsupported_obs
):
    diagonalizing_gates, new_base = diagonalize_observable(
        observable.base, diagonalization_wires, unsupported_obs
    )

    new_observable = copy(observable)
    new_observable._hyperparameters["base"] = new_base

    return diagonalizing_gates, new_observable


@_diagonalize_complex_observable.register
def _diagonalize_tensor(
    observable: Tensor, diagonalization_wires, unsupported_obs=_default_unsupported_obs
):
    diagonalizing_gates, new_obs = get_obs_and_gates(
        observable.obs, diagonalization_wires, unsupported_obs
    )

    new_observable = copy(observable)
    new_observable.obs = new_obs

    return diagonalizing_gates, new_observable


@_diagonalize_complex_observable.register
def _diagonalize_hamiltonian(
    observable: qml.ops.Hamiltonian,
    diagonalization_wires,
    unsupported_obs=_default_unsupported_obs,
):
    diagonalizing_gates, new_ops = get_obs_and_gates(
        observable.ops, diagonalization_wires, unsupported_obs
    )

    new_observable = copy(observable)
    new_observable._ops = new_ops

    return diagonalizing_gates, new_observable


@_diagonalize_complex_observable.register
def _diagonalize_composite_op(
    observable: CompositeOp, diagonalization_wires, unsupported_obs=_default_unsupported_obs
):
    diagonalizing_gates, new_operands = get_obs_and_gates(
        observable.operands, diagonalization_wires, unsupported_obs
    )

    new_observable = copy(observable)
    new_observable.operands = new_operands

    return diagonalizing_gates, new_observable
