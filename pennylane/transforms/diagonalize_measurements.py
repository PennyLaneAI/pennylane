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
"""Transform to diagonalize measurements on a tape, assuming all measurements are commuting."""

from copy import copy
from functools import singledispatch

import pennylane as qml
from pennylane.exceptions import QuantumFunctionError
from pennylane.ops import CompositeOp, LinearCombination, SymbolicOp
from pennylane.pauli import diagonalize_qwc_pauli_words
from pennylane.tape.tape import (
    _validate_computational_basis_sampling,
    rotations_and_diagonal_measurements,
)
from pennylane.transforms.core import transform

# pylint: disable=unused-argument

_default_supported_obs = (qml.Z, qml.Identity)


def null_postprocessing(results):
    """A postprocessing function returned by a transform that only converts the batch of results
    into a result for a single ``QuantumTape``.
    """
    return results[0]


@transform
def diagonalize_measurements(tape, supported_base_obs=_default_supported_obs, to_eigvals=False):
    """Diagonalize a set of measurements into the standard basis. Raises an error if the
    measurements do not commute.

    See the usage details for more information on which measurements are supported.

    Args:
        tape (QNode or QuantumScript or Callable): The quantum circuit to modify the measurements of.
        supported_base_obs (Optional, Iterable(Operator)): A list of supported base observable classes.
            Allowed observables are ``qml.X``, ``qml.Y``, ``qml.Z``, ``qml.Hadamard`` and ``qml.Identity``.
            Z and Identity are always treated as supported, regardless of input. If no list is provided,
            the transform will diagonalize everything into the Z basis. If a list is provided, only
            unsupported observables will be diagonalized to the Z basis.

    Returns:
        qnode (QNode) or tuple[List[QuantumScript], function]: The transformed circuit as described in :func:`qml.transform <pennylane.transform>`.

    .. note::
        An error will be raised if non-commuting terms are encountered. To avoid non-commuting
        terms in circuit measurements, the :func:`split_non_commuting <pennylane.transforms.split_non_commuting>`
        transform can be applied.

        This transform will diagonalize what it can, i.e., ``qml.X``, ``qml.Y``, ``qml.Z``,
        ``qml.Hadamard``, ``qml.Identity``, or a linear combination of them. Any unrecognized
        observable will not raise an error, deferring to the device's validation for supported
        measurements later on. Lastly, if ``diagonalize_measurements`` produces additional gates
        that the device does not support, the :func:`~pennylane.devices.preprocess.decompose`
        transform should be applied to ensure that the additional gates are decomposed to those
        that the device supports.

    **Examples:**

    This transform allows us to transform QNode measurements into the measurement basis by adding
    the relevant diagonalizing gates to the end of the tape operations.

    .. code-block:: python3

        from pennylane.transforms import diagonalize_measurements

        dev = qml.device("default.qubit")

        @diagonalize_measurements
        @qml.qnode(dev)
        def circuit(x):
            qml.RY(x[0], wires=0)
            qml.RX(x[1], wires=1)
            return qml.expval(qml.X(0) @ qml.Z(1)), qml.var(0.5 * qml.Y(2) + qml.X(0))

    Applying the transform appends the relevant gates to the end of the circuit to allow
    measurements to be in the Z basis, so the original circuit

    >>> print(qml.draw(circuit, level=0)([np.pi/4, np.pi/4]))
    0: ──RY(0.79)─┤ ╭<X@Z> ╭Var[𝓗(0.50)]
    1: ──RX(0.79)─┤ ╰<X@Z> │
    2: ───────────┤        ╰Var[𝓗(0.50)]

    becomes

    >>> print(qml.draw(circuit)([np.pi/4, np.pi/4]))
    0: ──RY(0.79)──H────┤ ╭<Z@Z> ╭Var[𝓗(0.50)]
    1: ──RX(0.79)───────┤ ╰<Z@Z> │
    2: ──Z─────────S──H─┤        ╰Var[𝓗(0.50)]

    >>> circuit([np.pi/4, np.pi/4])
    (0.5, 0.75)

    .. details::
        :title: Usage Details

        The transform diagonalizes observables from the local Pauli basis only, i.e. it diagonalizes
        X, Y, Z, and Hadamard. Any other observable will be unaffected:

        .. code-block:: python3

            measurements = [
                qml.expval(qml.X(0) + qml.Hermitian([[1, 0], [0, 1]], wires=[1]))
            ]
            tape = qml.tape.QuantumScript(measurements=measurements)
            tapes, processsing_fn = diagonalize_measurements(tape)

        >>> tapes[0].operations
        [H(0)]
        >>> tapes[0].measurements
        [expval(Z(0) + Hermitian(array([[1, 0], [0, 1]]), wires=[1]))]

        The transform can also diagonalize only a subset of these operators. By default, the only
        supported base observable is Z. What if a backend device can handle
        X, Y and Z, but doesn't provide support for Hadamard? We can set this by passing
        ``supported_base_obs`` to the transform. Let's create a tape with some measurements:

        .. code-block:: python3

            measurements = [
                qml.expval(qml.X(0) + qml.Hadamard(1)),
                qml.expval(qml.X(0) + 0.2 * qml.Hadamard(1)),
                qml.var(qml.Y(2) + qml.X(0)),
            ]
            tape = qml.tape.QuantumScript(measurements=measurements)
            tapes, processing_fn = diagonalize_measurements(
                tape,
                supported_base_obs=[qml.X, qml.Y, qml.Z]
            )

        Now ``tapes`` is a tuple containing a single tape with the updated measurements,
        where only the Hadamard gate has been diagonalized:

        >>> tapes[0].measurements
        [expval(X(0) + Z(1)), expval(X(0) + 0.2 * Z(1)), var(Y(2) + X(0))]
    """

    bad_obs_input = [
        o for o in supported_base_obs if o not in {qml.X, qml.Y, qml.Z, qml.Hadamard, qml.Identity}
    ]

    if bad_obs_input:
        raise ValueError(
            "Supported base observables must be a subset of [X, Y, Z, Hadamard, and Identity] "
            f"but received {list(bad_obs_input)}"
        )

    diagonalize_all = set(supported_base_obs).issubset(set(_default_supported_obs))

    if to_eigvals and not diagonalize_all:
        raise ValueError(
            "Using to_eigvals=True requires diagonalizing all observables to the "
            "measurement basis. Observables "
            f"{set(supported_base_obs)-set(_default_supported_obs)} can't "
            "be supported when using eigvals."
        )

    if (
        all(m.obs.pauli_rep is not None for m in tape.measurements if m.obs is not None)
        and diagonalize_all
    ):
        try:
            if tape.samples_computational_basis and len(tape.measurements) > 1:
                _validate_computational_basis_sampling(tape)
            diagonalizing_gates, new_measurements = _diagonalize_all_pauli_obs(
                tape, to_eigvals=to_eigvals
            )
        except QuantumFunctionError:
            # the pauli_rep based method sometimes fails unnecessarily -
            # if it fails, fall back on the less efficient method (which may also fail)
            diagonalizing_gates, new_measurements = _diagonalize_subset_of_pauli_obs(
                tape, supported_base_obs, to_eigvals=to_eigvals
            )

    else:
        diagonalizing_gates, new_measurements = _diagonalize_subset_of_pauli_obs(
            tape, supported_base_obs, to_eigvals=to_eigvals
        )

    new_operations = tape.operations + diagonalizing_gates

    new_tape = tape.copy(operations=new_operations, measurements=new_measurements)

    return (new_tape,), null_postprocessing


def _diagonalize_all_pauli_obs(tape, to_eigvals=False):
    """Takes a tape and changes all observables to the measurement basis. Assumes all
    measurements on the tape are qwc.

    Args:
        tape: the observable to be diagonalized
        to_eigvals: whether the diagonalization should create measurements using
            eigvals and wires rather than observables

    Returns:
        diagonalizing_gates: A list of operations to be applied to diagonalize the observable
        new_measurements: the relevant measurement to perform after applying diagonalzing_gates to get the
            correct measurement output
    """
    new_measurements = []

    diagonalizing_gates, diagonal_measurements = rotations_and_diagonal_measurements(tape)
    for m in diagonal_measurements:
        if m.obs is not None:
            gates, new_obs = _change_obs_to_Z(m.obs)
            if to_eigvals:
                new_meas = type(m)(eigvals=m.eigvals(), wires=m.wires)
            else:
                new_meas = type(m)(new_obs)
            diagonalizing_gates.extend(gates)
            new_measurements.append(new_meas)
        else:
            new_measurements.append(m)

    return diagonalizing_gates, new_measurements


def _diagonalize_subset_of_pauli_obs(tape, supported_base_obs, to_eigvals=False):
    """Takes a tape and changes a subset of observables to the measurement basis. Assumes all
    measurements on the tape are qwc.

    Args:
        tape: the observable to be diagonalized
        supported_base_obs (Optional, Iterable(Operator)): A list of supported base observable classes.
            Allowed observables are ``qml.X``, ``qml.Y``, ``qml.Z``, ``qml.Hadamard`` and ``qml.Identity``.
            Z and Identity are always treated as supported, regardless of input. If no list is provided,
            the transform will diagonalize everything into the Z basis. If a list is provided, only
            unsupported observables will be diagonalized to the Z basis.
        to_eigvals: whether the diagonalization should create measurements using
            eigvals and wires rather than observables

    Returns:
        diagonalizing_gates: A list of operations to be applied to diagonalize the observable
        new_measurements: the relevant measurement to perform after applying diagonalzing_gates to get the
            correct measurement output

    Raises:
        ValueError: if non-commuting observables are ecountered on the tape

    """
    supported_base_obs = set(list(supported_base_obs) + [qml.Z, qml.Identity])

    wires_sampled_in_computational_basis = []
    comp_basis_sampling_meas = [m for m in tape.measurements if m.samples_computational_basis]
    if any(m.wires == qml.wires.Wires([]) for m in comp_basis_sampling_meas):
        wires_sampled_in_computational_basis = tape.wires
    elif comp_basis_sampling_meas:
        for m in comp_basis_sampling_meas:
            wires_sampled_in_computational_basis.extend(list(m.wires))

    _visited_obs = (
        {qml.Z(w) for w in wires_sampled_in_computational_basis},
        set(wires_sampled_in_computational_basis),
    )  # tracks which observables and wires are already used and shouldn't be diagonalized
    diagonalizing_gates = []
    new_measurements = []

    for m in tape.measurements:
        if m.obs:
            gates, new_obs, _visited_obs = _diagonalize_observable(
                m.obs, _visited_obs, supported_base_obs
            )
            diagonalizing_gates.extend(gates)
            if to_eigvals:
                new_meas = type(m)(eigvals=m.eigvals(), wires=m.wires)
            else:
                new_meas = type(m)(new_obs)
            new_measurements.append(new_meas)
        else:
            new_measurements.append(m)

    return diagonalizing_gates, new_measurements


@singledispatch
def _change_obs_to_Z(observable):
    diagonalizing_gates, new_observable = diagonalize_qwc_pauli_words([observable])

    return diagonalizing_gates, new_observable[0]


@_change_obs_to_Z.register
def _change_symbolic_op(observable: SymbolicOp):
    diagonalizing_gates, [new_base] = diagonalize_qwc_pauli_words([observable.base])

    params, hyperparams = observable.parameters, observable.hyperparameters
    hyperparams = copy(hyperparams)
    hyperparams["base"] = new_base

    new_observable = observable.__class__(*params, **hyperparams)

    return diagonalizing_gates, new_observable


@_change_obs_to_Z.register
def _change_linear_combination(observable: LinearCombination):
    coeffs, obs = observable.terms()

    diagonalizing_gates, new_operands = diagonalize_qwc_pauli_words(obs)

    new_observable = LinearCombination(coeffs, new_operands)

    return diagonalizing_gates, new_observable


@_change_obs_to_Z.register
def _change_composite_op(observable: CompositeOp):
    diagonalizing_gates, new_operands = diagonalize_qwc_pauli_words(observable.operands)

    new_observable = observable.__class__(*new_operands)

    return diagonalizing_gates, new_observable


def _check_if_diagonalizing(obs, _visited_obs, switch_basis):
    """Checks if the observable should be diagonalized based on whether its basis should
    be switched, and whether the same observable has already been diagonalized.

    Args:
        obs: the observable to be diagonalized
        _visited_obs: a tuple containing a list of all observables that have already
            been encountered, and a list of all wires that have already been encountered
        switch_basis: whether the observable should be switched for one measuring in
            the Z-basis on the same wires.

    Returns:
        diagonalize (bool): whether or not to apply diagonalizing gates for the observable
        _visited_obs (tuple(Iterables): an up-to-date record of the observables and wires
            encountered on the tape so far
    """

    if obs in _visited_obs[0] or isinstance(obs, qml.Identity):
        # its already been encountered before, and if need be, diagonalized - we will
        # not be applying any gates or updating _visited_obs
        # same if it's just an Identity
        return False, _visited_obs

    # a different observable has been diagonalized on the same wire - error
    if obs.wires[0] in _visited_obs[1]:
        raise ValueError(
            f"Expected measurements on the same wire to commute, but {obs} "
            f"overlaps with another non-commuting observable on the tape."
        )

    # we diagonalize if it's an operator we are switching the basis for
    # we update _visited_obs to indicate that we've encountered that observable regardless
    _visited_obs[0].add(obs)
    _visited_obs[1].add(obs.wires[0])

    return switch_basis, _visited_obs


def _diagonalize_observable(
    observable,
    _visited_obs=None,
    supported_base_obs=_default_supported_obs,
):
    """Takes an observable and changes all unsupported obs to the measurement
    basis. Applies diagonalizing gates if the observable being switched to the
    measurement basis hasn't already been diagonalized for a previous observable
    on the tape.

    Args:
        observable: the observable to be diagonalized
        _visited_obs: a tuple containing a list of all observables that have already
            been encountered, and a list of all wires that have already been encountered
        supported_base_obs (Optional, Iterable(Str)): A list of names of supported base observables.
            Allowed names are 'PauliX', 'PauliY', 'PauliZ' and 'Hadamard'. If no list is provided,
            the function will diagonalize everything.

    Returns:
        diagonalizing_gates: A list of operations to be applied to diagonalize the observable
        new_obs: the relevant measurement to perform after applying diagonalzing_gates to get the
            correct measurement output
        _visited_obs: an up-to-date record of the observables and wires
            encountered on the tape so far
    """

    if _visited_obs is None:
        _visited_obs = (set(), set())

    if not isinstance(observable, (qml.X, qml.Y, qml.Z, qml.Hadamard, qml.Identity)):
        return _diagonalize_non_basic_observable(
            observable, _visited_obs, supported_base_obs=supported_base_obs
        )

    # also validates that the wire hasn't already been diagonalized and updated _visited_obs
    switch_basis = type(observable) not in supported_base_obs
    diagonalize, _visited_obs = _check_if_diagonalizing(observable, _visited_obs, switch_basis)

    if isinstance(observable, qml.Z):  # maybe kind of redundant
        return [], observable, _visited_obs

    new_obs = qml.Z(wires=observable.wires) if switch_basis else observable
    diagonalizing_gates = observable.diagonalizing_gates() if diagonalize else []

    return diagonalizing_gates, new_obs, _visited_obs


def _get_obs_and_gates(obs_list, _visited_obs, supported_base_obs=_default_supported_obs):
    """Calls _diagonalize_observable on each observable in a list, and returns the full result
    for the list of observables"""

    new_obs = []
    diagonalizing_gates = []

    for o in obs_list:
        gates, obs, _visited_obs = _diagonalize_observable(o, _visited_obs, supported_base_obs)
        if gates:
            diagonalizing_gates.extend(gates)
        new_obs.append(obs)

    return diagonalizing_gates, new_obs, _visited_obs


@singledispatch
def _diagonalize_non_basic_observable(
    observable, _visited_obs, supported_base_obs=_default_supported_obs
):
    """Takes an observable other than X, Y, Z, H, and I, and diagonalize it.

    For composite observables consisting of multiple other observables, it changes all
    unsupported obs to the measurement basis. Applies diagonalizing gates if changing
    the basis of an observable whose diagonalizing gates have not already been applied.
    For other observables, simply skips and returns the observable as is.

    """
    _visited_obs[0].add(observable)
    _visited_obs[1].add(observable.wires[0])
    return [], observable, _visited_obs


@_diagonalize_non_basic_observable.register
def _diagonalize_symbolic_op(
    observable: SymbolicOp, _visited_obs, supported_base_obs=_default_supported_obs
):
    diagonalizing_gates, new_base, _visited_obs = _diagonalize_observable(
        observable.base, _visited_obs, supported_base_obs
    )

    params, hyperparams = observable.parameters, observable.hyperparameters
    hyperparams = copy(hyperparams)
    hyperparams["base"] = new_base

    new_observable = observable.__class__(*params, **hyperparams)

    return diagonalizing_gates, new_observable, _visited_obs


@_diagonalize_non_basic_observable.register
def _diagonalize_linear_combination(
    observable: LinearCombination, _visited_obs, supported_base_obs=_default_supported_obs
):

    coeffs, obs = observable.terms()

    diagonalizing_gates, new_obs, _visited_obs = _get_obs_and_gates(
        obs, _visited_obs, supported_base_obs
    )

    new_observable = LinearCombination(coeffs, new_obs)

    return diagonalizing_gates, new_observable, _visited_obs


@_diagonalize_non_basic_observable.register
def _diagonalize_composite_op(
    observable: CompositeOp, _visited_obs, supported_base_obs=_default_supported_obs
):
    diagonalizing_gates, new_operands, _visited_obs = _get_obs_and_gates(
        observable.operands, _visited_obs, supported_base_obs
    )

    new_observable = observable.__class__(*new_operands)

    return diagonalizing_gates, new_observable, _visited_obs
