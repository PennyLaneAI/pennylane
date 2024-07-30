"""Transform to diagonalize measurements on a tape, assuming all measurements are commuting."""

from copy import copy
from functools import singledispatch

import pennylane as qml
from pennylane.operation import Tensor
from pennylane.ops import CompositeOp, LinearCombination, SymbolicOp
from pennylane.transforms.core import transform

# pylint: disable=protected-access

_default_supported_obs = frozenset(["PauliZ", "Identity"])


def null_postprocessing(results):
    """A postprocessing function returned by a transform that only converts the batch of results
    into a result for a single ``QuantumTape``.
    """
    return results[0]


@transform
def diagonalize_tape_measurements(tape, supported_base_obs=None):
    """Diagonalize the measurements on a tape if they are not supported. Raises an error if the
    measurements do not commute.

    Args:
        tape (QNode or QuantumScript or Callable): The quantum circuit to modify the measurements of.
        supported_base_obs (Optional, Iterable(Str)): A list of names of supported base observables.
            Allowed names are 'PauliX', 'PauliY', 'PauliZ' and 'Hadamard'. If no list is provided,
            the transform will diagonalize everything.

    Returns:
        qnode (QNode) or tuple[List[QuantumScript], function]: The transformed circuit as described in :func:`qml.transform <pennylane.transform>`.

    .. note::
        This transform will raise an error if it encounters non-commuting terms. To avoid non-commuting terms in
        circuit measurements, the :func:`split_non_commuting <pennylane.transforms.split_non_commuting>` transform
        can be applied.


    **Examples:**

    This transform allows us to transform QNode measurements into the measurement basis by adding
    the relevant diagonalizing gates to the end of the tape operations.

    .. code-block:: python3

        from pennylane.transforms import diagonalize_tape_measurements

        dev = qml.device("default.qubit", wires=2)

        @diagonalize_tape_measurements
        @qml.qnode(dev)
        def circuit(x):
            qml.RY(x[0], wires=0)
            qml.RX(x[1], wires=1)
            return qml.expval(qml.X(0) @ qml.Z(1)), qml.var(0.5 * qml.Y(2) + qml.X(0))

    Instead of decorating the QNode, we can also create a new function that yields the same
    result in the following way:

    .. code-block:: python3

        @qml.qnode(dev)
        def circuit(x):
            qml.RY(x[0], wires=0)
            qml.RX(x[1], wires=1)
            return qml.expval(qml.X(0) @ qml.Z(1)), qml.var(0.5 * qml.Y(2) + qml.X(0))

        diagonalized_circuit = diagonalize_tape_measurements(circuit)

    Applying the transform appends the relevant gates to the end of the cirucit to allow
    measurements to be in the Z basis, so the original circuit

    >>> print(qml.draw(circuit)([np.pi/4, np.pi/4]))
    0: ──RY(0.79)─┤ ╭<X@Z> ╭Var[(0.50*Y)+X]
    1: ──RX(0.79)─┤ ╰<X@Z> │
    2: ───────────┤        ╰Var[(0.50*Y)+X]

    becomes

    >>> print(qml.draw(diagonalized_circuit)([np.pi/4, np.pi/4]))
    0: ──RY(0.79)──H────┤ ╭<Z@Z> ╭Var[(0.50*Z)+Z]
    1: ──RX(0.79)───────┤ ╰<Z@Z> │
    2: ──Z─────────S──H─┤        ╰Var[(0.50*Z)+Z]

    >>> diagonalized_circuit([np.pi/4, np.pi/4])
    (tensor(0.5, requires_grad=True), tensor(0.75, requires_grad=True))

    .. details::
        :title: Usage Details

        The transform can also diagonalize only a subset of the operators. Be default, the only
        supported base observable is PauliZ. What if a backend device can handle
        X, Y and Z, but doesn't provide support for Hadamard? We can set this by passing
        ``supported_base_obs`` to the transform. Let's create a tape with some measurements:

        .. code-block:: python3

            measurements = [
                qml.expval(qml.X(0) + qml.Hadamard(1)),
                qml.expval(qml.X(0) + 0.2 * qml.Hadamard(1)),
                qml.var(qml.Y(2) + qml.X(0)),
            ]
            tape = qml.tape.QuantumScript(measurements=measurements)
            tapes, processing_fn = diagonalize_tape_measurements(tape,
                                                                 supported_base_obs=['PauliX', 'PauliY', 'PauliZ'])

        Now ``tapes`` is a tuple containing a single tape with the updated measurements,
        where only the Hadamard gate has been diagonalized:

        >>> tapes[0].measurements
        [expval(X(0) + Z(1)), expval(X(0) + 0.2 * Z(1)), var(Y(2) + X(0))]
    """

    if supported_base_obs is None:
        supported_base_obs = ["PauliZ"]

    bad_obs_input = [
        o
        for o in supported_base_obs
        if not o in {"PauliX", "PauliY", "PauliZ", "Hadamard", "Identity"}
    ]

    if bad_obs_input:
        raise ValueError(
            "Supported base observables must be a subset of ['PauliX', 'PauliY', 'PauliZ', 'Hadamard'] "
            f"but received {list(bad_obs_input)}"
        )

    supported_base_obs.append("Identity")

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
        # same if its just an Identity
        return False, _visited_obs

    # a different observable has been diagonalized on the same wire - error
    if obs.wires[0] in _visited_obs[1]:
        raise ValueError(
            f"Expected only a single observable per wire, but {obs} "
            f"overlaps with another observable on the tape."
        )

    # we diagonalize if it's an operator we are switching the basis for
    # we update _visited_obs to indicate that we've encountered that observable regardless
    _visited_obs[0].append(obs)
    _visited_obs[1].append(obs.wires[0])

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
        _visited_obs = ([], [])

    if not isinstance(observable, (qml.X, qml.Y, qml.Z, qml.Hadamard, qml.Identity)):
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
def _diagonalize_compound_observable(
    observable, _visited_obs, supported_base_obs=_default_supported_obs
):
    """Takes an observable consisting of multiple other observables, and changes all
    unsupported obs to the measurement basis. Applies diagonalizing gates if changing
    the basis of an observable whose diagonalizing gates have not already been applied."""

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
    diagonalizing_gates, new_obs, _visited_obs = _get_obs_and_gates(
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
    diagonalizing_gates, new_ops, _visited_obs = _get_obs_and_gates(
        observable.ops, _visited_obs, supported_base_obs
    )

    new_observable = copy(observable)
    new_observable._ops = new_ops

    return diagonalizing_gates, new_observable, _visited_obs


@_diagonalize_compound_observable.register
def _diagonalize_composite_op(
    observable: LinearCombination, _visited_obs, supported_base_obs=_default_supported_obs
):

    coeffs, obs = observable.terms()

    diagonalizing_gates, new_obs, _visited_obs = _get_obs_and_gates(
        obs, _visited_obs, supported_base_obs
    )

    new_observable = LinearCombination(coeffs, new_obs)

    return diagonalizing_gates, new_observable, _visited_obs


@_diagonalize_compound_observable.register
def _diagonalize_composite_op(
    observable: CompositeOp, _visited_obs, supported_base_obs=_default_supported_obs
):
    diagonalizing_gates, new_operands, _visited_obs = _get_obs_and_gates(
        observable.operands, _visited_obs, supported_base_obs
    )

    new_observable = observable.__class__(*new_operands)

    return diagonalizing_gates, new_observable, _visited_obs
