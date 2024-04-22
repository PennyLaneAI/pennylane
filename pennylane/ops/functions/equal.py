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
"""
This module contains the qml.equal function.
"""
# pylint: disable=too-many-arguments,too-many-return-statements,too-many-branches
from collections.abc import Iterable
from functools import singledispatch
from typing import Union

import pennylane as qml
from pennylane.measurements import MeasurementProcess
from pennylane.measurements.classical_shadow import ShadowExpvalMP
from pennylane.measurements.mid_measure import MidMeasureMP, MeasurementValue
from pennylane.measurements.mutual_info import MutualInfoMP
from pennylane.measurements.vn_entropy import VnEntropyMP
from pennylane.measurements.counts import CountsMP
from pennylane.pulse.parametrized_evolution import ParametrizedEvolution
from pennylane.operation import Observable, Operator, Tensor
from pennylane.ops import (
    Hamiltonian,
    LinearCombination,
    Controlled,
    Pow,
    Adjoint,
    Exp,
    SProd,
    CompositeOp,
)
from pennylane.templates.subroutines import ControlledSequence
from pennylane.tape import QuantumTape


def equal(
    op1: Union[Operator, MeasurementProcess, QuantumTape],
    op2: Union[Operator, MeasurementProcess, QuantumTape],
    check_interface=True,
    check_trainability=True,
    rtol=1e-5,
    atol=1e-9,
):
    r"""Function for determining operator or measurement equality.

    .. Warning::

        The ``qml.equal`` function is based on a comparison of the type and attributes
        of the measurement or operator, not a mathematical representation. While
        comparisons between some classes, such as ``Tensor`` and ``Hamiltonian``, are
        supported, mathematically equivalent operators defined via different classes
        may return False when compared via ``qml.equal``.

        To be more thorough would require the matrix forms to be calculated, which may
        drastically increase runtime.

    .. Warning::

        The kwargs ``check_interface`` and ``check_trainability`` can only be set when
        comparing ``Operation`` objects. Comparisons of ``MeasurementProcess``
        or ``Observable`` objects will use the default value of ``True`` for both, regardless
        of what the user specifies when calling the function. For subclasses of ``SymbolicOp``
        or ``CompositeOp`` with an ``Operation`` as a base, the kwargs will be applied to the base
        comparison.

    Args:
        op1 (.Operator or .MeasurementProcess or .QuantumTape): First object to compare
        op2 (.Operator or .MeasurementProcess or .QuantumTape): Second object to compare
        check_interface (bool, optional): Whether to compare interfaces. Default: ``True``. Not used for comparing ``MeasurementProcess``, ``Hamiltonian`` or ``Tensor`` objects.
        check_trainability (bool, optional): Whether to compare trainability status. Default: ``True``. Not used for comparing ``MeasurementProcess``, ``Hamiltonian`` or ``Tensor`` objects.
        rtol (float, optional): Relative tolerance for parameters. Not used for comparing ``MeasurementProcess``, ``Hamiltonian`` or ``Tensor`` objects.
        atol (float, optional): Absolute tolerance for parameters. Not used for comparing ``MeasurementProcess``, ``Hamiltonian`` or ``Tensor`` objects.

    Returns:
        bool: ``True`` if the operators or measurement processes are equal, else ``False``

    **Example**

    Given two operators or measurement processes, ``qml.equal`` determines their equality.

    >>> op1 = qml.RX(np.array(.12), wires=0)
    >>> op2 = qml.RY(np.array(1.23), wires=0)
    >>> qml.equal(op1, op1), qml.equal(op1, op2)
    (True, False)

    >>> T1 = qml.X(0) @ qml.Y(1)
    >>> T2 = qml.Y(1) @ qml.X(0)
    >>> T3 = qml.X(1) @ qml.Y(0)
    >>> qml.equal(T1, T2), qml.equal(T1, T3)
    (True, False)

    >>> T = qml.X(0) @ qml.Y(1)
    >>> H = qml.Hamiltonian([1], [qml.X(0) @ qml.Y(1)])
    >>> qml.equal(T, H)
    True

    >>> H1 = qml.Hamiltonian([0.5, 0.5], [qml.Z(0) @ qml.Y(1), qml.Y(1) @ qml.Z(0) @ qml.Identity("a")])
    >>> H2 = qml.Hamiltonian([1], [qml.Z(0) @ qml.Y(1)])
    >>> H3 = qml.Hamiltonian([2], [qml.Z(0) @ qml.Y(1)])
    >>> qml.equal(H1, H2), qml.equal(H1, H3)
    (True, False)

    >>> qml.equal(qml.expval(qml.X(0)), qml.expval(qml.X(0)))
    True
    >>> qml.equal(qml.probs(wires=(0,1)), qml.probs(wires=(1,2)))
    False
    >>> qml.equal(qml.classical_shadow(wires=[0,1]), qml.classical_shadow(wires=[0,1]))
    True
    >>> tape1 = qml.tape.QuantumScript([qml.RX(1.2, wires=0)], [qml.expval(qml.Z(0))])
    >>> tape2 = qml.tape.QuantumScript([qml.RX(1.2 + 1e-6, wires=0)], [qml.expval(qml.Z(0))])
    >>> qml.equal(tape1, tape2, tol=0, atol=1e-7)
    False
    >>> qml.equal(tape1, tape2, tol=0, atol=1e-5)
    True

    .. details::
        :title: Usage Details

        You can use the optional arguments to get more specific results. Additionally, they are
        applied when comparing the base of ``SymbolicOp`` and ``CompositeOp`` operators such as
        ``Controlled``, ``Pow``, ``SProd``, ``Prod``, etc., if the base is an ``Operation``. These arguments
        are, however, not used for comparing ``MeasurementProcess``, ``Hamiltonian`` or ``Tensor``
        objects.

        Consider the following comparisons:

        >>> op1 = qml.RX(torch.tensor(1.2), wires=0)
        >>> op2 = qml.RX(jax.numpy.array(1.2), wires=0)
        >>> qml.equal(op1, op2)
        False

        >>> qml.equal(op1, op2, check_interface=False, check_trainability=False)
        True

        >>> op3 = qml.RX(np.array(1.2, requires_grad=True), wires=0)
        >>> op4 = qml.RX(np.array(1.2, requires_grad=False), wires=0)
        >>> qml.equal(op3, op4)
        False

        >>> qml.equal(op3, op4, check_trainability=False)
        True

        >>> qml.equal(Controlled(op3, control_wires=1), Controlled(op4, control_wires=1))
        False

        >>> qml.equal(Controlled(op3, control_wires=1), Controlled(op4, control_wires=1), check_trainability=False)
        True
    """

    # types don't have to be the same type, they just both have to be Observables
    if not isinstance(op2, type(op1)) and not isinstance(op1, Observable):
        return False

    if isinstance(op2, (Hamiltonian, Tensor)):
        return _equal(op2, op1)

    return _equal(
        op1,
        op2,
        check_interface=check_interface,
        check_trainability=check_trainability,
        atol=atol,
        rtol=rtol,
    )


@singledispatch
def _equal(
    op1,
    op2,
    check_interface=True,
    check_trainability=True,
    rtol=1e-5,
    atol=1e-9,
):  # pylint: disable=unused-argument
    raise NotImplementedError(f"Comparison of {type(op1)} and {type(op2)} not implemented")


@_equal.register
def _equal_circuit(
    op1: qml.tape.QuantumScript,
    op2: qml.tape.QuantumScript,
    check_interface=True,
    check_trainability=True,
    rtol=1e-5,
    atol=1e-9,
):
    # operations
    if len(op1.operations) != len(op2.operations):
        return False
    for comparands in zip(op1.operations, op2.operations):
        if not qml.equal(
            comparands[0],
            comparands[1],
            check_interface=check_interface,
            check_trainability=check_trainability,
            rtol=rtol,
            atol=atol,
        ):
            return False
    # measurements
    if len(op1.measurements) != len(op2.measurements):
        return False
    for comparands in zip(op1.measurements, op2.measurements):
        if not qml.equal(
            comparands[0],
            comparands[1],
            check_interface=check_interface,
            check_trainability=check_trainability,
            rtol=rtol,
            atol=atol,
        ):
            return False

    if op1.shots != op2.shots:
        return False
    if op1.trainable_params != op2.trainable_params:
        return False
    return True


@_equal.register
def _equal_operators(
    op1: Operator,
    op2: Operator,
    check_interface=True,
    check_trainability=True,
    rtol=1e-5,
    atol=1e-9,
):
    """Default function to determine whether two Operator objects are equal."""
    if not isinstance(
        op2, type(op1)
    ):  # clarifies cases involving PauliX/Y/Z (Observable/Operation)
        return False

    if isinstance(op1, qml.Identity):
        # All Identities are equivalent, independent of wires.
        # We already know op1 and op2 are of the same type, so no need to check
        # that op2 is also an Identity
        return True

    if op1.arithmetic_depth != op2.arithmetic_depth:
        return False

    if op1.arithmetic_depth > 0:
        # Other dispatches cover cases of operations with arithmetic depth > 0.
        # If any new operations are added with arithmetic depth > 0, a new dispatch
        # should be created for them.
        return False
    if not all(
        qml.math.allclose(d1, d2, rtol=rtol, atol=atol) for d1, d2 in zip(op1.data, op2.data)
    ):
        return False
    if op1.wires != op2.wires:
        return False

    if op1.hyperparameters != op2.hyperparameters:
        return False

    if check_trainability:
        for params_1, params_2 in zip(op1.data, op2.data):
            if qml.math.requires_grad(params_1) != qml.math.requires_grad(params_2):
                return False

    if check_interface:
        for params_1, params_2 in zip(op1.data, op2.data):
            if qml.math.get_interface(params_1) != qml.math.get_interface(params_2):
                return False

    return True


@_equal.register
# pylint: disable=unused-argument, protected-access
def _equal_prod_and_sum(op1: CompositeOp, op2: CompositeOp, **kwargs):
    """Determine whether two Prod or Sum objects are equal"""
    if op1.pauli_rep is not None and (op1.pauli_rep == op2.pauli_rep):  # shortcut check
        return True

    if len(op1.operands) != len(op2.operands):
        return False

    # organizes by wire indicies while respecting commutation relations
    sorted_ops1 = op1._sort(op1.operands)
    sorted_ops2 = op2._sort(op2.operands)

    return all(equal(o1, o2, **kwargs) for o1, o2 in zip(sorted_ops1, sorted_ops2))


@_equal.register
def _equal_controlled(op1: Controlled, op2: Controlled, **kwargs):
    """Determine whether two Controlled or ControlledOp objects are equal"""
    # work wires and control_wire/control_value combinations compared here
    # op.base.wires compared in return
    if [
        dict(zip(op1.control_wires, op1.control_values)),
        op1.work_wires,
        op1.arithmetic_depth,
    ] != [
        dict(zip(op2.control_wires, op2.control_values)),
        op2.work_wires,
        op2.arithmetic_depth,
    ]:
        return False

    return qml.equal(op1.base, op2.base, **kwargs)


@_equal.register
def _equal_controlled_sequence(op1: ControlledSequence, op2: ControlledSequence, **kwargs):
    """Determine whether two ControlledSequences are equal"""
    if [op1.wires, op1.arithmetic_depth] != [
        op2.wires,
        op2.arithmetic_depth,
    ]:
        return False

    return qml.equal(op1.base, op2.base, **kwargs)


@_equal.register
# pylint: disable=unused-argument
def _equal_pow(op1: Pow, op2: Pow, **kwargs):
    """Determine whether two Pow objects are equal"""
    if op1.z != op2.z:
        return False
    return qml.equal(op1.base, op2.base)


@_equal.register
# pylint: disable=unused-argument
def _equal_adjoint(op1: Adjoint, op2: Adjoint, **kwargs):
    """Determine whether two Adjoint objects are equal"""
    # first line of top-level equal function already confirms both are Adjoint - only need to compare bases
    return qml.equal(op1.base, op2.base)


@_equal.register
# pylint: disable=unused-argument
def _equal_exp(op1: Exp, op2: Exp, **kwargs):
    """Determine whether two Exp objects are equal"""
    rtol, atol = (kwargs["rtol"], kwargs["atol"])

    if not qml.math.allclose(op1.coeff, op2.coeff, rtol=rtol, atol=atol):
        return False
    return qml.equal(op1.base, op2.base)


@_equal.register
# pylint: disable=unused-argument
def _equal_sprod(op1: SProd, op2: SProd, **kwargs):
    """Determine whether two SProd objects are equal"""
    rtol, atol = (kwargs["rtol"], kwargs["atol"])

    if op1.pauli_rep is not None and (op1.pauli_rep == op2.pauli_rep):  # shortcut check
        return True

    if not qml.math.allclose(op1.scalar, op2.scalar, rtol=rtol, atol=atol):
        return False
    return qml.equal(op1.base, op2.base)


@_equal.register
# pylint: disable=unused-argument
def _equal_tensor(op1: Tensor, op2: Observable, **kwargs):
    """Determine whether a Tensor object is equal to a Hamiltonian/Tensor"""
    if not isinstance(op2, Observable):
        return False

    if isinstance(op2, (Hamiltonian, LinearCombination)):
        return op2.compare(op1)

    if isinstance(op2, Tensor):
        return op1._obs_data() == op2._obs_data()  # pylint: disable=protected-access

    return False


@_equal.register
# pylint: disable=unused-argument
def _equal_hamiltonian(op1: Hamiltonian, op2: Observable, **kwargs):
    """Determine whether a Hamiltonian object is equal to a Hamiltonian/Tensor objects"""
    if not isinstance(op2, Observable):
        return False
    return op1.compare(op2)


@_equal.register
def _equal_parametrized_evolution(op1: ParametrizedEvolution, op2: ParametrizedEvolution, **kwargs):
    # check times match
    if op1.t is None or op2.t is None:
        if not (op1.t is None and op2.t is None):
            return False
    elif not qml.math.allclose(op1.t, op2.t):
        return False

    # check parameters passed to operator match
    if not _equal_operators(op1, op2, **kwargs):
        return False

    # check H.coeffs match
    if not all(c1 == c2 for c1, c2 in zip(op1.H.coeffs, op2.H.coeffs)):
        return False

    # check that all the base operators on op1.H and op2.H match
    return all(equal(o1, o2, **kwargs) for o1, o2 in zip(op1.H.ops, op2.H.ops))


@_equal.register
# pylint: disable=unused-argument
def _equal_measurements(
    op1: MeasurementProcess,
    op2: MeasurementProcess,
    check_interface=True,
    check_trainability=True,
    rtol=1e-5,
    atol=1e-9,
):
    """Determine whether two MeasurementProcess objects are equal"""

    if op1.obs is not None and op2.obs is not None:
        return equal(
            op1.obs,
            op2.obs,
            check_interface=check_interface,
            check_trainability=check_trainability,
            rtol=rtol,
            atol=atol,
        )

    if op1.mv is not None and op2.mv is not None:
        # qml.equal doesn't check if the MeasurementValues have the same processing functions
        if isinstance(op1.mv, MeasurementValue) and isinstance(op2.mv, MeasurementValue):
            return op1.mv.measurements == op2.mv.measurements

        if isinstance(op1.mv, Iterable) and isinstance(op2.mv, Iterable):
            if len(op1.mv) == len(op2.mv):
                return all(mv1.measurements == mv2.measurements for mv1, mv2 in zip(op1.mv, op2.mv))

        return False

    if op1.wires != op2.wires:
        return False

    if op1.obs is None and op2.obs is None:
        # only compare eigvals if both observables are None.
        # Can be expensive to compute for large observables
        if op1.eigvals() is not None and op2.eigvals() is not None:
            return qml.math.allclose(op1.eigvals(), op2.eigvals(), rtol=rtol, atol=atol)

        return op1.eigvals() is None and op2.eigvals() is None

    return False


@_equal.register
def _equal_mid_measure(op1: MidMeasureMP, op2: MidMeasureMP, **_):
    return (
        op1.wires == op2.wires
        and op1.id == op2.id
        and op1.reset == op2.reset
        and op1.postselect == op2.postselect
    )


@_equal.register
# pylint: disable=unused-argument
def _(op1: VnEntropyMP, op2: VnEntropyMP, **kwargs):
    """Determine whether two MeasurementProcess objects are equal"""
    eq_m = _equal_measurements(op1, op2, **kwargs)
    log_base_match = op1.log_base == op2.log_base
    return eq_m and log_base_match


@_equal.register
# pylint: disable=unused-argument
def _(op1: MutualInfoMP, op2: MutualInfoMP, **kwargs):
    """Determine whether two MeasurementProcess objects are equal"""
    eq_m = _equal_measurements(op1, op2, **kwargs)
    log_base_match = op1.log_base == op2.log_base
    return eq_m and log_base_match


@_equal.register
# pylint: disable=unused-argument
def _equal_shadow_measurements(op1: ShadowExpvalMP, op2: ShadowExpvalMP, **_):
    """Determine whether two ShadowExpvalMP objects are equal"""

    wires_match = op1.wires == op2.wires

    if isinstance(op1.H, Operator) and isinstance(op2.H, Operator):
        H_match = equal(op1.H, op2.H)
    elif isinstance(op1.H, Iterable) and isinstance(op2.H, Iterable):
        H_match = all(equal(o1, o2) for o1, o2 in zip(op1.H, op2.H))
    else:
        return False

    k_match = op1.k == op2.k

    return wires_match and H_match and k_match


@_equal.register
def _equal_counts(op1: CountsMP, op2: CountsMP, **kwargs):
    return _equal_measurements(op1, op2, **kwargs) and op1.all_outcomes == op2.all_outcomes


@_equal.register
# pylint: disable=unused-argument
def _equal_basis_rotation(
    op1: qml.BasisRotation,
    op2: qml.BasisRotation,
    check_interface=True,
    check_trainability=True,
    rtol=1e-5,
    atol=1e-9,
):
    if not qml.math.allclose(
        op1.hyperparameters["unitary_matrix"],
        op2.hyperparameters["unitary_matrix"],
        atol=atol,
        rtol=rtol,
    ):
        return False
    if op1.wires != op2.wires:
        return False
    if check_interface:
        if qml.math.get_interface(op1.hyperparameters["unitary_matrix"]) != qml.math.get_interface(
            op2.hyperparameters["unitary_matrix"]
        ):
            return False
    return True


@_equal.register
def _equal_hilbert_schmidt(
    op1: qml.HilbertSchmidt,
    op2: qml.HilbertSchmidt,
    check_interface=True,
    check_trainability=True,
    rtol=1e-5,
    atol=1e-9,
):
    if not all(
        qml.math.allclose(d1, d2, rtol=rtol, atol=atol) for d1, d2 in zip(op1.data, op2.data)
    ):
        return False

    if check_trainability:
        for params_1, params_2 in zip(op1.data, op2.data):
            if qml.math.requires_grad(params_1) != qml.math.requires_grad(params_2):
                return False

    if check_interface:
        for params_1, params_2 in zip(op1.data, op2.data):
            if qml.math.get_interface(params_1) != qml.math.get_interface(params_2):
                return False

    equal_kwargs = {
        "check_interface": check_interface,
        "check_trainability": check_trainability,
        "atol": atol,
        "rtol": rtol,
    }
    # Check hyperparameters using qml.equal rather than == where necessary
    if op1.hyperparameters["v_wires"] != op2.hyperparameters["v_wires"]:
        return False
    if not qml.equal(op1.hyperparameters["u_tape"], op2.hyperparameters["u_tape"], **equal_kwargs):
        return False
    if not qml.equal(op1.hyperparameters["v_tape"], op2.hyperparameters["v_tape"], **equal_kwargs):
        return False
    if op1.hyperparameters["v_function"] != op2.hyperparameters["v_function"]:
        return False

    return True
