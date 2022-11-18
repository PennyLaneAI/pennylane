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
# pylint: disable=too-many-arguments,too-many-return-statements, too-many-branches
from typing import Union

from functools import singledispatch
import pennylane as qml
from pennylane.measurements import MeasurementProcess, ShadowMeasurementProcess
from pennylane.operation import Operation, Operator, Observable
from pennylane.ops.qubit.hamiltonian import Hamiltonian


def equal(
    op1: Union[Operator, MeasurementProcess],
    op2: Union[Operator, MeasurementProcess],
    check_interface=True,
    check_trainability=True,
    rtol=1e-5,
    atol=1e-9,
):
    r"""Function for determining operator or measurement equality.

    .. Warning::

        The compare method does **not** check if the matrix representation
        of a :class:`~.Hermitian` observable is equal to an equivalent
        observable expressed in terms of Pauli matrices, or as a
        linear combination of Hermitians.
        To do so would require the matrix form of Hamiltonians and Tensors
        be calculated, which would drastically increase runtime.

        The kwargs ``check_interface`` and ``check_trainability`` can only be set when
        comparing ``Operation`` objects. Comparisons of ``MeasurementProcess`` or ``Observable``
        objects will use the defualt value of ``True`` for both, regardless of what the user
        specifies when calling the function.

    Args:
        op1 (.Operator or .MeasurementProcess): First object to compare
        op2 (.Operator or .MeasurementProcess): Second object to compare
        check_interface (bool, optional): Whether to compare interfaces. Default: ``True``. Can only be set by user when comparing ``Operations``.
        check_trainability (bool, optional): Whether to compare trainability status. Default: ``True``. Can only be set by user when comparing ``Operations``.
        rtol (float, optional): Relative tolerance for parameters
        atol (float, optional): Absolute tolerance for parameters

    Returns:
        bool: ``True`` if the operators or measurement processes are equal, else ``False``

    **Examples**

    Given two operators or measurement processes, ``qml.equal`` determines their equality.

    >>> op1 = qml.RX(np.array(.12), wires=0)
    >>> op2 = qml.RY(np.array(1.23), wires=0)
    >>> qml.equal(op1, op1), qml.equal(op1, op2)
    (True, False)

    >>> T1 = qml.PauliX(0) @ qml.PauliY(1)
    >>> T2 = qml.PauliY(1) @ qml.PauliX(0)
    >>> T3 = qml.PauliX(1) @ qml.PauliY(0)
    >>> qml.equal(T1, T2), qml.equal(T1, T3)
    (True, False)

    >>> T = qml.PauliX(0) @ qml.PauliY(1)
    >>> H = qml.Hamiltonian([1], [qml.PauliX(0) @ qml.PauliY(1)])
    >>> qml.equal(T, H)
    True

    >>> H1 = qml.Hamiltonian([0.5, 0.5], [qml.PauliZ(0) @ qml.PauliY(1), qml.PauliY(1) @ qml.PauliZ(0) @ qml.Identity("a")])
    >>> H2 = qml.Hamiltonian([1], [qml.PauliZ(0) @ qml.PauliY(1)])
    >>> H3 = qml.Hamiltonian([2], [qml.PauliZ(0) @ qml.PauliY(1)])
    >>> qml.equal(H1, H2), qml.equal(H1, H3)
    (True, False)

    >>> qml.equal(qml.expval(qml.PauliX(0)), qml.expval(qml.PauliX(0)) )
    True
    >>> qml.equal(qml.probs(wires=(0,1)), qml.probs(wires=(1,2)) )
    False
    >>> qml.equal(qml.classical_shadow(wires=[0,1]), qml.classical_shadow(wires=[0,1]) )
    True



    .. details::
        :title: Usage Details

        You can use the optional arguments when comparing ``Operations`` to get more specific results.

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
    """

    # types don't have to be the same type, they just both have to be Observables
    if isinstance(op1, Observable) and isinstance(op2, Observable):
        pass
    # for everything else, types must match
    else:
        if not isinstance(op2, type(op1)):
            return False

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
    **kwargs,
):
    """Calls relevant comparison function using singledispatch."""
    raise NotImplementedError(f"Comparison between {type(op1)} and {type(op2)} not implemented.")


@_equal.register
def _equal_operation(
    op1: Operation,
    op2: Operation,
    check_interface=True,
    check_trainability=True,
    rtol=1e-5,
    atol=1e-9,
):
    """Determine whether two Operations objects are equal."""

    if op1.arithmetic_depth != op2.arithmetic_depth:
        return False

    if op1.arithmetic_depth > 0:
        raise NotImplementedError(
            "Comparison of operators with an arithmetic depth larger than 0 is not yet implemented."
        )
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

    return getattr(op1, "inverse", False) == getattr(op2, "inverse", False)


@_equal.register
# pylint: disable=unused-argument
def _equal_observables(op1: Observable, op2: Observable, **kwargs):
    """Determine whether two Observables objects are equal"""
    return _obs_comparison_data(op1) == _obs_comparison_data(op2)


@_equal.register
# pylint: disable=unused-argument
def _equal_measurements(op1: MeasurementProcess, op2: MeasurementProcess, **kwargs):

    """Determine whether two MeasurementProcess objects are equal"""

    return_types_match = op1.return_type == op2.return_type
    if op1.obs is not None and op2.obs is not None:
        observables_match = equal(op1.obs, op2.obs)
    # check obs equality when either one is None (False) or both are None (True)
    else:
        observables_match = op1.obs == op2.obs
    wires_match = op1.wires == op2.wires
    eigvals_match = qml.math.allequal(op1.eigvals(), op2.eigvals())
    log_base_match = getattr(op1, "log_base", None) == getattr(op2, "log_base", None)

    return (
        return_types_match
        and observables_match
        and wires_match
        and eigvals_match
        and log_base_match
    )


@_equal.register
# pylint: disable=unused-argument
def _equal_shadow_measurements(
    op1: ShadowMeasurementProcess, op2: ShadowMeasurementProcess, **kwargs
):
    """Determine whether two ShadowMeasurementProcess objects are equal"""

    return_types_match = op1.return_type == op2.return_type
    wires_match = op1.wires == op2.wires
    H_match = op1.H == op2.H
    k_match = op1.k == op2.k

    return return_types_match and wires_match and H_match and k_match


def _obs_comparison_data(obs):
    if isinstance(obs, Hamiltonian):
        obs.simplify()
        obs = obs._obs_data()  # pylint: disable=protected-access
    else:
        obs = {(1, frozenset(obs._obs_data()))}  # pylint: disable=protected-access
    return obs
