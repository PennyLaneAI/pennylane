# Copyright 2019 Xanadu Quantum Technologies Inc.

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
This submodule contains functionality for running Variational Quantum Eigensolver (VQE)
computations using PennyLane.
"""
import numpy as np
from pennylane.ops import Observable
from pennylane.measure import expval
from pennylane.qnode import QNode


class Hamiltonian:
    r"""Lightweight class for representing Hamiltonians for Variational Quantum Eigensolver problems.

    Hamiltonians can be expressed as linear combinations of observables, e.g.,
    :math:`\sum_{k=0}^{N-1}` c_k O_k`.

    This class keeps track of the terms (coefficients and observables) separately.

    Args:
        coeffs (Iterable[float]): coefficients of the Hamiltonian expression
        observables (Iterable[Observable]): observables in the Hamiltonian expression
    """

    def __init__(self, coeffs, observables):

        if len(coeffs) != len(observables):
            raise ValueError(
                "Could not create valid Hamiltonian; "
                "number of coefficients and operators does not match."
            )

        if any(np.imag(coeffs) != 0):
            raise ValueError(
                "Could not create valid Hamiltonian; " "coefficients are not real-valued."
            )

        for obs in observables:
            if not isinstance(obs, Observable):
                raise ValueError(
                    "Could not create circuits. Some or all observables are not valid."
                )

        self._coeffs = coeffs
        self._ops = observables

    @property
    def coeffs(self):
        """Return the coefficients defining the Hamiltonian.

        Returns:
            Iterable[float]): coefficients in the Hamiltonian expression
        """
        return self._coeffs

    @property
    def ops(self):
        """Return the operators defining the Hamiltonian.

        Returns:
            Iterable[Observable]): observables in the Hamiltonian expression
        """
        return self._ops

    @property
    def terms(self):
        r"""The terms of the Hamiltonian expression :math:`\sum_{k=0}^{N-1}` c_k O_k`

        Returns:
            (tuple, tuple): tuples of coefficients and operations, each of length N
        """
        return self.coeffs, self.ops


def circuits(ansatz, observables, device, interface="numpy"):
    """Create a set of callable functions which evaluate quantum circuits based on
    ``ansatz`` and ``observables``.

    One circuit function is created for every observable ``obs`` in ``observables``.
    The circuits have the following structure:

    .. code-block:: python
        @qml.qnode(device)
        def circuit(params):
            ansatz(*params, wires=range(device.num_wires))
            return qml.expval(obs)

    Args:
        ansatz (callable): the ansatz for the circuit before the final measurement step
        observables (Iterable[:class:`~.Observable`]): observables to measure during the final step of each circuit
        device (:class:`~.Device`): device where the circuits should be executed
        interface (str): which interface to use for the :class:`~.QNode`s of the circuits

    Returns:
        tuple: callable functions which evaluate each observable
    """
    if not callable(ansatz):
        raise ValueError(
            "Could not create quantum circuits. The ansatz is not a callable function."
        )

    # TODO: can be more clever/efficient here for observables which are jointly measurable
    qnodes = []
    for obs in observables:
        if not isinstance(obs, Observable):
            raise ValueError("Could not create circuits. Some or all observables are not valid.")

        def circuit(*params, obs=obs):
            ansatz(*params, wires=range(device.num_wires))
            return expval(obs)

        qnode = QNode(circuit, device)

        if interface == "tf":
            qnode = qnode.to_tf()
        elif interface == "torch":
            qnode = qnode.to_torch()

        qnodes.append(qnode)

    return qnodes


def aggregate(coeffs, qnodes, params):
    r"""Aggregate a collection of coefficients and expectation values into a single scalar.

    Suppose that the kth :class:`~.QNode` in ``qnodes`` returns the expectation value
    :math:`\langle O_k \rangle`, and that the kth element of ``coeffs`` is :math:`c_k`.
    Then this function returns the sum :math:`\sum_{k} c_k O_k`.

    Args:
        coeffs (Iterable[float]): the coefficients of each summand
        qnodes (Iterable[:class:`~.QNode`]): Callable :class:`~.QNode` functions which return
            an expectation value. All QNodes provided must use the same interface.
        params (Iterable[float, tf.Tensor, torch.Tensor]): parameter values to be
            used as arguments to each individual :class:`~.QNode`

    Returns:
        float: the result of evaluating each :class:`~.QNode` and summing with the ``coeffs``
    """
    interfaces = {q.interface for q in qnodes}

    if len(interfaces) != 1:
        raise ValueError("Provided QNodes must all use the same interface.")

    expvals = [c*circuit(*params) for c, circuit in zip(coeffs, qnodes)]
    return sum(expvals)


def cost(params, ansatz, hamiltonian, device, interface="numpy"):
    """
    Evaluate the VQE cost function, i.e., the expectation value of a Hamiltonian.

    This function evaluates the expectation value of ``hamiltonian`` for the circuit
    specified by ``ansatz`` with circuit parameters ``params``.

    Args:
        params (Iterable[float, tf.Tensor, torch.Tensor]): the parameters of the circuit
            to use when evaluating the expectation value
        ansatz (function or :class:`~.template`): Callable function which contains a
            series of PennyLane operations, but no measurements.
        hamiltonian (:class:`~.Hamiltonian`): the Hamiltonian whose expectation value
            is to be evaluated
        interface (str): which interface to use for the underlying circuits

    Returns:
        float: the result of evaluating each :class:`~.QNode` and summing with the ``coeffs``
    """
    coeffs, observables = hamiltonian.terms
    qnodes = circuits(ansatz, observables, device, interface)
    return aggregate(coeffs, qnodes, params)
