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
from pennylane.ops import Hermitian
from pennylane import expval

sum_fn = {"numpy", np.sum}
try:
    from tensorflow.math import reduce_sum
    sum_fn["tf"] = reduce_sum
except:
    pass

try:
    from torch import sum
    sum_fn["torch"] = sum
except:
    pass


class Hamiltonian:
    """
    Lightweight class for representing Hamiltonians for Variational Quantum Eigensolver problems.

    Hamiltonians can be expressed as linear combinations of observables, e.g.,
    :math:`\sum_{k=0}^{N-1}` c_k O_k`.
    This class keeps track of the terms (coefficients and observables) separately.

    Args:
        coeffs (Iterable[float]): coefficients of the Hamiltonian expression
        ops (Iterable[Observable]): observables in the Hamiltonian expression
        tol (float): tolerance used to determine if ops are Hermitian
    """

    def __init__(self, coeffs, ops, tol=1e-5):

        if len(coeffs) != len(ops):
            raise ValueError("Could not create valid Hamiltonian; "
                             "number of coefficients and operators does not match.")
        if any(np.imag(coeffs) != 0):
            raise ValueError("Could not create valid Hamiltonian; "
                             "coefficients are not real-valued.")
        # TODO: to make easier, add a boolean attribute `hermitian` to all pennylane ops
        numeric_ops = [op.parameters[0] for op in ops if isinstance(op, Hermitian)]
        for A in numeric_ops:
            if not np.allclose(A, np.conj(A).T, rtol=tol, atol=0):
                raise ValueError("Could not create valid Hamiltonian; "
                                 "one or more ops are not Hermitian.")

        self._coeffs = coeffs
        self._ops = ops

    @property
    def coeffs(self):
        return self._coeffs

    @property
    def ops(self):
        return self._ops

    @property
    def terms(self):
        """
        The terms of the Hamiltonian expression :math:`\sum_{k=0}^{N-1}` c_k O_k`

        Returns:
            (tuple, tuple): tuples of coefficients and operations, each of length N
        """
        return self.coeffs, self.ops


def qnodes(ansatz, observables, device, interface='numpy'):
    """
    Create a set of :class:`~.QNode`s whose circuits are based on ``ansatz`` and ``observables``.

    One :class:`~.QNode` is created for every observable ``obs`` in ``observables``. The QNode has the following
    circuit structure:

    .. code-block:: python
        @qml.qnode(device)
        def circuit(params):
            ansatz
            return qml.expval(obs)

    Args:
        ansatz (Iterable[:class:`~.Observable`]): the ansatz for the circuit before the final measurement step
        observables (Iterable[:class:`~.Observable`]): observables to measure during the final step of each circuit
        device (:class:`~.Device`): device where the :class:`~.QNode`s should be executed
        interface (str): which interface to use for the :class:`~.QNode`s

    Returns:
        tuple(:class:`~.QNode`): callable functions which evaluate each observable
    """
    # TODO: can be more clever/efficient here for observables which are jointly measurable
    qnodes = []
    for obs in observables:
        @qml.qnode(device=device, interface=interface)
        def circuit(params):
            ansatz(params)
            return expval(obs)
        qnodes += circuit
    return qnodes


def aggregate(coeffs, qnodes, params, interface="numpy"):
    """
    Aggregate a collection of coefficients and expectation values into a single scalar.

    Suppose that the kth :class:`~.QNode` in ``qnodes`` returns the expectation value :math:`\langle O_k \rangle`, and
    that the kth element of ``coeffs`` is :math:`c_k`.
    Then this function returns the sum :math:`\sum_{k} c_k O_k`.

    Args:
        coeffs (Iterable[float]): the coefficients of each summand
        qnodes (Iterable[:class:`~.QNode`]): callable :class:`~.QNode` functions which return an expectation value
        params (Iterable[float, tf.Tensor, torch.Tensor]): parameter values to be used as arguments to each
            individual :class:`~.QNode`
        interface (str): which interface to use for the :class:`~.QNode`s

    Returns:
        float: the result of evaluating each :class:`~.QNode` and summing with the ``coeffs``
    """
    expvals = [circuit(params) for circuit in qnodes]
    return sum_fn[interface](coeffs * expvals)


def cost(params, ansatz, hamiltonian, device, interface="numpy"):
    """
    Evaluate the VQE cost function, i.e., the expectation value of a Hamiltonian.

    This function evaluates the expectation value of ``hamiltonian`` for the circuit specified by ``ansatz``
    with circuit parameters ``params``.

    Args:
        params (Iterable[float, tf.Tensor, torch.Tensor]): the parameters of the circuit to use when evaluating the
            expectation value
        ansatz (function or :class:`~.template`): Callable function which contains a series of PennyLane operations,
            but no measurements.
        hamiltonian (:class:`~.Hamiltonian`): the Hamiltonian whose expectation value is to be evaluated
        interface (str): which interface to use for the underlying circuits

    Returns:
        float: the result of evaluating each :class:`~.QNode` and summing with the ``coeffs``
    """
    coeffs, observables = hamiltonian.terms
    qnodes = qnodes(ansatz, observables, device, interface)
    return aggregate(coeffs, qnodes, params, interface)
