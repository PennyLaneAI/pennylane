# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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
# pylint: disable=too-many-arguments, too-few-public-methods
import numpy as np
import pennylane as qml
from pennylane.operation import Observable, Tensor


OBS_MAP = {"PauliX": "X", "PauliY": "Y", "PauliZ": "Z", "Hadamard": "H", "Identity": "I"}


class Hamiltonian:
    r"""Lightweight class for representing Hamiltonians for Variational Quantum
    Eigensolver problems.

    Hamiltonians can be expressed as linear combinations of observables, e.g.,
    :math:`\sum_{k=0}^{N-1} c_k O_k`.

    This class keeps track of the terms (coefficients and observables) separately.

    Args:
        coeffs (Iterable[float]): coefficients of the Hamiltonian expression
        observables (Iterable[Observable]): observables in the Hamiltonian expression

    .. seealso:: :class:`~.VQECost`, :func:`~.generate_hamiltonian`

    **Example:**

    A Hamiltonian can be created by simply passing the list of coefficients
    as well as the list of observables:

    >>> coeffs = [0.2, -0.543]
    >>> obs = [qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.Hadamard(2)]
    >>> H = qml.Hamiltonian(coeffs, obs)
    >>> print(H)
    (0.2) [X0 Z1] + (-0.543) [Z0 H2]

    Alternatively, the :func:`~.generate_hamiltonian` function from the
    :doc:`/introduction/chemistry` module can be used to generate a molecular
    Hamiltonian.
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

    def __str__(self):
        terms = []

        for i, obs in enumerate(self.ops):
            coeff = "({}) [{{}}]".format(self.coeffs[i])

            if isinstance(obs, Tensor):
                obs_strs = ["{}{}".format(OBS_MAP[i.name], i.wires[0]) for i in obs.obs]
                term = " ".join(obs_strs)
            elif isinstance(obs, Observable):
                term = "{}{}".format(OBS_MAP[obs.name], obs.wires[0])

            terms.append(coeff.format(term))

        return "\n+ ".join(terms)


class VQECost:
    """Create a VQE cost function, i.e., a cost function returning the
    expectation value of a Hamiltonian.

    Args:
        ansatz (callable): The ansatz for the circuit before the final measurement step.
            Note that the ansatz **must** have the following signature:

            .. code-block:: python

                ansatz(params, **kwargs)

            where ``params`` are the trainable weights of the variational circuit, and
            ``kwargs`` are any additional keyword arguments that need to be passed
            to the template.
        hamiltonian (~.Hamiltonian): Hamiltonian operator whose expectation value should be measured
        device (Device, Sequence[Device]): Corresponding device(s) where the resulting
            cost function should be executed. This can either be a single device, or a list
            of devices of length matching the number of terms in the Hamiltonian.
        interface (str, None): Which interface to use.
            This affects the types of objects that can be passed to/returned to the cost function.
            Supports all interfaces supported by the :func:`~.qnode` decorator.
        diff_method (str, None): The method of differentiation to use with the created cost function.
            Supports all differentiation methods supported by the :func:`~.qnode` decorator.

    Returns:
        callable: a cost function with signature ``cost_fn(params, **kwargs)`` that evaluates
        the expectation of the Hamiltonian on the provided device(s)

    .. seealso:: :class:`~.Hamiltonian`, :func:`~.generate_hamiltonian`, :func:`~.map`, :func:`~.dot`

    **Example:**

    First, we create a device and design an ansatz:

    .. code-block:: python

        dev = qml.device('default.qubit', wires=4)

        def ansatz(params, **kwargs):
            qml.BasisState(np.array([1, 1, 0, 0]), wires=[0, 1, 2, 3])
            for i in range(4):
                qml.Rot(*params[i], wires=i)
            qml.CNOT(wires=[2, 3])
            qml.CNOT(wires=[2, 0])
            qml.CNOT(wires=[3, 1])

    Now we can create the Hamiltonian that defines the VQE problem:

    .. code-block:: python3

        coeffs = [0.2, -0.543]
        obs = [
            qml.PauliX(0) @ qml.PauliZ(1) @ qml.PauliY(3),
            qml.PauliZ(0) @ qml.Hadamard(2)
        ]
        H = qml.vqe.Hamiltonian(coeffs, obs)

    Alternatively, the :func:`~.generate_hamiltonian` function from the
    :doc:`/introduction/chemistry` module can be used to generate a molecular
    Hamiltonian.

    Next, we can define the cost function:

    >>> cost = qml.VQECost(ansatz, hamiltonian, dev, interface="torch")
    >>> params = torch.rand([4, 3])
    >>> cost(params)
    tensor(0.0245, dtype=torch.float64)

    The cost function can be minimized using any gradient descent-based
    :doc:`optimizer </introduction/optimizers>`.
    """

    def __init__(
        self, ansatz, hamiltonian, device, interface="autograd", diff_method="best", **kwargs
    ):
        coeffs, observables = hamiltonian.terms
        self.hamiltonian = hamiltonian
        """Hamiltonian: the hamiltonian defining the VQE problem."""

        self.qnodes = qml.map(
            ansatz, observables, device, interface=interface, diff_method=diff_method, **kwargs
        )
        """QNodeCollection: The QNodes to be evaluated. Each QNode corresponds to the
        the expectation value of each observable term after applying the circuit ansatz.
        """

        self.cost_fn = qml.dot(coeffs, self.qnodes)

    def __call__(self, *args, **kwargs):
        return self.cost_fn(*args, **kwargs)

    def metric_tensor(self, args, kwargs=None, diag_approx=False, only_construct=False):
        """Evaluate the value of the metric tensor.

        Args:
            args (tuple[Any]): positional (differentiable) arguments
            kwargs (dict[str, Any]): auxiliary arguments
            diag_approx (bool): iff True, use the diagonal approximation
            only_construct (bool): Iff True, construct the circuits used for computing
                the metric tensor but do not execute them, and return None.

        Returns:
            array[float]: metric tensor
        """
        # We know that for VQE, all the qnodes share the same ansatz so we select the first
        return self.qnodes.qnodes[0].metric_tensor(
            args=args, kwargs=kwargs, diag_approx=diag_approx, only_construct=only_construct
        )
