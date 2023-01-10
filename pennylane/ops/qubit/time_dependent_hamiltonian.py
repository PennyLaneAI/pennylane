"""
This submodule contains the time-dependent hamiltonian class
"""

from jax import numpy as jnp

import pennylane as qml
from pennylane.operation import Observable, Tensor
from pennylane.ops.qubit.hamiltonian import Hamiltonian


# pylint: disable= too-many-instance-attributes
class TDHamiltonian(Observable):
    r"""Operator representing a time-dependent Hamiltonian.

    The Hamiltonian is represented as a linear combination of other operators, e.g.,
    :math:`H(v, t) = H_\text{drift} + \sum_j f_j(v, t) H_j`, where the :math:`v, t` are trainable parameters,
    and t is time. Including time as one of the trainable parameters is optional.

    Args:
        coeffs (Union[tensor_like, callable]): coefficients of the Hamiltonian expression. #ToDo: change from tensor_like to scalar, or update to support tensor_like coefficients
        observables (Iterable[Observable]): observables in the Hamiltonian expression, of same length as coeffs
        id (str): Custom string to label a specific operator instance.
        do_queue (bool): indicates whether the operator should be
            recorded when created in a tape context

    **Example:**

    A time-dependent Hamiltonian can be created by passing a list of coefficients and callables,
    as well as the list of observables.

    >>> def f1(params, t): return jnp.polyval(params, t)
    >>> coeffs = [0.2, f1]
    >>> obs = [qml.PauliX(0) @ qml.PauliZ(1), qml.PauliY(0) @ qml.Identity(1)]
    >>> H = TDHamiltonian(coeffs, obs)
    >>> print(H)
      (-0.543) [Z0 H2]   # ToDo: does not print nicely, and this output is incorrect
    + (0.2) [X0 Z1]

    The coefficients can be a trainable tensor, for example:  # ToDo: Can parameters be a trainable tensor? Update example regardless.

    >>> coeffs = tf.Variable([0.2, -0.543], dtype=tf.double)
    >>> obs = [qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.Hadamard(2)]
    >>> H = qml.Hamiltonian(coeffs, obs)
    >>> print(H)
      (-0.543) [Z0 H2]
    + (0.2) [X0 Z1]

    In many cases, Hamiltonians can be constructed using Pythonic arithmetic operations.
    For example:

    >>> qml.Hamiltonian([1.], [qml.PauliX(0)]) + 2 * qml.PauliZ(0) @ qml.PauliZ(1)

    is equivalent to the following Hamiltonian:

    >>> qml.Hamiltonian([1, 2], [qml.PauliX(0), qml.PauliZ(0) @ qml.PauliZ(1)])

    Addition of TDHamiltonians with TDHamiltonians, Hamiltonians or other observables is possible with
    tensor-valued coefficients, i.e.,

    >>> H1 = qml.Hamiltonian(torch.tensor([1.]), [qml.PauliX(0)])
    >>> H2 = qml.Hamiltonian(torch.tensor([2., 3.]), [qml.PauliY(0), qml.PauliX(1)])
    >>> obs3 = [qml.PauliX(0), qml.PauliY(0), qml.PauliX(1)]
    >>> H3 = qml.Hamiltonian(torch.tensor([1., 2., 3.]), obs3)
    >>> H3.compare(H1 + H2)  # ToDo: compare function not implemented, modify documentation

    """

    num_wires = qml.operation.AnyWires
    grad_method = "A"  # supports analytic gradients

    def __init__(
        self,
        coeffs,
        observables,
        id=None,
        do_queue=True,
    ):

        if qml.math.shape(coeffs)[0] != len(observables):
            raise ValueError(
                "Could not create valid Hamiltonian; "
                "number of coefficients and operators does not match."
            )

        for obs in observables:
            if not isinstance(obs, Observable):
                raise ValueError(
                    "Could not create circuits. Some or all observables are not valid."
                )

        self._ops = list(observables)
        self._coeffs = coeffs

        self.H_drift_coeffs = []
        self.H_drift_ops = []
        self.H_ts_fs = []
        self.H_ts_ops = []

        # ToDo: so far this only deals with scalar coefficients for H_drift - should it accept tensor_like?

        for coeff, obs in zip(coeffs, observables):
            if callable(coeff):
                self.H_ts_fs.append(coeff)
                self.H_ts_ops.append(obs)
            else:
                self.H_drift_coeffs.append(coeff)
                self.H_drift_ops.append(obs)

        if len(self.H_ts_ops) == 0:
            raise RuntimeError(
                "Found no time-dependent terms while initializing time dependent Hamiltonian "
                f"with coefficients {self.coeffs}"
            )

        self.H_drift = self._H_drift
        self.H_ts = self._H_ts

        self._hyperparameters = {"ops": self._ops}

        self._wires = qml.wires.Wires.all_wires([op.wires for op in self.ops], sort=True)

        self.return_type = None

        super().__init__(*self._coeffs, wires=self._wires, id=id, do_queue=do_queue)

    def __call__(self, params, t, wire_order=None):
        # TODO: jax-optimize this?
        if wire_order is None:
            wire_order = self.wires

        if len(self.H_ts_ops) == 1:
            H = jnp.array(
                [qml.matrix(self.H_drift, wire_order=wire_order)]
                + [qml.matrix(self.H_ts(params, t), wire_order=wire_order)]
            )
            return jnp.sum(H, axis=0)

        H = jnp.array(
            [qml.matrix(self.H_drift, wire_order=wire_order)]
            + [qml.matrix(self.H_ts(params, t), wire_order=wire_order)]
        )
        return jnp.sum(H, axis=0)

    @staticmethod
    def _get_terms(coeffs, obs):
        """Takes a list of scalar coefficients and list of Observables. Returns a qml.Sum of qml.SProd operators
        (or a single qml.SProd operator in the event that there is only one term)."""
        terms_list = [qml.s_prod(coeff, ob) for coeff, ob in zip(coeffs, obs)]
        if len(terms_list) == 0:
            return None
        if len(terms_list) == 1:
            return terms_list[0]
        return qml.op_sum(*terms_list)

    @property
    def _H_drift(self):
        """The terms without any time dependence. Returns a qml.Sum operator of qml.SProd operators
        (or a single qml.SProd operator in the event that there is only one term in H_drift)."""
        return self._get_terms(self.H_drift_coeffs, self.H_drift_ops)

    @property
    def _H_ts(self):
        """The time-dependent terms of the Hamiltonian. Returns a function that can be evaluated
        to get a snapshot of the operator at a set time and with set parameters. When the function is
        evaluated, the returned operator is a qml.Sum of qml.S_Prod operators (or a single qml.SProd
        operator in the event that there is only one term in H_ts)."""

        def snapshot(params, t):
            coeffs = [f(params, t) for f in self.H_ts_fs]
            return self._get_terms(coeffs, self.H_ts_ops)

        return snapshot

    @property
    def coeffs(self):
        """Return the coefficients defining the Hamiltonian, including the unevaluated
        functions for the time-dependent terms.

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
    def name(self):
        return "Time-Dependent Hamiltonian"

    def __add__(self, H):
        r"""The addition operation between a time dependent Hamiltonian and a
        Hamiltonian or time-dependent Hamiltonian."""
        ops = self.ops.copy()
        coeffs = self.coeffs.copy()

        if isinstance(H, (Hamiltonian, TDHamiltonian)):
            coeffs.extend(H.coeffs.copy())
            ops.extend(H.ops.copy())
            return TDHamiltonian(coeffs, ops)

        if isinstance(H, (Tensor, Observable)):
            coeffs.append(qml.math.cast_like([1.0], coeffs)[0])
            ops.append(H)
            return TDHamiltonian(coeffs, ops)

        raise ValueError(f"Cannot add Hamiltonian and {type(H)}")

    __radd__ = __add__
