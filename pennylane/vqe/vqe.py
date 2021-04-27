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
This submodule contains functionality for running Variational Quantum Eigensolver (VQE)
computations using PennyLane.
"""
# pylint: disable=too-many-arguments, too-few-public-methods
from collections.abc import Sequence
import itertools
import warnings

import pennylane as qml
from pennylane import numpy as np
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
        simplify (bool): Specifies whether the Hamiltonian is simplified upon initialization
                         (like-terms are combined). The default value is `False`.

    .. seealso:: :class:`~.ExpvalCost`, :func:`~.molecular_hamiltonian`

    **Example:**

    A Hamiltonian can be created by simply passing the list of coefficients
    as well as the list of observables:

    >>> coeffs = [0.2, -0.543]
    >>> obs = [qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.Hadamard(2)]
    >>> H = qml.Hamiltonian(coeffs, obs)
    >>> print(H)
      (-0.543) [Z0 H2]
    + (0.2) [X0 Z1]

    The user can also provide custom observables:

    >>> obs_matrix = np.array([[0.5, 1.0j, 0.0, -3j],
                               [-1.0j, -1.1, 0.0, -0.1],
                               [0.0, 0.0, -0.9, 12.0],
                               [3j, -0.1, 12.0, 0.0]])
    >>> obs = qml.Hermitian(obs_matrix, wires=[0, 1])
    >>> H = qml.Hamiltonian((0.8, ), (obs, ))
    >>> print(H)
    (0.8) [Hermitian0'1]

    Alternatively, the :func:`~.molecular_hamiltonian` function from the
    :doc:`/introduction/chemistry` module can be used to generate a molecular
    Hamiltonian.
    """

    def __init__(self, coeffs, observables, simplify=False):

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

        self._coeffs = list(coeffs)
        self._ops = list(observables)

        if simplify:
            self.simplify()

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
        r"""The terms of the Hamiltonian expression :math:`\sum_{k=0}^{N-1} c_k O_k`

        Returns:
            (tuple, tuple): tuples of coefficients and operations, each of length N
        """
        return self.coeffs, self.ops

    @property
    def wires(self):
        r"""The sorted union of wires from all operators.

        Returns:
            (Wires): Combined wires present in all terms, sorted.
        """
        return qml.wires.Wires.all_wires([op.wires for op in self.ops], sort=True)

    def simplify(self):
        r"""Simplifies the Hamiltonian by combining like-terms.

        **Example**

        >>> ops = [qml.PauliY(2), qml.PauliX(0) @ qml.Identity(1), qml.PauliX(0)]
        >>> H = qml.Hamiltonian([1, 1, -2], ops)
        >>> H.simplify()
        >>> print(H)
          (-1) [X0]
        + (1) [Y2]
        """

        coeffs = []
        ops = []

        for c, op in zip(self.coeffs, self.ops):
            op = op if isinstance(op, Tensor) else Tensor(op)

            ind = None
            for i, other in enumerate(ops):
                if op.compare(other):
                    ind = i
                    break

            if ind is not None:
                coeffs[ind] += c
                if np.allclose([coeffs[ind]], [0]):
                    del coeffs[ind]
                    del ops[ind]
            else:
                ops.append(op.prune())
                coeffs.append(c)

        self._coeffs = coeffs
        self._ops = ops

    def __str__(self):
        # Lambda function that formats the wires
        wires_print = lambda ob: "'".join(map(str, ob.wires.tolist()))

        paired_coeff_obs = list(zip(self.coeffs, self.ops))
        paired_coeff_obs.sort(key=lambda pair: (len(pair[1].wires), pair[0]))

        terms_ls = []

        for coeff, obs in paired_coeff_obs:

            if isinstance(obs, Tensor):
                obs_strs = [f"{OBS_MAP.get(ob.name, ob.name)}{wires_print(ob)}" for ob in obs.obs]
                ob_str = " ".join(obs_strs)
            elif isinstance(obs, Observable):
                ob_str = f"{OBS_MAP.get(obs.name, obs.name)}{wires_print(obs)}"

            term_str = f"({coeff}) [{ob_str}]"

            terms_ls.append(term_str)

        return "  " + "\n+ ".join(terms_ls)

    def _obs_data(self):
        r"""Extracts the data from a Hamiltonian and serializes it in an order-independent fashion.

        This allows for comparison between Hamiltonians that are equivalent, but are defined with terms and tensors
        expressed in different orders. For example, `qml.PauliX(0) @ qml.PauliZ(1)` and
        `qml.PauliZ(1) @ qml.PauliX(0)` are equivalent observables with different orderings.

        .. Note::

            In order to store the data from each term of the Hamiltonian in an order-independent serialization,
            we make use of sets. Note that all data contained within each term must be immutable, hence the use of
            strings and frozensets.

        **Example**

        >>> H = qml.Hamiltonian([1, 1], [qml.PauliX(0) @ qml.PauliX(1), qml.PauliZ(0)])
        >>> print(H._obs_data())
        {(1, frozenset({('PauliZ', <Wires = [1]>, ())})),
        (1, frozenset({('PauliX', <Wires = [1]>, ()), ('PauliX', <Wires = [0]>, ())}))}
        """
        data = set()

        for co, op in zip(*self.terms):
            obs = op.non_identity_obs if isinstance(op, Tensor) else [op]
            tensor = []
            for ob in obs:
                parameters = tuple(
                    param.tostring() for param in ob.parameters
                )  # Converts params into immutable type
                tensor.append((ob.name, ob.wires, parameters))
            data.add((co, frozenset(tensor)))

        return data

    def compare(self, H):
        r"""Compares with another :class:`~Hamiltonian`, :class:`~.Observable`, or :class:`~.Tensor`,
        to determine if they are equivalent.

        Hamiltonians/observables are equivalent if they represent the same operator
        (their matrix representations are equal), and they are defined on the same wires.

        .. Warning::

            The compare method does **not** check if the matrix representation
            of a :class:`~.Hermitian` observable is equal to an equivalent
            observable expressed in terms of Pauli matrices, or as a
            linear combination of Hermitians.
            To do so would require the matrix form of Hamiltonians and Tensors
            be calculated, which would drastically increase runtime.

        Returns:
            (bool): True if equivalent.

        **Examples**

        >>> A = np.array([[1, 0], [0, -1]])
        >>> H = qml.Hamiltonian(
        ...     [0.5, 0.5],
        ...     [qml.Hermitian(A, 0) @ qml.PauliY(1), qml.PauliY(1) @ qml.Hermitian(A, 0) @ qml.Identity("a")]
        ... )
        >>> obs = qml.Hermitian(A, 0) @ qml.PauliY(1)
        >>> print(H.compare(obs))
        True

        >>> H1 = qml.Hamiltonian([1, 1], [qml.PauliX(0), qml.PauliZ(1)])
        >>> H2 = qml.Hamiltonian([1, 1], [qml.PauliZ(0), qml.PauliX(1)])
        >>> H1.compare(H2)
        False

        >>> ob1 = qml.Hamiltonian([1], [qml.PauliX(0)])
        >>> ob2 = qml.Hermitian(np.array([[0, 1], [1, 0]]), 0)
        >>> ob1.compare(ob2)
        False
        """
        if isinstance(H, Hamiltonian):
            self.simplify()
            H.simplify()
            return self._obs_data() == H._obs_data()  # pylint: disable=protected-access

        if isinstance(H, (Tensor, Observable)):
            self.simplify()
            return self._obs_data() == {
                (1, frozenset(H._obs_data()))  # pylint: disable=protected-access
            }

        raise ValueError("Can only compare a Hamiltonian, and a Hamiltonian/Observable/Tensor.")

    def __matmul__(self, H):
        r"""The tensor product operation between a Hamiltonian and a Hamiltonian/Tensor/Observable."""
        coeffs1 = self.coeffs.copy()
        terms1 = self.ops.copy()

        if isinstance(H, Hamiltonian):
            coeffs2 = H.coeffs
            terms2 = H.ops

            coeffs = [c[0] * c[1] for c in itertools.product(coeffs1, coeffs2)]
            term_list = itertools.product(terms1, terms2)
            terms = [qml.operation.Tensor(t[0], t[1]) for t in term_list]

            return qml.Hamiltonian(coeffs, terms, simplify=True)

        if isinstance(H, (Tensor, Observable)):
            coeffs = coeffs1
            terms = [term @ H for term in terms1]

            return qml.Hamiltonian(coeffs, terms, simplify=True)

        raise ValueError(f"Cannot tensor product Hamiltonian and {type(H)}")

    def __add__(self, H):
        r"""The addition operation between a Hamiltonian and a Hamiltonian/Tensor/Observable."""
        coeffs = self.coeffs.copy()
        ops = self.ops.copy()

        if isinstance(H, Hamiltonian):
            coeffs.extend(H.coeffs.copy())
            ops.extend(H.ops.copy())
            return qml.Hamiltonian(coeffs, ops, simplify=True)

        if isinstance(H, (Tensor, Observable)):
            coeffs.append(1)
            ops.append(H)
            return qml.Hamiltonian(coeffs, ops, simplify=True)

        raise ValueError(f"Cannot add Hamiltonian and {type(H)}")

    def __mul__(self, a):
        r"""The scalar multiplication operation between a scalar and a Hamiltonian."""
        if isinstance(a, (int, float)):
            coeffs = [a * c for c in self.coeffs.copy()]
            return qml.Hamiltonian(coeffs, self.ops.copy())

        raise ValueError(f"Cannot multiply Hamiltonian by {type(a)}")

    __rmul__ = __mul__

    def __sub__(self, H):
        r"""The subtraction operation between a Hamiltonian and a Hamiltonian/Tensor/Observable."""
        if isinstance(H, (Hamiltonian, Tensor, Observable)):
            return self.__add__(H.__mul__(-1))
        raise ValueError(f"Cannot subtract {type(H)} from Hamiltonian")

    def __iadd__(self, H):
        r"""The inplace addition operation between a Hamiltonian and a Hamiltonian/Tensor/Observable."""
        if isinstance(H, Hamiltonian):
            self._coeffs.extend(H.coeffs.copy())
            self._ops.extend(H.ops.copy())
            self.simplify()
            return self

        if isinstance(H, (Tensor, Observable)):
            self._coeffs.append(1)
            self._ops.append(H)
            self.simplify()
            return self

        raise ValueError(f"Cannot add Hamiltonian and {type(H)}")

    def __imul__(self, a):
        r"""The inplace scalar multiplication operation between a scalar and a Hamiltonian."""
        if isinstance(a, (int, float)):
            self._coeffs = [a * c for c in self.coeffs]
            return self

        raise ValueError(f"Cannot multiply Hamiltonian by {type(a)}")

    def __isub__(self, H):
        r"""The inplace subtraction operation between a Hamiltonian and a Hamiltonian/Tensor/Observable."""
        if isinstance(H, (Hamiltonian, Tensor, Observable)):
            self.__iadd__(H.__mul__(-1))
            return self
        raise ValueError(f"Cannot subtract {type(H)} from Hamiltonian")


class ExpvalCost:
    """Create a cost function that gives the expectation value of an input Hamiltonian.

    This cost function is useful for a range of problems including VQE and QAOA.

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
        optimize (bool): Whether to optimize the observables composing the Hamiltonian by
            separating them into qubit-wise commuting groups. Each group can then be executed
            within a single QNode, resulting in fewer QNodes to evaluate.

    Returns:
        callable: a cost function with signature ``cost_fn(params, **kwargs)`` that evaluates
        the expectation of the Hamiltonian on the provided device(s)

    .. seealso:: :class:`~.Hamiltonian`, :func:`~.molecular_hamiltonian`, :func:`~.map`, :func:`~.dot`

    **Example:**

    To construct an ``ExpvalCost`` cost function, we require a Hamiltonian to measure, and an ansatz
    for our variational circuit.

    We can construct a Hamiltonian manually,

    .. code-block:: python

        coeffs = [0.2, -0.543]
        obs = [
            qml.PauliX(0) @ qml.PauliZ(1) @ qml.PauliY(3),
            qml.PauliZ(0) @ qml.Hadamard(2)
        ]
        H = qml.vqe.Hamiltonian(coeffs, obs)

    Alternatively, the :func:`~.molecular_hamiltonian` function from the
    :doc:`/introduction/chemistry` module can be used to generate a molecular Hamiltonian.

    Once we have our Hamiltonian, we can select an ansatz and construct
    the cost function.

    >>> ansatz = qml.templates.StronglyEntanglingLayers
    >>> dev = qml.device("default.qubit", wires=4)
    >>> cost = qml.ExpvalCost(ansatz, H, dev, interface="torch")
    >>> params = torch.rand([2, 4, 3])
    >>> cost(params)
    tensor(-0.2316, dtype=torch.float64)

    The cost function can then be minimized using any gradient descent-based
    :doc:`optimizer </introduction/optimizers>`.

    .. UsageDetails::

        **Optimizing observables:**

        Setting ``optimize=True`` can be used to decrease the number of device executions. The
        observables composing the Hamiltonian can be separated into groups that are qubit-wise
        commuting using the :mod:`~.grouping` module. These groups can be executed together on a
        *single* qnode, resulting in a lower device overhead:

        .. code-block:: python

            commuting_obs = [qml.PauliX(0), qml.PauliX(0) @ qml.PauliZ(1)]
            H = qml.vqe.Hamiltonian([1, 1], commuting_obs)

            dev = qml.device("default.qubit", wires=2)
            ansatz = qml.templates.StronglyEntanglingLayers

            cost_opt = qml.ExpvalCost(ansatz, H, dev, optimize=True)
            cost_no_opt = qml.ExpvalCost(ansatz, H, dev, optimize=False)

            params = qml.init.strong_ent_layers_uniform(3, 2)

        Grouping these commuting observables leads to fewer device executions:

        >>> cost_opt(params)
        >>> ex_opt = dev.num_executions
        >>> cost_no_opt(params)
        >>> ex_no_opt = dev.num_executions - ex_opt
        >>> print("Number of executions:", ex_no_opt)
        Number of executions: 2
        >>> print("Number of executions (optimized):", ex_opt)
        Number of executions (optimized): 1
    """

    def __init__(
        self,
        ansatz,
        hamiltonian,
        device,
        interface="autograd",
        diff_method="best",
        optimize=False,
        **kwargs,
    ):
        if kwargs.get("measure", "expval") != "expval":
            raise ValueError("ExpvalCost can only be used to construct sums of expectation values.")

        coeffs, observables = hamiltonian.terms

        self.hamiltonian = hamiltonian
        """Hamiltonian: the input Hamiltonian."""

        self.qnodes = None
        """QNodeCollection: The QNodes to be evaluated. Each QNode corresponds to the expectation
        value of each observable term after applying the circuit ansatz."""

        self._multiple_devices = isinstance(device, Sequence)
        """Bool: Records if multiple devices are input"""

        if all(c == 0 for c in coeffs) or not coeffs:
            self.cost_fn = lambda *args, **kwargs: np.array(0)
            return

        self._optimize = optimize

        self.qnodes = qml.map(
            ansatz, observables, device, interface=interface, diff_method=diff_method, **kwargs
        )

        if self._optimize:

            if self._multiple_devices:
                raise ValueError("Using multiple devices is not supported when optimize=True")

            obs_groupings, coeffs_groupings = qml.grouping.group_observables(observables, coeffs)
            d = device[0] if self._multiple_devices else device
            w = d.wires.tolist()

            @qml.qnode(device, interface=interface, diff_method=diff_method, **kwargs)
            def circuit(*qnode_args, obs, **qnode_kwargs):
                """Converting ansatz into a full circuit including measurements"""
                ansatz(*qnode_args, wires=w, **qnode_kwargs)
                return [qml.expval(o) for o in obs]

            def cost_fn(*qnode_args, **qnode_kwargs):
                """Combine results from grouped QNode executions with grouped coefficients"""
                total = 0
                for o, c in zip(obs_groupings, coeffs_groupings):
                    res = circuit(*qnode_args, obs=o, **qnode_kwargs)
                    total += sum([r * c_ for r, c_ in zip(res, c)])
                return total

            self.cost_fn = cost_fn

        else:
            self.cost_fn = qml.dot(coeffs, self.qnodes)

    def __call__(self, *args, **kwargs):
        return self.cost_fn(*args, **kwargs)


class VQECost(ExpvalCost):
    """Create a cost function that gives the expectation value of an input Hamiltonian.

    .. warning::
        Use of :class:`~.VQECost` is deprecated and should be replaced with
        :class:`~.ExpvalCost`.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "Use of VQECost is deprecated and should be replaced with ExpvalCost",
            DeprecationWarning,
            2,
        )
        super().__init__(*args, **kwargs)
