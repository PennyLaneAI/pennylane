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
import warnings
import itertools
from copy import copy

import pennylane as qml
from pennylane import numpy as np
from pennylane.operation import Observable, Tensor
from pennylane.queuing import QueuingError
from pennylane.wires import Wires

OBS_MAP = {"PauliX": "X", "PauliY": "Y", "PauliZ": "Z", "Hadamard": "H", "Identity": "I"}


def _compute_grouping_indices(observables, grouping_type="qwc", method="rlf"):

    # todo: directly compute the
    # indices, instead of extracting groups of observables first
    observable_groups = qml.grouping.group_observables(
        observables, coefficients=None, grouping_type=grouping_type, method=method
    )

    observables = copy(observables)

    indices = []
    available_indices = list(range(len(observables)))
    for partition in observable_groups:
        indices_this_group = []
        for pauli_word in partition:
            # find index of this pauli word in remaining original observables,
            for observable in observables:
                if qml.grouping.utils.are_identical_pauli_words(pauli_word, observable):
                    ind = observables.index(observable)
                    indices_this_group.append(available_indices[ind])
                    # delete this observable and its index, so it cannot be found again
                    observables.pop(ind)
                    available_indices.pop(ind)
                    break
        indices.append(indices_this_group)

    return indices


class Hamiltonian(qml.operation.Observable):
    r"""Operator representing a Hamiltonian.

    The Hamiltonian is represented as a linear combination of other operators, e.g.,
    :math:`\sum_{k=0}^{N-1} c_k O_k`, where the :math:`c_k` are trainable parameters.

    Args:
        coeffs (tensor_like): coefficients of the Hamiltonian expression
        observables (Iterable[Observable]): observables in the Hamiltonian expression, of same length as coeffs
        simplify (bool): Specifies whether the Hamiltonian is simplified upon initialization
                         (like-terms are combined). The default value is `False`.
        grouping_type (str): If not None, compute and store information on how to group commuting
            observables upon initialization. This information may be accessed when QNodes containing this
            Hamiltonian are executed on devices. The string refers to the type of binary relation between Pauli words.
            Can be ``'qwc'`` (qubit-wise commuting), ``'commuting'``, or ``'anticommuting'``.
        method (str): The graph coloring heuristic to use in solving minimum clique cover for grouping, which
            can be ``'lf'`` (Largest First) or ``'rlf'`` (Recursive Largest First).
        id (str): name to be assigned to this Hamiltonian instance

    **Example:**

    A Hamiltonian can be created by simply passing the list of coefficients
    as well as the list of observables:

    >>> coeffs = [0.2, -0.543]
    >>> obs = [qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.Hadamard(2)]
    >>> H = qml.Hamiltonian(coeffs, obs)
    >>> print(H)
      (-0.543) [Z0 H2]
    + (0.2) [X0 Z1]

    The coefficients can be a trainable tensor, for example:

    >>> coeffs = tf.Variable([0.2, -0.543], dtype=tf.double)
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
    (0.8) [Hermitian0,1]

    Alternatively, the :func:`~.molecular_hamiltonian` function from the
    :doc:`/introduction/chemistry` module can be used to generate a molecular
    Hamiltonian.

    In many cases, Hamiltonians can be constructed using Pythonic arithmetic operations.
    For example:

    >>> qml.Hamiltonian([1.], [qml.PauliX(0)]) + 2 * qml.PauliZ(0) @ qml.PauliZ(1)

    is equivalent to the following Hamiltonian:

    >>> qml.Hamiltonian([1, 2], [qml.PauliX(0), qml.PauliZ(0) @ qml.PauliZ(1)])

    While scalar multiplication requires native python floats or integer types,
    addition, subtraction, and tensor multiplication of Hamiltonians with Hamiltonians or
    other observables is possible with tensor-valued coefficients, i.e.,

    >>> H1 = qml.Hamiltonian(torch.tensor([1.]), [qml.PauliX(0)])
    >>> H2 = qml.Hamiltonian(torch.tensor([2., 3.]), [qml.PauliY(0), qml.PauliX(1)])
    >>> obs3 = [qml.PauliX(0), qml.PauliY(0), qml.PauliX(1)]
    >>> H3 = qml.Hamiltonian(torch.tensor([1., 2., 3.]), obs3)
    >>> H3.compare(H1 + H2)
    True

    A Hamiltonian can store information on which commuting observables should be measured together in
    a circuit:

    >>> obs = [qml.PauliX(0), qml.PauliX(1), qml.PauliZ(0)]
    >>> coeffs = np.array([1., 2., 3.])
    >>> H = qml.Hamiltonian(coeffs, obs, grouping_type='qwc')
    >>> H.grouping_indices
    [[0, 1], [2]]

    This attribute can be used to compute groups of coefficients and observables:

    >>> grouped_coeffs = [coeffs[indices] for indices in H.grouping_indices]
    >>> grouped_obs = [[H.ops[i] for i in indices] for indices in H.grouping_indices]
    >>> grouped_coeffs
    [tensor([1., 2.], requires_grad=True), tensor([3.], requires_grad=True)]
    >>> grouped_obs
    [[qml.PauliX(0), qml.PauliX(1)], [qml.PauliZ(0)]]

    Devices that evaluate a Hamiltonian expectation by splitting it into its local observables can
    use this information to reduce the number of circuits evaluated.

    Note that one can compute the ``grouping_indices`` for an already initialized Hamiltonian by
    using the :func:`compute_grouping <pennylane.Hamiltonian.compute_grouping>` method.
    """

    num_wires = qml.operation.AnyWires
    num_params = 1
    par_domain = "A"
    grad_method = "A"  # supports analytic gradients

    def __init__(
        self,
        coeffs,
        observables,
        simplify=False,
        grouping_type=None,
        method="rlf",
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

        self._coeffs = coeffs
        self._ops = list(observables)
        self._wires = qml.wires.Wires.all_wires([op.wires for op in self.ops], sort=True)

        self.return_type = None

        # attribute to store indices used to form groups of
        # commuting observables, since recomputation is costly
        self._grouping_indices = None

        if simplify:
            self.simplify()
        if grouping_type is not None:
            self._grouping_indices = qml.transforms.invisible(_compute_grouping_indices)(
                self.ops, grouping_type=grouping_type, method=method
            )

        coeffs_flat = [self._coeffs[i] for i in range(qml.math.shape(self._coeffs)[0])]
        # overwrite this attribute, now that we have the correct info
        self.num_params = qml.math.shape(self._coeffs)[0]

        # create the operator using each coefficient as a separate parameter;
        # this causes H.data to be a list of tensor scalars,
        # while H.coeffs is the original tensor
        super().__init__(*coeffs_flat, wires=self._wires, id=id, do_queue=do_queue)

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
        return self._wires

    @property
    def name(self):
        return "Hamiltonian"

    @property
    def grouping_indices(self):
        """Return the grouping indices attribute.

        Returns:
            list[list[int]]: indices needed to form groups of commuting observables
        """
        return self._grouping_indices

    def compute_grouping(self, grouping_type="qwc", method="rlf"):
        """
        Compute groups of indices corresponding to commuting observables of this
        Hamiltonian, and store it in the ``grouping_indices`` attribute.

        Args:
            grouping_type (str): The type of binary relation between Pauli words used to compute the grouping.
                Can be ``'qwc'``, ``'commuting'``, or ``'anticommuting'``.
            method (str): The graph coloring heuristic to use in solving minimum clique cover for grouping, which
                can be ``'lf'`` (Largest First) or ``'rlf'`` (Recursive Largest First).
        """

        self._grouping_indices = qml.transforms.invisible(_compute_grouping_indices)(
            self.ops, grouping_type=grouping_type, method=method
        )

    def simplify(self):
        r"""Simplifies the Hamiltonian by combining like-terms.

        **Example**

        >>> ops = [qml.PauliY(2), qml.PauliX(0) @ qml.Identity(1), qml.PauliX(0)]
        >>> H = qml.Hamiltonian([1, 1, -2], ops)
        >>> H.simplify()
        >>> print(H)
          (-1) [X0]
        + (1) [Y2]

        .. warning::

            Calling this method will reset ``grouping_indices`` to None, since
            the observables it refers to are updated.
        """
        data = []
        ops = []

        for i in range(len(self.ops)):  # pylint: disable=consider-using-enumerate
            op = self.ops[i]
            c = self.coeffs[i]
            op = op if isinstance(op, Tensor) else Tensor(op)

            ind = None
            for j, o in enumerate(ops):
                if op.compare(o):
                    ind = j
                    break

            if ind is not None:
                data[ind] += c
                if np.isclose(qml.math.toarray(data[ind]), np.array(0.0)):
                    del data[ind]
                    del ops[ind]
            else:
                ops.append(op.prune())
                data.append(c)

        self._coeffs = qml.math.stack(data) if data else []
        self.data = data
        self._ops = ops
        # reset grouping, since the indices refer to the old observables and coefficients
        self._grouping_indices = None

    def __str__(self):
        # Lambda function that formats the wires
        wires_print = lambda ob: ",".join(map(str, ob.wires.tolist()))

        list_of_coeffs = self.data  # list of scalar tensors
        paired_coeff_obs = list(zip(list_of_coeffs, self.ops))
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

    def __repr__(self):
        # Constructor-call-like representation
        return f"<Hamiltonian: terms={qml.math.shape(self.coeffs)[0]}, wires={self.wires.tolist()}>"

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

        coeffs_arr = qml.math.toarray(self.coeffs)
        for co, op in zip(coeffs_arr, self.ops):
            obs = op.non_identity_obs if isinstance(op, Tensor) else [op]
            tensor = []
            for ob in obs:
                parameters = tuple(
                    str(param) for param in ob.parameters
                )  # Converts params into immutable type
                tensor.append((ob.name, ob.wires, parameters))
            data.add((co, frozenset(tensor)))

        return data

    def compare(self, other):
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
        if isinstance(other, Hamiltonian):
            self.simplify()
            other.simplify()
            return self._obs_data() == other._obs_data()  # pylint: disable=protected-access

        if isinstance(other, (Tensor, Observable)):
            self.simplify()
            return self._obs_data() == {
                (1, frozenset(other._obs_data()))  # pylint: disable=protected-access
            }

        raise ValueError("Can only compare a Hamiltonian, and a Hamiltonian/Observable/Tensor.")

    def __matmul__(self, H):
        r"""The tensor product operation between a Hamiltonian and a Hamiltonian/Tensor/Observable."""
        coeffs1 = copy(self.coeffs)
        ops1 = self.ops.copy()

        if isinstance(H, Hamiltonian):
            shared_wires = Wires.shared_wires([self.wires, H.wires])
            if len(shared_wires) > 0:
                raise ValueError(
                    "Hamiltonians can only be multiplied together if they act on "
                    "different sets of wires"
                )

            coeffs2 = H.coeffs
            ops2 = H.ops

            coeffs = qml.math.kron(coeffs1, coeffs2)
            ops_list = itertools.product(ops1, ops2)
            terms = [qml.operation.Tensor(t[0], t[1]) for t in ops_list]

            return qml.Hamiltonian(coeffs, terms, simplify=True)

        if isinstance(H, (Tensor, Observable)):
            terms = [op @ H for op in ops1]

            return qml.Hamiltonian(coeffs1, terms, simplify=True)

        raise ValueError(f"Cannot tensor product Hamiltonian and {type(H)}")

    def __add__(self, H):
        r"""The addition operation between a Hamiltonian and a Hamiltonian/Tensor/Observable."""
        ops = self.ops.copy()
        self_coeffs = copy(self.coeffs)

        if isinstance(H, Hamiltonian):
            coeffs = qml.math.concatenate([self_coeffs, copy(H.coeffs)], axis=0)
            ops.extend(H.ops.copy())
            return qml.Hamiltonian(coeffs, ops, simplify=True)

        if isinstance(H, (Tensor, Observable)):
            coeffs = qml.math.concatenate(
                [self_coeffs, qml.math.cast_like([1.0], self_coeffs)], axis=0
            )
            ops.append(H)
            return qml.Hamiltonian(coeffs, ops, simplify=True)

        raise ValueError(f"Cannot add Hamiltonian and {type(H)}")

    def __mul__(self, a):
        r"""The scalar multiplication operation between a scalar and a Hamiltonian."""
        if isinstance(a, (int, float)):
            self_coeffs = copy(self.coeffs)
            coeffs = qml.math.multiply(qml.math.cast_like([a], self_coeffs), self_coeffs)
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
            self._coeffs = qml.math.concatenate([self._coeffs, H.coeffs], axis=0)
            self._ops.extend(H.ops.copy())
            self.simplify()
            return self

        if isinstance(H, (Tensor, Observable)):
            self._coeffs = qml.math.concatenate(
                [self._coeffs, qml.math.cast_like([1.0], self._coeffs)], axis=0
            )
            self._ops.append(H)
            self.simplify()
            return self

        raise ValueError(f"Cannot add Hamiltonian and {type(H)}")

    def __imul__(self, a):
        r"""The inplace scalar multiplication operation between a scalar and a Hamiltonian."""
        if isinstance(a, (int, float)):
            self._coeffs = qml.math.multiply(qml.math.cast_like([a], self._coeffs), self._coeffs)
            return self

        raise ValueError(f"Cannot multiply Hamiltonian by {type(a)}")

    def __isub__(self, H):
        r"""The inplace subtraction operation between a Hamiltonian and a Hamiltonian/Tensor/Observable."""
        if isinstance(H, (Hamiltonian, Tensor, Observable)):
            self.__iadd__(H.__mul__(-1))
            return self
        raise ValueError(f"Cannot subtract {type(H)} from Hamiltonian")

    def queue(self, context=qml.QueuingContext):
        """Queues a qml.Hamiltonian instance"""
        for o in self.ops:
            try:
                context.update_info(o, owner=self)
            except QueuingError:
                o.queue(context=context)
                context.update_info(o, owner=self)
            except NotImplementedError:
                pass

        context.append(self, owns=tuple(self.ops))
        return self


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

        if np.isclose(qml.math.toarray(qml.math.count_nonzero(coeffs)), 0):
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
            UserWarning,
            2,
        )
        super().__init__(*args, **kwargs)
