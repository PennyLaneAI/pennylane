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
This submodule contains the discrete-variable quantum operations that perform
arithmetic operations on their input states.
"""
# pylint: disable=too-many-arguments,too-many-instance-attributes
import itertools
import numbers
from collections.abc import Iterable
from copy import copy
import functools
from typing import List
from warnings import warn
import numpy as np
import scipy


import pennylane as qml
from pennylane.operation import Observable, Tensor
from pennylane.wires import Wires

OBS_MAP = {"PauliX": "X", "PauliY": "Y", "PauliZ": "Z", "Hadamard": "H", "Identity": "I"}


def _compute_grouping_indices(observables, grouping_type="qwc", method="rlf"):
    # todo: directly compute the
    # indices, instead of extracting groups of observables first
    observable_groups = qml.pauli.group_observables(
        observables, coefficients=None, grouping_type=grouping_type, method=method
    )

    observables = copy(observables)

    indices = []
    available_indices = list(range(len(observables)))
    for partition in observable_groups:  # pylint:disable=too-many-nested-blocks
        indices_this_group = []
        for pauli_word in partition:
            # find index of this pauli word in remaining original observables,
            for ind, observable in enumerate(observables):
                if qml.pauli.are_identical_pauli_words(pauli_word, observable):
                    indices_this_group.append(available_indices[ind])
                    # delete this observable and its index, so it cannot be found again
                    observables.pop(ind)
                    available_indices.pop(ind)
                    break
        indices.append(tuple(indices_this_group))

    return tuple(indices)


class Hamiltonian(Observable):
    r"""Operator representing a Hamiltonian.

    The Hamiltonian is represented as a linear combination of other operators, e.g.,
    :math:`\sum_{k=0}^{N-1} c_k O_k`, where the :math:`c_k` are trainable parameters.

    .. warning::

        As of ``v0.36``, ``qml.Hamiltonian`` dispatches to :class:`~.pennylane.ops.op_math.LinearCombination`
        by default. For further details, see :doc:`Updated Operators </news/new_opmath/>`.

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
            can be ``'lf'`` (Largest First) or ``'rlf'`` (Recursive Largest First). Ignored if ``grouping_type=None``.
        id (str): name to be assigned to this Hamiltonian instance

    **Example:**

    .. note::
        As of ``v0.36``, ``qml.Hamiltonian`` dispatches to :class:`~.pennylane.ops.op_math.LinearCombination`
        by default, so the following examples assume this behaviour.

    ``qml.Hamiltonian`` takes in a list of coefficients and a list of operators.

    >>> coeffs = [0.2, -0.543]
    >>> obs = [qml.X(0) @ qml.Z(1), qml.Z(0) @ qml.Hadamard(2)]
    >>> H = qml.Hamiltonian(coeffs, obs)
    >>> print(H)
    0.2 * (X(0) @ Z(1)) + -0.543 * (Z(0) @ Hadamard(wires=[2]))

    The coefficients can be a trainable tensor, for example:

    >>> coeffs = tf.Variable([0.2, -0.543], dtype=tf.double)
    >>> obs = [qml.X(0) @ qml.Z(1), qml.Z(0) @ qml.Hadamard(2)]
    >>> H = qml.Hamiltonian(coeffs, obs)
    >>> print(H)
    0.2 * (X(0) @ Z(1)) + -0.543 * (Z(0) @ Hadamard(wires=[2]))

    A ``qml.Hamiltonian`` stores information on which commuting observables should be measured
    together in a circuit:

    >>> obs = [qml.X(0), qml.X(1), qml.Z(0)]
    >>> coeffs = np.array([1., 2., 3.])
    >>> H = qml.Hamiltonian(coeffs, obs, grouping_type='qwc')
    >>> H.grouping_indices
    ((0, 1), (2,))

    This attribute can be used to compute groups of coefficients and observables:

    >>> grouped_coeffs = [coeffs[list(indices)] for indices in H.grouping_indices]
    >>> grouped_obs = [[H.ops[i] for i in indices] for indices in H.grouping_indices]
    >>> grouped_coeffs
    [array([1., 2.]), array([3.])]
    >>> grouped_obs
    [[X(0), X(1)], [Z(0)]]

    Devices that evaluate a ``qml.Hamiltonian`` expectation by splitting it into its local
    observables can use this information to reduce the number of circuits evaluated.

    Note that one can compute the ``grouping_indices`` for an already initialized ``qml.Hamiltonian``
    by using the :func:`compute_grouping <pennylane.ops.LinearCombination.compute_grouping>` method.

    .. details::
        :title: Old Hamiltonian behaviour

        The following code examples show the behaviour of ``qml.Hamiltonian`` using old operator
        arithmetic. See :doc:`Updated Operators </news/new_opmath/>` for more details. The old
        behaviour can be reactivated by calling

        >>> qml.operation.disable_new_opmath()

        Alternatively, ``qml.ops.Hamiltonian`` provides a permanent access point for Hamiltonian
        behaviour before ``v0.36``.

        >>> coeffs = [0.2, -0.543]
        >>> obs = [qml.X(0) @ qml.Z(1), qml.Z(0) @ qml.Hadamard(2)]
        >>> H = qml.Hamiltonian(coeffs, obs)
        >>> print(H)
          (-0.543) [Z0 H2]
        + (0.2) [X0 Z1]

        The coefficients can be a trainable tensor, for example:

        >>> coeffs = tf.Variable([0.2, -0.543], dtype=tf.double)
        >>> obs = [qml.X(0) @ qml.Z(1), qml.Z(0) @ qml.Hadamard(2)]
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

        >>> qml.Hamiltonian([1.], [qml.X(0)]) + 2 * qml.Z(0) @ qml.Z(1)

        is equivalent to the following Hamiltonian:

        >>> qml.Hamiltonian([1, 2], [qml.X(0), qml.Z(0) @ qml.Z(1)])

        While scalar multiplication requires native python floats or integer types,
        addition, subtraction, and tensor multiplication of Hamiltonians with Hamiltonians or
        other observables is possible with tensor-valued coefficients, i.e.,

        >>> H1 = qml.Hamiltonian(torch.tensor([1.]), [qml.X(0)])
        >>> H2 = qml.Hamiltonian(torch.tensor([2., 3.]), [qml.Y(0), qml.X(1)])
        >>> obs3 = [qml.X(0), qml.Y(0), qml.X(1)]
        >>> H3 = qml.Hamiltonian(torch.tensor([1., 2., 3.]), obs3)
        >>> H3.compare(H1 + H2)
        True

        A Hamiltonian can store information on which commuting observables should be measured together in
        a circuit:

        >>> obs = [qml.X(0), qml.X(1), qml.Z(0)]
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
        [[qml.X(0), qml.X(1)], [qml.Z(0)]]

        Devices that evaluate a Hamiltonian expectation by splitting it into its local observables can
        use this information to reduce the number of circuits evaluated.

        Note that one can compute the ``grouping_indices`` for an already initialized Hamiltonian by
        using the :func:`compute_grouping <pennylane.Hamiltonian.compute_grouping>` method.

    """

    num_wires = qml.operation.AnyWires
    grad_method = "A"  # supports analytic gradients
    batch_size = None
    ndim_params = None  # could be (0,) * len(coeffs), but it is not needed. Define at class-level

    def _flatten(self):
        # note that we are unable to restore grouping type or method without creating new properties
        return (self.data, self._ops), (self.grouping_indices,)

    @classmethod
    def _unflatten(cls, data, metadata):
        new_op = cls(data[0], data[1])
        new_op._grouping_indices = metadata[0]  # pylint: disable=protected-access
        return new_op

    def __init__(
        self,
        coeffs,
        observables: List[Observable],
        simplify=False,
        grouping_type=None,
        method="rlf",
        id=None,
    ):
        if qml.operation.active_new_opmath():
            warn(
                "Using 'qml.ops.Hamiltonian' with new operator arithmetic is deprecated. "
                "Instead, use 'qml.Hamiltonian'. "
                "Please visit https://docs.pennylane.ai/en/stable/news/new_opmath.html for more information and help troubleshooting.",
                qml.PennyLaneDeprecationWarning,
            )

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

        # TODO: avoid having multiple ways to store ops and coeffs,
        # ideally only use parameters for coeffs, and hyperparameters for ops
        self._hyperparameters = {"ops": self._ops}

        self._wires = qml.wires.Wires.all_wires([op.wires for op in self.ops], sort=True)

        # attribute to store indices used to form groups of
        # commuting observables, since recomputation is costly
        self._grouping_indices = None

        if simplify:
            # simplify upon initialization changes ops such that they wouldnt be
            # removed in self.queue() anymore, removing them here manually.
            if qml.QueuingManager.recording():
                for o in observables:
                    qml.QueuingManager.remove(o)

            with qml.QueuingManager.stop_recording():
                self.simplify()

        if grouping_type is not None:
            with qml.QueuingManager.stop_recording():
                self._grouping_indices = _compute_grouping_indices(
                    self.ops, grouping_type=grouping_type, method=method
                )

        coeffs_flat = [self._coeffs[i] for i in range(qml.math.shape(self._coeffs)[0])]

        # create the operator using each coefficient as a separate parameter;
        # this causes H.data to be a list of tensor scalars,
        # while H.coeffs is the original tensor

        super().__init__(*coeffs_flat, wires=self._wires, id=id)
        self._pauli_rep = "unset"

    @property
    def pauli_rep(self):
        if self._pauli_rep != "unset":
            return self._pauli_rep

        if any(op.pauli_rep is None for op in self.ops):
            self._pauli_rep = None
            return self._pauli_rep

        ps = qml.pauli.PauliSentence()
        for coeff, term in zip(*self.terms()):
            ps += term.pauli_rep * coeff

        self._pauli_rep = ps
        return self._pauli_rep

    def _check_batching(self):
        """Override for Hamiltonian, batching is not yet supported."""

    def label(self, decimals=None, base_label=None, cache=None):
        decimals = None if (len(self.parameters) > 3) else decimals
        return super().label(decimals=decimals, base_label=base_label or "𝓗", cache=cache)

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

    def terms(self):
        r"""Representation of the operator as a linear combination of other operators.

         .. math:: O = \sum_i c_i O_i

         .. seealso:: :meth:`~.Hamiltonian.terms`

        Returns:
            tuple[Iterable[tensor_like or float], list[.Operator]]: coefficients and operations

        **Example**
        >>> coeffs = [1., 2.]
        >>> ops = [qml.X(0), qml.Z(0)]
        >>> H = qml.Hamiltonian(coeffs, ops)

        >>> H.terms()
        [1., 2.], [qml.X(0), qml.Z(0)]

        The coefficients are differentiable and can be stored as tensors:
        >>> import tensorflow as tf
        >>> H = qml.Hamiltonian([tf.Variable(1.), tf.Variable(2.)], [qml.X(0), qml.Z(0)])
        >>> t = H.terms()

        >>> t[0]
        [<tf.Tensor: shape=(), dtype=float32, numpy=1.0>, <tf.Tensor: shape=(), dtype=float32, numpy=2.0>]
        """
        return self.parameters, self.ops

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

    @grouping_indices.setter
    def grouping_indices(self, value):
        """Set the grouping indices, if known without explicit computation, or if
        computation was done externally. The groups are not verified.

        **Example**

        Examples of valid groupings for the Hamiltonian

        >>> H = qml.Hamiltonian([qml.X('a'), qml.X('b'), qml.Y('b')])

        are

        >>> H.grouping_indices = [[0, 1], [2]]

        or

        >>> H.grouping_indices = [[0, 2], [1]]

        since both ``qml.X('a'), qml.X('b')`` and ``qml.X('a'), qml.Y('b')`` commute.


        Args:
            value (list[list[int]]): List of lists of indexes of the observables in ``self.ops``. Each sublist
                represents a group of commuting observables.
        """

        if (
            not isinstance(value, Iterable)
            or any(not isinstance(sublist, Iterable) for sublist in value)
            or any(i not in range(len(self.ops)) for i in [i for sl in value for i in sl])
        ):
            raise ValueError(
                f"The grouped index value needs to be a tuple of tuples of integers between 0 and the "
                f"number of observables in the Hamiltonian; got {value}"
            )
        # make sure all tuples so can be hashable
        self._grouping_indices = tuple(tuple(sublist) for sublist in value)

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

        with qml.QueuingManager.stop_recording():
            self._grouping_indices = _compute_grouping_indices(
                self.ops, grouping_type=grouping_type, method=method
            )

    def sparse_matrix(self, wire_order=None):
        r"""Computes the sparse matrix representation of a Hamiltonian in the computational basis.

        Args:
            wire_order (Iterable): global wire order, must contain all wire labels from the operator's wires.
                If not provided, the default order of the wires (self.wires) of the Hamiltonian is used.

        Returns:
            csr_matrix: a sparse matrix in scipy Compressed Sparse Row (CSR) format with dimension
            :math:`(2^n, 2^n)`, where :math:`n` is the number of wires

        **Example:**

        >>> coeffs = [1, -0.45]
        >>> obs = [qml.Z(0) @ qml.Z(1), qml.Y(0) @ qml.Z(1)]
        >>> H = qml.Hamiltonian(coeffs, obs)
        >>> H_sparse = H.sparse_matrix()
        >>> H_sparse
        <4x4 sparse matrix of type '<class 'numpy.complex128'>'
                with 8 stored elements in Compressed Sparse Row format>

        The resulting sparse matrix can be either used directly or transformed into a numpy array:

        >>> H_sparse.toarray()
        array([[ 1.+0.j  ,  0.+0.j  ,  0.+0.45j,  0.+0.j  ],
               [ 0.+0.j  , -1.+0.j  ,  0.+0.j  ,  0.-0.45j],
               [ 0.-0.45j,  0.+0.j  , -1.+0.j  ,  0.+0.j  ],
               [ 0.+0.j  ,  0.+0.45j,  0.+0.j  ,  1.+0.j  ]])
        """
        if wire_order is None:
            wires = self.wires
        else:
            wires = wire_order
        n = len(wires)
        matrix = scipy.sparse.csr_matrix((2**n, 2**n), dtype="complex128")

        coeffs = qml.math.toarray(self.data)

        temp_mats = []
        for coeff, op in zip(coeffs, self.ops):
            obs = []
            for o in qml.operation.Tensor(op).obs:
                if len(o.wires) > 1:
                    # todo: deal with operations created from multi-qubit operations such as Hermitian
                    raise ValueError(
                        f"Can only sparsify Hamiltonians whose constituent observables consist of "
                        f"(tensor products of) single-qubit operators; got {op}."
                    )
                obs.append(o.matrix())

            # Array to store the single-wire observables which will be Kronecker producted together
            mat = []
            # i_count tracks the number of consecutive single-wire identity matrices encountered
            # in order to avoid unnecessary Kronecker products, since I_n x I_m = I_{n+m}
            i_count = 0
            for wire_lab in wires:
                if wire_lab in op.wires:
                    if i_count > 0:
                        mat.append(scipy.sparse.eye(2**i_count, format="coo"))
                    i_count = 0
                    idx = op.wires.index(wire_lab)
                    # obs is an array storing the single-wire observables which
                    # make up the full Hamiltonian term
                    sp_obs = scipy.sparse.coo_matrix(obs[idx])
                    mat.append(sp_obs)
                else:
                    i_count += 1

            if i_count > 0:
                mat.append(scipy.sparse.eye(2**i_count, format="coo"))

            red_mat = (
                functools.reduce(lambda i, j: scipy.sparse.kron(i, j, format="coo"), mat) * coeff
            )

            temp_mats.append(red_mat.tocsr())
            # Value of 100 arrived at empirically to balance time savings vs memory use. At this point
            # the `temp_mats` are summed into the final result and the temporary storage array is
            # cleared.
            if (len(temp_mats) % 100) == 0:
                matrix += sum(temp_mats)
                temp_mats = []

        matrix += sum(temp_mats)
        return matrix

    def simplify(self):
        r"""Simplifies the Hamiltonian by combining like-terms.

        **Example**

        >>> ops = [qml.Y(2), qml.X(0) @ qml.Identity(1), qml.X(0)]
        >>> H = qml.Hamiltonian([1, 1, -2], ops)
        >>> H.simplify()
        >>> print(H)
          (-1) [X0]
        + (1) [Y2]

        .. warning::

            Calling this method will reset ``grouping_indices`` to None, since
            the observables it refers to are updated.
        """

        # Todo: make simplify return a new operation, so
        # it does not mutate this one

        new_coeffs = []
        new_ops = []

        for i in range(len(self.ops)):  # pylint: disable=consider-using-enumerate
            op = self.ops[i]
            c = self.coeffs[i]
            op = op if isinstance(op, Tensor) else Tensor(op)

            ind = next((j for j, o in enumerate(new_ops) if op.compare(o)), None)
            if ind is not None:
                new_coeffs[ind] += c
                if np.isclose(qml.math.toarray(new_coeffs[ind]), np.array(0.0)):
                    del new_coeffs[ind]
                    del new_ops[ind]
            else:
                new_ops.append(op.prune())
                new_coeffs.append(c)

        # hotfix: We `self.data`, since `self.parameters` returns a copy of the data and is now returned in
        # self.terms(). To be improved soon.
        self.data = tuple(new_coeffs)
        # hotfix: We overwrite the hyperparameter entry, which is now returned in self.terms().
        # To be improved soon.
        self.hyperparameters["ops"] = new_ops

        self._coeffs = qml.math.stack(new_coeffs) if new_coeffs else []
        self._ops = new_ops
        self._wires = qml.wires.Wires.all_wires([op.wires for op in self.ops], sort=True)
        # reset grouping, since the indices refer to the old observables and coefficients
        self._grouping_indices = None
        return self

    def __str__(self):
        def wires_print(ob: Observable):
            """Function that formats the wires."""
            return ",".join(map(str, ob.wires.tolist()))

        list_of_coeffs = self.data  # list of scalar tensors
        paired_coeff_obs = list(zip(list_of_coeffs, self.ops))
        paired_coeff_obs.sort(key=lambda pair: (len(pair[1].wires), qml.math.real(pair[0])))

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

    def _ipython_display_(self):  # pragma: no-cover
        """Displays __str__ in ipython instead of __repr__
        See https://ipython.readthedocs.io/en/stable/config/integrating.html
        """
        if len(self.ops) < 15:
            print(str(self))
        else:  # pragma: no-cover
            print(repr(self))

    def _obs_data(self):
        r"""Extracts the data from a Hamiltonian and serializes it in an order-independent fashion.

        This allows for comparison between Hamiltonians that are equivalent, but are defined with terms and tensors
        expressed in different orders. For example, `qml.X(0) @ qml.Z(1)` and
        `qml.Z(1) @ qml.X(0)` are equivalent observables with different orderings.

        .. Note::

            In order to store the data from each term of the Hamiltonian in an order-independent serialization,
            we make use of sets. Note that all data contained within each term must be immutable, hence the use of
            strings and frozensets.

        **Example**

        >>> H = qml.Hamiltonian([1, 1], [qml.X(0) @ qml.X(1), qml.Z(0)])
        >>> print(H._obs_data())
        {(1, frozenset({('PauliX', <Wires = [1]>, ()), ('PauliX', <Wires = [0]>, ())})),
         (1, frozenset({('PauliZ', <Wires = [0]>, ())}))}
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
                if isinstance(ob, qml.GellMann):
                    parameters += (ob.hyperparameters["index"],)
                tensor.append((ob.name, ob.wires, parameters))
            data.add((co, frozenset(tensor)))

        return data

    def compare(self, other):
        r"""Determines whether the operator is equivalent to another.

        Currently only supported for :class:`~Hamiltonian`, :class:`~.Observable`, or :class:`~.Tensor`.
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

        >>> H = qml.Hamiltonian(
        ...     [0.5, 0.5],
        ...     [qml.Z(0) @ qml.Y(1), qml.Y(1) @ qml.Z(0) @ qml.Identity("a")]
        ... )
        >>> obs = qml.Z(0) @ qml.Y(1)
        >>> print(H.compare(obs))
        True

        >>> H1 = qml.Hamiltonian([1, 1], [qml.X(0), qml.Z(1)])
        >>> H2 = qml.Hamiltonian([1, 1], [qml.Z(0), qml.X(1)])
        >>> H1.compare(H2)
        False

        >>> ob1 = qml.Hamiltonian([1], [qml.X(0)])
        >>> ob2 = qml.Hermitian(np.array([[0, 1], [1, 0]]), 0)
        >>> ob1.compare(ob2)
        False
        """

        if isinstance(other, qml.operation.Operator):
            if (pr1 := self.pauli_rep) is not None and (pr2 := other.pauli_rep) is not None:
                pr1.simplify()
                pr2.simplify()
                return pr1 == pr2

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

        qml.QueuingManager.remove(H)
        qml.QueuingManager.remove(self)

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
            return Hamiltonian(coeffs, terms, simplify=True)

        if isinstance(H, (Tensor, Observable)):
            terms = [op @ copy(H) for op in ops1]

            return Hamiltonian(coeffs1, terms, simplify=True)

        return NotImplemented

    def __rmatmul__(self, H):
        r"""The tensor product operation (from the right) between a Hamiltonian and
        a Hamiltonian/Tensor/Observable (ie. Hamiltonian.__rmul__(H) = H @ Hamiltonian).
        """
        if isinstance(H, Hamiltonian):  # can't be accessed by '@'
            return H.__matmul__(self)

        coeffs1 = copy(self.coeffs)
        ops1 = self.ops.copy()

        if isinstance(H, (Tensor, Observable)):
            terms = [copy(H) @ op for op in ops1]

            return Hamiltonian(coeffs1, terms, simplify=True)

        return NotImplemented

    def __add__(self, H):
        r"""The addition operation between a Hamiltonian and a Hamiltonian/Tensor/Observable."""
        ops = self.ops.copy()
        self_coeffs = copy(self.coeffs)

        if isinstance(H, numbers.Number) and H == 0:
            return self

        if isinstance(H, Hamiltonian):
            coeffs = qml.math.concatenate([self_coeffs, copy(H.coeffs)], axis=0)
            ops.extend(H.ops.copy())
            return Hamiltonian(coeffs, ops, simplify=True)

        if isinstance(H, (Tensor, Observable)):
            coeffs = qml.math.concatenate(
                [self_coeffs, qml.math.cast_like([1.0], self_coeffs)], axis=0
            )
            ops.append(H)
            return Hamiltonian(coeffs, ops, simplify=True)

        return NotImplemented

    __radd__ = __add__

    def __mul__(self, a):
        r"""The scalar multiplication operation between a scalar and a Hamiltonian."""
        if isinstance(a, (int, float)):
            self_coeffs = copy(self.coeffs)
            coeffs = qml.math.multiply(a, self_coeffs)
            return Hamiltonian(coeffs, self.ops.copy())

        return NotImplemented

    __rmul__ = __mul__

    def __sub__(self, H):
        r"""The subtraction operation between a Hamiltonian and a Hamiltonian/Tensor/Observable."""
        if isinstance(H, (Hamiltonian, Tensor, Observable)):
            return self + (-1 * H)
        return NotImplemented

    def __iadd__(self, H):
        r"""The inplace addition operation between a Hamiltonian and a Hamiltonian/Tensor/Observable."""
        if isinstance(H, numbers.Number) and H == 0:
            return self

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

        return NotImplemented

    def __imul__(self, a):
        r"""The inplace scalar multiplication operation between a scalar and a Hamiltonian."""
        if isinstance(a, (int, float)):
            self._coeffs = qml.math.multiply(a, self._coeffs)
            if self.pauli_rep is not None:
                self._pauli_rep = qml.math.multiply(a, self._pauli_rep)
            return self

        return NotImplemented

    def __isub__(self, H):
        r"""The inplace subtraction operation between a Hamiltonian and a Hamiltonian/Tensor/Observable."""
        if isinstance(H, (Hamiltonian, Tensor, Observable)):
            self.__iadd__(H.__mul__(-1))
            return self
        return NotImplemented

    def queue(self, context=qml.QueuingManager):
        """Queues a qml.Hamiltonian instance"""
        for o in self.ops:
            context.remove(o)
        context.append(self)
        return self

    def map_wires(self, wire_map: dict):
        """Returns a copy of the current hamiltonian with its wires changed according to the given
        wire map.

        Args:
            wire_map (dict): dictionary containing the old wires as keys and the new wires as values

        Returns:
            .Hamiltonian: new hamiltonian
        """
        cls = self.__class__
        new_op = cls.__new__(cls)
        new_op.data = copy(self.data)
        new_op._wires = Wires(  # pylint: disable=protected-access
            [wire_map.get(wire, wire) for wire in self.wires]
        )
        new_op._ops = [  # pylint: disable=protected-access
            op.map_wires(wire_map) for op in self.ops
        ]
        for attr, value in vars(self).items():
            if attr not in {"data", "_wires", "_ops"}:
                setattr(new_op, attr, value)
        new_op.hyperparameters["ops"] = new_op._ops  # pylint: disable=protected-access
        new_op._pauli_rep = "unset"  # pylint: disable=protected-access
        return new_op
