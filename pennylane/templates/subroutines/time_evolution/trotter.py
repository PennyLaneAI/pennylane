# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
Contains templates for Suzuki-Trotter approximation based subroutines.
"""
import copy
from collections import defaultdict

from pennylane import math
from pennylane import ops as qml_ops
from pennylane.capture.autograph import wraps
from pennylane.decomposition import add_decomps, register_resources, resource_rep
from pennylane.operation import Operation, Operator
from pennylane.queuing import QueuingManager, apply
from pennylane.resource import Resources, ResourcesOperation
from pennylane.resource.error import (
    ErrorOperation,
    SpectralNormError,
    _commutator_error,
    _one_norm_error,
)
from pennylane.tape import QuantumScript, make_qscript
from pennylane.wires import Wires


def _scalar(order):
    """Compute the scalar used in the recursive expression.

    Args:
        order (int): order of Trotter product (assume order is an even integer > 2).

    Returns:
        float: scalar to be used in the recursive expression.
    """
    root = 1 / (order - 1)
    return (4 - 4**root) ** -1


@QueuingManager.stop_recording()
def _recursive_expression(x, order, ops):
    """Generate a list of operations using the
    recursive expression which defines the Trotter product.

    Args:
        x (complex): the evolution 'time'
        order (int): the order of the Trotter expansion
        ops (Iterable(~.Operators)): a list of terms in the Hamiltonian

    Returns:
        list: the approximation as product of exponentials of the Hamiltonian terms
    """
    if order == 1:
        return [qml_ops.Evolution(op, -x) for op in ops]

    if order == 2:
        return [qml_ops.Evolution(op, -x * 0.5) for op in ops + ops[::-1]]

    scalar_1 = _scalar(order)
    scalar_2 = 1 - 4 * scalar_1

    ops_lst_1 = _recursive_expression(scalar_1 * x, order - 2, ops)
    ops_lst_2 = _recursive_expression(scalar_2 * x, order - 2, ops)

    return (2 * ops_lst_1) + ops_lst_2 + (2 * ops_lst_1)


class TrotterProduct(ErrorOperation, ResourcesOperation):
    r"""An operation representing the Suzuki-Trotter product approximation for the complex matrix
    exponential of a given Hamiltonian.

    The Suzuki-Trotter product formula provides a method to approximate the matrix exponential of
    Hamiltonian expressed as a linear combination of operands which in general do not commute.
    Consider the Hamiltonian :math:`H = \Sigma^{N}_{j=0} O_{j}`, the product formula is constructed
    using symmetrized products of the terms in the Hamiltonian. The symmetrized products of order
    :math:`m \in [1, 2, 4, ..., 2k]` with :math:`k \in \mathbb{N}` are given by:

    .. math::

        \begin{align}
            S_{1}(t) &= \Pi_{j=0}^{N} \ e^{i t O_{j}} \\
            S_{2}(t) &= \Pi_{j=0}^{N} \ e^{i \frac{t}{2} O_{j}} \cdot \Pi_{j=N}^{0} \ e^{i \frac{t}{2} O_{j}} \\
            &\vdots \\
            S_{m}(t) &= S_{m-2}(p_{m}t)^{2} \cdot S_{m-2}((1-4p_{m})t) \cdot S_{m-2}(p_{m}t)^{2},
        \end{align}

    where the coefficient is :math:`p_{m} = 1 / (4 - \sqrt[m - 1]{4})`. The :math:`m^{\text{th}}` order,
    :math:`n`\ -step Suzuki-Trotter approximation is then defined as:

    .. math:: e^{iHt} \approx \left [S_{m}(t / n)  \right ]^{n}.

    For more details see `J. Math. Phys. 32, 400 (1991) <https://pubs.aip.org/aip/jmp/article-abstract/32/2/400/229229>`_.

    Args:
        hamiltonian (Union[.Hamiltonian, .Sum, .SProd]): The Hamiltonian written as a linear combination
            of operators with known matrix exponentials.
        time (float): The time of evolution, namely the parameter :math:`t` in :math:`e^{iHt}`
        n (int): An integer representing the number of Trotter steps to perform
        order (int): An integer (:math:`m`) representing the order of the approximation (must be 1 or even)
        check_hermitian (bool): A flag to enable the validation check to ensure this is a valid unitary operator

    Raises:
        TypeError: The ``hamiltonian`` is not of type :class:`~.Sum`.
        ValueError: The ``hamiltonian`` has only one term or no terms.
        ValueError: One or more of the terms in ``hamiltonian`` are not Hermitian
            (only for ``check_hermitian=True``)
        ValueError: The ``order`` is not one or a positive even integer.

    **Example**

    .. code-block:: python

        coeffs = [0.25, 0.75]
        ops = [qml.X(0), qml.Z(0)]
        H = qml.dot(coeffs, ops)

        dev = qml.device("default.qubit", wires=2)
        @qml.qnode(dev)
        def my_circ():
            # Prepare some state
            qml.Hadamard(0)

            # Evolve according to H
            qml.TrotterProduct(H, time=2.4, order=2)

            # Measure some quantity
            return qml.state()

    >>> my_circ()
    array([-0.132...+0.597...j,  0.        +0.j        , -0.132...-0.779...j,  0.        +0.j        ])

    .. warning::

        The Trotter-Suzuki decomposition depends on the order of the summed observables. Two
        mathematically identical :class:`~.LinearCombination` objects may undergo different time
        evolutions due to the order in which those observables are stored. The order of observables
        can be queried using the :attr:`~.Sum.operands` attribute. Also see the advanced example
        below.

    .. warning::

        ``TrotterProduct`` does not automatically simplify the input Hamiltonian. This allows
        for a more fine-grained control over the decomposition but also risks an increased
        runtime and/or number of gates. Simplification can be performed manually by
        applying :func:`~.simplify` to your Hamiltonian before using it in ``TrotterProduct``.

    .. details::
        :title: Usage Details

        An *upper-bound* for the error in approximating time-evolution using this operator can be
        computed by calling :func:`~.TrotterProduct.error()`. It is computed using two different
        methods; the "one-norm-bound" scaling method and the "commutator-bound" scaling method.
        (see `Childs et al. (2021) <https://arxiv.org/abs/1912.08854>`_)

        >>> hamiltonian = qml.dot([1.0, 0.5, -0.25], [qml.X(0), qml.Y(0), qml.Z(0)])
        >>> op = qml.TrotterProduct(hamiltonian, time=0.01, order=2)
        >>> op.error(method="one-norm-bound")
        SpectralNormError(8.039062500000003e-06)
        >>> op.error(method="commutator-bound")
        SpectralNormError(6.166666666666668e-06)

        This operation is similar to the :class:`~.ApproxTimeEvolution`. One can recover the behaviour
        of :class:`~.ApproxTimeEvolution` by taking the adjoint:

        >>> qml.adjoint(qml.TrotterProduct(hamiltonian, time, order=1, n=n)) # doctest: +SKIP

        The grouping of terms in the ``operands`` attribute of the ``hamiltonian`` impacts
        the structure of the gates created by ``TrotterProduct``. To understand this, first
        consider this simple two-qubit Hamiltonian with four Pauli word terms:

        >>> coeffs = [0.5, 0.2, 0.1, -0.6]
        >>> ops = [qml.X(0), qml.Y(1), qml.Y(0) @ qml.Z(1), qml.X(0) @ qml.Y(1)]
        >>> H_flat = qml.dot(coeffs, ops)
        >>> H_flat
        (
            0.5 * X(0)
        + 0.2 * Y(1)
        + 0.1 * (Y(0) @ Z(1))
        + -0.6 * (X(0) @ Y(1))
        )
        >>> print(*H_flat.operands, sep="\n")
        0.5 * X(0)
        0.2 * Y(1)
        0.1 * (Y(0) @ Z(1))
        -0.6 * (X(0) @ Y(1))

        As we can see, each Pauli word contributes an individual operand. As a result, the
        ``TrotterProduct`` (of first order, for simplicity) of this Hamiltonian will contain four
        exponentials per Trotter step:

        >>> qml.TrotterProduct(H_flat, 1., n=1, order=1).decomposition()
        [Evolution(1j -0.6 * (X(0) @ Y(1))), Evolution(1j 0.1 * (Y(0) @ Z(1))), Evolution(1j 0.2 * Y(1)), Evolution(1j 0.5 * X(0))]

        If we first create two operands with two Pauli words each and then sum those, this is
        reflected in the structure of the operator:

        >>> H_grouped = qml.sum(qml.dot(coeffs[:2], ops[:2]), qml.dot(coeffs[2:], ops[2:]))
        >>> print(*H_grouped.operands, sep="\n")
        0.5 * X(0) + 0.2 * Y(1)
        0.1 * (Y(0) @ Z(1)) + -0.6 * (X(0) @ Y(1))

        The ``TrotterProduct`` accordingly has a different structure as well:

        >>> qml.TrotterProduct(H_grouped, 1., n=1, order=1).decomposition()
        [Evolution(1j 0.1 * (Y(0) @ Z(1)) + -0.6 * (X(0) @ Y(1))), Evolution(1j 0.5 * X(0) + 0.2 * Y(1))]


        As we can see, the ``operands`` structure of the Hamiltonian directly impacts the
        constructed Trotter circuit, and in general, those circuits will be different
        approximations to the true time evolution.

        We can also compute the gradient with respect to the coefficients of the Hamiltonian and the
        evolution time. Note that this is currently only possible with backprop. 

        .. code-block:: python

            @qml.qnode(dev, diff_method="backprop")
            def my_circ(c1, c2, time):
                # Prepare H:
                H = qml.dot([c1, c2], [qml.X(0), qml.Z(0)])

                # Prepare some state
                qml.Hadamard(0)

                # Evolve according to H
                qml.TrotterProduct(H, time, order=2)

                # Measure some quantity
                return qml.expval(qml.Z(0) @ qml.Z(1))

        >>> args = qml.numpy.array([1.23, 4.5, 0.1])
        >>> qml.grad(my_circ)(*tuple(args))
        (tensor(0.077..., requires_grad=True), tensor(0.015..., requires_grad=True), tensor(1.642..., requires_grad=True))

    """

    resource_keys = {"n", "order", "ops"}

    @classmethod
    def _primitive_bind_call(cls, *args, **kwargs):
        # accepts no wires, so bypasses the wire processing.
        return cls._primitive.bind(*args, **kwargs)

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self, hamiltonian, time, n=1, order=1, check_hermitian=True, id=None
    ):
        r"""Initialize the TrotterProduct class"""

        if order <= 0 or order != 1 and order % 2 != 0:
            raise ValueError(
                f"The order of a TrotterProduct must be 1 or a positive even integer, got {order}."
            )

        if isinstance(hamiltonian, qml_ops.LinearCombination):
            coeffs, ops = hamiltonian.terms()
            if len(coeffs) < 2:
                raise ValueError(
                    "There should be at least 2 terms in the Hamiltonian. Otherwise use `qml.exp`"
                )
            if QueuingManager.recording():
                QueuingManager.remove(hamiltonian)
            hamiltonian = qml_ops.functions.dot(coeffs, ops)

        if isinstance(hamiltonian, qml_ops.op_math.SProd):
            if QueuingManager.recording():
                QueuingManager.remove(hamiltonian)
            hamiltonian = hamiltonian.simplify()
            if len(hamiltonian.terms()[0]) < 2:
                raise ValueError(
                    "There should be at least 2 terms in the Hamiltonian. Otherwise use `qml.exp`"
                )

        if not isinstance(hamiltonian, qml_ops.Sum):
            raise TypeError(
                f"The given operator must be a PennyLane ~.Sum or ~.SProd, got {hamiltonian}"
            )

        if check_hermitian:
            for op in hamiltonian.operands:
                if not op.is_hermitian:
                    raise ValueError(
                        "One or more of the terms in the Hamiltonian may not be Hermitian"
                    )

        self._hyperparameters = {
            "n": n,
            "order": order,
            "base": hamiltonian,
            "check_hermitian": check_hermitian,
        }

        super().__init__(*hamiltonian.data, time, wires=hamiltonian.wires, id=id)

    @property
    def resource_params(self) -> dict:
        return {
            "n": self.hyperparameters["n"],
            "order": self.hyperparameters["order"],
            "ops": self.hyperparameters["base"].operands,
        }

    def map_wires(self, wire_map: dict):
        # pylint: disable=protected-access
        new_op = copy.deepcopy(self)
        new_op._wires = Wires([wire_map.get(wire, wire) for wire in self.wires])
        new_op._hyperparameters["base"] = qml_ops.functions.map_wires(
            new_op._hyperparameters["base"], wire_map
        )
        return new_op

    def queue(self, context=QueuingManager):
        context.remove(self.hyperparameters["base"])
        context.append(self)
        return self

    def resources(self) -> Resources:
        r"""The resource requirements for a given instance of the Suzuki-Trotter product.

        Returns:
            :class:`~.resource.Resources`: The resources for an instance of ``TrotterProduct``.
        """
        with QueuingManager.stop_recording():
            decomp = self.compute_decomposition(*self.parameters, **self.hyperparameters)

        num_wires = len(self.wires)
        num_gates = len(decomp)

        depth = QuantumScript(decomp).graph.get_depth()

        gate_types = defaultdict(int)
        gate_sizes = defaultdict(int)

        for op in decomp:
            gate_types[op.name] += 1
            gate_sizes[len(op.wires)] += 1

        return Resources(num_wires, num_gates, gate_types, gate_sizes, depth)

    def error(
        self, method: str = "commutator-bound", fast: bool = True
    ):  # pylint: disable=arguments-differ
        r"""Compute an *upper-bound* on the spectral norm error for approximating the
        time-evolution of the base Hamiltonian using the Suzuki-Trotter product formula.

        The error in the Suzuki-Trotter product formula is defined as

        .. math:: || \ e^{iHt} - \left [S_{m}(t / n)  \right ]^{n} \ ||,

        Where the norm :math:`||\cdot||` is the spectral norm. This function supports two methods
        from literature for upper-bounding the error, the "one-norm" error bound and the "commutator"
        error bound.

        **Example:**

        The "one-norm" error bound can be computed by passing the kwarg :code:`method="one-norm-bound"`, and
        is defined according to Section 2.3 (lemma 6, equation 22 and 23) of
        `Childs et al. (2021) <https://arxiv.org/abs/1912.08854>`_.

        >>> hamiltonian = qml.dot([1.0, 0.5, -0.25], [qml.X(0), qml.Y(0), qml.Z(0)])
        >>> op = qml.TrotterProduct(hamiltonian, time=0.01, order=2)
        >>> op.error(method="one-norm-bound")
        SpectralNormError(8.039062500000003e-06)

        The "commutator" error bound can be computed by passing the kwarg :code:`method="commutator-bound"`, and
        is defined according to Appendix C (equation 189) `Childs et al. (2021) <https://arxiv.org/abs/1912.08854>`_.

        >>> hamiltonian = qml.dot([1.0, 0.5, -0.25], [qml.X(0), qml.Y(0), qml.Z(0)])
        >>> op = qml.TrotterProduct(hamiltonian, time=0.01, order=2)
        >>> op.error(method="commutator-bound")
        SpectralNormError(6.166666666666668e-06)

        Args:
            method (str, optional): Options include "one-norm-bound" and "commutator-bound" and specify the
                method with which the error is computed. Defaults to "commutator-bound".
            fast (bool, optional): Uses more approximations to speed up computation. Defaults to True.

        Raises:
            ValueError: The method is not supported.

        Returns:
            SpectralNormError: The spectral norm error.
        """
        base_unitary = self.hyperparameters["base"]
        t, p, n = (self.parameters[-1], self.hyperparameters["order"], self.hyperparameters["n"])

        parameters = [t] + base_unitary.parameters
        if any(
            math.get_interface(param) == "tensorflow" for param in parameters
        ):  # pragma: no cover (TensorFlow tests were disabled during deprecation)
            raise TypeError(
                "Calculating error bound for Tensorflow objects is currently not supported"
            )

        terms = base_unitary.operands
        if method == "one-norm-bound":
            return SpectralNormError(_one_norm_error(terms, t, p, n, fast=fast))

        if method == "commutator-bound":
            return SpectralNormError(_commutator_error(terms, t, p, n, fast=fast))

        raise ValueError(
            f"The '{method}' method is not supported for computing the error. Please select a valid method for computing the error."
        )

    def _flatten(self):
        """Serialize the operation into trainable and non-trainable components.

        Returns:
            data, metadata: The trainable and non-trainable components.

        See ``Operator._unflatten``.

        The data component can be recursive and include other operations. For example, the trainable component of ``Adjoint(RX(1, wires=0))``
        will be the operator ``RX(1, wires=0)``.

        The metadata **must** be hashable.  If the hyperparameters contain a non-hashable component, then this
        method and ``Operator._unflatten`` should be overridden to provide a hashable version of the hyperparameters.

        **Example:**

        >>> op = qml.Rot(1.2, 2.3, 3.4, wires=0)
        >>> qml.Rot._unflatten(*op._flatten())
        Rot(1.2, 2.3, 3.4, wires=[0])
        >>> op = qml.PauliRot(1.2, "XY", wires=(0,1))
        >>> qml.PauliRot._unflatten(*op._flatten())
        PauliRot(1.2, XY, wires=[0, 1])

        Operators that have trainable components that differ from their ``Operator.data`` must implement their own
        ``_flatten`` methods.

        >>> op = qml.ctrl(qml.U2(3.4, 4.5, wires="a"), ("b", "c") )
        >>> op._flatten()
        ((U2(3.4, 4.5, wires=['a']),), (Wires(['b', 'c']), (True, True), Wires([]), 'borrowed'))

        """
        hamiltonian = self.hyperparameters["base"]
        time = self.data[-1]

        hashable_hyperparameters = tuple(
            item for item in self.hyperparameters.items() if item[0] != "base"
        )
        return (hamiltonian, time), hashable_hyperparameters

    @classmethod
    def _unflatten(cls, data, metadata):
        return cls(*data, **dict(metadata))

    @staticmethod
    def compute_decomposition(*args, **kwargs):
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.

        .. note::

            Operations making up the decomposition should be queued within the
            ``compute_decomposition`` method.

        .. seealso:: :meth:`~.Operator.decomposition`.

        Args:
            *params (list): trainable parameters of the operator, as stored in the ``parameters`` attribute
            wires (Iterable[Any], Wires): wires that the operator acts on
            **hyperparams (dict): non-trainable hyperparameters of the operator, as stored in the ``hyperparameters`` attribute

        Returns:
            list[Operator]: decomposition of the operator
        """
        time = args[-1]
        n = kwargs["n"]
        order = kwargs["order"]
        ops = kwargs["base"].operands

        decomp = _recursive_expression(time / n, order, ops)[::-1] * n

        if QueuingManager.recording():
            for op in decomp:  # apply operators in reverse order of expression
                apply(op)

        return decomp


def _trotter_product_decomposition_resources(n, order, ops):
    reps = {}

    def _count(op, ops):
        count = 0
        for nxt in ops:
            if nxt == op:
                count += 1
        return count

    if order == 1:
        for op in ops:
            reps[resource_rep(qml_ops.op_math.Evolution, base=op, num_steps=None)] = n * _count(
                op, ops
            )
        return reps
    if order == 2:
        for op in ops:
            reps[resource_rep(qml_ops.op_math.Evolution, base=op, num_steps=None)] = (
                n * 2 * _count(op, ops)
            )
        return reps
    for op in ops:
        reps[resource_rep(qml_ops.op_math.Evolution, base=op, num_steps=None)] = (
            n * _count(op, ops) * 2 * 5 * (order - 2) / 2
        )
    return reps


@register_resources(_trotter_product_decomposition_resources)
def _trotter_product_decomposition(*args, **kwargs):
    time = args[-1]
    n = kwargs["n"]
    order = kwargs["order"]
    ops = kwargs["base"].operands

    def _recursive(x, order, ops):
        if order == 1:
            for op in ops[::-1]:
                qml_ops.Evolution(op, -x)
            return

        if order == 2:
            for op in ops + ops[::-1]:
                qml_ops.Evolution(op, -x * 0.5)
            return

        scalar_1 = _scalar(order)
        scalar_2 = 1 - 4 * scalar_1

        for _ in range(2):
            _recursive(scalar_1 * x, order - 2, ops)
        _recursive(scalar_2 * x, order - 2, ops)
        for _ in range(2):
            _recursive(scalar_1 * x, order - 2, ops)

    for _ in range(n):
        _recursive(time / n, order, ops)


add_decomps(TrotterProduct, _trotter_product_decomposition)


class TrotterizedQfunc(Operation):
    r"""An operation representing the Suzuki-Trotter product approximation applied to a set of
    operations defined in a function.

    The Suzuki-Trotter product formula provides a method to approximate the matrix exponential of
    a Hamiltonian expressed as a linear combination of terms which in general do not commute. Consider
    the Hamiltonian :math:`H = \Sigma^{N}_{j=0} O_{j}`, the product formula is constructed using
    symmetrized products of the terms in the Hamiltonian. The symmetrized products of order
    :math:`m \in [1, 2, 4, ..., 2k]` with :math:`k \in \mathbb{N}` are given by:

    .. math::

        \begin{align}
            S_{1}(t) &= \Pi_{j=0}^{N} \ e^{i t O_{j}} \\
            S_{2}(t) &= \Pi_{j=0}^{N} \ e^{i \frac{t}{2} O_{j}} \cdot \Pi_{j=N}^{0} \ e^{i \frac{t}{2} O_{j}} \\
            &\vdots \\
            S_{m}(t) &= S_{m-2}(p_{m}t)^{2} \cdot S_{m-2}((1-4p_{m})t) \cdot S_{m-2}(p_{m}t)^{2},
        \end{align}

    where the coefficient is :math:`p_{m} = 1 / (4 - \sqrt[m - 1]{4})`. The :math:`m`th order,
    :math:`n`-step Suzuki-Trotter approximation is then defined as:

    .. math:: e^{iHt} \approx \left [S_{m}(t / n)  \right ]^{n}.

    For more details see `J. Math. Phys. 32, 400 (1991) <https://pubs.aip.org/aip/jmp/article-abstract/32/2/400/229229>`_.

    Suppose we have direct access to the operators which represent the exponentiated terms of
    a hamiltonian:

    .. math:: \{ \hat{U}_{j} = e^{i t O_{j}}, j \in [1, N] \}.

    Given a quantum circuit which uses these :math:`\hat{U}_{j}` operators to represent the
    first order expansion :math:`S_{1}(t)`; this class expands it to any higher order Suzuki-Trotter product.

    .. warning::

        :code:`TrotterizedQfunc` requires that the input function has a very specific function signature.
        The first argument should be a time parameter which will be modified according to the Suzuki-Trotter
        product formula. The wires required by the circuit should be either the last explicit argument or the
        first keyword argument. :code:`qfunc((time, arg1, ..., arg_n, wires=[...], kwarg_1, ..., kwarg_n))`

    .. warning::

        :code:`TrotterizedQfunc` currently does not support pickling. Instead please decompose the operation
        first before attempting to pickle the quantum circuit.

    Args:
        time (float): the time of evolution, namely the parameter :math:`t` in :math:`e^{iHt}`
        *trainable_args (tuple): the trainable arguments of the first-order expansion function
        qfunc (Callable): the first-order expansion given as a callable function which queues operations
        wires (Iterable): the set of wires the operation will act upon (should be identical to qfunc wires)
        n (int): an integer representing the number of Trotter steps to perform
        order (int): an integer (:math:`m`) representing the order of the approximation (must be 1 or even)
        reverse (bool): if true, reverse the order of the operations queued by :code:`qfunc`
        **non_trainable_kwargs (dict): non-trainable keyword arguments of the first-order expansion function

    Raises:
        ValueError: A qfunc must be provided to be trotterized.

    See also :func:`~.trotterize`.

    **Example**

    .. code-block:: python

        def first_order_expansion(time, theta, phi, wires=[0, 1, 2], flip=False):
            "This is the first order expansion (U_1)."
            qml.RX(time*theta, wires[0])
            qml.RY(time*phi, wires[1])
            if flip:
                qml.CNOT(wires=wires[:2])

        @qml.qnode(qml.device("default.qubit"))
        def my_circuit(time, angles, num_trotter_steps):
            qml.TrotterizedQfunc(
                time,
                *angles,
                qfunc=first_order_expansion,
                n=num_trotter_steps,
                order=2,
                wires=['a', 'b'],
                flip=True,
            )
            return qml.state()

    We can visualize the circuit to see the Suzuki-Trotter product formula being applied:

    >>> time = 0.1
    >>> angles = (0.12, -3.45)
    >>> print(qml.draw(my_circuit, level="device")(time, angles, num_trotter_steps=1))
    a: ──RX(0.01)──╭●─╭●──RX(0.01)──┤  State
    b: ──RY(-0.17)─╰X─╰X──RY(-0.17)─┤  State
    >>>
    >>> print(qml.draw(my_circuit, level="device")(time, angles, num_trotter_steps=3))
    a: ──RX(0.00)──╭●─╭●──RX(0.00)───RX(0.00)──╭●─╭●──RX(0.00)───RX(0.00)──╭●─╭●──RX(0.00)──┤  State
    b: ──RY(-0.06)─╰X─╰X──RY(-0.06)──RY(-0.06)─╰X─╰X──RY(-0.06)──RY(-0.06)─╰X─╰X──RY(-0.06)─┤  State

    """

    # pylint: disable = too-many-arguments
    def __init__(
        self,
        time,
        *trainable_args,
        qfunc=None,
        n=1,
        order=2,
        reverse=False,
        id=None,
        **non_trainable_kwargs,
    ):
        # This class requires the input function (qfunc) has a very specific
        # signature: qfunc(time, arg1, ..., arg_n, wires=[...], kwarg_1, ..., kwarg_n)

        if qfunc is None:
            raise ValueError("The qfunc must be provided to be trotterized.")

        if order <= 0 or order != 1 and order % 2 != 0:
            raise ValueError(f"The order must be 1 or a positive even integer, got {order}.")

        try:
            wires = non_trainable_kwargs.pop("wires")
        except KeyError:
            wires = trainable_args[-1]
            trainable_args = trainable_args[:-1]  # exclude the wires from the args

        self._hyperparameters = non_trainable_kwargs
        self._hyperparameters["n"] = n
        self._hyperparameters["order"] = order
        self._hyperparameters["qfunc"] = qfunc
        self._hyperparameters["reverse"] = reverse

        super().__init__(time, *trainable_args, wires=wires, id=id)

    def decomposition(self) -> list[Operator]:
        """The decomposition"""
        n = self.hyperparameters["n"]
        order = self.hyperparameters["order"]
        qfunc = self.hyperparameters["qfunc"]
        reverse = self.hyperparameters["reverse"]

        time = self.parameters[0]
        qfunc_args = self.parameters[1:]

        base_hyper_params = ("n", "order", "qfunc", "reverse")
        qfunc_kwargs = {k: v for k, v in self.hyperparameters.items() if not k in base_hyper_params}

        decomp = (
            _recursive_qfunc(
                time / n, order, qfunc, self.wires, reverse, *qfunc_args, **qfunc_kwargs
            )
            * n
        )

        if QueuingManager.recording():
            for op in decomp:
                apply(op)

        return decomp

    def _flatten(self):
        hashable_hyperparameters = tuple(self.hyperparameters.items()) + (("wires", self.wires),)
        return self.data, hashable_hyperparameters

    @classmethod
    def _unflatten(cls, data, metadata):
        return cls(*data, **dict(metadata))


def trotterize(qfunc, n=1, order=2, reverse=False):
    r"""Generates higher order Suzuki-Trotter product formulas from a set of
    operations defined in a function.

    The Suzuki-Trotter product formula provides a method to approximate the matrix exponential of a
    Hamiltonian expressed as a linear combination of terms which in general do not commute. Consider
    the Hamiltonian :math:`H = \Sigma^{N}_{j=0} O_{j}`, the product formula is constructed using
    symmetrized products of the terms in the Hamiltonian. The symmetrized products of order
    :math:`m \in [1, 2, 4, ..., 2k]` with :math:`k \in \mathbb{N}` are given by:

    .. math::

        \begin{align}
            S_{1}(t) &= \Pi_{j=0}^{N} \ e^{i t O_{j}} \\
            S_{2}(t) &= \Pi_{j=0}^{N} \ e^{i \frac{t}{2} O_{j}} \cdot \Pi_{j=N}^{0} \ e^{i \frac{t}{2} O_{j}} \\
            &\vdots \\
            S_{m}(t) &= S_{m-2}(p_{m}t)^{2} \cdot S_{m-2}((1-4p_{m})t) \cdot S_{m-2}(p_{m}t)^{2},
        \end{align}

    where the coefficient is :math:`p_{m} = 1 / (4 - \sqrt[m - 1]{4})`. The :math:`m^{\text{th}}` order,
    :math:`n`\-step Suzuki-Trotter approximation is then defined as:

    .. math:: e^{iHt} \approx \left [S_{m}(t / n)  \right ]^{n}.

    For more details see `J. Math. Phys. 32, 400 (1991) <https://pubs.aip.org/aip/jmp/article-abstract/32/2/400/229229>`_.

    Suppose we have direct access to the operators which represent the exponentiated terms of
    a Hamiltonian:

    .. math:: \{ \hat{U}_{j} = e^{i t O_{j}}, j \in [1, N] \}.

    Given a quantum circuit which uses these :math:`\hat{U}_{j}` operators to represent the
    first order expansion :math:`S_{1}(t)`, this function expands it to any higher order Suzuki-Trotter product.

    .. warning::

        :code:`trotterize()` requires the :code:`qfunc` argument to be a function with a specific call
        signature. The first argument of the :code:`qfunc` function should be a time parameter which will be modified according to the
        Suzuki-Trotter product formula. The wires required by the circuit should be either the last
        positional argument or the first keyword argument:
        :code:`qfunc((time, arg1, ..., arg_n, wires=[...], kwarg_1, ..., kwarg_n))`

    Args:
        qfunc (Callable): the first-order expansion given as a callable function which queues operations
        n (int): an integer representing the number of Trotter steps to perform
        order (int): an integer (:math:`m`) representing the order of the approximation (must be 1 or even)
        reverse (bool): if true, reverse the order of the operations queued by :code:`qfunc`
        name (str): an optional name for the instance

    Returns:
        (Callable): a function with the same signature as :code:`qfunc`, when called it queues an instance of
            :class:`~.TrotterizedQfunc`

    **Example**

    .. code-block:: python

        def first_order_expansion(time, theta, phi, wires, flip=False):
            "This is the first order expansion (U_1)."
            qml.RX(time*theta, wires[0])
            qml.RY(time*phi, wires[1])
            if flip:
                qml.CNOT(wires=wires[:2])

        @qml.qnode(qml.device("default.qubit"))
        def my_circuit(time, theta, phi, num_trotter_steps):
            qml.trotterize(
                first_order_expansion,
                n=num_trotter_steps,
                order=2,
            )(time, theta, phi, wires=['a', 'b'], flip=True)
            return qml.state()

    We can visualize the circuit to see the Suzuki-Trotter product formula being applied:

    >>> time = 0.1
    >>> theta, phi = (0.12, -3.45)
    >>>
    >>> print(qml.draw(my_circuit, level="device")(time, theta, phi, num_trotter_steps=1))
    a: ──RX(0.01)──╭●─╭●──RX(0.01)──┤  State
    b: ──RY(-0.17)─╰X─╰X──RY(-0.17)─┤  State
    >>>
    >>> print(qml.draw(my_circuit, level="device")(time, theta, phi, num_trotter_steps=3))
    a: ──RX(0.00)──╭●─╭●──RX(0.00)───RX(0.00)──╭●─╭●──RX(0.00)───RX(0.00)──╭●─╭●──RX(0.00)──┤  State
    b: ──RY(-0.06)─╰X─╰X──RY(-0.06)──RY(-0.06)─╰X─╰X──RY(-0.06)──RY(-0.06)─╰X─╰X──RY(-0.06)─┤  State

    """

    @wraps(qfunc)
    def wrapper(time, *args, **kwargs):

        special_keys = ["n", "order", "qfunc", "reverse"]
        if any(key in kwargs for key in special_keys):
            raise ValueError(
                f"Cannot use any of the specailized names:\n {special_keys}\nas keyword"
                f"arguments to the qfunc: {kwargs}"
            )

        return TrotterizedQfunc(
            time, *args, qfunc=qfunc, n=n, order=order, reverse=reverse, **kwargs
        )

    return wrapper


@QueuingManager.stop_recording()
def _recursive_qfunc(time, order, qfunc, wires, reverse, *qfunc_args, **qfunc_kwargs):
    """Generate a list of operations using the
    recursive expression which defines the Trotter product.
    Args:
        time (float): the evolution 'time'
        order (int): the order of the Trotter expansion
        ops (Iterable(~.Operators)): a list of terms in the Hamiltonian
    Returns:
        list: the approximation as a product of exponentials of the Hamiltonian terms
    """
    if order == 1:
        tape = make_qscript(qfunc)(time, *qfunc_args, wires=wires, **qfunc_kwargs)
        return tape.operations[::-1] if reverse else tape.operations

    if order == 2:
        tape = make_qscript(qfunc)(time / 2, *qfunc_args, wires=wires, **qfunc_kwargs)
        return (
            tape.operations[::-1] + tape.operations
            if reverse
            else tape.operations + tape.operations[::-1]
        )

    scalar_1 = _scalar(order)
    scalar_2 = 1 - 4 * scalar_1

    ops_lst_1 = _recursive_qfunc(
        scalar_1 * time, order - 2, qfunc, wires, reverse, *qfunc_args, **qfunc_kwargs
    )
    ops_lst_2 = _recursive_qfunc(
        scalar_2 * time, order - 2, qfunc, wires, reverse, *qfunc_args, **qfunc_kwargs
    )

    return (2 * ops_lst_1) + ops_lst_2 + (2 * ops_lst_1)
