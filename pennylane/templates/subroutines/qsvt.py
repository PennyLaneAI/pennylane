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
Contains the QSVT template and qsvt wrapper function.
"""
# pylint: disable=too-many-arguments
import copy

import numpy as np

import pennylane as qml
from pennylane.operation import AnyWires, Operation
from pennylane.ops.op_math import adjoint
from pennylane.queuing import QueuingManager
from pennylane.wires import Wires


def qsvt(A, poly, encoding_wires, block_encoding):
    r"""
    Implements the Quantum Singular Value Transformation (QSVT) for a matrix or Hamiltonian `A`, using a polynomial
    defined by `poly` and a block encoding specified by `block_encoding`. QSVT applies polynomial
    transformations to the singular values of `A`.

    The function calculates the required phase angles from the polynomial using `qml.math.poly_to_angles`.

    Args:

        A (Union[tensor_like, LinearCombination]): The matrix on which the QSVT will be applied.
            This can be an array or an object that has a Pauli representation.

        poly (tensor_like): Polynomial coefficients defining the transformation, represented in increasing order of degree.
            This means the first coefficient corresponds to the constant term, the second to the linear term, and so on.

        encoding_wires (Sequence[int]): The qubit wires used for the block encoding. See Usage Details bellow for
            more information on `encoding_wires` depending on the block encoding used.

        block_encoding (str): Specifies the type of block encoding to use. Options include:
                - "prepselprep": Embeds the hamiltonian `A` using `PrepSelPrep`.
                - "qubitization": Embeds the hamiltonian `A` using `Qubitization`.
                - "embedding": Embeds the matrix `A` using `BlockEncode`.
                - "fable": Embeds the matrix `A` using `FABLE`.

    Returns:
        (Operator): A quantum operator implementing QSVT on the matrix `A` with the specified encoding and projector phases.

    Example:

    .. code-block::

        # P(x) = -x + 0.5 x^3 + 0.5 x^5
        poly = np.array([0,-1, 0, 0.5, 0 , 0.5])

        hamiltonian = qml.dot([0.3, 0.7], [qml.Z(1), qml.X(1) @ qml.Z(2)])

        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit():
            qml.qsvt(hamiltonian, poly, encoding_wires=[0], block_encoding="qubitization")
            return qml.state()

        matrix = qml.matrix(circuit, wire_order=[0,1,2])()

    .. code-block:: pycon

        >>> print(matrix[:4, :4].real)
        [[-0.16254  0.      -0.37926  0.     ]
         [ 0.      -0.16254  0.       0.37926]
         [-0.37926  0.       0.16254  0.     ]
         [ 0.       0.37926  0.       0.16254]]


    .. details::
        :title: Usage Details

        If the input to the algorithm ``A`` is a Hamiltonian, the valid ``block_encoding`` values are
        ``"prepselprep"`` and ``"qubitization"``. In this case, ``encoding_wires`` refers to the
        ``control`` parameter in the templates :class:`~pennylane.PrepSelPrep` and :class:`~pennylane.Qubitization`,
        respectively. These wires represent the auxiliary qubits necessary for the block encoding of
        the Hamiltonian. The number of ``encoding_wires`` required must be :math:`\lceil \log_2(m) \rceil`,
        where :math:`m` is the number of terms in the Hamiltonian.

        .. code-block:: python

            # P(x) = -1 + 0.2 x^2 - 0.3 x^4
            poly = np.array([-1, 0, 0.2, 0 , 0.5])

            hamiltonian = qml.dot([0.3, 0.4, 0.3], [qml.Z(2), qml.X(2) @ qml.Z(3), qml.X(2)])

            dev = qml.device("default.qubit")

            @qml.qnode(dev)
            def circuit():
                qml.qsvt(hamiltonian, poly, encoding_wires=[0,1], block_encoding="prepselprep")
                return qml.state()

            matrix = qml.matrix(circuit, wire_order=[0,1,2,3])()


        .. code-block:: pycon

            >>> print(np.round(matrix[:4, :4],4).real)
            [[-0.7158  0.      0.      0.    ]
             [ 0.     -0.975   0.      0.    ]
             [ 0.      0.     -0.7158  0.    ]
             [ 0.     -0.      0.     -0.975 ]]


        Alternatively, if the input ``A`` is a matrix, the valid values for ``block_encoding`` are
        ``"embedding"`` and ``"fable"``. In this case, the ``encoding_wires`` parameter corresponds to
        the ``wires`` attribute in the templates :class:`~pennylane.BlockEncode` and :class:`~pennylane.FABLE`, respectively.
        Note that for QSVT to work, the imput matrix must be hermitian.

        .. code-block:: python

            # P(x) = -0.1 + 0.2 x^2 + 0.5 x^4
            poly = np.array([-0.1, 0, 0.2, 0, 0.5])

            A = np.array([[-0.1, 0, 0, 0.3], [0, 0.2, 0, 0], [0, 0, -0.2, -0.2], [0.3, 0, -0.2, 0.3]]) / 6

            dev = qml.device("default.qubit")
            @qml.qnode(dev)
            def circuit():
                qml.qsvt(A, poly, encoding_wires=[0, 1, 2, 3, 4], block_encoding="fable")
                return qml.state()

            matrix = qml.matrix(circuit, wire_order=[0, 1, 2, 3, 4])()

        .. code-block:: pycon

            >>> print(np.round(matrix[:4, :4], 4).real)
            [[-0.0994 -0.     -0.0003  0.0003]
             [-0.     -0.0998  0.      0.    ]
             [-0.0003 -0.     -0.0996 -0.0001]
             [ 0.0003 -0.     -0.0001 -0.0988]]


    """

    angles = qml.math.poly_to_angles(poly, "QSVT")
    projectors = []

    # If the input A is a Hamiltonian
    if hasattr(A, "pauli_rep"):

        if block_encoding == "qubitization":
            encoding = qml.Qubitization(A, control=encoding_wires)

        else:
            encoding = qml.PrepSelPrep(A, control=encoding_wires)

        projectors = [
            qml.PCPhase(angles[i], dim=2 ** len(A.wires), wires=encoding_wires + A.wires)
            for i in range(len(angles))
        ]

    else:

        if qml.math.shape(A) == () or qml.math.shape(A) == (1,):
            A = qml.math.reshape(A, [1, 1])

        A = qml.math.array(A)

        if block_encoding == "embedding":

            c, r = qml.math.shape(A)

            for idx, phi in enumerate(angles):
                dim = c if idx % 2 else r
                projectors.append(qml.PCPhase(phi, dim=dim, wires=encoding_wires))

            encoding = qml.BlockEncode(A, wires=encoding_wires)

        else:

            # It is normalized to ensure that the block encoding is the desired one.
            s = int(np.ceil(np.log2(max(len(A), len(A[0])))))
            encoding = qml.FABLE(2**s * A, wires=encoding_wires)

            projectors = [
                qml.PCPhase(angles[i], dim=len(A), wires=encoding_wires) for i in range(len(angles))
            ]

    return QSVT(encoding, projectors)


class QSVT(Operation):
    r"""QSVT(UA,projectors)
    Implements the
    `quantum singular value transformation <https://arxiv.org/abs/1806.01838>`__ (QSVT) circuit.

    .. note ::

        This template allows users to define hardware-compatible block encoding and
        projector-controlled phase shift circuits. For a QSVT implementation that is
        tailored for simulators see :func:`~.qsvt` .

    Given an :class:`~.Operator` :math:`U`, which block encodes the matrix :math:`A`, and a list of
    projector-controlled phase shift operations :math:`\vec{\Pi}_\phi`, this template applies a
    circuit for the quantum singular value transformation as follows.

    When the number of projector-controlled phase shifts is even (:math:`d` is odd), the QSVT
    circuit is defined as:

    .. math::

        U_{QSVT} = \tilde{\Pi}_{\phi_1}U\left[\prod^{(d-1)/2}_{k=1}\Pi_{\phi_{2k}}U^\dagger
        \tilde{\Pi}_{\phi_{2k+1}}U\right]\Pi_{\phi_{d+1}}.


    And when the number of projector-controlled phase shifts is odd (:math:`d` is even):

    .. math::

        U_{QSVT} = \left[\prod^{d/2}_{k=1}\Pi_{\phi_{2k-1}}U^\dagger\tilde{\Pi}_{\phi_{2k}}U\right]
        \Pi_{\phi_{d+1}}.

    This circuit applies a polynomial transformation (:math:`Poly^{SV}`) to the singular values of
    the block encoded matrix:

    .. math::

        \begin{align}
             U_{QSVT}(A, \vec{\phi}) &=
             \begin{bmatrix}
                Poly^{SV}(A) & \cdot \\
                \cdot & \cdot
            \end{bmatrix}.
        \end{align}

    .. seealso::

        :func:`~.qsvt` and `A Grand Unification of Quantum Algorithms <https://arxiv.org/pdf/2105.02859.pdf>`_.

    Args:
        UA (Operator): the block encoding circuit, specified as an :class:`~.Operator`,
            like :class:`~.BlockEncode`
        projectors (Sequence[Operator]): a list of projector-controlled phase
            shifts that implement the desired polynomial

    Raises:
        ValueError: if the input block encoding is not an operator

    **Example**

    To implement QSVT in a circuit, we can use the following method:

    >>> dev = qml.device("default.qubit", wires=2)
    >>> A = np.array([[0.1]])
    >>> block_encode = qml.BlockEncode(A, wires=[0, 1])
    >>> shifts = [qml.PCPhase(i + 0.1, dim=1, wires=[0, 1]) for i in range(3)]
    >>> @qml.qnode(dev)
    >>> def example_circuit():
    ...    qml.QSVT(block_encode, shifts)
    ...    return qml.expval(qml.Z(0))

    The resulting circuit implements QSVT.

    >>> print(qml.draw(example_circuit)())
    0: ─╭QSVT─┤  <Z>
    1: ─╰QSVT─┤

    To see the implementation details, we can expand the circuit:

    >>> q_script = qml.tape.QuantumScript(ops=[qml.QSVT(block_encode, shifts)])
    >>> print(q_script.expand().draw(decimals=2))
    0: ─╭∏_ϕ(0.10)─╭BlockEncode(M0)─╭∏_ϕ(1.10)─╭BlockEncode(M0)†─╭∏_ϕ(2.10)─┤
    1: ─╰∏_ϕ(0.10)─╰BlockEncode(M0)─╰∏_ϕ(1.10)─╰BlockEncode(M0)†─╰∏_ϕ(2.10)─┤

    When working with this class directly, we can make use of any PennyLane operation
    to represent our block-encoding and our phase-shifts.

    >>> dev = qml.device("default.qubit", wires=[0])
    >>> block_encoding = qml.Hadamard(wires=0)  # note H is a block encoding of 1/sqrt(2)
    >>> phase_shifts = [qml.RZ(-2 * theta, wires=0) for theta in (1.23, -0.5, 4)]  # -2*theta to match convention
    >>>
    >>> @qml.qnode(dev)
    >>> def example_circuit():
    ...     qml.QSVT(block_encoding, phase_shifts)
    ...     return qml.expval(qml.Z(0))
    >>>
    >>> example_circuit()
    tensor(0.54030231, requires_grad=True)

    Once again, we can visualize the circuit as follows:

    >>> print(qml.draw(example_circuit)())
    0: ──QSVT─┤  <Z>

    To see the implementation details, we can expand the circuit:

    >>> q_script = qml.tape.QuantumScript(ops=[qml.QSVT(block_encoding, phase_shifts)])
    >>> print(q_script.expand().draw(decimals=2))
    0: ──RZ(-2.46)──H──RZ(1.00)──H†──RZ(-8.00)─┤
    """

    num_wires = AnyWires
    """int: Number of wires that the operator acts on."""

    grad_method = None
    """Gradient computation method."""

    def _flatten(self):
        data = (self.hyperparameters["UA"], self.hyperparameters["projectors"])
        return data, tuple()

    @classmethod
    def _primitive_bind_call(cls, *args, **kwargs):
        return cls._primitive.bind(*args, **kwargs)

    @classmethod
    def _unflatten(cls, data, _) -> "QSVT":
        return cls(*data)

    def __init__(self, UA, projectors, id=None):
        if not isinstance(UA, qml.operation.Operator):
            raise ValueError("Input block encoding must be an Operator")

        self._hyperparameters = {
            "UA": UA,
            "projectors": projectors,
        }

        total_wires = qml.wires.Wires.all_wires(
            [proj.wires for proj in projectors]
        ) + qml.wires.Wires(UA.wires)

        super().__init__(wires=total_wires, id=id)

    def map_wires(self, wire_map: dict):
        # pylint: disable=protected-access
        new_op = copy.deepcopy(self)
        new_op._wires = Wires([wire_map.get(wire, wire) for wire in self.wires])
        new_op._hyperparameters["UA"] = qml.map_wires(new_op._hyperparameters["UA"], wire_map)
        new_op._hyperparameters["projectors"] = [
            qml.map_wires(proj, wire_map) for proj in new_op._hyperparameters["projectors"]
        ]
        return new_op

    @property
    def data(self):
        r"""Flattened list of operator data in this QSVT operation.

        This ensures that the backend of a ``QuantumScript`` which contains a
        ``QSVT`` operation can be inferred with respect to the types of the
        ``QSVT`` block encoding and projector-controlled phase shift data.
        """
        return tuple(datum for op in self._operators for datum in op.data)

    @data.setter
    def data(self, new_data):
        # We need to check if ``new_data`` is empty because ``Operator.__init__()``  will attempt to
        # assign the QSVT data to an empty tuple (since no positional arguments are provided).
        if new_data:
            for op in self._operators:
                if op.num_params > 0:
                    op.data = new_data[: op.num_params]
                    new_data = new_data[op.num_params :]

    def __copy__(self):
        # Override Operator.__copy__() to avoid setting the "data" property before the new instance
        # is assigned hyper-parameters since QSVT data is derived from the hyper-parameters.
        clone = QSVT.__new__(QSVT)

        # Ensure the operators in the hyper-parameters are copied instead of aliased.
        clone._hyperparameters = {
            "UA": copy.copy(self._hyperparameters["UA"]),
            "projectors": list(map(copy.copy, self._hyperparameters["projectors"])),
        }

        for attr, value in vars(self).items():
            if attr != "_hyperparameters":
                setattr(clone, attr, value)

        return clone

    @property
    def _operators(self) -> list[qml.operation.Operator]:
        """Flattened list of operators that compose this QSVT operation."""
        return [self._hyperparameters["UA"], *self._hyperparameters["projectors"]]

    @staticmethod
    def compute_decomposition(
        *_data, UA, projectors, **_kwargs
    ):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a product of other operators.

        The :class:`~.QSVT` is decomposed into alternating block encoding
        and projector-controlled phase shift operators. This is defined by the following
        equations, where :math:`U` is the block encoding operation and both :math:`\Pi_\phi` and
        :math:`\tilde{\Pi}_\phi` are projector-controlled phase shifts with angle :math:`\phi`.

        When the number of projector-controlled phase shifts is even (:math:`d` is odd), the QSVT
        circuit is defined as:

        .. math::

            U_{QSVT} = \Pi_{\phi_1}U\left[\prod^{(d-1)/2}_{k=1}\Pi_{\phi_{2k}}U^\dagger
            \tilde{\Pi}_{\phi_{2k+1}}U\right]\Pi_{\phi_{d+1}}.


        And when the number of projector-controlled phase shifts is odd (:math:`d` is even):

        .. math::

            U_{QSVT} = \left[\prod^{d/2}_{k=1}\Pi_{\phi_{2k-1}}U^\dagger\tilde{\Pi}_{\phi_{2k}}U\right]
            \Pi_{\phi_{d+1}}.

        .. seealso:: :meth:`~.QSVT.decomposition`.

        Args:
            UA (Operator): the block encoding circuit, specified as a :class:`~.Operator`
            projectors (list[Operator]): a list of projector-controlled phase
                shift circuits that implement the desired polynomial

        Returns:
            list[.Operator]: decomposition of the operator
        """

        op_list = []
        UA_adj = copy.copy(UA)

        for idx, op in enumerate(projectors[:-1]):
            if qml.QueuingManager.recording():
                qml.apply(op)
            op_list.append(op)

            if idx % 2 == 0:
                if qml.QueuingManager.recording():
                    qml.apply(UA)
                op_list.append(UA)

            else:
                op_list.append(adjoint(UA_adj))

        if qml.QueuingManager.recording():
            qml.apply(projectors[-1])
        op_list.append(projectors[-1])

        return op_list

    def label(self, decimals=None, base_label=None, cache=None):
        op_label = base_label or self.__class__.__name__
        return op_label

    def queue(self, context=QueuingManager):
        context.remove(self._hyperparameters["UA"])
        for op in self._hyperparameters["projectors"]:
            context.remove(op)
        context.append(self)
        return self

    @staticmethod
    def compute_matrix(*args, **kwargs):
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.Operator.matrix` and :func:`~.matrix`

        Args:
            *params (list): trainable parameters of the operator, as stored in the ``parameters`` attribute
            **hyperparams (dict): non-trainable hyperparameters of the operator, as stored in the ``hyperparameters`` attribute

        Returns:
            tensor_like: matrix representation
        """
        # pylint: disable=unused-argument
        op_list = []
        UA = kwargs["UA"]
        projectors = kwargs["projectors"]

        with (
            QueuingManager.stop_recording()
        ):  # incase this method is called in a queue context, this prevents
            UA_copy = copy.copy(UA)  # us from queuing operators unnecessarily

            for idx, op in enumerate(projectors[:-1]):
                op_list.append(op)
                if idx % 2 == 0:
                    op_list.append(UA)
                else:
                    op_list.append(adjoint(UA_copy))

            op_list.append(projectors[-1])
            mat = qml.matrix(qml.prod(*tuple(op_list[::-1])))

        return mat
