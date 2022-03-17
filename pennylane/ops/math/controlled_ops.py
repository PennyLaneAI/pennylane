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

import pennylane as qml

from . import Controlled

import warnings

class ControlledQubitUnitary(Controlled):
    r"""ControlledQubitUnitary(U, control_wires, wires, control_values)
    Apply an arbitrary fixed unitary to ``wires`` with control from the ``control_wires``.

    In addition to default ``Operation`` instance attributes, the following are
    available for ``ControlledQubitUnitary``:

    * ``control_wires``: wires that act as control for the operation
    * ``U``: unitary applied to the target wires

    **Details:**

    * Number of wires: Any (the operation can act on any number of wires)
    * Number of parameters: 1
    * Gradient recipe: None

    Args:
        U (array[complex]): square unitary matrix
        control_wires (Union[Wires, Sequence[int], or int]): the control wire(s)
        wires (Union[Wires, Sequence[int], or int]): the wire(s) the unitary acts on
        control_values (Bool, Sequence[Bool]): The values to control on, denoted by ``0``, ``1``,
            ``True``, or ``False``.  Must be same length as ``control_wires``

    **Example**

    The following shows how a single-qubit unitary can be applied to wire ``2`` with control on
    both wires ``0`` and ``1``:

    >>> U = np.array([[ 0.94877869,  0.31594146], [-0.31594146,  0.94877869]])
    >>> qml.ControlledQubitUnitary(U, control_wires=[0, 1], wires=2)

    Typically controlled operations apply a desired gate if the control qubits
    are all in the state :math:`\vert 1\rangle`. However, there are some situations where
    it is necessary to apply a gate conditioned on all qubits being in the
    :math:`\vert 0\rangle` state, or a mix of the two.

    The state on which to control can be changed by passing a string of bits to
    `control_values`. For example, if we want to apply a single-qubit unitary to
    wire ``3`` conditioned on three wires where the first is in state ``0``, the
    second is in state ``1``, and the third in state ``1``, we can write:

    >>> qml.ControlledQubitUnitary(U, control_wires=[0, 1, 2], wires=3, control_values='011')

    """

    grad_method = None
    """Gradient computation method."""

    def __init__(
        self,
        *params,
        control_wires=None,
        wires=None,
        control_values=None,
        do_queue=True,
        id=None
    ):
        super().__init__(qml.QubitUnitary(*params, wires=wires), control_wires, control_values=control_values,
            do_queue=do_queue, id=id)
        self._name = self.__class__.__name__


class CNOT(Controlled):
    r"""CNOT(wires)
    The controlled-NOT operator

    .. math:: CNOT = \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0\\
            0 & 0 & 0 & 1\\
            0 & 0 & 1 & 0
        \end{bmatrix}.

    .. note:: The first wire provided corresponds to the **control qubit**.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 0

    Args:
        wires (Sequence[int]): the wires the operation acts on
    """
    
    def __init__(self, wires=None, do_queue=True, id=None):
        super().__init__(qml.PauliX(wires[1]), wires[0], do_queue=do_queue, id=id)
        self._name = self.__class__.__name__

class CZ(Controlled):
    r"""CZ(wires)
    The controlled-Z operator

    .. math:: CZ = \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0\\
            0 & 0 & 1 & 0\\
            0 & 0 & 0 & -1
        \end{bmatrix}.

    .. note:: The first wire provided corresponds to the **control qubit**.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 0

    Args:
        wires (Sequence[int]): the wires the operation acts on
    """
    def __init__(self, wires=None, do_queue=True, id=None):
        super().__init__(qml.PauliZ(wires[1]), wires[0], do_queue=do_queue, id=id)
        self._name = self.__class__.__name__

class CY(Controlled):
    r"""CY(wires)
    The controlled-Y operator

    .. math:: CY = \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0\\
            0 & 0 & 0 & -i\\
            0 & 0 & i & 0
        \end{bmatrix}.

    .. note:: The first wire provided corresponds to the **control qubit**.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 0

    Args:
        wires (Sequence[int]): the wires the operation acts on
    """

    def __init__(self, wires=None, do_queue=True, id=None):
        super().__init__(qml.PauliY(wires[1]), wires[0], do_queue=do_queue, id=id)

class CSWAP(Controlled):
    r"""CSWAP(wires)
    The controlled-swap operator

    .. math:: CSWAP = \begin{bmatrix}
            1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
            0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
        \end{bmatrix}.

    .. note:: The first wire provided corresponds to the **control qubit**.

    **Details:**

    * Number of wires: 3
    * Number of parameters: 0

    Args:
        wires (Sequence[int]): the wires the operation acts on
    """
    def __init__(self, wires=None, do_queue=True, id=None):
        super().__init__(qml.SWAP(wires=wires[1:]), wires[0], do_queue=do_queue, id=id)
        self._name = self.__class__.__name__


class Toffoli(Controlled):
    r"""Toffoli(wires)
    Toffoli (controlled-controlled-X) gate.

    .. math::

        Toffoli =
        \begin{pmatrix}
        1 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
        0 & 1 & 0 & 0 & 0 & 0 & 0 & 0\\
        0 & 0 & 1 & 0 & 0 & 0 & 0 & 0\\
        0 & 0 & 0 & 1 & 0 & 0 & 0 & 0\\
        0 & 0 & 0 & 0 & 1 & 0 & 0 & 0\\
        0 & 0 & 0 & 0 & 0 & 1 & 0 & 0\\
        0 & 0 & 0 & 0 & 0 & 0 & 0 & 1\\
        0 & 0 & 0 & 0 & 0 & 0 & 1 & 0
        \end{pmatrix}

    **Details:**

    * Number of wires: 3
    * Number of parameters: 0

    Args:
        wires (Sequence[int]): the subsystem the gate acts on
    """
    def __init__(self, wires=None, do_queue=True, id=None):
        super().__init__(qml.PauliX(wires[-1]), wires[0:-1], do_queue=do_queue, id=id)
        self._name = self.__class__.__name__


class MultiControlledX(Controlled):
    r"""MultiControlledX(control_wires, wires, control_values)
    Apply a Pauli X gate controlled on an arbitrary computational basis state.

    **Details:**

    * Number of wires: Any (the operation can act on any number of wires)
    * Number of parameters: 0
    * Gradient recipe: None

    Args:
        control_wires (Union[Wires, Sequence[int], or int]): Deprecated way to indicate the control wires.
            Now users should use "wires" to indicate both the control wires and the target wire.
        wires (Union[Wires, Sequence[int], or int]): control wire(s) followed by a single target wire where
            the operation acts on
        control_values (Bool, Iterable[Bool]): The values to control on, denoted by ``0``, ``1``,
            ``True``, or ``False``.  Must be same length as ``control_wires``
        work_wires (Union[Wires, Sequence[int], or int]): optional work wires used to decompose
            the operation into a series of Toffoli gates


    .. note::

        If ``MultiControlledX`` is not supported on the targeted device, PennyLane will decompose
        the operation into :class:`~.Toffoli` and/or :class:`~.CNOT` gates. When controlling on
        three or more wires, the Toffoli-based decompositions described in Lemmas 7.2 and 7.3 of
        `Barenco et al. <https://arxiv.org/abs/quant-ph/9503016>`__ will be used. These methods
        require at least one work wire.

        The number of work wires provided determines the decomposition method used and the resulting
        number of Toffoli gates required. When ``MultiControlledX`` is controlling on :math:`n`
        wires:

        #. If at least :math:`n - 2` work wires are provided, the decomposition in Lemma 7.2 will be
           applied using the first :math:`n - 2` work wires.
        #. If fewer than :math:`n - 2` work wires are provided, a combination of Lemmas 7.3 and 7.2
           will be applied using only the first work wire.

        These methods present a tradeoff between qubit number and depth. The method in point 1
        requires fewer Toffoli gates but a greater number of qubits.

        Note that the state of the work wires before and after the decomposition takes place is
        unchanged.

    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        control_wires=None,
        wires=None,
        control_values=None,
        work_wires=None,
        do_queue=True,
        id=None
    ):
        if control_wires is None:
            super().__init__(qml.PauliX(wires[-1]), wires[0:-1], work_wires=work_wires,
                control_values=control_values, do_queue=do_queue, id=id)
        else:
            warnings.warn(
                "The control_wires keyword will be removed soon. "
                "Use wires = (control_wires, target_wire) instead. "
                "See the documentation for more information.",
                category=UserWarning,
            )
            super().__init__(qml.PauliX(wires), control_wires, work_wires=work_wires,
                control_values=control_values, do_queue=do_queue, id=id)
        self._name = self.__class__.__name__


##### PARAMETRIC OPS #######################


class ControlledPhaseShift(Controlled):
    r"""
    A qubit controlled phase shift.

    .. math:: CR_\phi(\phi) = \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 1 & 0 \\
                0 & 0 & 0 & e^{i\phi}
            \end{bmatrix}.

    .. note:: The first wire provided corresponds to the **control qubit**.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Gradient recipe: :math:`\frac{d}{d\phi}f(CR_\phi(\phi)) = \frac{1}{2}\left[f(CR_\phi(\phi+\pi/2)) - f(CR_\phi(\phi-\pi/2))\right]`
      where :math:`f` is an expectation value depending on :math:`CR_{\phi}(\phi)`.

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int]): the wire the operation acts on
        do_queue (bool): Indicates whether the operator should be
            immediately pushed into the Operator queue (optional)
        id (str or None): String representing the operation (optional)
    """
    grad_method = "A"
    parameter_frequencies = [(1,)]

    def __init__(self, phi, wires, do_queue=True, id=None):
        super().__init__(qml.PhaseShift(phi, wires[-1]), wires[:-1], do_queue=do_queue,
            id=id)
        self._name = self.__class__.__name__

    @staticmethod
    def compute_eigvals(phi):  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.ControlledPhaseShift.eigvals`


        Args:
            phi (tensor_like or float): phase shift

        Returns:
            tensor_like: eigenvalues

        **Example**

        >>> qml.ControlledPhaseShift.compute_eigvals(torch.tensor(0.5))
        tensor([1.0000+0.0000j, 1.0000+0.0000j, 1.0000+0.0000j, 0.8776+0.4794j])
        """
        if qml.math.get_interface(phi) == "tensorflow":
            phi = qml.math.cast_like(phi, 1j)

        exp_part = qml.math.exp(1j * phi)

        return qml.math.stack([1, 1, 1, exp_part]) 

    
CPhase = ControlledPhaseShift

class CRX(Controlled):
    r"""
    The controlled-RX operator

    .. math::

        \begin{align}
            CR_x(\phi) &=
            \begin{bmatrix}
            & 1 & 0 & 0 & 0 \\
            & 0 & 1 & 0 & 0\\
            & 0 & 0 & \cos(\phi/2) & -i\sin(\phi/2)\\
            & 0 & 0 & -i\sin(\phi/2) & \cos(\phi/2)
            \end{bmatrix}.
        \end{align}

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Gradient recipe: The controlled-RX operator satisfies a four-term parameter-shift rule
      (see Appendix F, https://arxiv.org/abs/2104.05695):

      .. math::

          \frac{d}{d\phi}f(CR_x(\phi)) = c_+ \left[f(CR_x(\phi+a)) - f(CR_x(\phi-a))\right] - c_- \left[f(CR_x(\phi+b)) - f(CR_x(\phi-b))\right]

      where :math:`f` is an expectation value depending on :math:`CR_x(\phi)`, and

      - :math:`a = \pi/2`
      - :math:`b = 3\pi/2`
      - :math:`c_{\pm} = (\sqrt{2} \pm 1)/{4\sqrt{2}}`

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int]): the wire the operation acts on
        do_queue (bool): Indicates whether the operator should be
            immediately pushed into the Operator queue (optional)
        id (str or None): String representing the operation (optional)
    """

    grad_method = "A"
    parameter_frequencies = [(0.5, 1.0)]

    def __init__(self, phi, wires, do_queue=True, id=None):
        super().__init__(qml.RX(phi, wires[-1]), wires[:-1],
            do_queue=do_queue, id=id)
        self._name = self.__class__.__name__


class CRY(Controlled):
    r"""
    The controlled-RY operator

    .. math::

        \begin{align}
            CR_y(\phi) &=
            \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0\\
                0 & 0 & \cos(\phi/2) & -\sin(\phi/2)\\
                0 & 0 & \sin(\phi/2) & \cos(\phi/2)
            \end{bmatrix}.
        \end{align}

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Gradient recipe: The controlled-RY operator satisfies a four-term parameter-shift rule
      (see Appendix F, https://arxiv.org/abs/2104.05695):

      .. math::

          \frac{d}{d\phi}f(CR_y(\phi)) = c_+ \left[f(CR_y(\phi+a)) - f(CR_y(\phi-a))\right] - c_- \left[f(CR_y(\phi+b)) - f(CR_y(\phi-b))\right]

      where :math:`f` is an expectation value depending on :math:`CR_y(\phi)`, and

      - :math:`a = \pi/2`
      - :math:`b = 3\pi/2`
      - :math:`c_{\pm} = (\sqrt{2} \pm 1)/{4\sqrt{2}}`

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int]): the wire the operation acts on
        do_queue (bool): Indicates whether the operator should be
            immediately pushed into the Operator queue (optional)
        id (str or None): String representing the operation (optional)
    """
    parameter_frequencies = [(0.5, 1.0)]

    def __init__(self, phi, wires, do_queue=True, id=None):
        super().__init__(qml.RY(phi, wires[-1]), wires[:-1], do_queue=do_queue, id=id)
        self._name = self.__class__.__name__


class CRZ(Controlled):
    r"""
    The controlled-RZ operator

    .. math::

        \begin{align}
             CR_z(\phi) &=
             \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0\\
                0 & 0 & e^{-i\phi/2} & 0\\
                0 & 0 & 0 & e^{i\phi/2}
            \end{bmatrix}.
        \end{align}


    .. note:: The subscripts of the operations in the formula refer to the wires they act on, e.g. 1 corresponds to the first element in ``wires`` that is the **control qubit**.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Gradient recipe: The controlled-RZ operator satisfies a four-term parameter-shift rule
      (see Appendix F, https://arxiv.org/abs/2104.05695):

      .. math::

          \frac{d}{d\phi}f(CR_z(\phi)) = c_+ \left[f(CR_z(\phi+a)) - f(CR_z(\phi-a))\right] - c_- \left[f(CR_z(\phi+b)) - f(CR_z(\phi-b))\right]

      where :math:`f` is an expectation value depending on :math:`CR_z(\phi)`, and

      - :math:`a = \pi/2`
      - :math:`b = 3\pi/2`
      - :math:`c_{\pm} = (\sqrt{2} \pm 1)/{4\sqrt{2}}`

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int]): the wire the operation acts on
        do_queue (bool): Indicates whether the operator should be
            immediately pushed into the Operator queue (optional)
        id (str or None): String representing the operation (optional)
    """
    grad_method = "A"
    parameter_frequencies = [(0.5, 1.0)]

    def __init__(self, phi, wires, do_queue=True, id=None):
        super().__init__(qml.RZ(phi, wires[-1]), wires[-1],
            do_queue=do_queue, id=id)
        self._name = self.__class__.__name__

    @staticmethod
    def compute_eigvals(theta):  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.CRZ.eigvals`


        Args:
            theta (tensor_like or float): rotation angle

        Returns:
            tensor_like: eigenvalues

        **Example**

        >>> qml.CRZ.compute_eigvals(torch.tensor(0.5))
        tensor([1.0000+0.0000j, 1.0000+0.0000j, 0.9689-0.2474j, 0.9689+0.2474j])
        """
        theta = qml.math.flatten(qml.math.stack([theta]))[0]

        if qml.math.get_interface(theta) == "tensorflow":
            theta = qml.math.cast_like(theta, 1j)

        exp_part = qml.math.exp(-0.5j * theta)

        return qml.math.stack([1, 1, exp_part, qml.math.conj(exp_part)])


class CRot(Controlled):
    r"""
    The controlled-Rot operator

    .. math:: CR(\phi, \theta, \omega) = \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0\\
            0 & 0 & e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2)\\
            0 & 0 & e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
        \end{bmatrix}.

    .. note:: The first wire provided corresponds to the **control qubit**.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 3
    * Gradient recipe: The controlled-Rot operator satisfies a four-term parameter-shift rule
      (see Appendix F, https://arxiv.org/abs/2104.05695):

      .. math::

          \frac{d}{d\mathbf{x}_i}f(CR(\mathbf{x}_i)) = c_+ \left[f(CR(\mathbf{x}_i+a)) - f(CR(\mathbf{x}_i-a))\right] - c_- \left[f(CR(\mathbf{x}_i+b)) - f(CR(\mathbf{x}_i-b))\right]

      where :math:`f` is an expectation value depending on :math:`CR(\mathbf{x}_i)`, and

      - :math:`\mathbf{x} = (\phi, \theta, \omega)` and `i` is an index to :math:`\mathbf{x}`
      - :math:`a = \pi/2`
      - :math:`b = 3\pi/2`
      - :math:`c_{\pm} = (\sqrt{2} \pm 1)/{4\sqrt{2}}`

    Args:
        phi (float): rotation angle :math:`\phi`
        theta (float): rotation angle :math:`\theta`
        omega (float): rotation angle :math:`\omega`
        wires (Sequence[int]): the wire the operation acts on
        do_queue (bool): Indicates whether the operator should be
            immediately pushed into the Operator queue (optional)
        id (str or None): String representing the operation (optional)
    """

    grad_method = "A"
    parameter_frequencies = [(0.5, 1.0), (0.5, 1.0), (0.5, 1.0)]

    def __init__(self, phi, theta, omega, wires, do_queue=True, id=None):
        super().__init__(qml.Rot(phi, theta, omega, wires[-1]), wires[:-1],
            do_queue=do_queue, id=id)
        self._name = self.__class__.__name__

