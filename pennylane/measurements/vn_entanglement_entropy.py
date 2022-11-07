# pylint: disable=protected-access
"""
This module contains the qml.mutual_info measurement.
"""
import pennylane as qml

from .measurements import MeasurementProcess, EntanglementEntropy


def vn_entanglement_entropy(wires0, wires1, log_base=None):
    r"""Entanglement entropy between the subsystems prior to measurement:

    .. math::

        S(\rho_A) = -Tr[\rho_A log \rho_A] = -Tr[\rho_B log \rho_B] = S(\rho_B)

    where :math:`S` is the von Neumann entropy; :math:`\rho_A = Tr_B[\rho_{AB}]` and
    :math:`\rho_B = Tr_A[\rho_{AB}]` are the reduced density matrices for each partition.

    The Von Neumann entanglement entropy is a measure of the degree of quantum entanglement between
    two subsystems constituting a pure bipartite quantum state. The entropy of entanglement is the
    Von Neumann entropy of the reduced density matrix for any of the subsystems. If it is non-zero,
    it indicates the two subsystems are entangled.

    Args:
        wires0 (Sequence[int] or int): the wires of the first subsystem
        wires1 (Sequence[int] or int): the wires of the second subsystem
        log_base (float): Base for the logarithm.

    **Example:**

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(params):
            qml.RY(params, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.vn_entanglement_entropy(wires0=[0], wires1=[1])

    Executing this QNode:

    >>> circuit(np.pi / 2)
    0.69314718

    It is also possible to get the gradient of the previous QNode:

    >>> param = np.array(np.pi / 4, requires_grad=True)
    >>> qml.grad(circuit)(param)
    0.62322524

    .. note::

        Calculating the derivative of :func:`~.vn_entanglement_entropy` is currently supported when
        using the classical backpropagation differentiation method (``diff_method="backprop"``)
        with a compatible device and finite differences (``diff_method="finite-diff"``).

    .. seealso:: :func:`~.vn_entropy`, :func:`pennylane.qinfo.transforms.vn_entanglement_entropy` and :func:`pennylane.math.vn_entanglement_entropy`
    """
    # the subsystems cannot overlap
    if [wire for wire in wires0 if wire in wires1]:
        raise qml.QuantumFunctionError(
            "Subsystems for computing entanglement entropy must not overlap."
        )

    wires0 = qml.wires.Wires(wires0)
    wires1 = qml.wires.Wires(wires1)
    return MeasurementProcess(EntanglementEntropy, wires=[wires0, wires1], log_base=log_base)
