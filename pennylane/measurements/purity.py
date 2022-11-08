"""
This module contains the purity measurement.
"""

from pennylane.wires import Wires
from .measurements import MeasurementProcess, Purity


def purity(wires):
    r"""Purity of the system prior to measurement.

    .. math::
        \gamma = Tr(\rho^2)

    where :math:`\rho` is the density matrix. The purity of a normalized quantum state satisfies
    :math:`\frac{1}{d} \leq \gamma \leq 1`, where :math:`d` is the dimension of the Hilbert space.
    A pure state has a :math:`\gamma` of 1.

    It is possible to compute the purity of a sub-system from a given state. To find the purity of
    the overall state, include all wires in the "wires" argument.

    Args:
        wires (Sequence[int] or int): The wires of the subsystem

    **Example:**

    .. code-block:: python

        dev = qml.device("default.mixed", wires=2)

        @qml.qnode(dev)
        def noisy_circuit_purity(p):
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.BitFlip(p, wires=0)
            qml.BitFlip(p, wires=1)
            return qml.purity(wires=[0, 1])

        @qml.qnode(dev)
        def circuit_purity(x):
            qml.IsingXX(x, wires=[0, 1])
            return purity(wires=[0])

    >>> noisy_circuit_purity(0.2)
    0.5648000000000398

    >>> circuit_purity(np.pi / 2)
    0.5

    It is also possible to get the gradient of the previous QNode:

    >>> param = np.array(np.pi / 4, requires_grad=True)
    >>> qml.grad(circuit_purity)(param)
    -0.5

    .. seealso:: :func:`pennylane.qinfo.transforms.purity` and :func:`pennylane.math.purity`
    """
    wires = Wires(wires)
    return MeasurementProcess(Purity, wires=wires)
