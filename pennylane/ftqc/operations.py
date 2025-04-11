
from pennylane.math import allclose, is_abstract, requires_grad
from pennylane.operation import Operation
from pennylane.ops import RX, RZ


def _can_replace(x, y):
    """
    Convenience function that returns true if x is close to y and if
    x does not require grad
    """
    return not is_abstract(x) and not requires_grad(x) and allclose(x, y)


class RotXZX(Operation):
    r"""
    Arbitrary single qubit rotation with angles XZX

    .. math::

        R(\phi,\theta,\omega) = RX(\omega)RZ(\theta)RX(\phi)

    **Details:**

    * Number of wires: 1
    * Number of parameters: 3
    * Number of dimensions per parameter: (0, 0, 0)
    * Gradient recipe: :math:`\frac{d}{d\phi}f(R(\phi, \theta, \omega)) = \frac{1}{2}\left[f(R(\phi+\pi/2, \theta, \omega)) - f(R(\phi-\pi/2, \theta, \omega))\right]`
      where :math:`f` is an expectation value depending on :math:`R(\phi, \theta, \omega)`.
      This gradient recipe applies for each angle argument :math:`\{\phi, \theta, \omega\}`.

    .. note::

        If the ``RotXZX`` gate is not supported on the targeted device, PennyLane
        will attempt to decompose the gate into :class:`~.RX` and :class:`~.RZ` gates.

    Args:
        phi (float): rotation angle :math:`\phi`
        theta (float): rotation angle :math:`\theta`
        omega (float): rotation angle :math:`\omega`
        wires (Any, Wires): the wire the operation acts on
        id (str or None): String representing the operation (optional)
    """
    num_wires = 1
    num_params = 3
    """int: Number of trainable parameters that the operator depends on."""

    ndim_params = (0, 0, 0)
    """tuple[int]: Number of dimensions per trainable parameter that the operator depends on."""

    grad_method = "A"
    parameter_frequencies = [(1,), (1,), (1,)]

    def __init__(self, phi, theta, omega, wires, id=None):
        super().__init__(phi, theta, omega, wires=wires, id=id)

    @property
    def resource_params(self) -> dict:
        pass

    @staticmethod
    def compute_decomposition(phi, theta, omega, wires):
        r"""Representation of the operator as a product of other operators (static method). :

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.Rot.decomposition`.

        Args:
            phi (float): rotation angle :math:`\phi`
            theta (float): rotation angle :math:`\theta`
            omega (float): rotation angle :math:`\omega`
            wires (Any, Wires): the wire the operation acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.RotXZX.compute_decomposition(1.2, 2.3, 3.4, wires=0)
        [RX(1.2, wires=[0]), RZ(2.3, wires=[0]), RX(3.4, wires=[0])]

        """
        decomp_ops = [
            RX(phi, wires=wires),
            RZ(theta, wires=wires),
            RX(omega, wires=wires),
        ]
        return decomp_ops

    def adjoint(self):
        phi, theta, omega = self.parameters
        return RotXZX(-omega, -theta, -phi, wires=self.wires)

    def single_qubit_rot_angles(self):
        return self.data