# Copyright 2025 Xanadu Quantum Technologies Inc.

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
Contains FTQC/MBQC-specific operations
"""

from pennylane.decomposition import add_decomps, register_resources
from pennylane.operation import Operation
from pennylane.ops import RX, RZ


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

    # pylint: disable = too-many-arguments, too-many-positional-arguments
    def __init__(self, phi, theta, omega, wires, id=None):
        super().__init__(phi, theta, omega, wires=wires, id=id)

    @property
    def resource_params(self) -> dict:
        return {}

    @staticmethod
    def compute_decomposition(phi, theta, omega, wires):  # pylint: disable=arguments-differ
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

        >>> from pennylane.ftqc import RotXZX
        >>> RotXZX.compute_decomposition(1.2, 2.3, 3.4, wires=0)
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


@register_resources({RX: 2, RZ: 1})
def _xzx_decompose(phi, theta, omega, wires, **__):
    RX(phi, wires=wires)
    RZ(theta, wires=wires)
    RX(omega, wires=wires)


add_decomps(RotXZX, _xzx_decompose)
