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
r"""
Contains the ``Interferometer`` template.
"""
import pennylane as qml

# pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
from pennylane.templates.decorator import template
from pennylane.ops import Beamsplitter, Rotation
from pennylane.wires import Wires


def _preprocess(theta, phi, varphi, wires):
    """Validate and pre-process inputs as follows:

    * Check the shape of the three weight tensors.

    Args:
        theta (tensor_like): trainable parameters of the template
        phi (tensor_like): trainable parameters of the template
        varphi (tensor_like): trainable parameters of the template
        wires (Wires): wires that the template acts on

    Returns:
        tuple: shape of varphi tensor
    """

    n_wires = len(wires)
    n_if = n_wires * (n_wires - 1) // 2

    shape = qml.math.shape(theta)
    if shape != (n_if,):
        raise ValueError(f"Theta must be of shape {(n_if,)}; got {shape}.")

    shape = qml.math.shape(phi)
    if shape != (n_if,):
        raise ValueError(f"Phi must be of shape {(n_if,)}; got {shape}.")

    shape_varphi = qml.math.shape(varphi)
    if shape_varphi != (n_wires,):
        raise ValueError(f"Varphi must be of shape {(n_wires,)}; got {shape_varphi}.")

    return shape_varphi


@template
def Interferometer(theta, phi, varphi, wires, mesh="rectangular", beamsplitter="pennylane"):
    r"""General linear interferometer, an array of beamsplitters and phase shifters.

    For :math:`M` wires, the general interferometer is specified by
    providing :math:`M(M-1)/2` transmittivity angles :math:`\theta` and the same number of
    phase angles :math:`\phi`, as well as :math:`M-1` additional rotation
    parameters :math:`\varphi`.

    By specifying the keyword argument ``mesh``, the scheme used to implement the interferometer
    may be adjusted:

    * ``mesh='rectangular'`` (default): uses the scheme described in
      `Clements et al. <https://dx.doi.org/10.1364/OPTICA.3.001460>`__, resulting in a *rectangular* array of
      :math:`M(M-1)/2` beamsplitters arranged in :math:`M` slices and ordered from left
      to right and top to bottom in each slice. The first beamsplitter acts on
      wires :math:`0` and :math:`1`:

      .. figure:: ../../_static/clements.png
          :align: center
          :width: 30%
          :target: javascript:void(0);


    * ``mesh='triangular'``: uses the scheme described in `Reck et al. <https://dx.doi.org/10.1103/PhysRevLett.73.58>`__,
      resulting in a *triangular* array of :math:`M(M-1)/2` beamsplitters arranged in
      :math:`2M-3` slices and ordered from left to right and top to bottom. The
      first and fourth beamsplitters act on wires :math:`M-1` and :math:`M`, the second
      on :math:`M-2` and :math:`M-1`, and the third on :math:`M-3` and :math:`M-2`, and
      so on.

      .. figure:: ../../_static/reck.png
          :align: center
          :width: 30%
          :target: javascript:void(0);

    In both schemes, the network of :class:`~pennylane.ops.Beamsplitter` operations is followed by
    :math:`M` local :class:`~pennylane.ops.Rotation` Operations.

    The rectangular decomposition is generally advantageous, as it has a lower
    circuit depth (:math:`M` vs :math:`2M-3`) and optical depth than the triangular
    decomposition, resulting in reduced optical loss.

    This is an example of a 4-mode interferometer with beamsplitters :math:`B` and rotations :math:`R`,
    using ``mesh='rectangular'``:

    .. figure:: ../../_static/layer_interferometer.png
        :align: center
        :width: 60%
        :target: javascript:void(0);

    .. note::

        The decomposition as formulated in `Clements et al. <https://dx.doi.org/10.1364/OPTICA.3.001460>`__ uses a different
        convention for a beamsplitter :math:`T(\theta, \phi)` than PennyLane, namely:

        .. math:: T(\theta, \phi) = BS(\theta, 0) R(\phi)

        For the universality of the decomposition, the used convention is irrelevant, but
        for a given set of angles the resulting interferometers will be different.

        If an interferometer consistent with the convention from `Clements et al. <https://dx.doi.org/10.1364/OPTICA.3.001460>`__
        is needed, the optional keyword argument ``beamsplitter='clements'`` can be specified. This
        will result in each :class:`~pennylane.ops.Beamsplitter` being preceded by a :class:`~pennylane.ops.Rotation` and
        thus increase the number of elementary operations in the circuit.

    Args:
        theta (tensor_like): size :math:`(M(M-1)/2,)` tensor of transmittivity angles :math:`\theta`
        phi (tensor_like): size :math:`(M(M-1)/2,)` tensor of phase angles :math:`\phi`
        varphi (tensor_like): size :math:`(M,)` tensor of rotation angles :math:`\varphi`
        wires (Iterable or Wires): Wires that the template acts on. Accepts an iterable of numbers or strings, or
            a Wires object.
        mesh (string): the type of mesh to use
        beamsplitter (str): if ``clements``, the beamsplitter convention from
          Clements et al. 2016 (https://dx.doi.org/10.1364/OPTICA.3.001460) is used; if ``pennylane``, the
          beamsplitter is implemented via PennyLane's ``Beamsplitter`` operation.

    Raises:
        ValueError: if inputs do not have the correct format
    """

    wires = Wires(wires)
    M = len(wires)

    shape_varphi = _preprocess(theta, phi, varphi, wires)

    if M == 1:
        # the interferometer is a single rotation
        Rotation(varphi[0], wires=wires[0])
        return

    n = 0  # keep track of free parameters

    if mesh == "rectangular":
        # Apply the Clements beamsplitter array
        # The array depth is N
        for l in range(M):
            for k, (w1, w2) in enumerate(zip(wires[:-1], wires[1:])):
                # skip even or odd pairs depending on layer
                if (l + k) % 2 != 1:
                    if beamsplitter == "clements":
                        Rotation(phi[n], wires=Wires(w1))
                        Beamsplitter(theta[n], 0, wires=Wires([w1, w2]))
                    elif beamsplitter == "pennylane":
                        Beamsplitter(theta[n], phi[n], wires=Wires([w1, w2]))
                    else:
                        raise ValueError(f"did not recognize beamsplitter {beamsplitter}")
                    n += 1

    elif mesh == "triangular":
        # apply the Reck beamsplitter array
        # The array depth is 2*N-3
        for l in range(2 * M - 3):
            for k in range(abs(l + 1 - (M - 1)), M - 1, 2):
                if beamsplitter == "clements":
                    Rotation(phi[n], wires=wires[k])
                    Beamsplitter(theta[n], 0, wires=wires.subset([k, k + 1]))
                elif beamsplitter == "pennylane":
                    Beamsplitter(theta[n], phi[n], wires=wires.subset([k, k + 1]))
                else:
                    raise ValueError(f"did not recognize beamsplitter {beamsplitter} ")
                n += 1
    else:
        raise ValueError(f"did not recognize mesh {mesh}")

    # apply the final local phase shifts to all modes
    for i in range(shape_varphi[0]):
        act_on = wires[i]
        Rotation(varphi[i], wires=act_on)
