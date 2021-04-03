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
Contains the ``SqueezingEmbedding`` template.
"""
# pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
import pennylane as qml
from pennylane.templates.decorator import template
from pennylane.templates import broadcast
from pennylane.wires import Wires


def _preprocess(features, wires, method, c):
    """Validate and pre-process inputs as follows:

    * Check that the features tensor is one-dimensional.
    * Check that the first dimension of the features tensor
      has length :math:`n` or less, where :math:`n` is the number of qubits.
    * Create a parameter tensor which combines the features with a tensor of constants.

    Args:
        features (tensor_like): input features to pre-process
        wires (Wires): wires that template acts on
        method (str): indicates whether amplitude or phase encoding is used
        c (float): value of constant

    Returns:
        tensor_like: 2-dimensional tensor containing the features and constants
    """
    shape = qml.math.shape(features)
    constants = [c] * shape[0]

    if len(shape) != 1:
        raise ValueError(f"Features must be one-dimensional; got shape {shape}.")

    n_features = shape[0]
    if n_features != len(wires):
        raise ValueError(f"Features must be of length {len(wires)}; got length {n_features}.")

    if method == "amplitude":
        pars = qml.math.stack([features, constants], axis=1)

    elif method == "phase":
        pars = qml.math.stack([constants, features], axis=1)

    else:
        raise ValueError(f"did not recognize method {method}")

    return pars


@template
def SqueezingEmbedding(features, wires, method="amplitude", c=0.1):
    r"""Encodes :math:`N` features into the squeezing amplitudes :math:`r \geq 0` or phases :math:`\phi \in [0, 2\pi)`
    of :math:`M` modes, where :math:`N\leq M`.

    The mathematical definition of the squeezing gate is given by the operator

    .. math::

        S(z) = \exp\left(\frac{r}{2}\left(e^{-i\phi}\a^2 -e^{i\phi}{\ad}^{2} \right) \right),

    where :math:`\a` and :math:`\ad` are the bosonic creation and annihilation operators.

    ``features`` has to be an iterable of at most ``len(wires)`` floats. If there are fewer entries in
    ``features`` than wires, the circuit does not apply the remaining squeezing gates.

    Args:
        features (tensor_like): Array of features of size (N,)
        wires (Iterable or Wires): Wires that the template acts on. Accepts an iterable of numbers or strings, or
            a Wires object.
        method (str): ``'phase'`` encodes the input into the phase of single-mode squeezing, while
            ``'amplitude'`` uses the amplitude
        c (float): value of the phase of all squeezing gates if ``execution='amplitude'``, or the
            amplitude of all squeezing gates if ``execution='phase'``

    Raises:
        ValueError: if inputs do not have the correct format
    """

    wires = Wires(wires)
    pars = _preprocess(features, wires, method, c)

    broadcast(unitary=qml.Squeezing, pattern="single", wires=wires, parameters=pars)
