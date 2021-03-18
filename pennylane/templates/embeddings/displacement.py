# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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
Contains the ``DisplacementEmbedding`` template.
"""
# pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
import pennylane as qml
from pennylane.templates.decorator import template
from pennylane.templates import broadcast
from pennylane.wires import Wires
from pennylane.templates.utils import (
    check_shape,
    check_is_in_options,
    get_shape,
)


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

    if qml.tape_mode_active():
        shape = qml.math.shape(features)
        constants = [c] * shape[0]

        if len(shape) != 1:
            raise ValueError(f"Features must be a one-dimensional tensor; got shape {shape}.")

        n_features = shape[0]
        if n_features != len(wires):
            raise ValueError(f"Features must be of length {len(wires)}; got length {n_features}.")

        if method == "amplitude":
            pars = qml.math.stack([features, constants], axis=1)

        elif method == "phase":
            pars = qml.math.stack([constants, features], axis=1)

        else:
            raise ValueError(f"did not recognize method {method}")

    else:

        expected_shape = (len(wires),)
        check_shape(
            features,
            expected_shape,
            bound="max",
            msg="Features must be of shape {} or smaller; got {}."
            "".format(expected_shape, get_shape(features)),
        )

        check_is_in_options(
            method,
            ["amplitude", "phase"],
            msg="did not recognize option {} for 'method'" "".format(method),
        )

        constants = [c] * len(features)

        if method == "amplitude":
            pars = list(zip(features, constants))

        elif method == "phase":
            pars = list(zip(constants, features))

    return pars


@template
def DisplacementEmbedding(features, wires, method="amplitude", c=0.1):
    r"""Encodes :math:`N` features into the displacement amplitudes :math:`r` or phases :math:`\phi` of :math:`M` modes,
    where :math:`N\leq M`.

    The mathematical definition of the displacement gate is given by the operator

    .. math::
            D(\alpha) = \exp(r (e^{i\phi}\ad -e^{-i\phi}\a)),

    where :math:`\a` and :math:`\ad` are the bosonic creation and annihilation operators.

    ``features`` has to be an array of at most ``len(wires)`` floats. If there are fewer entries in
    ``features`` than wires, the circuit does not apply the remaining displacement gates.

    Args:
        features (tensor_like): Array of features of size (N,)
        wires (Iterable or Wires): Wires that the template acts on. Accepts an iterable of numbers or strings, or
            a Wires object.
        method (str): ``'phase'`` encodes the input into the phase of single-mode displacement, while
            ``'amplitude'`` uses the amplitude
        c (float): value of the phase of all displacement gates if ``execution='amplitude'``, or
            the amplitude of all displacement gates if ``execution='phase'``

    Raises:
        ValueError: if inputs do not have the correct format
    """

    wires = Wires(wires)
    pars = _preprocess(features, wires, method, c)
    broadcast(unitary=qml.Displacement, pattern="single", wires=wires, parameters=pars)
