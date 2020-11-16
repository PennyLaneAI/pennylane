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
Contains the ``SqueezingEmbedding`` template.
"""
# pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
import pennylane as qml
from pennylane.templates.decorator import template
from pennylane.templates import broadcast
from pennylane.templates.utils import (
    check_shape,
    check_is_in_options,
    get_shape,
)
from pennylane.wires import Wires


def _preprocess(features, wires, method, c):
    """Validate and pre-process inputs."""


    if qml.tape_mode_active():

        features = qml.proc.TensorBox(features)
        constants = [c] * len(features)

        if len(features.shape) != 1:
            raise ValueError(f"Features must be one-dimensional; got shape {features.shape}.")

        n_features = features.shape[0]
        if n_features != len(wires):
            raise ValueError(f"Features must be of length {len(wires)}; got length {n_features}.")

        if method == "amplitude":
            pars = features.stack([features, constants], axis=1).data

        elif method == "phase":
            pars = features.stack([constants, features], axis=1).data

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
        features (array): Array of features of size (N,)
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

    broadcast(
        unitary=qml.Squeezing,
        pattern="single",
        wires=wires,
        parameters=pars)

