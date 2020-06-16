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
from pennylane.templates.decorator import template
from pennylane.ops import Displacement
from pennylane.templates import broadcast
from pennylane.templates.utils import (
    check_shape,
    check_no_variable,
    check_wires,
    check_is_in_options,
    get_shape,
)


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
        features (array): Array of features of size (N,)
        wires (Sequence[int]): sequence of mode indices that the template acts on
        method (str): ``'phase'`` encodes the input into the phase of single-mode displacement, while
            ``'amplitude'`` uses the amplitude
        c (float): value of the phase of all displacement gates if ``execution='amplitude'``, or
            the amplitude of all displacement gates if ``execution='phase'``

    Raises:
        ValueError: if inputs do not have the correct format
   """

    #############
    # Input checks

    check_no_variable(method, msg="'method' cannot be differentiable")
    check_no_variable(c, msg="'c' cannot be differentiable")

    wires = check_wires(wires)

    expected_shape = (len(wires),)
    check_shape(
        features,
        expected_shape,
        bound="max",
        msg="'features' must be of shape {} or smaller; got {}."
        "".format(expected_shape, get_shape(features)),
    )

    check_is_in_options(
        method,
        ["amplitude", "phase"],
        msg="did not recognize option {} for 'method'" "".format(method),
    )

    #############

    constants = [c] * len(features)

    if method == "amplitude":
        broadcast(
            unitary=Displacement,
            pattern="single",
            wires=wires,
            parameters=list(zip(features, constants)),
        )

    elif method == "phase":
        broadcast(
            unitary=Displacement,
            pattern="single",
            wires=wires,
            parameters=list(zip(constants, features)),
        )
