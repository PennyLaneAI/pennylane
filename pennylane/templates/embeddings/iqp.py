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
Contains the ``IQPEmbedding`` template.
"""
# pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
from collections import Sequence
from pennylane.templates.decorator import template
from pennylane.ops import RZ, CNOT, Hadamard
from pennylane.templates import broadcast
from pennylane.templates.utils import (
    check_shape,
    check_wires,
    check_type,
    get_shape,
)



@template
def zz(parameter, wires):
    """Template for decomposition of ZZ coupling.

    Args:
        parameter (float): parameter of z rotation
        wires (list[int]): qubit indices that the template acts on
    """
    CNOT(wires=wires)
    RZ(2 * parameter, wires=wires[0])
    CNOT(wires=wires)


@template
def IQPEmbedding(features, n_repeats, wires, pattern=None):
    r"""
    Encodes :math:`N` features into :math:`n=N` qubits using diagonal gates of an IQP circuit, as proposed by
    `Havlicek et al. <https://arxiv.org/pdf/1804.11326.pdf>`_.

    An IQP circuit is a quantum circuit of a block of Hadamards, followed by a block of gates that are
    diagonal in the computational basis. Here, the diagonal gates are chosen as
    two-qubit ZZ interactions :math:`e^{-i x_i \sigma_z \otimes \sigma_z}` .

    * If ``pattern`` is None, the default pattern will be used, in which the entangling gates connect successive
      pairs of neighbours:

        |

        .. figure:: ../../_static/templates/iqp.png
            :align: center
            :width: 60%
            :target: javascript:void(0);

        |

    * Else, pattern is a dictionary of the form {"i": (a, b)}, indicating that ``feature[i]`` will be applied by a
      ZZ gate acting on wires ``[a, b]``. Since diagonal gates commute, the order in which the features are applied
      does not play a role.

    Args:
        features (array): array of features to encode
        n_repeats (int):
        wires (Sequence[int] or int): qubit indices that the template acts on
        pattern (dict): if provided, the features are encoded in entanglers arranged in the pattern specified in the dict

    Raises:
        ValueError: if inputs do not have the correct format

    .. UsageDetails::

        .. code-block:: python

            from pennylane import IQPEmbedding

            features = [1., 2.]





    """
    #############
    # Input checks

    wires = check_wires(wires)

    msg = "n_repeats must be a positive integer"
    check_type(n_repeats, [int], msg=msg)
    msg = "features must be a sequence of floats"
    check_type(features, [Sequence], msg=msg)
    for f in features:
        check_type(f, [int], msg=msg)

    if pattern is None:
        # ring pattern has only one entangler for two wires
        if len(features) == 2:
            expected_shape = (1,)
        else:
            expected_shape = (len(wires),)
    else:
        expected_shape = (len(pattern),)

    check_shape(
        features,
        expected_shape,
        msg="'features' must be of shape {}; got {}"
            "".format((len(wires),), get_shape(features)),
    )

    #####################

    for i in range(n_repeats):

        # first block of hadamards
        broadcast(unitary=Hadamard, pattern="single")

        # diagonal gates encoding the features
        if pattern is None:

            if len(wires) == 1:
                # for one wire, use a z rotation
                RZ(features[0], wires=0)
            else:
                # use ring pattern of zz entanglers
                broadcast(unitary=zz, pattern="ring")
        else:
            broadcast(unitary=zz, pattern=pattern)
