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
"""
Base class for Channels.
"""

import abc

import numpy as np

from .base import Operation


class Channel(Operation, abc.ABC):
    r"""Base class for quantum channels.

    Quantum channels have to define an additional numerical representation
    as Kraus matrices.

    Args:
        params (tuple[tensor_like]): trainable parameters
        wires (Iterable[Any] or Any): Wire label(s) that the operator acts on.
            If not given, args[-1] is interpreted as wires.
    """

    @staticmethod
    @abc.abstractmethod
    def compute_kraus_matrices(*params, **hyperparams) -> list[np.ndarray]:
        """Kraus matrices representing a quantum channel, specified in
        the computational basis.

        This is a static method that should be defined for all
        new channels, and which allows matrices to be computed
        directly without instantiating the channel first.

        To return the Kraus matrices of an *instantiated* channel,
        please use the :meth:`~.Operator.kraus_matrices()` method instead.

        .. note::
            This method gets overwritten by subclasses to define the kraus matrix representation
            of a particular operator.

        Args:
            *params (list): trainable parameters of the operator, as stored in the ``parameters`` attribute
            **hyperparams (dict): non-trainable hyperparameters of the operator,
                as stored in the ``hyperparameters`` attribute

        Returns:
            list (array): list of Kraus matrices

        **Example**

        >>> qp.AmplitudeDamping.compute_kraus_matrices(0.1)
        [array([[1.       , 0.       ],
                [0.       , 0.9486833]]),
         array([[0.        , 0.31622777],
                [0.        , 0.        ]])]
        """
        raise NotImplementedError

    def kraus_matrices(self):
        r"""Kraus matrices of an instantiated channel
        in the computational basis.

        Returns:
            list (array): list of Kraus matrices

        ** Example**

        >>> U = qp.AmplitudeDamping(0.1, wires=1)
        >>> U.kraus_matrices()
        [array([[1.       , 0.       ],
                [0.       , 0.9486833]]),
         array([[0.        , 0.31622777],
                [0.        , 0.        ]])]
        """
        return self.compute_kraus_matrices(*self.parameters, **self.hyperparameters)
