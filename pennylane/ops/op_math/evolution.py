# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
This submodule defines the Evolution class.
"""

from copy import copy
from warnings import warn

import pennylane as qp
from pennylane import math
from pennylane.exceptions import GeneratorUndefinedError

from .exp import Exp


class Evolution(Exp):
    r"""Create an exponential operator that defines a generator, of the form :math:`e^{-ix\hat{G}}`

    Args:
        base (~.operation.Operator): The operator to be used as a generator, G.
        param (float): The evolution parameter, x. This parameter is expected not to have
            any complex component.
        id (str): id for the Evolution operator. Default is None.

    Returns:
       :class:`Evolution`: A :class:`~.operation.Operator` representing an operator exponential of the form :math:`e^{-ix\hat{G}}`,
       where x is real.


    **Usage Details**

    In contrast to the general :class:`~.Exp` class, the ``Evolution`` operator :math:`e^{-ix\hat{G}}` is constrained to have a single trainable
    parameter, x. Any parameters contained in the base operator are not trainable. This allows the operator
    to be differentiated with regard to the evolution parameter. Defining a mathematically identical operator
    using the :class:`~.Exp` class will be incompatible with a variety of PennyLane functions that require only a single
    trainable parameter.

    **Example**

    This symbolic operator can be used to make general rotation operators:

    >>> theta = np.array(1.23)
    >>> op = Evolution(qp.X(0), 0.5 * theta)
    >>> qp.math.allclose(op.matrix(), qp.RX(theta, wires=0).matrix())
    True

    Or to define a time evolution operator for a time-independent Hamiltonian:

    >>> H = qp.Hamiltonian([1, 1], [qp.Y(0), qp.X(1)])
    >>> t = 10e-6
    >>> U = Evolution(H, t)

    If the base operator is Hermitian, then the gate can be used in a circuit,
    though it may not be supported by the device and may not be differentiable.

    >>> @qp.qnode(qp.device('default.qubit', wires=1))
    ... def circuit(x):
    ...     qp.ops.Evolution(qp.X(0), 0.5 * x)
    ...     return qp.expval(qp.Z(0))
    >>> print(qp.draw(circuit)(1.23))
    0: ──Exp(-0.61j X)─┤  <Z>

    """

    _name = "Evolution"
    num_params = 1

    def __init__(self, generator, param=1, id=None):
        super().__init__(generator, coeff=-1j * param, id=id)
        self._data = (param,)

    def __repr__(self):
        return (
            f"Evolution({self.coeff} {self.base})"
            if self.base.arithmetic_depth > 0
            else f"Evolution({self.coeff} {self.base.name})"
        )

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, new_data):
        self._data = new_data

    @property
    def param(self):
        """A real coefficient with ``1j`` factored out."""
        return self.data[0]

    @property
    def coeff(self):
        return -1j * self.data[0]

    def label(self, decimals=None, base_label=None, cache=None):
        param = (
            -self.data[0]
            if decimals is None
            else format(math.toarray(-self.data[0]), f".{decimals}f")
        )
        return base_label or f"Exp({param}j {self.base.label(decimals=decimals, cache=cache)})"

    def simplify(self):
        new_base = self.base.simplify()
        if isinstance(new_base, qp.ops.op_math.SProd):
            return Evolution(new_base.base, self.param * new_base.scalar)
        return Evolution(new_base, self.param)

    def pow(self, z):
        return Evolution(self.base, self.param * z)

    @property
    def has_generator(self):
        return not qp.math.real(self.coeff)

    def generator(self):
        r"""Generator of an operator that is in single-parameter-form.

        For example, for operator

        .. math::

            U(\phi) = e^{-i\phi (0.5 Y + Z\otimes X)}

        we get the generator

        >>> U = qp.ops.op_math.Evolution(0.5 * qp.Y(0) + qp.Z(0) @ qp.X(1), 1)
        >>> print(U)
        Evolution(-1j 0.5 * Y(0) + Z(0) @ X(1))
        >>> U.generator()
        -1 * (0.5 * Y(0) + Z(0) @ X(1))

        """
        if not self.base.is_verified_hermitian:
            warn(f"The base {self.base} may not be hermitian.")
        if qp.math.real(self.coeff):
            raise GeneratorUndefinedError(
                f"The operator coefficient {self.coeff} is not imaginary; the expected format is exp(-ixG)."
                f"The generator is not defined."
            )
        return -1 * self.base

    def __copy__(self):
        copied = super().__copy__()
        copied._data = copy(self._data)
        return copied
