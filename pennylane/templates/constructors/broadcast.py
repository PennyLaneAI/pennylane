# Copyright 2018-2019 Xanadu Quantum Technologies Inc.

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
Contains the ``broadcast`` template constructor.
"""
# pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
from collections import Iterable

from pennylane.templates.decorator import template
from pennylane.templates.utils import _check_wires, _check_type, _get_shape


@template
def Broadcast(block, wires, parameters=None, kwargs={}):
    r"""Applies a (potentially parametrized) single-qubit unitary ``block`` to each wire.

    .. figure:: ../../_static/templates/constructors/broadcast.png
        :align: center
        :width: 20%
        :target: javascript:void(0);

    If the block does not depend on parameters, ``parameters`` is set to ``None``.

    If ``parameters`` is not ``None``, it represents sets of parameters
    :math:`p_m = [\varphi^m_1, \varphi^m_2, ..., \varphi_D^m]`, one set for each wire :math:`m = 1, \dots , M`.
    The block acting on the :math:`m` th wire is :math:`U(\varphi^m_1, \varphi^m_2, ..., \varphi_D^m)`. Hence,
    ``parameters`` must be a list or array of length :math:`M`.

    The block must be a function of a specific signature. It is called
    by :mod:`~.pennylane.templates.constructors.broadcast` as follows:

    .. code-block:: python

        block(parameter1, ... parameterD, wires, **kwargs)

    Therefore, the first :math:`D` positional arguments must be the :math:`D` parameters that are fed into
    the block, and the last positional argument must be ``wires``. If :math:`D=0` (i.e., the block
    is not parametrized), ``wires`` is the *only* positional argument. The ``block`` function
    can take user-defined keyword arguments.

    Typically, ``block`` is either a quantum operation (such as :meth:`~.pennylane.ops.RX`), or a
    user-supplied template (i.e., a sequence of quantum operations). For more details, see *UsageDetails* below.

    Args:
        block (function): quantum gate or template
        parameters (iterable or None): sequence of parameters for each gate applied
        wires (Sequence[int] or int): wire indices that the unitaries act upon

    Raises:
        ValueError: if inputs do not have the correct format

    .. UsageDetails::

        In the simplest case the block is typically an :meth:`~.pennylane.operation.Operation` object
        implementing a single qubit gate.

        .. code-block:: python

            import pennylane as qml
            from pennylane.templates import broadcast

            dev = qml.device('default.qubit', wires=3)

            @qml.qnode(dev)
            def circuit(pars):
                broadcast(block=qml.RX, wires=[0,1,2], parameters=pars)
                return qml.expval(qml.PauliZ(0))

            circuit([1, 1, 2])


        Alternatively, one can use a sequence of gates by creating a template using the
        :meth:`~.pennylane.templates.template` decorator.

        .. code-block:: python

            from pennylane.templates import template

            @template
            def mytemplate(pars, wires):
                qml.Hadamard(wires=wires)
                qml.RY(pars, wires=wires)

            dev = qml.device('default.qubit', wires=3)

            @qml.qnode(dev)
            def circuit(pars):
                broadcast(block=mytemplate, wires=[0,1,2], parameters=pars)
                return qml.expval(qml.PauliZ(0))

            print(circuit([1, 1, 0.1]))

        **Constant unitaries**

        If the ``block`` argument does not take parameters, no ``parameters`` argument is passed to
        :mod:`~.pennylane.templates.constructors.broadcast`:

        .. code-block:: python

            dev = qml.device('default.qubit', wires=3)

            @qml.qnode(dev)
            def circuit():
                broadcast(block=qml.Hadamard, wires=[0,1,2])
                return qml.expval(qml.PauliZ(0))

            circuit()


        **Multiple parameters in block**

        The block, whether it is a single gate or a user-defined template,
        can take multiple parameters. For example:

        .. code-block:: python

            from pennylane.templates import template

            @template
            def mytemplate(pars1, pars2, wires):
                qml.Hadamard(wires=wires)
                qml.RY(pars1, wires=wires)
                qml.RX(pars2, wires=wires)

            @qml.qnode(dev)
            def circuit(pars):
                broadcast(block=mytemplate, wires=[0,1,2], parameters=pars)
                return qml.expval(qml.PauliZ(0))

            circuit([[1, 1], [2, 1], [0.1, 1]])

        As mentioned above, in general the block **must** have the following signature:

        .. code-block:: python

            block(parameter1, parameter2, ... parameterD, wires, **kwargs)

        If ``block`` does not depend on parameters (:math:`D=0`), the signature is

        .. code-block:: python

            block(wires, **kwargs)

        As a result, ``parameters`` must be a list or array of length-:math:`D` lists or arrays.

        If :math:`D` becomes large, the signature can be simplified by wrapping each entry in ``parameters``:

        .. code-block:: python

            @template
            def mytemplate(pars, wires):
                qml.Hadamard(wires=wires)
                qml.RY(pars[0], wires=wires)
                qml.RX(pars[1], wires=wires)

            @qml.qnode(dev)
            def circuit(pars):
                broadcast(block=mytemplate, wires=[0,1,2], parameters=pars)
                return qml.expval(qml.PauliZ(0))

            print(circuit([[[1, 1]], [[2, 1]], [[0.1, 1]]]))

        If the number of parameters for each wire does not match the block, an error gets thrown:

        .. code-block:: python

                @template
                def mytemplate(pars1, pars2, wires):
                    qml.Hadamard(wires=wires)
                    qml.RY(pars1, wires=wires)
                    qml.RX(pars2, wires=wires)

                @qml.qnode(dev)
                def circuit(pars):
                    broadcast(block=mytemplate, wires=[0, 1, 2], parameters=pars)
                    return qml.expval(qml.PauliZ(0))


        >>> circuit([1, 2, 3]))
        TypeError: mytemplate() missing 1 required positional argument: 'pars2'

        **Keyword arguments**

        The block can be a template that takes additional keyword arguments.

        .. code-block:: python

            @template
            def mytemplate(wires, h=True):
                if h:
                    qml.Hadamard(wires=wires)
                qml.T(wires=wires)

            @qml.qnode(dev)
            def circuit(hadamard=None):
                broadcast(block=mytemplate, wires=[0, 1, 2], kwargs={'h': hadamard})
                return qml.expval(qml.PauliZ(0))

            circuit(hadamard=False)

    """

    #########
    # Input checks

    wires = _check_wires(wires)

    _check_type(
        parameters,
        [Iterable, type(None)],
        msg="'parameters' must be either of type None or "
        "Iterable; got {}".format(type(parameters)),
    )

    if parameters is not None:
        shape = _get_shape(parameters)
        if shape[0] != len(wires):
            raise ValueError(
                "'parameters' must contain one entry for each of the {} wires; got shape {}".format(
                    len(wires), shape
                )
            )
        # repackage for consistent unpacking
        if len(shape) == 1:
            parameters = [[p] for p in parameters]
    else:
        parameters = [[] for _ in range(len(wires))]

    #########

    for w, p in zip(wires, parameters):
        block(*p, wires=w, **kwargs)
