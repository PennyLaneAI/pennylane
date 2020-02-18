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
#pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
from collections import Iterable

from pennylane.templates.decorator import template
from pennylane.templates.utils import (_check_wires,
                                       _check_type,
                                       _get_shape)


@template
def broadcast_double(block, wires, parameters=None, even=True, kwargs={}):
    r"""Applies a two-qubit unitary to pairs of wires.

    If ``even`` is ``True``, start with the wire pair ``[0, 1]``, else with the pair ``[1, 2]``.

    .. figure:: ../../_static/templates/constructors/broadcast_double.png
        :align: center
        :width: 40%
        :target: javascript:void(0);

    Each unitary ``block`` applied to a pair of wires may depend on parameters. These are passed as a list
    by the ``parameters`` argument. For :math:`M` wires and ``even=True``, parameters for
    :math:`\lfloor \frac{M}{2} \rfloor` unitaries must be supplied, while for ``even=False``
    there are :math:`\lfloor \frac{M-1}{2} \rfloor` unitaries.

    Typically, ``block`` is either a two-qubit quantum operation (such as :meth:`~.pennylane.ops.CNOT`), or a
    user-supplied template acting on two qubits. For more details, see *UsageDetails* below.

    Args:
        block (function): quantum gate or template
        parameters (iterable or None): sequence of parameters for each gate applied
        even (boolean): if ``True``, pairs are formed starting with the first wire index, else with
                        the second wire index
        wires (Sequence[int] or int): wire indices that the unitaries act upon

    Raises:
        ValueError: if inputs do not have the correct format

    .. UsageDetails::

        In the simplest case the block is typically an :meth:`~.pennylane.operation.Operation` object
        implementing a two-qubit gate. For example, for 4 wires and an ``CRot`` gate, we need 2 sets of 3 parameters:

        .. code-block:: python

            import pennylane as qml
            from pennylane.templates import broadcast_double

            dev = qml.device('default.qubit', wires=4)

            @qml.qnode(dev)
            def circuit(pars):
                broadcast_double(block=qml.CRot, wires=[0,1,2,3], even=True, parameters=pars)
                return qml.expval(qml.PauliZ(0))

            parameters_1 = [1, 2, 3]
            parameters_2 = [-1, 4, 2]
            circuit([parameters_1, parameters_2])

        If ``even=False``, the same example requires :math:`\lfloor \frac{4-1}{2} \rfloor = 1` parameter sets.

        .. code-block:: python

            @qml.qnode(dev)
            def circuit(pars):
                broadcast_double(block=qml.CRot, wires=[0,1,2,3], even=False, parameters=pars)
                return qml.expval(qml.PauliZ(0))

            parameters_1 = [1, 2, 3]
            circuit([parameters_1])

        Alternatively, one can use a sequence of gates by creating a template using the
        :meth:`~.pennylane.templates.template` decorator.

        .. code-block:: python

            from pennylane.templates import template

            @template
            def mytemplate(pars, wires):
                qml.Hadamard(wires=wires[0])
                qml.RY(pars, wires=wires[1])

            dev = qml.device('default.qubit', wires=4)

            @qml.qnode(dev)
            def circuit(pars):
                broadcast_double(block=mytemplate, wires=[0,1,2,3], parameters=pars)
                return qml.expval(qml.PauliZ(0))

            parameters_1 = 1
            parameters_2 = -1
            circuit([parameters_1, parameters_2])

        **Constant unitaries**

        If the ``block`` argument does not take parameters, no ``parameters`` argument is passed to
        :mod:`~.pennylane.templates.constructors.broadcast`:

        .. code-block:: python

            @qml.qnode(dev)
            def circuit():
                broadcast_double(block=qml.CNOT, wires=[0,1,2,3])
                return qml.expval(qml.PauliZ(0))

            circuit()


        In general, a block taking D trainable parameters **must** have the following signature:

        .. code-block:: python

            block(p1, p2, ... pD, wires, **kwargs)

        If ``block`` does not depend on parameters (:math:`D=0`), the signature is

        .. code-block:: python

            block(wires, **kwargs)

        As a result, ``parameters`` must be a list of length-:math:`D` lists.

        If :math:`D` becomes large, the signature can be simplified by wrapping each entry in ``parameters``:

        .. code-block:: python

            @template
            def mytemplate(pars, wires):
                qml.Hadamard(wires=wires[0])
                qml.RY(pars[0], wires=wires[1])
                qml.RX(pars[1], wires=wires[0])

            @qml.qnode(dev)
            def circuit(pars):
                broadcast_double(block=mytemplate, wires=[0,1,2,3], parameters=pars)
                return qml.expval(qml.PauliZ(0))

            parameters_1 = [[1, 1]]
            parameters_2 = [[2, 1]]
            circuit([parameters_1, parameters_2])

        If the number of parameters for a wire pair does not match the block's requirements, an error gets thrown:

        .. code-block:: python

                @template
                def mytemplate(pars1, pars2, wires):
                    qml.Hadamard(wires=wires)
                    qml.RY(pars1, wires=wires)
                    qml.RX(pars2, wires=wires)

                @qml.qnode(dev)
                def circuit(pars):
                    broadcast_double(block=mytemplate, wires=[0, 1, 2, 3], parameters=pars)
                    return qml.expval(qml.PauliZ(0))

        >>> circuit([1, 2, 3])
        TypeError: mytemplate() missing 1 required positional argument: 'pars2'

        **Keyword arguments**

        The block can be a template that takes additional keyword arguments.

        .. code-block:: python

            @template
            def mytemplate(wires, h=True):
                if h:
                    qml.Hadamard(wires=wires[0])
                qml.T(wires=wires[1])

            @qml.qnode(dev)
            def circuit(hadamard=None):
                broadcast_double(block=mytemplate, wires=[0, 1, 2, 3], kwargs={'h': hadamard})
                return qml.expval(qml.PauliZ(0))

            circuit(hadamard=False)

    """

    #########
    # Input checks

    wires = _check_wires(wires)

    _check_type(parameters, [Iterable, type(None)], msg="'parameters' must be either of type None or "
                                                        "Iterable; got {}".format(type(parameters)))

    if even:
        n_pars = len(wires)//2
    else:
        n_pars = (len(wires)-1)//2
    if parameters is not None:
        shape = _get_shape(parameters)
        if shape[0] != n_pars:
            raise ValueError("'parameters' must contain one entry for each of the {} wire pairs; got shape {}"
                             .format(n_pars, shape))
        # repackage for consistent unpacking
        if len(shape) == 1:
            parameters = [[p] for p in parameters]
    else:
        parameters = [[] for _ in range(len(wires))]

    #########

    if even:
        start_with = 0
    else:
        start_with = 1

    # extract the pairs of wires that the block acts on
    wire_pairs = [[wires[i], wires[i+1]] for i in range(start_with, len(wires)-1, 2)]

    for w, p in zip(wire_pairs, parameters):
        block(*p, wires=w, **kwargs)


