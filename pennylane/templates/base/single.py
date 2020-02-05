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
Contains the ``Single`` base template.
"""
#pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
from collections import Iterable

from pennylane.templates.decorator import template
from pennylane.templates.utils import (_check_wires,
                                       _check_type,
                                       _get_shape)


@template
def Single(unitary, wires, parameters=None, kwargs={}):
    """
    Applies ``unitary`` :math:`U` to each wire, feeding it values in ``parameters``.

    .. figure:: ../../_static/single_parametrized.png
        :align: center
        :width: 60%
        :target: javascript:void(0);

    If ``parameters`` is ``None``, the unitary is applied to each wire as it is:

    .. figure:: ../../_static/single_constant.png
        :align: center
        :width: 60%
        :target: javascript:void(0);

    The unitary is typically a ``~.pennylane.operation.Operation'' object representing a single qubit gate, like
    shown in the following example:

    .. code-block:: python

        import pennylane as qml
        from pennylane.templates import Single

        dev = qml.device('default.qubit', wires=3)

        @qml.qnode(dev)
        def circuit(pars):
            Single(unitary=qml.RX, wires=[0,1,2], parameters=pars)
            return qml.expval(qml.PauliZ(0))

        circuit([1, 1, 2])


    Alternatively, one can use a sequence of gates by creating a template, and feeding it into

    .. code-block:: python

        from pennylane.templates import template

        @template
        def mytemplate(pars, wires):
            qml.Hadamard(wires=wires)
            qml.RY(pars, wires=wires)

        dev = qml.device('default.qubit', wires=3)

        @qml.qnode(dev)
        def circuit(pars):
            Single(unitary=mytemplate, wires=[0,1,2], parameters=pars)
            return qml.expval(qml.PauliZ(0))

        print(circuit([1, 1, 0.1]))

    Args:
        unitary (qml.Operation):
        parameters (list or None):
        wires (Sequence[int] or int):

    Raises:
        ValueError: if inputs do not have the correct format

    UsageDetails::

        **Constant unitaries**

        If the ``unitary`` argument does not take parameters, no ``parameters`` argument is passed to the
        ``Single`` template:

        .. code-block:: python

            dev = qml.device('default.qubit', wires=3)

            @qml.qnode(dev)
            def circuit():
                Single(unitary=qml.Hadamard, wires=[0,1,2])
                return qml.expval(qml.PauliZ(0))

            circuit()


        **Multiple parameters in unitary**

        A unitary, whether it is a gate or a template, can take multiple parameters. For example:

        .. code-block:: python

            from pennylane.templates import template

            @template
            def mytemplate(pars1, pars2, wires):
                qml.Hadamard(wires=wires)
                qml.RY(pars1, wires=wires)
                qml.RX(pars2, wires=wires)

            @qml.qnode(dev)
            def circuit(pars):
                Single(unitary=mytemplate, wires=[0,1,2], parameters=pars)
                return qml.expval(qml.PauliZ(0))

            circuit([[1, 1], [2, 1], [0.1, 1]])

        In more general, the unitary **must** have the following signature:

        .. code-block:: python

            unitary(parameter1, parameter2, ... parameterN, wires, **kwargs)

        Note that if ``unitary`` does not depend on parameters (:math:`N=0`), the signature is

        .. code-block:: python

            unitary(wires, **kwargs)

        Overall, ``parameters`` must be a list of length-:math:`N` lists.

        If :math:`N` becomes large, the signature can be simplified by wrapping each entry in ``parameters``:

        .. code-block:: python

            from pennylane.templates import template

            @template
            def mytemplate(pars, wires):
                qml.Hadamard(wires=wires)
                qml.RY(pars[0], wires=wires)
                qml.RX(pars[1], wires=wires)

            @qml.qnode(dev)
            def circuit(pars):
                Single(unitary=mytemplate, wires=[0,1,2], parameters=pars)
                return qml.expval(qml.PauliZ(0))

            print(circuit([[[1, 1]], [[2, 1]], [[0.1, 1]]]))

    If the number of parameters for each wire does not match the template or gate, an error gets thrown:

    .. code-block:: python

            @template
            def mytemplate(pars1, pars2, wires):
                qml.Hadamard(wires=wires)
                qml.RY(pars1, wires=wires)
                qml.RX(pars2, wires=wires)

            @qml.qnode(dev)
            def circuit(pars):
                Single(unitary=mytemplate, wires=[0, 1, 2], parameters=pars)
                return qml.expval(qml.PauliZ(0))


    >>> circuit([1, 2, 3]))
    TypeError: mytemplate() missing 1 required positional argument: 'pars2'

    **Keyword arguments**

    The unitary can be a template that takes additional keyword arguments.

    .. code-block:: python

        @template
        def mytemplate(wires, h=True):
            if h:
                qml.Hadamard(wires=wires)
            qml.T(wires=wires)

        @qml.qnode(dev)
        def circuit(hadamard=None):
            Single(unitary=mytemplate, wires=[0, 1, 2], kwargs={'h': hadamard})
            return qml.expval(qml.PauliZ(0))

        circuit(hadamard=False)

    """

    #########
    # Input checks

    wires = _check_wires(wires)

    _check_type(parameters, [Iterable, type(None)], msg="'parameters' must be either of type None or "
                                                        "Iterable; got {}".format(type(parameters)))
#    _check_type(unitary, [list, FunctionType], msg="unitary must be a ``~.pennylane.operation.Operation`` "
#                                                   "or a template function; got {}".format(type(unitary)))

    if parameters is not None:
        shape = _get_shape(parameters)
        if shape[0] != len(wires):
            raise ValueError("'parameters' must contain one entry for each of the {} wires; got shape {}"
                             .format(len(wires), shape))
        # repackage for consistent unpacking
        if len(shape) == 1:
            parameters = [[p] for p in parameters]
    else:
        parameters = [[] for _ in range(len(wires))]

    #########

    for w, p in zip(wires, parameters):
        unitary(*p, wires=w, **kwargs)


