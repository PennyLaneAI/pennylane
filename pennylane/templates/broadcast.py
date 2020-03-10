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
Contains the ``broadcast`` template constructor.
To add a new pattern:
* extend the variables ``OPTIONS``, ``n_parameters`` and ``wire_sequence``,
* update the list in the docstring and add a usage example at the end of the docstring's
  ``UsageDetails`` section,
* add tests to parametrizations in :func:`test_templates_broadcast`.
"""
# pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
from collections import Iterable

from pennylane.templates.decorator import template
from pennylane.templates.utils import _check_wires, _check_type, _get_shape, _check_is_in_options


# helpers to define pattern wire sequences


def wires_ring(wires):
    """Wire sequence for the ring pattern"""

    if len(wires) in [0, 1]:
        return []
    elif len(wires) == 2:
        # exception: for 2 wires ring is set equal to chain,
        # to avoid duplication of gate
        return [[wires[0], wires[1]]]
    else:
        ring = list(wires) + list(wires[0:1])
        sequence = [[ring[i], ring[i + 1]] for i in range(len(wires))]
        return sequence


def wires_pyramid(wires):
    """Wire sequence for the pyramid pattern."""
    sequence = []
    for layer in range(len(wires) // 2):
        temp = wires[layer : len(wires) - layer]
        sequence += [[temp[i], temp[i + 1]] for i in range(0, len(temp) - 1, 2)]
    return sequence


def wires_all_to_all(wires):
    """Wire sequence for the all-to-all pattern"""
    sequence = []
    for i in range(len(wires)):
        for j in range(i + 1, len(wires)):
            sequence += [[wires[i], wires[j]]]
    return sequence


@template
def broadcast(unitary, wires, pattern, parameters=None, kwargs=None):
    r"""Applies a unitary multiple times to a specific pattern of wires.

    The unitary, defined by the argument ``unitary``, is either a quantum operation
    (such as :meth:`~.pennylane.ops.RX`), or a
    user-supplied template. Depending on the chosen pattern, ``unitary`` is applied to a wire or a subset of wires:

    * ``pattern= 'single'`` applies a single-wire unitary to each one of the :math:`M` wires:

      .. figure:: ../../_static/templates/broadcast_single.png
            :align: center
            :width: 20%
            :target: javascript:void(0);

    * ``pattern= 'double'`` applies a two-wire unitary to :math:`\lfloor \frac{M}{2} \rfloor`
      subsequent pairs of wires:

      .. figure:: ../../_static/templates/broadcast_double.png
          :align: center
          :width: 20%
          :target: javascript:void(0);

    * ``pattern= 'double_odd'`` applies a two-wire unitary to :math:`\lfloor \frac{M-1}{2} \rfloor`
      subsequent pairs of wires, starting with the second wire:

      .. figure:: ../../_static/templates/broadcast_double_odd.png
          :align: center
          :width: 20%
          :target: javascript:void(0);

    * ``pattern= 'chain'`` applies a two-wire unitary to all :math:`M-1` neighbouring pairs of wires:

      .. figure:: ../../_static/templates/broadcast_chain.png
          :align: center
          :width: 20%
          :target: javascript:void(0);

    * ``pattern= 'ring'`` applies a two-wire unitary to all :math:`M` neighbouring pairs of wires,
      where the last wire is considered to be a neighbour to the first one:

      .. figure:: ../../_static/templates/broadcast_ring.png
          :align: center
          :width: 20%
          :target: javascript:void(0);

      .. note:: For 2 wires, the ring pattern is automatically replaced by ``pattern = 'chain'`` to avoid
                a mere repetition of the unitary.

    * ``pattern= 'pyramid'`` applies a two-wire unitary to wire pairs shaped in a pyramid declining to the right:

      .. figure:: ../../_static/templates/broadcast_pyramid.png
          :align: center
          :width: 20%
          :target: javascript:void(0);

    * ``pattern= 'all_to_all'`` applies a two-wire unitary to wire pairs that connect all wires to each other:

      .. figure:: ../../_static/templates/broadcast_alltoall.png
          :align: center
          :width: 20%
          :target: javascript:void(0);

    Each ``unitary`` may depend on a different set of parameters. These are passed as a list by the ``parameters``
    argument.

    For more details, see *Usage Details* below.

    Args:
        unitary (func): quantum gate or template
        pattern (str): specifies the wire pattern of the broadcast
        parameters (list): sequence of parameters for each gate applied
        wires (Sequence[int] or int): wire indices that the unitaries act upon
        kwargs (dict): dictionary of auxilliary parameters for ``unitary``

    Raises:
        ValueError: if inputs do not have the correct format

    .. UsageDetails::

        **Broadcasting single gates**

        In the simplest case the unitary is typically an :meth:`~.pennylane.operation.Operation` object
        implementing a quantum gate.

        .. code-block:: python

            import pennylane as qml
            from pennylane import broadcast

            dev = qml.device('default.qubit', wires=3)

            @qml.qnode(dev)
            def circuit(pars):
                broadcast(unitary=qml.RX, pattern="single", wires=[0,1,2], parameters=pars)
                return qml.expval(qml.PauliZ(0))

            circuit([1, 1, 2])

        This is equivalent to the following circuit:

        .. code-block:: python

            @qml.qnode(dev)
            def circuit(pars):
                qml.RX(pars[0], wires=[0])
                qml.RX(pars[1], wires=[1])
                qml.RX(pars[2], wires=[2])
                return qml.expval(qml.PauliZ(0))

            circuit([1, 1, 2])

        **Broadcasting templates**

        Alternatively, one can broadcast a built-in or user-defined template:

        .. code-block:: python

            from pennylane.templates import template

            @template
            def mytemplate(pars, wires):
                qml.Hadamard(wires=wires)
                qml.RY(pars, wires=wires)

            dev = qml.device('default.qubit', wires=3)

            @qml.qnode(dev)
            def circuit(pars):
                broadcast(unitary=mytemplate, pattern="single", wires=[0,1,2], parameters=pars)
                return qml.expval(qml.PauliZ(0))

            print(circuit([1, 1, 0.1]))

        **Constant unitaries**

        If the ``unitary`` argument does not take parameters, no ``parameters`` argument is passed to
        :func:`~.pennylane.broadcast`:

        .. code-block:: python

            dev = qml.device('default.qubit', wires=3)

            @qml.qnode(dev)
            def circuit():
                broadcast(unitary=qml.Hadamard, pattern="single", wires=[0,1,2])
                return qml.expval(qml.PauliZ(0))

            circuit()

        **Multiple parameters in unitary**

        The unitary, whether it is a single gate or a user-defined template,
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
                broadcast(unitary=mytemplate, pattern="single", wires=[0,1,2], parameters=pars)
                return qml.expval(qml.PauliZ(0))

            circuit([[1, 1], [2, 1], [0.1, 1]])

        In general, the unitary takes D parameters and **must** have the following signature:

        .. code-block:: python

            unitary(parameter1, parameter2, ... parameterD, wires, **kwargs)

        If ``unitary`` does not depend on parameters (:math:`D=0`), the signature is

        .. code-block:: python

            unitary(wires, **kwargs)

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
                broadcast(unitary=mytemplate, pattern="single", wires=[0,1,2], parameters=pars)
                return qml.expval(qml.PauliZ(0))

            print(circuit([[[1, 1]], [[2, 1]], [[0.1, 1]]]))

        If the number of parameters for each wire does not match the unitary, an error gets thrown:

        .. code-block:: python

                @template
                def mytemplate(pars1, pars2, wires):
                    qml.Hadamard(wires=wires)
                    qml.RY(pars1, wires=wires)
                    qml.RX(pars2, wires=wires)

                @qml.qnode(dev)
                def circuit(pars):
                    broadcast(unitary=mytemplate, pattern="single", wires=[0, 1, 2], parameters=pars)
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
                broadcast(unitary=mytemplate, pattern="single", wires=[0, 1, 2], kwargs={'h': hadamard})
                return qml.expval(qml.PauliZ(0))

            circuit(hadamard=False)

        **Different patterns**

        The basic usage of the different patterns works as follows:

        * Double pattern

            .. code-block:: python

                dev = qml.device('default.qubit', wires=4)

                @qml.qnode(dev)
                def circuit(pars):
                    broadcast(unitary=qml.CRot, pattern='double',
                              wires=[0,1,2,3], parameters=pars)
                    return qml.expval(qml.PauliZ(0))

                pars1 = [-1, 2.5, 3]
                pars2 = [-1, 4, 2]

                circuit([pars1, pars2])

        * Double-odd pattern

            .. code-block:: python

                dev = qml.device('default.qubit', wires=4)

                @qml.qnode(dev)
                def circuit(pars):
                    broadcast(unitary=qml.CRot, pattern='double_odd',
                              wires=[0,1,2,3], parameters=pars)
                    return qml.expval(qml.PauliZ(0))

                pars1 = [-5.3, 2.3, 3]

                circuit([pars1])

        * Chain pattern

            .. code-block:: python

                dev = qml.device('default.qubit', wires=4)

                @qml.qnode(dev)
                def circuit(pars):
                    broadcast(unitary=qml.CRot, pattern='chain',
                              wires=[0,1,2,3], parameters=pars)
                    return qml.expval(qml.PauliZ(0))

                pars1 = [1.8, 2, 3]
                pars2 = [-1, 3, 1]
                pars3 = [2, -1.2, 4]

                circuit([pars1, pars2, pars3])

        * Ring pattern

          In general, the number of parameter sequences has to match
          the number of wires:

            .. code-block:: python

                dev = qml.device('default.qubit', wires=3)

                @qml.qnode(dev)
                def circuit(pars):
                    broadcast(unitary=qml.CRot, pattern='ring',
                              wires=[0,1,2], parameters=pars)
                    return qml.expval(qml.PauliZ(0))

                pars1 = [1, -2.2, 3]
                pars2 = [-1, 3, 1]
                pars3 = [2.6, 1, 4]

                circuit([pars1, pars2, pars3])

          However, there is an exception for 2 wires, where only one set of parameters is needed.
          This avoids repeating a gate over the
          same wires twice:

            .. code-block:: python

                dev = qml.device('default.qubit', wires=2)


                @qml.qnode(dev)
                def circuit(pars):
                    broadcast(unitary=qml.CRot, pattern='ring',
                              wires=[0,1], parameters=pars)
                    return qml.expval(qml.PauliZ(0))

                pars1 = [-3.2, 2, 1.2]

                circuit([pars1])

        * Pyramid pattern

            .. code-block:: python

                @qml.qnode(dev)
                def circuit(pars):
                    broadcast(unitary=qml.CRot, pattern='pyramid',
                              wires=[0,1,2,3], parameters=pars)
                    return qml.expval(qml.PauliZ(0))

                pars1 = [1.1, 2, 3]
                pars2 = [-1, 3, 1]
                pars3 = [2, 1, 4.2]

                circuit([pars1, pars2, pars3])

        * All-to-all pattern

            .. code-block:: python

                @qml.qnode(dev)
                def circuit(pars):
                    broadcast(unitary=qml.CRot, pattern='ring',
                              wires=[0,1,2,3], parameters=pars)
                    return qml.expval(qml.PauliZ(0))

                pars1 = [1, 2, 3]
                pars2 = [-1, 3, 1]
                pars3 = [2, 1, 4]
                pars4 = [-1, -2, -3]
                pars5 = [2, 1, 4]
                pars6 = [3, -2, -3]

                circuit([pars1, pars2, pars3, pars4, pars5, pars6])
    """

    OPTIONS = ["single", "double", "double_odd", "chain", "ring", "pyramid", "all_to_all"]

    #########
    # Input checks

    wires = _check_wires(wires)

    _check_type(
        parameters,
        [Iterable, type(None)],
        msg="'parameters' must be either of type None or "
        "Iterable; got {}".format(type(parameters)),
    )

    _check_type(
        pattern, [str], msg="'pattern' must be a string; got {}".format(type(pattern)),
    )

    if kwargs is None:
        kwargs = {}

    _check_type(
        kwargs, [dict], msg="'kwargs' must be a dictionary; got {}".format(type(kwargs)),
    )

    _check_is_in_options(
        pattern, OPTIONS, msg="did not recognize option {} for 'pattern'".format(pattern),
    )

    n_parameters = {
        "single": len(wires),
        "double": 0 if len(wires) in [0, 1] else len(wires) // 2,
        "double_odd": 0 if len(wires) in [0, 1] else (len(wires) - 1) // 2,
        "chain": 0 if len(wires) in [0, 1] else len(wires) - 1,
        "ring": 0 if len(wires) in [0, 1] else (1 if len(wires) == 2 else len(wires)),
        "pyramid": 0 if len(wires) in [0, 1] else sum(i + 1 for i in range(len(wires) // 2)),
        "all_to_all": 0 if len(wires) in [0, 1] else len(wires) * (len(wires) - 1) // 2,
    }

    # check that enough parameters for pattern
    if parameters is not None:
        shape = _get_shape(parameters)

        # specific error message for ring edge case of 2 wires
        if (pattern == "ring") and (len(wires) == 2) and (shape[0] != 1):
            raise ValueError(
                "the ring pattern with 2 wires is an exception and only applies one unitary"
            )

        if shape[0] != n_parameters[pattern]:
            raise ValueError(
                "'parameters' must contain entries for {} unitaries; got {} entries".format(
                    n_parameters[pattern], shape[0]
                )
            )

        # repackage for consistent unpacking
        if len(shape) == 1:
            parameters = [[p] for p in parameters]
    else:
        parameters = [[] for _ in range(n_parameters[pattern])]

    #########

    # define wire sequence for patterns
    wire_sequence = {
        "single": wires,
        "double": [[wires[i], wires[i + 1]] for i in range(0, len(wires) - 1, 2)],
        "double_odd": [[wires[i], wires[i + 1]] for i in range(1, len(wires) - 1, 2)],
        "chain": [[wires[i], wires[i + 1]] for i in range(len(wires) - 1)],
        "ring": wires_ring(wires),
        "pyramid": wires_pyramid(wires),
        "all_to_all": wires_all_to_all(wires),
    }

    # broadcast the unitary
    for w, p in zip(wire_sequence[pattern], parameters):
        unitary(*p, wires=w, **kwargs)
