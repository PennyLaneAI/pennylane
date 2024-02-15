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
r"""
Contains the ``broadcast`` template constructor.
To add a new pattern:
* extend the variables ``OPTIONS``, ``n_parameters`` and ``wire_sequence``,
* update the list in the docstring and add a usage example at the end of the docstring's
  ``details`` section,
* add tests to parametrizations in :func:`test_templates_broadcast`.
"""
# pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
import pennylane as qml
from pennylane.wires import Wires

OPTIONS = {"single", "double", "double_odd", "chain", "ring", "pyramid", "all_to_all", "custom"}

###################
# helpers to define pattern wire sequences


def wires_ring(wires):
    """Wire sequence for the ring pattern"""

    if len(wires) in [0, 1]:
        return []

    if len(wires) == 2:
        # deviation from the rule: for 2 wires ring is set equal to chain,
        # to avoid duplication of single gate
        return [wires.subset([0, 1])]

    sequence = [wires.subset([i, i + 1], periodic_boundary=True) for i in range(len(wires))]
    return sequence


def wires_pyramid(wires):
    """Wire sequence for the pyramid pattern."""
    sequence = []
    for layer in range(len(wires) // 2):
        block = wires[layer : len(wires) - layer]
        sequence += [block.subset([i, i + 1]) for i in range(0, len(block) - 1, 2)]
    return sequence


def wires_all_to_all(wires):
    """Wire sequence for the all-to-all pattern"""
    sequence = []
    for i in range(len(wires)):
        for j in range(i + 1, len(wires)):
            sequence += [wires.subset([i, j])]
    return sequence


# define wire sequences for patterns
PATTERN_TO_WIRES = {
    "single": lambda wires: [wires.subset([i]) for i in range(len(wires))],
    "double": lambda wires: [wires.subset([i, i + 1]) for i in range(0, len(wires) - 1, 2)],
    "double_odd": lambda wires: [wires.subset([i, i + 1]) for i in range(1, len(wires) - 1, 2)],
    "chain": lambda wires: [wires.subset([i, i + 1]) for i in range(len(wires) - 1)],
    "ring": wires_ring,
    "pyramid": wires_pyramid,
    "all_to_all": wires_all_to_all,
    "custom": lambda wires: wires,
}

# define required number of parameters
PATTERN_TO_NUM_PARAMS = {
    "single": len,  # Use the length of the given wires.
    "double": lambda wires: 0 if len(wires) in [0, 1] else len(wires) // 2,
    "double_odd": lambda wires: 0 if len(wires) in [0, 1] else (len(wires) - 1) // 2,
    "chain": lambda wires: 0 if len(wires) in [0, 1] else len(wires) - 1,
    "ring": lambda wires: 0 if len(wires) in [0, 1] else (1 if len(wires) == 2 else len(wires)),
    "pyramid": lambda w: 0 if len(w) in [0, 1] else sum(i + 1 for i in range(len(w) // 2)),
    "all_to_all": lambda wires: 0 if len(wires) in [0, 1] else len(wires) * (len(wires) - 1) // 2,
    "custom": lambda wires: len(wires) if wires is not None else None,
}
###################


def _preprocess(parameters, pattern, wires):
    """Validate and pre-process inputs as follows:

    * Check that pattern is recognised, or use default pattern if None.
    * Check the dimension of the parameters
    * Create wire sequence of the pattern.

    Args:
        parameters (tensor_like): trainable parameters of the template
        pattern (str): specifies the wire pattern
        wires (Wires): wires that template acts on

    Returns:
        wire_sequence, parameters: preprocessed pattern and parameters
    """

    if isinstance(pattern, str):
        _wires = wires
        if pattern not in OPTIONS:
            raise ValueError(f"did not recognize pattern {pattern}")
    else:
        # turn custom pattern into list of Wires objects
        _wires = [Wires(w) for w in pattern]
        # set "pattern" to "custom", indicating that custom settings have to be used
        pattern = "custom"

    # check that there are enough parameters for pattern
    if parameters is not None:
        shape = qml.math.shape(parameters)

        # expand dimension so that parameter sets for each unitary can be unpacked
        if len(shape) == 1:
            parameters = qml.math.expand_dims(parameters, 1)

        # specific error message for ring edge case of 2 wires
        if (pattern == "ring") and (len(wires) == 2) and (shape[0] != 1):
            raise ValueError(
                "the ring pattern with 2 wires is an exception and only applies one unitary"
            )
        num_params = PATTERN_TO_NUM_PARAMS[pattern](_wires)
        if shape[0] != num_params:
            raise ValueError(
                f"Parameters must contain entries for {num_params} unitaries; got {shape[0]} entries"
            )

    wire_sequence = PATTERN_TO_WIRES[pattern](_wires)
    return wire_sequence, parameters


def broadcast(unitary, wires, pattern, parameters=None, kwargs=None):
    r"""Applies a unitary multiple times to a specific pattern of wires.

    The unitary, defined by the argument ``unitary``, is either a quantum operation
    (such as :meth:`~.pennylane.ops.RX`), or a
    user-supplied template. Depending on the chosen pattern, ``unitary`` is applied to a wire or a subset of wires:

    * ``pattern="single"`` applies a single-wire unitary to each one of the :math:`M` wires:

      .. figure:: ../../_static/templates/broadcast_single.png
            :align: center
            :width: 20%
            :target: javascript:void(0);

    * ``pattern="double"`` applies a two-wire unitary to :math:`\lfloor \frac{M}{2} \rfloor`
      subsequent pairs of wires:

      .. figure:: ../../_static/templates/broadcast_double.png
          :align: center
          :width: 20%
          :target: javascript:void(0);

    * ``pattern="double_odd"`` applies a two-wire unitary to :math:`\lfloor \frac{M-1}{2} \rfloor`
      subsequent pairs of wires, starting with the second wire:

      .. figure:: ../../_static/templates/broadcast_double_odd.png
          :align: center
          :width: 20%
          :target: javascript:void(0);

    * ``pattern="chain"`` applies a two-wire unitary to all :math:`M-1` neighbouring pairs of wires:

      .. figure:: ../../_static/templates/broadcast_chain.png
          :align: center
          :width: 20%
          :target: javascript:void(0);

    * ``pattern="ring"`` applies a two-wire unitary to all :math:`M` neighbouring pairs of wires,
      where the last wire is considered to be a neighbour to the first one:

      .. figure:: ../../_static/templates/broadcast_ring.png
          :align: center
          :width: 20%
          :target: javascript:void(0);

      .. note:: For 2 wires, the ring pattern is automatically replaced by ``pattern = 'chain'`` to avoid
                a mere repetition of the unitary.

    * ``pattern="pyramid"`` applies a two-wire unitary to wire pairs shaped in a pyramid declining to the right:

      .. figure:: ../../_static/templates/broadcast_pyramid.png
          :align: center
          :width: 20%
          :target: javascript:void(0);

    * ``pattern="all_to_all"`` applies a two-wire unitary to wire pairs that connect all wires to each other:

      .. figure:: ../../_static/templates/broadcast_alltoall.png
          :align: center
          :width: 20%
          :target: javascript:void(0);

    * A custom pattern can be passed by providing a list of wire lists to ``pattern``. The ``unitary`` is applied
      to each set of wires specified in the list.

      .. figure:: ../../_static/templates/broadcast_custom.png
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
        wires (Iterable or Wires): Wires that the template acts on. Accepts an iterable of numbers or strings, or
            a Wires object.
        kwargs (dict): dictionary of auxilliary parameters for ``unitary``

    Raises:
        ValueError: if inputs do not have the correct format

    .. details::
        :title: Usage Details

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
                return qml.expval(qml.Z(0))

            circuit([1, 1, 2])

        This is equivalent to the following circuit:

        .. code-block:: python

            @qml.qnode(dev)
            def circuit(pars):
                qml.RX(pars[0], wires=[0])
                qml.RX(pars[1], wires=[1])
                qml.RX(pars[2], wires=[2])
                return qml.expval(qml.Z(0))

            circuit([1, 1, 2])

        **Broadcasting templates**

        Alternatively, one can broadcast a built-in or user-defined template:

        .. code-block:: python

            def mytemplate(pars, wires):
                qml.Hadamard(wires=wires)
                qml.RY(pars, wires=wires)

            dev = qml.device('default.qubit', wires=3)

            @qml.qnode(dev)
            def circuit(pars):
                broadcast(unitary=mytemplate, pattern="single", wires=[0,1,2], parameters=pars)
                return qml.expval(qml.Z(0))

            print(circuit([1, 1, 0.1]))

        **Constant unitaries**

        If the ``unitary`` argument does not take parameters, no ``parameters`` argument is passed to
        :func:`~.pennylane.broadcast`:

        .. code-block:: python

            dev = qml.device('default.qubit', wires=3)

            @qml.qnode(dev)
            def circuit():
                broadcast(unitary=qml.Hadamard, pattern="single", wires=[0,1,2])
                return qml.expval(qml.Z(0))

            circuit()

        **Multiple parameters in unitary**

        The unitary, whether it is a single gate or a user-defined template,
        can take multiple parameters. For example:

        .. code-block:: python

            def mytemplate(pars1, pars2, wires):
                qml.Hadamard(wires=wires)
                qml.RY(pars1, wires=wires)
                qml.RX(pars2, wires=wires)

            @qml.qnode(dev)
            def circuit(pars):
                broadcast(unitary=mytemplate, pattern="single", wires=[0,1,2], parameters=pars)
                return qml.expval(qml.Z(0))

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

            def mytemplate(pars, wires):
                qml.Hadamard(wires=wires)
                qml.RY(pars[0], wires=wires)
                qml.RX(pars[1], wires=wires)

            @qml.qnode(dev)
            def circuit(pars):
                broadcast(unitary=mytemplate, pattern="single", wires=[0,1,2], parameters=pars)
                return qml.expval(qml.Z(0))

            print(circuit([[[1, 1]], [[2, 1]], [[0.1, 1]]]))

        If the number of parameters for each wire does not match the unitary, an error gets thrown:

        .. code-block:: python

            def mytemplate(pars1, pars2, wires):
                qml.Hadamard(wires=wires)
                qml.RY(pars1, wires=wires)
                qml.RX(pars2, wires=wires)

            @qml.qnode(dev)
            def circuit(pars):
                broadcast(unitary=mytemplate, pattern="single", wires=[0, 1, 2], parameters=pars)
                return qml.expval(qml.Z(0))

        >>> circuit([1, 2, 3]))
        TypeError: mytemplate() missing 1 required positional argument: 'pars2'

        **Keyword arguments**

        The unitary can be a template that takes additional keyword arguments.

        .. code-block:: python

            def mytemplate(wires, h=True):
                if h:
                    qml.Hadamard(wires=wires)
                qml.T(wires=wires)

            @qml.qnode(dev)
            def circuit(hadamard=None):
                broadcast(unitary=mytemplate, pattern="single", wires=[0, 1, 2], kwargs={'h': hadamard})
                return qml.expval(qml.Z(0))

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
                  return qml.expval(qml.Z(0))

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
                  return qml.expval(qml.Z(0))

              pars1 = [-5.3, 2.3, 3]

              circuit([pars1])

        * Chain pattern

          .. code-block:: python

              dev = qml.device('default.qubit', wires=4)

              @qml.qnode(dev)
              def circuit(pars):
                  broadcast(unitary=qml.CRot, pattern='chain',
                            wires=[0,1,2,3], parameters=pars)
                  return qml.expval(qml.Z(0))

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
                  return qml.expval(qml.Z(0))

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
                  return qml.expval(qml.Z(0))

              pars1 = [-3.2, 2, 1.2]

              circuit([pars1])

        * Pyramid pattern

          .. code-block:: python

              dev = qml.device('default.qubit', wires=4)

              @qml.qnode(dev)
              def circuit(pars):
                  broadcast(unitary=qml.CRot, pattern='pyramid',
                            wires=[0,1,2,3], parameters=pars)
                  return qml.expval(qml.Z(0))

              pars1 = [1.1, 2, 3]
              pars2 = [-1, 3, 1]
              pars3 = [2, 1, 4.2]

              circuit([pars1, pars2, pars3])

        * All-to-all pattern

          .. code-block:: python

              dev = qml.device('default.qubit', wires=4)

              @qml.qnode(dev)
              def circuit(pars):
                  broadcast(unitary=qml.CRot, pattern="all_to_all",
                            wires=[0,1,2,3], parameters=pars)
                  return qml.expval(qml.Z(0))

              pars1 = [1, 2, 3]
              pars2 = [-1, 3, 1]
              pars3 = [2, 1, 4]
              pars4 = [-1, -2, -3]
              pars5 = [2, 1, 4]
              pars6 = [3, -2, -3]

              circuit([pars1, pars2, pars3, pars4, pars5, pars6])

        * Custom pattern

          For a custom pattern, the wire lists for each application of the unitary is
          passed to ``pattern``:

          .. code-block:: python

              dev = qml.device('default.qubit', wires=5)

              pattern = [[0, 1], [3, 4]]

              @qml.qnode(dev)
              def circuit():
                  broadcast(unitary=qml.CNOT, pattern=pattern,
                            wires=range(5))
                  return qml.expval(qml.Z(0))

              circuit()

          When using a parametrized unitary, make sure that the number of wire lists in ``pattern`` corresponds to the
          number of parameters in ``parameters``.

          .. code-block:: python

                pattern = [[0, 1], [3, 4]]

                @qml.qnode(dev)
                def circuit(pars):
                    broadcast(unitary=qml.CRot, pattern=pattern,
                              wires=range(5), parameters=pars)
                    return qml.expval(qml.Z(0))

                pars1 = [1, 2, 3]
                pars2 = [-1, 3, 1]
                pars = [pars1, pars2]

                assert len(pars) == len(pattern)

                circuit(pars)
    """
    # We deliberately disable iterating using enumerate here, since
    # it causes a slowdown when iterating over TensorFlow variables.
    # pylint: disable=consider-using-enumerate
    wires = Wires(wires)
    if kwargs is None:
        kwargs = {}

    wire_sequence, parameters = _preprocess(parameters, pattern, wires)

    if parameters is None:
        for i in range(len(wire_sequence)):
            unitary(wires=wire_sequence[i], **kwargs)
    else:
        for i in range(len(wire_sequence)):
            unitary(*parameters[i], wires=wire_sequence[i], **kwargs)
