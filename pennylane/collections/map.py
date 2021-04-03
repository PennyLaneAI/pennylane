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
Contains the map function, for mapping templates over observables and devices.
"""
# pylint: disable=too-many-arguments
from collections.abc import Sequence

import pennylane as qml
from pennylane.operation import Observable

from .qnode_collection import QNodeCollection


MEASURE_MAP = {"expval": qml.expval, "var": qml.var, "sample": qml.sample}


def map(
    template,
    observables,
    device,
    measure="expval",
    interface="autograd",
    diff_method="best",
    **kwargs
):
    """Map a quantum template over a list of observables to create
    a :class:`QNodeCollection`.

    The number of QNodes within the created QNode collection will match the number
    of observables passed. The device and the measurement type will either be
    applied to all QNodes in the collection, or can be provided as a list for more
    fine-grained control.

    Args:
        template (callable): the ansatz for the circuit before the final measurement step
            Note that the template **must** have the following signature:

            .. code-block:: python

                template(params, wires, **kwargs)

            where ``params`` are the trainable weights of the variational circuit,
            ``wires`` is a list of integers representing the wires of the device, and
            ``kwargs`` are any additional keyword arguments that need to be passed
            to the template.
        observables (Iterable[:class:`~.Observable`]): observables to measure during the
            final step of each circuit
        device (Device, Sequence[Device]): Corresponding device(s) where the resulting
            QNodeCollection should be executed. This can either be a single device, or a list
            of devices of length ``len(observables)``.
        measure (str, Union(List[str], Tuple[str])): Measurement(s) to perform. Options include
            :func:`'expval' <pennylane.expval>`, :func:`'var' <pennylane.var>`,
            and :func:`'sample' <pennylane.sample>`.
            This can either be a single measurement type, in which case it is applied
            to all observables, or a list of measurements of length ``len(observables)``.
        interface (str, None): which interface to use for the returned QNode collection.
            This affects the types of objects that can be passed to/returned from the QNode.
            Supports all interfaces supported by the :func:`~.qnode` decorator.
        diff_method (str, None): the method of differentiation to use in the created QNodeCollection.
            Supports all differentiation methods supported by the :func:`~.qnode` decorator.

    Returns:
        QNodeCollection: a collection of QNodes executing the circuit template with
        the specified measurements

    **Example:**

    Let's define a custom two-wire template:

    .. code-block:: python

        def my_template(params, wires, **kwargs):
            qml.RX(params[0], wires=wires[0])
            qml.RX(params[1], wires=wires[1])
            qml.CNOT(wires=wires)

    We want to compute the expectation values over the following list
    of two-qubit observables:

    >>> obs_list = [qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.PauliX(1)]

    This can be easily done with the ``map`` function:

    >>> dev = qml.device("default.qubit", wires=2)
    >>> qnodes = qml.map(my_template, obs_list, dev, measure="expval")

    The returned :class:`~.QNodeCollection` can be evaluated, returning the results from each
    mapped QNode as an array:

    >>> params = [0.54, 0.12]
    >>> qnodes(params)
    array([-0.06154835  0.99280864])
    """
    if not callable(template):
        raise ValueError("Could not create QNodes. The template is not a callable function.")

    qnodes = QNodeCollection()

    if not isinstance(device, Sequence):
        # broadcast the single device over all observables
        device = [device] * len(observables)

    if not isinstance(measure, (list, tuple)):
        # broadcast the single measurement over all observables
        measure = [measure] * len(observables)

    for obs, m, dev in zip(observables, measure, device):
        # Generate QNodes from all pairs of observables, measurements, and devices.
        if not isinstance(obs, Observable):
            raise ValueError("Could not create QNodes. Some or all observables are not valid.")

        # Need to convert wires to a list, because
        # the torch/tf interface complains about Wires objects being fed to qnodes
        # TODO: Allow for Wires argument to be passed through here
        wires = dev.wires.tolist()

        # Note: in the following template definition, we pass the observable, measurement,
        # and wires as *default arguments* to named parameters. This is to avoid
        # Python's late binding closure behaviour
        # (see https://docs.python-guide.org/writing/gotchas/#late-binding-closures)
        def circuit(
            params, _obs=obs, _m=m, _wires=wires, **circuit_kwargs
        ):  # pylint: disable=dangerous-default-value, function-redefined
            template(params, wires=_wires, **circuit_kwargs)
            return MEASURE_MAP[_m](_obs)

        qnode = qml.QNode(circuit, dev, interface=interface, diff_method=diff_method, **kwargs)
        qnodes.append(qnode)

    return qnodes
