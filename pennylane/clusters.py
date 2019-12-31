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
"""
Contains the high-level QNode processing.
"""
from collections.abc import Sequence

from pennylane.qnodes import QNode
from pennylane.measure import expval, var, sample
from pennylane.operation import Observable


MEASURE_MAP = {"expval": expval, "var": var, "sample": sample}


def map(template, observables, device, measure="expval", interface="autograd", diff_method="best"):
    """Map a quantum template over a list of observables to create
    a :class:`QNodeCluster`.

    The number of QNodes within the created QNode cluster will match the number
    of observables passed. The device and the measurement type will either be
    applied to all QNodes in the cluster, or can be provided as a list for more
    fine-grained control.

    Note that the template **must** have the following signature:

    .. code-block:: python

        template(*params, wires)

    The template must also act on all wires of the provided device.

    **Example:**

    Let's define a custom two-wire template:

    .. code-block:: python

        def my_template(*params, wires):
            for i in range(2):
                qml.RX(params[i], wires=wires[i])

            qml.CNOT(wires=wires)

    We want to compute the expectation values over the following list
    of two-qubit observables:

    >>> obs_list = [qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.PauliX(1)]

    This can be easily done with the ``map`` function:

    >>> dev = qml.device("default.qubit", wires=2)
    >>> qnodes = qml.map(my_template, obs_list, dev, measure="expval")
    >>> qnodes(0.54, 0.12)
    array([-0.06154835  0.99280864])

    Args:
        template (callable): the ansatz for the circuit before the final measurement step
        observables (Iterable[:class:`~.Observable`]): observables to measure during the
            final step of each circuit
        device (Device, List[Device]): Corresponding device(s) where the resulting
            QNodeCluster should be executed. This can either be a single device, or a list
            of devices of length ``len(observables)``.
        measure (str, List[str]): Measurement(s) to perform. Options include
            :func:`'expval' <~.expval>`, :func:`'var' <~.var>`, and :func`'sample' <~.sample>`.
            This can either be a single measurement type, in which case it is applied
            to all observables, or a list of measurements of length ``len(observables)``.
        interface (str, None): which interface to use for the QNodeCluster.
            This affects the types of objects that can be passed to/returned from the QNode:

            * ``interface='autograd'``: Allows autograd to backpropogate
              through the QNode. The QNode accepts default Python types
              (floats, ints, lists) as well as NumPy array arguments,
              and returns NumPy arrays.

            * ``interface='torch'``: Allows PyTorch to backpropogate
              through the QNode.The QNode accepts and returns Torch tensors.

            * ``interface='tfe'``: Allows TensorFlow in eager mode to backpropogate
              through the QNode.The QNode accepts and returns
              TensorFlow ``tf.Variable`` and ``tf.tensor`` objects.

            * ``None``: The QNode accepts default Python types
              (floats, ints, lists) as well as NumPy array arguments,
              and returns NumPy arrays. It does not connect to any
              machine learning library automatically for backpropagation.

        diff_method (str, None): the method of differentiation to use in the created QNode.

            * ``"best"``: Best available method. Uses the device directly to compute
              the gradient if supported, otherwise will use the analytic parameter-shift
              rule where possible with finite-difference as a fallback.

            * ``"parameter-shift"``: Use the analytic parameter-shift
              rule where possible with finite-difference as a fallback.

            * ``"finite-diff"``: Uses numerical finite-differences.

            * ``None``: a non-differentiable QNode is returned.

    Returns:
        QNodeCluster: a cluster of QNodes executing the circuit template with
        the specified measurements
    """
    if not callable(template):
        raise ValueError(
            "Could not create QNodes. The template is not a callable function."
        )

    if not isinstance(measure, (list, tuple)):
        measure = [measure]*len(observables)

    if not isinstance(device, Sequence):
        device = [device]*len(observables)

    qnodes = QNodeCluster()

    for obs, m, dev in zip(observables, measure, device):
        if not isinstance(obs, Observable):
            raise ValueError("Could not create QNodes. Some or all observables are not valid.")

        def circuit(*params, obs=obs, m=m):
            template(*params, wires=list(range(dev.num_wires)))
            return MEASURE_MAP[m](obs)

        qnode = QNode(circuit, dev, interface=interface, diff_method=diff_method)
        qnodes.append(qnode)

    return qnodes


def apply(func, qnode_cluster):
    return lambda *params: func(qnode_cluster(*params))


def sum(x):
    if x.interface == "tf":
        import tensorflow as tf
        return lambda *params: tf.sum(x(*params))

    elif x.interface == "torch":
        import torch
        return lambda *params: torch.sum(x(*params))

    elif x.interface in ("autograd", "numpy"):
        from autograd import numpy as np
        return lambda *params: np.sum(x(*params))


def _get_dot_func(interface):
    if interface == "tf":
        import tensorflow as tf
        func = lambda a, b: tf.tensordot(a, b, 1)

    elif interface == "torch":
        import torch
        func = torch.dot

    elif interface in ("autograd", "numpy"):
        from autograd import numpy as np
        func = np.dot

    return func


def dot(x, y):
    if isinstance(x, QNodeCluster) and isinstance(y, QNodeCluster):

        if x.interface != y.interface:
            raise ValueError("QNodeClusters have non-matching interfaces")

        return lambda *params: _get_dot_func(x.interface)(x(*params), y(*params))

    if isinstance(x, QNodeCluster):
        return lambda *params: _get_dot_func(x.interface)(x(*params), y)

    if isinstance(y, QNodeCluster):
        return lambda *params: _get_dot_func(y.interface)(x, y(*params))


class QNodeCluster(Sequence):

    def __init__(self, qnodes=None):
        self.qnodes = qnodes or []

    @property
    def interface(self):
        if not self.qnodes:
            return None

        return self.qnodes[0].interface

    def append(self, qnode):
        if self.qnodes and (qnode.interface != self.interface):
            raise ValueError("Could not append QNode. QNode uses the {} interface, "
                             "QNode cluster uses the {} interface".format(qnode.interface, self.interface))

        self.qnodes.append(qnode)

    def __call__(self, *args, **kwargs):
        results = []

        for q in self.qnodes:
            # TODO: allow asynchronous QNode evaluation here
            results.append(q(*args, **kwargs))

        if self.interface == "tf":
            import tensorflow as tf
            return tf.stack(results)
        elif self.interface == "torch":
            import torch
            return torch.stack(results, dim=0)
        elif self.interface in ("autograd", "numpy"):
            from autograd import numpy as np
            return np.stack(results)

        return results

    def __len__(self):
        return len(self.qnodes)

    def __getitem__(self, idx):
        return self.qnodes[idx]
