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
"""
Contains high-level QNode processing functions and classes.
"""
# pylint: disable=too-many-arguments,import-outside-toplevel
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
            QNodeCluster should be executed. This can either be a single device, or a list
            of devices of length ``len(observables)``.
        measure (str, Union(List[str], Tuple[str])): Measurement(s) to perform. Options include
            :func:`'expval' <pennylane.expval>`, :func:`'var' <pennylane.var>`,
            and :func:`'sample' <pennylane.sample>`.
            This can either be a single measurement type, in which case it is applied
            to all observables, or a list of measurements of length ``len(observables)``.
        interface (str, None): which interface to use for the returned QNode cluster.
            This affects the types of objects that can be passed to/returned from the QNode.
            Supports all interfaces supported by the :func:`~.qnode` decorator.
        diff_method (str, None): the method of differentiation to use in the created QNodeCluster.
            Supports all differentiation methods supported by the :func:`~.qnode` decorator.

    Returns:
        QNodeCluster: a cluster of QNodes executing the circuit template with
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

    The returned :class:`~.QNodeCluster` can be evaluated, returning the results from each
    mapped QNode as an array:

    >>> params = [0.54, 0.12]
    >>> qnodes(params)
    array([-0.06154835  0.99280864])
    """
    if not callable(template):
        raise ValueError("Could not create QNodes. The template is not a callable function.")

    qnodes = QNodeCluster()

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

        wires = list(range(dev.num_wires))

        # Note: in the following template definition, we pass the observable, measurement,
        # and wires as *default arguments* to named parameters. This is to avoid
        # Python's late binding closure behaviour
        # (see https://docs.python-guide.org/writing/gotchas/#late-binding-closures)
        def circuit(
            params, _obs=obs, _m=m, _wires=wires, **kwargs
        ):  # pylint: disable=dangerous-default-value, function-redefined
            template(params, wires=_wires, **kwargs)
            return MEASURE_MAP[_m](_obs)

        qnode = QNode(circuit, dev, interface=interface, diff_method=diff_method)
        qnodes.append(qnode)

    return qnodes


def apply(func, qnode_cluster):
    """Apply a function to the constituent QNodes of a :class:`QNodeCluster`.

    Args:
        func (callable): A function to be applied to the QNodeCluster results.
            This function must be supported by the corresponding QNodeCluster
            interface; i.e., a ``torch`` QNodeCluster can only be acted on functions
            that accept ``torch.tensor`` objects.
        qnode_cluster (QNodeCluster): a QNode cluster.

    **Example:**

    We can create a QNodeCluster using :func:`~.map`:

    >>> dev = qml.device("default.qubit", wires=2)
    >>> obs_list = [qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.PauliZ(1)]
    >>> qnodes = qml.map(qml.templates.StronglyEntanglingLayers, obs_list, dev, interface="torch")

    As we are using the ``'torch'`` interface, we now apply ``torch.sum``
    to the QNodeCluster:

    >>> cost = qml.apply(torch.sum, qnodes)

    This is a lazy composition --- no QNode evaluation has yet occured. Evaluation
    only occurs when the returned function ``cost`` is evaluated:

    >>> x = qml.init.strong_ent_layers_normal(3, 2)
    >>> cost(x)
    tensor(0.9092, dtype=torch.float64, grad_fn=<SumBackward0>)
    """
    return lambda params, **kwargs: func(qnode_cluster(params, **kwargs))


def sum(x):
    """Lazily sum the constituent QNodes of a :class:`QNodeCluster`.

    Args:
        x (QNodeCluster): a QNode cluster of independent QNodes.

    .. seealso:: :func:`~.apply`, :func:`~.dot`

    **Example:**

    We can create a QNodeCluster using :func:`~.map`:

    >>> dev = qml.device("default.qubit", wires=2)
    >>> obs_list = [qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.PauliZ(1)]
    >>> qnodes = qml.map(qml.templates.StronglyEntanglingLayers, obs_list, dev, interface="torch")

    For the cost function, we now sum the results of all QNodes in the cluster:

    >>> cost = qml.sum(qnodes)

    This is a lazy summation --- no QNode evaluation has yet occured. Evaluation
    only occurs when the returned function ``cost`` is evaluated:

    >>> x = qml.init.strong_ent_layers_normal(3, 2)
    >>> cost(x)
    tensor(0.9092, dtype=torch.float64, grad_fn=<SumBackward0>)
    """
    if hasattr(x, "interface") and x.interface is not None:
        if x.interface == "tf":
            import tensorflow as tf

            return apply(tf.reduce_sum, x)

        if x.interface == "torch":
            import torch

            return apply(torch.sum, x)

        if x.interface in ("autograd", "numpy"):
            from autograd import numpy as np

            return apply(np.sum, x)

        raise ValueError("Unknown interface {}".format(x.interface))

    import numpy as np

    return apply(np.sum, x)


def _get_dot_func(interface, x=None):
    """Helper function for :func:`~.dot` to determine
    the correct dot product function depending on the QNodeCluster
    interface.

    Args:
        interface (str): the interface to get the dot product function for
        x (Sequence): A non-QNodeCluster sequence. If it isn't the correct
            type for the interface, it is automatically converted.

    Returns:
        tuple[callable, Sequence or torch.Tensor or tf.Variable]: a tuple
        containing the required dot product function, as well as the
        (potentially converted) sequence.
    """
    if interface == "tf":
        import tensorflow as tf

        if x is not None and not isinstance(x, (tf.Tensor, tf.Variable)):
            x = tf.Variable(x, dtype=tf.float64)

        return lambda a, b: tf.tensordot(a, b, 1), x

    if interface == "torch":
        import torch

        if x is not None and not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float64)

        return torch.matmul, x

    if interface in ("autograd", "numpy"):
        from autograd import numpy as np

        if x is not None and not isinstance(x, np.ndarray):
            x = np.array(x)

        return np.dot, x

    if interface is None:
        import numpy as np

        return np.dot, x

    raise ValueError("Unknown interface {}".format(interface))


def dot(x, y):
    r"""Lazily perform the dot product between arrays, tensors, and :class:`QNodeCluster`.

    Using this function, lazy dot products can be computed between two :class:`QNodeCluster`
    objects, or a :class:`QNodeCluster` object and an array/tensor object. In the latter
    case, only one-dimensional arrays/tensors are supported.

    Args:
        x (array or tensor or QNodeCluster): A QNode cluster of independent QNodes,
            or an array/tensor object.
        y (array or tensor or QNodeCluster): A QNode cluster of independent QNodes,
            or an array/tensor object.

    .. seealso:: :func:`~.apply`, :func:`~.sum`

    **Example:**

    We can create a QNodeCluster using :func:`~.map`:

    >>> dev = qml.device("default.qubit", wires=2)
    >>> obs_list = [qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.PauliZ(1)]
    >>> qnodes = qml.map(qml.templates.StronglyEntanglingLayers, obs_list, dev, interface="torch")

    The returned QNodeCluster contains 2 QNodes, as we mapped the :func:`~.StronglyEntanglingLayers`
    over a list of two observables:

    >>> len(qnodes)
    2

    For the cost function, we now perform the dot product between a vector of coefficients
    and the QNodeCluster:

    >>> coeffs = torch.tensor([0.32, -0.2], dtype=torch.double)
    >>> cost = qml.dot(coeffs, qnodes)

    Note that ``cost`` is equivalent to computing :math:`\langle 0 | U(\theta)^\dagger H U(\theta) | 0\rangle`
    where

    * :math:`U(\theta)` is the unitary applied by the strongly entangling layers, and
    * :math:`H = 0.32 X\otimes Z - 0.2 Z \otimes Z`.

    This is a lazy dot product --- no QNode evaluation has yet occured. Evaluation
    only occurs when the returned function ``cost`` is evaluated:

    >>> x = qml.init.strong_ent_layers_normal(3, 2) # generate random parameters
    >>> cost(x)
    tensor(-0.2183, dtype=torch.float64, grad_fn=<DotBackward>)
    """
    if isinstance(x, QNodeCluster) and isinstance(y, QNodeCluster):

        if x.interface != y.interface:
            raise ValueError("QNodeClusters have non-matching interfaces")

        fn, _ = _get_dot_func(x.interface)
        return lambda params, **kwargs: fn(x(params, **kwargs), y(params, **kwargs))

    if isinstance(x, QNodeCluster):
        fn, y = _get_dot_func(x.interface, y)
        return lambda params, **kwargs: fn(x(params, **kwargs), y)

    if isinstance(y, QNodeCluster):
        fn, x = _get_dot_func(y.interface, x)
        return lambda params, **kwargs: fn(x, y(params, **kwargs))

    raise ValueError("At least one argument must be a QNodeCluster")


class QNodeCluster(Sequence):
    """Represents a sequence of independent QNodes that all share the same signature.
    When the cluster is evaluated, all QNodes are simultaneously evaluated
    with the same parameters.

    All QNodes within a QNodeCluster **must** use the same interface.

    .. note:: the recommended method of creating a QNodeCluster is via :func:`~.map`.

    Args:
        qnodes (None or List[QNode]): A list of QNodes sharing the same signature.
            If not provided, an empty QNode cluster is instantiated.

    .. seealso:: :func:`~.map`, :func:`~.apply`, :func:`~.sum`, :func:`~.dot`

    **Example:**

    A QNodeCluster can be created using a list of existing QNodes:

    >>> qnode = qml.QNodeCluster([qnode1, qnode2])

    Instantiating a QNode cluster with no arguments creates an empty cluster:

    >>> qnodes = qml.QNodeCluster()
    >>> len(qnodes)
    0

    QNodes can be appended:

    >>> qnodes.append(qnode1)
    >>> len(qnodes)
    1

    or extended:

    >>> qnodes.extend([qnode2, qnode3])
    >>> len(qnodes)
    3

    They can also be indexed:

    >>> qnodes[0]
    <QNode: device='default.qubit', func=circuit, wires=2, interface=torch>

    or looped over:

    >>> [i.num_wires for i in qnodes]
    [2, 2, 2]

    To evaluate a QNodeCluster, simply call the cluster, passing the parameters
    as required by the constituent QNode. For example, consider the
    following two QNodes with the same signature:

    .. code-block:: python3

        dev1 = qml.device("default.qubit", wires=1)
        dev2 = qml.device("default.qubit", wires=2)

        @qml.qnode(dev1)
        def qnode1(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=0)
            return qml.expval(qml.PauliZ(0))

        @qml.qnode(dev2)
        def qnode2(x, y):
            qml.Hadamard(wires=0)
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.var(qml.PauliZ(1))

    Creating a QNodeCluster,

    >>> qnodes = qml.QNodeCluster([qnode1, qnode2])

    We can evaluate this QNode cluster directly:

    >>> qnodes(0.5643, -0.45)
    [ 7.60844651e-01 -5.55111512e-17  1.00000000e+00]

    where the results from each QNode have been flattened and concatenated
    into a single one-dimensional list.
    """

    def __init__(self, qnodes=None):
        self.qnodes = []
        self.extend(qnodes or [])

    @property
    def interface(self):
        """str, None: automatic differentiation interface used by the cluster, if any"""
        if not self.qnodes:
            return None

        return self.qnodes[0].interface

    def append(self, qnode):
        """Appends a QNode to the cluster. The appended QNode *must* have the same
        interface as the QNode cluster."""
        self.extend([qnode])

    def extend(self, qnodes):
        """Extends the cluster by a list of QNodes. The appended QNodes *must* have the same
        interface as the QNode cluster."""
        if not all(i.interface == qnodes[0].interface for i in qnodes):
            raise ValueError("Provided QNodes do not all use the same interface")

        if self.qnodes and (qnodes[0].interface != self.interface):
            raise ValueError(
                "Interface mismatch. Provided QNodes use the {} interface, "
                "QNode cluster uses the {} interface".format(qnodes[0].interface, self.interface)
            )

        self.qnodes.extend(qnodes)

    def __call__(self, *args, **kwargs):
        results = []

        for q in self.qnodes:
            # TODO: allow asynchronous QNode evaluation here
            results.append(q(*args, **kwargs))

        if self.interface == "tf":
            import tensorflow as tf

            return tf.stack(results)

        if self.interface == "torch":
            import torch

            return torch.stack(results, dim=0)

        if self.interface in ("autograd", "numpy"):
            from autograd import numpy as np

            return np.vstack(results)

        return results

    def __len__(self):
        return len(self.qnodes)

    def __getitem__(self, idx):
        return self.qnodes[idx]
