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
import copy
import warnings

from pennylane.qnodes import QNode
from pennylane.measure import expval, var, sample
from pennylane.operation import Observable

MEASURE_MAP = {"expval": expval, "var": var, "sample": sample}


def map(template, observables, device, measure="expval", interface="autograd", diff_method="best"):
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
        # device = [copy.deepcopy(device) for _ in range(len(observables))]

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


def apply(func, qnode_collection):
    """Apply a function to the constituent QNodes of a :class:`QNodeCollection`.

    Args:
        func (callable): A function to be applied to the QNodeCollection results.
            This function must be supported by the corresponding QNodeCollection
            interface; i.e., a ``torch`` QNodeCollection can only be acted on functions
            that accept ``torch.tensor`` objects.
        qnode_collection (QNodeCollection): a QNode collection.

    **Example:**

    We can create a QNodeCollection using :func:`~.map`:

    >>> dev = qml.device("default.qubit", wires=2)
    >>> obs_list = [qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.PauliZ(1)]
    >>> qnodes = qml.map(qml.templates.StronglyEntanglingLayers, obs_list, dev, interface="torch")

    As we are using the ``'torch'`` interface, we now apply ``torch.sum``
    to the QNodeCollection:

    >>> cost = qml.apply(torch.sum, qnodes)

    This is a lazy composition --- no QNode evaluation has yet occured. Evaluation
    only occurs when the returned function ``cost`` is evaluated:

    >>> x = qml.init.strong_ent_layers_normal(3, 2)
    >>> cost(x)
    tensor(0.9092, dtype=torch.float64, grad_fn=<SumBackward0>)
    """
    new_func = lambda params, **kwargs: func(qnode_collection(params, **kwargs))
    new_func.interface = qnode_collection.interface
    return new_func


def sum(x):
    """Lazily sum the constituent QNodes of a :class:`QNodeCollection`.

    Args:
        x (QNodeCollection): a QNode collection of independent QNodes.

    .. seealso:: :func:`~.apply`, :func:`~.dot`

    **Example:**

    We can create a QNodeCollection using :func:`~.map`:

    >>> dev = qml.device("default.qubit", wires=2)
    >>> obs_list = [qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.PauliZ(1)]
    >>> qnodes = qml.map(qml.templates.StronglyEntanglingLayers, obs_list, dev, interface="torch")

    For the cost function, we now sum the results of all QNodes in the collection:

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
    the correct dot product function depending on the QNodeCollection
    interface.

    Args:
        interface (str): the interface to get the dot product function for
        x (Sequence): A non-QNodeCollection sequence. If it isn't the correct
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
    r"""Lazily perform the dot product between arrays, tensors, and :class:`QNodeCollection`.

    Using this function, lazy dot products can be computed between two :class:`QNodeCollection`
    objects, or a :class:`QNodeCollection` object and an array/tensor object. In the latter
    case, only one-dimensional arrays/tensors are supported.

    Args:
        x (array or tensor or QNodeCollection): A QNode collection of independent QNodes,
            or an array/tensor object.
        y (array or tensor or QNodeCollection): A QNode collection of independent QNodes,
            or an array/tensor object.

    .. seealso:: :func:`~.apply`, :func:`~.sum`

    **Example:**

    We can create a QNodeCollection using :func:`~.map`:

    >>> dev = qml.device("default.qubit", wires=2)
    >>> obs_list = [qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.PauliZ(1)]
    >>> qnodes = qml.map(qml.templates.StronglyEntanglingLayers, obs_list, dev, interface="torch")

    The returned QNodeCollection contains 2 QNodes, as we mapped the :func:`~.StronglyEntanglingLayers`
    over a list of two observables:

    >>> len(qnodes)
    2

    For the cost function, we now perform the dot product between a vector of coefficients
    and the QNodeCollection:

    >>> coeffs = torch.tensor([0.32, -0.2], dtype=torch.double)
    >>> cost = qml.dot(coeffs, qnodes)

    .. note::

        The ``cost`` function is equivalent to computing :math:`\langle 0 | U(\theta)^\dagger H U(\theta) | 0\rangle`
        where

        * :math:`U(\theta)` is the unitary applied by the strongly entangling layers, and
        * :math:`H = 0.32 X\otimes Z - 0.2 Z \otimes Z`.

    This is a lazy dot product --- no QNode evaluation has yet occured. Evaluation
    only occurs when the returned function ``cost`` is evaluated:

    >>> x = qml.init.strong_ent_layers_normal(3, 2) # generate random parameters
    >>> cost(x)
    tensor(-0.2183, dtype=torch.float64, grad_fn=<DotBackward>)
    """
    if hasattr(x, "interface") and hasattr(y, "interface"):

        if x.interface != y.interface:
            raise ValueError("QNodeCollections have non-matching interfaces")

        interface = x.interface
        fn, _ = _get_dot_func(interface)
        func = lambda params, **kwargs: fn(x(params, **kwargs), y(params, **kwargs))

    elif hasattr(x, "interface"):
        interface = x.interface
        fn, y = _get_dot_func(interface, y)
        func = lambda params, **kwargs: fn(x(params, **kwargs), y)

    elif hasattr(y, "interface"):
        interface = y.interface
        fn, x = _get_dot_func(interface, x)
        func = lambda params, **kwargs: fn(x, y(params, **kwargs))

    else:
        raise ValueError("At least one argument must be a QNodeCollection")

    func.interface = interface
    return func


class QNodeCollection(Sequence):
    """Represents a sequence of independent QNodes that all share the same signature.
    When the collection is evaluated, all QNodes are simultaneously evaluated
    with the same parameters.

    All QNodes within a QNodeCollection **must** use the same interface.

    .. note:: the recommended method of creating a QNodeCollection is via :func:`~.map`.

    Args:
        qnodes (None or List[QNode]): A list of QNodes sharing the same signature.
            If not provided, an empty QNode collection is instantiated.

    .. seealso:: :func:`~.map`, :func:`~.apply`, :func:`~.sum`, :func:`~.dot`

    **Example:**

    A QNodeCollection can be created using a list of existing QNodes:

    >>> qnode = qml.QNodeCollection([qnode1, qnode2])

    Instantiating a QNode collection with no arguments creates an empty collection:

    >>> qnodes = qml.QNodeCollection()
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

    To evaluate a QNodeCollection, simply call the collection, passing the parameters
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

    Creating a QNodeCollection,

    >>> qnodes = qml.QNodeCollection([qnode1, qnode2])

    We can evaluate this QNode collection directly:

    >>> qnodes(0.5643, -0.45)
    [ 7.60844651e-01 -5.55111512e-17  1.00000000e+00]

    where the results from each QNode have been flattened and concatenated
    into a single one-dimensional list.

    .. caution::

        **Asynchronous evaluation**

        By default, the QNodes within the QNode cluster are executed sequentially.

        However, experimental asynchronous support is now available using the
        `Dask <https://dask.org/>`_ parallelism library. This can be activated
        by passing the ``parallel=True`` keyword argument when evaluating the
        QNodeCollection.

        For example, let's create the following two QVM simulation devices:

        >>> qpu1 = qml.device("forest.qvm", device="Aspen-4-4Q-D", shots=1000)
        >>> qpu2 = qml.device("forest.qvm", device="Aspen-7-4Q-B", shots=1000)

        We can create a collection of QNodes with different observables by mapping
        an ansatz over these devices using :func:`~.map`:

        >>> obs_list = [qml.PauliX(0), qml.PauliZ(0) @ qml.PauliZ(1)]
        >>> qnodes = qml.map(qml.templates.StronglyEntanglingLayers, obs_list, [qpu1, qpu2])

        We can now creating some parameters and evaluate the collection:

        >>> params = qml.init.strong_ent_layers_normal(n_layers=4, n_wires=4)
        >>> qnodes(params)
        array([0.046875  , 0.93164062])

        The above collection was executed sequentially. Executing it in parallel:

        >>> qnodes(params, parallel=True)
        array([0.0234375 , 0.92578125])

        We can time both approaches from within IPython or a Jupyter notebook:

        >>> %timeit qnodes(params)
        5.16 s ± 162 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        >>> %timeit qnodes(params, parallel=True)
        2.99 s ± 40.7 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)


    .. warning::

        Asynchronous evaluation is experimental --- please report all bugs and issues
        to our GitHub page. It currently works with all interfaces, however backpropagation
        and gradient computation is limited to Autograd and PyTorch. **Quantum gradients
        using TensorFlow in asynchronous mode is currently not supported**.

        You will find the best speedups when using asynchronous mode when QNodes are to
        be evaluated on external hardware devices or external simulators. **It is not
        advised to use asynchronous mode with** ``default.qubit``**.**
    """

    def __init__(self, qnodes=None):
        self.qnodes = []
        self.extend(qnodes or [])

    @property
    def interface(self):
        """str, None: automatic differentiation interface used by the collection, if any"""
        if not self.qnodes:
            return None

        return self.qnodes[0].interface

    def append(self, qnode):
        """Appends a QNode to the collection. The appended QNode *must* have the same
        interface as the QNode collection."""
        self.extend([qnode])

    def extend(self, qnodes):
        """Extends the collection by a list of QNodes. The appended QNodes *must* have the same
        interface as the QNode collection."""
        if not all(i.interface == qnodes[0].interface for i in qnodes):
            raise ValueError("Provided QNodes do not all use the same interface")

        if self.qnodes and (qnodes[0].interface != self.interface):
            raise ValueError(
                "Interface mismatch. Provided QNodes use the {} interface, "
                "QNode collection uses the {} interface".format(qnodes[0].interface, self.interface)
            )

        self.qnodes.extend(qnodes)

    def evaluate(self, args, kwargs):
        """Evaluate all QNodes in the collection.

        Args:
            args (list): list containing the arguments
                to pass to all internal QNodes
            kwargs (dict): dictionary containing the keyword
                arguments to pass to all internal QNodes

        Returns:
            list: the results from each QNode
        """
        results = []
        _async = kwargs.pop("parallel", False)
        _scheduler = kwargs.pop("scheduler", "threads")

        if _async:
            import dask

            if self.interface == "tf":
                warnings.warn("Parallel execution of QNode clusters is "
                              "an experimental feature, and currently doesn't "
                              "work with TensorFlow backpropagation. Please use "
                              "the PyTorch or Autograd interfaces instead.", UserWarning)

            for q in self.qnodes:
                results.append(dask.delayed(q)(*args, **kwargs))

            return dask.compute(*results, scheduler=_scheduler)

        for q in self.qnodes:
            results.append(q(*args, **kwargs))

        return results

    @staticmethod
    def convert_results(results, interface):
        """Convert a list of results coming from multiple QNodes
        to the object required by each interface for auto-differentiation.

        Internally, this method makes use of ``tf.stack``, ``torch.stack``,
        and ``np.vstack``.

        Args:
            results (list): list containing the results from
                multiple QNodes
            interface (str): the interfaces of the underlying QNodes

        Returns:
            list or array or torch.Tensor or tf.Tensor: the converted
            and stacked results.
        """
        if interface == "tf":
            import tensorflow as tf

            return tf.stack(results)

        if interface == "torch":
            import torch

            return torch.stack(results, dim=0)

        if interface in ("autograd", "numpy"):
            from autograd import numpy as np

            return np.vstack(results)

        return results

    def __call__(self, *args, **kwargs):
        results = self.evaluate(args, kwargs)
        return self.convert_results(results, self.interface)

    def __len__(self):
        return len(self.qnodes)

    def __getitem__(self, idx):
        return self.qnodes[idx]
