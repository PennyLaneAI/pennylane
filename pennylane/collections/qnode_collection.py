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
Contains the QNodeCollection class.
"""
import warnings

# pylint: disable=too-many-arguments,import-outside-toplevel
from collections.abc import Sequence
from typing import List

from pennylane.qnode import QNode


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
            return qml.var(qml.PauliZ(1))

    Creating a QNodeCollection,

    >>> qnodes = qml.QNodeCollection([qnode1, qnode2])

    We can evaluate this QNode collection directly:

    >>> qnodes(0.5643, -0.45)
    array([0.76084465, 1.        ])

    where the results from each QNode have been flattened and concatenated
    into a single one-dimensional list.

    .. raw:: html

        <h2>Asynchronous evaluation</h2>

    .. warning::

        You will find the best speedups when using asynchronous mode when QNodes are to
        be evaluated on external hardware devices or external simulators. **It is not
        advised at this point to use asynchronous mode with** ``default.qubit`` **.**

    .. warning::

        Asynchronous evaluation is experimental --- please report all bugs and issues
        to our GitHub page. It currently works with all interfaces, however backpropagation
        and gradient computation is limited to Autograd and PyTorch. **Quantum gradients
        using TensorFlow in asynchronous mode is currently not supported**.

    By default, the QNodes within the QNodeCollection are executed sequentially.

    However, experimental asynchronous support is now available using the
    `Dask <https://dask.org/>`_ parallelism library. This can be activated
    by passing the ``parallel=True`` keyword argument when evaluating the
    QNodeCollection.

    For example, let's create the following two QVM simulation devices:

    >>> qpu1 = qml.device("forest.qvm", device="Aspen-4-4Q-D")
    >>> qpu2 = qml.device("forest.qvm", device="Aspen-7-4Q-B")

    We can create a collection of QNodes with different observables by mapping
    an ansatz over these devices using :func:`~.map`:

    >>> obs_list = [qml.PauliX(0), qml.PauliZ(0) @ qml.PauliZ(1)]
    >>> qnodes = qml.map(qml.templates.StronglyEntanglingLayers, obs_list, [qpu1, qpu2])

    We can now create some parameters and evaluate the collection:

    >>> shape = qml.templates.StronglyEntanglingLayers.shape(n_layers=4, n_wires=4)
    >>> params = np.random.random(shape)
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
    """

    def __init__(self, qnodes=None):
        self.qnodes: List[QNode] = []
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
                f"Interface mismatch. Provided QNodes use the {qnodes[0].interface} interface, "
                f"QNode collection uses the {self.interface} interface"
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
        parallel = kwargs.pop("parallel", False)
        _scheduler = kwargs.pop("scheduler", "threads")

        if parallel:
            try:
                import dask
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "Dask must be installed for parallel evaluation. "
                    "\nDask can be installed using pip:"
                    "\n\npip install dask[delayed]"
                ) from e

            if self.interface == "tf":
                warnings.warn(
                    "Parallel execution of QNodeCollections is "
                    "an experimental feature, and currently doesn't "
                    "work with TensorFlow backpropagation. Please use "
                    "the PyTorch or Autograd interfaces instead.",
                    UserWarning,
                )

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

        Internally, this method makes use of ``tf.stack``, ``torch.stack``, ``jnp.stack``,
        and ``np.stack``.

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

        if interface == "jax":
            import jax.numpy as jnp

            return jnp.stack(results)

        if interface in ("autograd", "numpy"):
            from autograd import numpy as np

            return np.stack(results)

        return results

    def __call__(self, *args, **kwargs):
        results = self.evaluate(args, kwargs)
        return self.convert_results(results, self.interface)

    def __len__(self):
        return len(self.qnodes)

    def __getitem__(self, idx):
        return self.qnodes[idx]
