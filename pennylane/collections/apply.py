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
Contains the apply function
"""


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

    >>> cost = qml.collections.apply(torch.sum, qnodes)

    This is a lazy composition --- no QNode evaluation has yet occured. Evaluation
    only occurs when the returned function ``cost`` is evaluated:

    >>> shape = qml.templates.StronglyEntanglingLayers.shape(layers=3, qubits=2)
    >>> x = np.random.random(shape)
    >>> cost(x)
    tensor(0.9092, dtype=torch.float64, grad_fn=<SumBackward0>)
    """
    new_func = lambda params, **kwargs: func(qnode_collection(params, **kwargs))
    new_func.interface = qnode_collection.interface
    return new_func
