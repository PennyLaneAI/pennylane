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

from abc import ABC

import pennylane as qml
from pennylane import numpy as np
from pennylane.numpy import random


class TesterQNode(ABC):
    r"""Base class for testing QNodes.

    The following attributes should be defined for each implementation:
    * ``qnode``: a qnode taking a single argument
    * ``input``: an input for the qnode
    * ``output``: result of qnode at input
    * ``input_shape``: the necessary shape for the qnode argument
    * ``output_shape``: the shape of the qnode output


    **Example:**

    >>> tester = qml.templates.tester_qnodes.test_qnode1()

    The object provides a default input and randomly generated inputs

    >>> tester.input
    1.0
    >>> tester.random_input()   
    array(-0.25604702)
    
    You can call it with the default input or randomly generated inputs:

    >>> tester()
    tensor(0.54030231, requires_grad=True)
    >>> tester(tester.input)
    tensor(0.54030231, requires_grad=True)

    You can also reset the random number generator to a specific seed:

    >>> tester.reset_random(seed=42)
    >>> tester.random_input()
    array(0.30471708)
    >>> tester.reset_random(seed=42)
    >>> tester.random_input()
    array(0.30471708)

    """

    def __init__(self, seed=None):
        self.rng = random.default_rng(seed)
    
    def qnode(self, arg):
        raise NotImplementedError

    def random_input(self):
        return self.rng.standard_normal(self.input_shape)

    def reset_random(self, seed=None):
        """ reset random number generator with a specific seed
        """
        self.rng = random.default_rng(seed)

    def __call__(self, arg=None, **kwargs):
        if arg is None:
            return self.qnode(self.input, **kwargs)

        return self.qnode(arg, **kwargs)


class test_qnode1(TesterQNode):
    r""" A bare-bones qnode with a scalar input and scalar output

    .. code-block:: python
        dev = qml.device('default.qubit', wires=1)

        @qml.qnode(dev)
        def circuit(x):
            qml.RY(x, wires=0)
            return qml.expval(qml.PauliZ(0))

    """

    def __init__(self, seed=None):
        super().__init__(seed=seed)

        dev = qml.device('default.qubit', wires=1)

        @qml.qnode(dev)
        def circuit(x):
            qml.RY(x, wires=0)
            return qml.expval(qml.PauliZ(0))

        self.qnode = circuit
        self.input_shape = tuple()
        self.output_shape = tuple()

        self.input = 1.0
        self.output = np.cos(self.input)

class test_qnode2(TesterQNode):
    r""" A qnode composed of ``BasicEntanglerLayers``

    ..code-block:: python
        dev = qml.device('default.qubit', wires=n_wires)

        @qml.qnode(dev)
        def circuit(weights):
            qml.templates.BasicEntanglerLayers(weights=weights, wires=range(n_wires))
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_wires)]

    """

    def __init__(self, seed=None):
        super().__init__(seed=seed)

        n_wires = 5
        n_layers = 2

        dev = qml.device('default.qubit', wires=n_wires)

        @qml.qnode(dev)
        def circuit(weights):
            qml.templates.BasicEntanglerLayers(weights=weights, wires=range(n_wires))
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_wires)]

        self.qnode = circuit
        self.input_shape = (2,5)
        self.output_shape = (5,)

        self.input = np.array([[0.07788672, 0.83725374, 0.09653275, 0.30368225, 0.71876906],
        [0.98153295, 0.74873539, 0.90897408, 0.33720815, 0.24980921]])
        self.output = np.array([0.32013105, 0.29318489, 0.17019053, 0.17737544, 0.18701874])

        
