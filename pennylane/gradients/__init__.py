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
Quantum gradient transforms are strategies for computing the gradient of a quantum
circuit that work by **transforming** the quantum circuit into one or more gradient circuits.
These gradient circuits, once executed and post-processed, return the gradient
of the original circuit.

Examples of quantum gradient transforms include finite-differences and parameter-shift
rules.

This module provides a selection of device-independent, differentiable quantum
gradient transforms. As such, these quantum gradient transforms can be used to
compute the gradients of quantum circuits on both simulators and hardware.

In addition, it also includes an API for writing your own quantum gradient
transforms.

These quantum gradient transforms can be used in two ways:

- Transforming quantum circuits directly
- Registering a quantum gradient strategy for use when performing autodifferentiation
  with a :class:`QNode <pennylane.QNode>`.

Overview
--------

Gradient transforms
^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: api

    finite_diff
    param_shift
    param_shift_cv
    param_shift_hessian

Custom gradients
^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: api

    gradient_transform
    hessian_transform

Utility functions
^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: api

    finite_diff_coeffs
    generate_shifted_tapes
    generate_multishifted_tapes
    generate_shift_rule
    generate_multi_shift_rule
    eigvals_to_frequencies
    compute_vjp
    batch_vjp
    vjp


Registering autodifferentiation gradients
-----------------------------------------

All PennyLane QNodes are automatically differentiable, and can be included
seamlessly within an autodiff pipeline. When creating a :class:`QNode <pennylane.QNode>`, the
strategy for determining the optimal differentiation strategy is *automated*,
and takes into account the circuit, device, autodiff framework, and metadata
(such as whether a finite number of shots are used).

.. code-block:: python

    dev = qml.device("default.qubit", wires=2, shots=1000)

    @qml.qnode(dev, interface="tf")
    def circuit(weights):
        ...

In particular:

- When using a simulator device with exact measurement statistics, backpropagation
  is preferred due to performance and memory improvements.

- When using a hardware device, or a simulator with a finite number of shots,
  a quantum gradient transform---such as the parameter-shift rule---is preferred.

If you would like to specify a particular quantum gradient transform to use
when differentiating your quantum circuit, this can be passed when
creating the QNode:

.. code-block:: python

    @qml.qnode(dev, gradient_fn=qml.gradients.param_shift)
    def circuit(weights):
        ...


When using your preferred autodiff framework to compute the gradient of your
hybrid quantum-classical cost function, the specified gradient transform
for each QNode will be used.

.. note::

    A single cost function may include multiple QNodes, each with their
    own quantum gradient transform registered.


Transforming QNodes
-------------------

Alternatively, quantum gradient transforms can be applied manually to QNodes.

.. code-block:: python

    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def circuit(weights):
        qml.RX(weights[0], wires=0)
        qml.RY(weights[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RX(weights[2], wires=1)
        return qml.probs(wires=1)

>>> weights = np.array([0.1, 0.2, 0.3], requires_grad=True)
>>> circuit(weights)
tensor([0.9658079, 0.0341921], requires_grad=True)
>>> qml.gradients.param_shift(circuit)(weights)
tensor([[-0.04673668, -0.09442394, -0.14409127],
        [ 0.04673668,  0.09442394,  0.14409127]], requires_grad=True)

Comparing this to autodifferentiation:

>>> qml.grad(circuit)(weights)
array([[-0.04673668, -0.09442394, -0.14409127],
       [ 0.04673668,  0.09442394,  0.14409127]])

Quantum gradient transforms can also be applied as decorators to QNodes,
if *only* gradient information is needed. Evaluating the QNode will then
automatically return the gradient:

.. code-block:: python

    dev = qml.device("default.qubit", wires=2)

    @qml.gradients.param_shift
    @qml.qnode(dev)
    def decorated_circuit(weights):
        qml.RX(weights[0], wires=0)
        qml.RY(weights[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RX(weights[2], wires=1)
        return qml.probs(wires=1)

>>> decorated_circuit(weights)
tensor([[-0.04673668, -0.09442394, -0.14409127],
        [ 0.04673668,  0.09442394,  0.14409127]], requires_grad=True)

.. note::

    If your circuit contains any operations not supported by the gradient
    transform, the transform will attempt to automatically decompose the
    circuit into only operations that support gradients.

.. note::

    If you wish to only return the purely **quantum** component of the
    gradient---that is, the gradient of the output with respect to
    **gate** arguments, not QNode arguments---pass ``hybrid=False``
    when applying the transform:

    >>> qml.gradients.param_shift(circuit, hybrid=False)(weights)


Differentiating gradient transforms
-----------------------------------

Gradient transforms are themselves differentiable, allowing higher-order
gradients to be computed:

.. code-block:: python

    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def circuit(weights):
        qml.RX(weights[0], wires=0)
        qml.RY(weights[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RX(weights[2], wires=1)
        return qml.expval(qml.PauliZ(1))

>>> weights = np.array([0.1, 0.2, 0.3], requires_grad=True)
>>> circuit(weights)
tensor(0.9316158, requires_grad=True)
>>> qml.gradients.param_shift(circuit)(weights)  # gradient
array([[-0.09347337, -0.18884787, -0.28818254]])
>>> qml.jacobian(qml.gradients.param_shift(circuit))(weights)  # hessian
array([[[-0.9316158 ,  0.01894799,  0.0289147 ],
        [ 0.01894799, -0.9316158 ,  0.05841749],
        [ 0.0289147 ,  0.05841749, -0.9316158 ]]])


Transforming tapes
------------------

Gradient transforms can be applied to low-level :class:`~.QuantumTape` objects,
a datastructure representing variational quantum algorithms:

.. code-block:: python

    weights = np.array([0.1, 0.2, 0.3], requires_grad=True)

    with qml.tape.JacobianTape() as tape:
        qml.RX(weights[0], wires=0)
        qml.RY(weights[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RX(weights[2], wires=1)
        qml.expval(qml.PauliZ(1))

Unlike when transforming a QNode, transforming a tape directly
will perform no implicit quantum device evaluation. Instead, it returns
the processed tapes, and a post-processing function, which together
define the gradient:

>>> gradient_tapes, fn = qml.gradients.param_shift(tape)
>>> gradient_tapes
[<QuantumTape: wires=[0, 1], params=3>,
 <QuantumTape: wires=[0, 1], params=3>,
 <QuantumTape: wires=[0, 1], params=3>,
 <QuantumTape: wires=[0, 1], params=3>,
 <QuantumTape: wires=[0, 1], params=3>,
 <QuantumTape: wires=[0, 1], params=3>]

This can be useful if the underlying circuits representing the gradient
computation need to be analyzed.

The output tapes can then be evaluated and post-processed to retrieve
the gradient:

>>> dev = qml.device("default.qubit", wires=2)
>>> fn(qml.execute(gradient_tapes, dev, None))
[[-0.09347337 -0.18884787 -0.28818254]]

Note that the post-processing function ``fn`` returned by the
gradient transform is applied to the flat list of results returned
from executing the gradient tapes.


Custom gradient transforms
--------------------------

Using the :class:`~.gradient_transform` decorator, custom gradient transforms
can be created:

.. code-block:: python

    @gradient_transform
    def my_custom_gradient(tape, **kwargs):
        ...
        return gradient_tapes, processing_fn

Once created, a custom gradient transform can be applied directly
to QNodes, or registered as the quantum gradient transform to use
during autodifferentiation.

For more details, please see the :class:`~.gradient_transform`
documentation.
"""
import pennylane as qml

from . import parameter_shift
from . import parameter_shift_cv
from . import parameter_shift_hessian
from . import finite_difference

from .gradient_transform import gradient_transform
from .hessian_transform import hessian_transform
from .finite_difference import finite_diff, finite_diff_coeffs
from .parameter_shift import param_shift
from .parameter_shift_cv import param_shift_cv
from .parameter_shift_hessian import param_shift_hessian
from .vjp import compute_vjp, batch_vjp, vjp
from .vjp_new import compute_vjp_new, batch_vjp_new, vjp_new
from .hamiltonian_grad import hamiltonian_grad
from .general_shift_rules import (
    eigvals_to_frequencies,
    generate_shift_rule,
    generate_multi_shift_rule,
    generate_shifted_tapes,
    generate_multishifted_tapes,
)
