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
    spsa_grad
    hadamard_grad
    stoch_pulse_grad
    pulse_odegen

Metric tensors
^^^^^^^^^^^^^^
.. currentmodule:: pennylane

.. autosummary::
    :toctree: api

    metric_tensor
    adjoint_metric_tensor

.. currentmodule:: pennylane.gradients


Utility functions
^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: api

    finite_diff_coeffs
    finite_diff_jvp
    generate_shifted_tapes
    generate_multishifted_tapes
    generate_shift_rule
    generate_multi_shift_rule
    eigvals_to_frequencies
    compute_vjp_single
    compute_vjp_multi
    batch_vjp
    vjp
    compute_jvp_single
    compute_jvp_multi
    batch_jvp
    jvp
    classical_jacobian
    classical_fisher
    quantum_fisher


Registering autodifferentiation gradients
-----------------------------------------

All PennyLane QNodes are automatically differentiable, and can be included
seamlessly within an autodiff pipeline. When creating a :class:`QNode <pennylane.QNode>`, the
strategy for determining the optimal differentiation strategy is *automated*,
and takes into account the circuit, device, autodiff framework, and metadata
(such as whether a finite number of shots are used).

.. code-block:: python

    dev = qml.device("default.qubit")

    @partial(qml.set_shots, shots=1000)
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

    @qml.qnode(dev, diff_method=qml.gradients.param_shift)
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

Alternatively, quantum gradient transforms can be applied manually to QNodes. This is not
recommended because PennyLane must compute the classical Jacobian of the parameters and multiply it with
the quantum Jacobian, we recommend using the ``diff_method`` kwargs with your favorite machine learning
framework.

.. code-block:: python

    dev = qml.device("default.qubit")

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

>>> qml.jacobian(circuit)(weights)
array([[-0.04673668, -0.09442394, -0.14409127],
       [ 0.04673668,  0.09442394,  0.14409127]])

Quantum gradient transforms can also be applied as decorators to QNodes,
if *only* gradient information is needed. Evaluating the QNode will then
automatically return the gradient:

.. code-block:: python

    dev = qml.device("default.qubit")

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
    (tensor([-0.04673668,  0.04673668], requires_grad=True),
     tensor([-0.09442394,  0.09442394], requires_grad=True),
     tensor([-0.14409127,  0.14409127], requires_grad=True))


Differentiating gradient transforms and higher-order derivatives
----------------------------------------------------------------

Gradient transforms are themselves differentiable, allowing higher-order
gradients to be computed:

.. code-block:: python

    dev = qml.device("default.qubit")

    @qml.qnode(dev)
    def circuit(weights):
        qml.RX(weights[0], wires=0)
        qml.RY(weights[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RX(weights[2], wires=1)
        return qml.expval(qml.Z(1))

>>> weights = np.array([0.1, 0.2, 0.3], requires_grad=True)
>>> circuit(weights)
tensor(0.9316158, requires_grad=True)
>>> qml.gradients.param_shift(circuit)(weights)  # gradient
tensor([-0.09347337, -0.18884787, -0.28818254], requires_grad=True)
>>> def stacked_output(weights):
...     return qml.numpy.stack(qml.gradients.param_shift(circuit)(weights))
>>> qml.jacobian(stacked_output)(weights)  # hessian
array([[-0.9316158 ,  0.01894799,  0.0289147 ],
       [ 0.01894799, -0.9316158 ,  0.05841749],
       [ 0.0289147 ,  0.05841749, -0.9316158 ]])

Another way to compute higher-order derivatives is by passing the ``max_diff`` and
``diff_method`` arguments to the QNode and by successive differentiation:

.. code-block:: python

    @qml.qnode(dev, diff_method="parameter-shift", max_diff=2)
    def circuit(weights):
        qml.RX(weights[0], wires=0)
        qml.RY(weights[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RX(weights[2], wires=1)
        return qml.expval(qml.Z(1))

>>> weights = np.array([0.1, 0.2, 0.3], requires_grad=True)
>>> qml.jacobian(qml.jacobian(circuit))(weights)  # hessian
array([[-0.9316158 ,  0.01894799,  0.0289147 ],
       [ 0.01894799, -0.9316158 ,  0.05841749],
       [ 0.0289147 ,  0.05841749, -0.9316158 ]])

Note that the ``max_diff`` argument only applies to gradient transforms and that its default value is ``1``; failing to
set its value correctly may yield incorrect results for higher-order derivatives. Also, passing
``diff_method="parameter-shift"`` is equivalent to passing ``diff_method=qml.gradients.param_shift``.

Transforming tapes
------------------

Gradient transforms can be applied to low-level :class:`~.QuantumTape` objects,
a datastructure representing variational quantum algorithms:

.. code-block:: python

    weights = np.array([0.1, 0.2, 0.3], requires_grad=True)

    ops = [
        qml.RX(weights[0], wires=0),
        qml.RY(weights[1], wires=1),
        qml.CNOT(wires=[0, 1]),
        qml.RX(weights[2], wires=1)]
    measurements = [qml.expval(qml.Z(1))]
    tape = qml.tape.QuantumTape(ops, measurements)

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

>>> dev = qml.device("default.qubit")
>>> fn(qml.execute(gradient_tapes, dev, None))
(tensor(-0.09347337, requires_grad=True),
 tensor(-0.18884787, requires_grad=True),
 tensor(-0.28818254, requires_grad=True))

Note that the post-processing function ``fn`` returned by the
gradient transform is applied to the flat list of results returned
from executing the gradient tapes.


Custom gradient transforms
--------------------------

Using the :func:`qml.transform <pennylane.transform>` decorator, custom gradient transforms
can be created:

.. code-block:: python

    from pennylane.tape import QuantumScriptBatch
    from pennylane.typing import PostprocessingFn

    @transform
    def my_custom_gradient(tape: qml.tape.QuantumScript, **kwargs) -> tuple[QuantumScriptBatch, PostprocessingFn]:
        ...
        return gradient_tapes, processing_fn

Once created, a custom gradient transform can be applied directly
to QNodes, or registered as the quantum gradient transform to use
during autodifferentiation.

For more details, please see the :func:`qml.transform <pennylane.transform>`
documentation.
"""
from . import (
    finite_difference,
    hadamard_gradient,
    parameter_shift,
    parameter_shift_cv,
    parameter_shift_hessian,
    pulse_gradient,
    pulse_gradient_odegen,
    spsa_gradient,
)
from .adjoint_metric_tensor import adjoint_metric_tensor
from .classical_jacobian import classical_jacobian
from .finite_difference import finite_diff, finite_diff_coeffs, finite_diff_jvp
from .fisher import classical_fisher, quantum_fisher
from .general_shift_rules import (
    eigvals_to_frequencies,
    generate_multi_shift_rule,
    generate_multishifted_tapes,
    generate_shift_rule,
    generate_shifted_tapes,
)
from .gradient_transform import SUPPORTED_GRADIENT_KWARGS
from .hadamard_gradient import hadamard_grad
from .jvp import batch_jvp, compute_jvp_multi, compute_jvp_single, jvp
from .metric_tensor import metric_tensor
from .parameter_shift import param_shift
from .parameter_shift_cv import param_shift_cv
from .parameter_shift_hessian import param_shift_hessian
from .pulse_gradient import stoch_pulse_grad
from .pulse_gradient_odegen import pulse_odegen
from .spsa_gradient import spsa_grad
from .vjp import batch_vjp, compute_vjp_multi, compute_vjp_single, vjp
