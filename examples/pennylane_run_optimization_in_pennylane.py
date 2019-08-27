r"""
.. _optimization_in_pennylane:

Optimization in PennyLane
=========================

Now that we are familiar with the different operations that are
available and how quantum gradients are calculated in PennyLane, we can
carry out cost/objective optimization.

The gradient descent algorithm has two steps:

1. Compute the gradient of cost function: :math:`\nabla_\alpha C`
2. Update the parameters :math:`\alpha` in proportion to this gradient:

   .. math:: \alpha \mapsto \alpha - \eta \nabla_\alpha C

The scaling factor :math:`\eta` is known as the *learning rate*.

   This procedure is carried out automatically by the PennyLane's 
   :class:`GradientDescentOptimizer <pennylane.optimize.GradientDescentOptimizer>`
   class object. This function needs to be called with one argument: the
   stepsize for the gradient descent algorithm.

Example
^^^^^^^^
"""

# first we import the essentials
import pennylane as qml
from pennylane import numpy as np

##############################################################################

dev1 = qml.device('default.qubit', wires = 2)
    
@qml.qnode(dev1)
def circuit(params):
    qml.Hadamard(wires = 1)
    qml.RX(params[0], wires = 0)
    qml.RX(params[1], wires = 1)
    qml.CNOT(wires = [0,1])
    return qml.expval.PauliZ(wires=0)

##############################################################################

# Define cost function
def cost(var):
    return circuit(var)

##############################################################################
# For simplicity, we have taken the cost function to be the expectation
# value of the Pauli-Z operator on the first qubit. Hence, the
# ``GradientDescentOptimizer`` will optimize ``params`` so as to get the
# lowest possible value of :math:`\langle\hat{\sigma}_z\rangle_0` for this
# circuit.
#
# Let’s explicitly look at what happens in a single step of the gradient
# descent algorithm for this example:

eta = 0.1
opt = qml.GradientDescentOptimizer(eta)
    
init_val = np.random.random(2)
new_val = opt.step(cost, init_val)
print("Initial value:", init_val)
print("Value after one step:", new_val)

##############################################################################

# Confirm that automatic update does what we expect
grad_circuit = qml.grad(circuit, argnum = 0)
new_val_manual = init_val - eta * grad_circuit(init_val)
assert np.allclose(new_val, new_val_manual)

##############################################################################
#
# In general, we will have to perform many optimization steps (10s to
# 100s) to find the optimum parameters of a cost function.
#
# .. important::
#
#     There are a number of other optimizers in the gradient descent family
#     which are available in PennyLane - as shown below. See
#     :mod:`pennylane.optimize`
#     for details and documentation of all available optimizers.

# various optimizers available in PennyLane
print(qml.optimize.__all__)

##############################################################################

