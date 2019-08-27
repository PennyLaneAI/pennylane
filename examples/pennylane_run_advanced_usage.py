r"""

.. _advanced_features:

Advanced Usage
==============

In the introductory tutorials, we explored the basic concepts of
PennyLane, including qubit- and CV-model quantum computations and gradient-based optimization. Here, we will highlight some of the more advanced features of Pennylane.
"""

##############################################################################
# Keyword arguments
# -----------------
#
# While automatic differentiation is a handy feature, sometimes we want certain parts of our
# computational pipeline (e.g., the inputs :math:`x` to a parameterized quantum function
# :math:`f(x;\bf{\theta})` or the training data for a machine learning model) to not be
# differentiated.
#
# PennyLane uses the pattern that *all positional arguments to quantum functions are available
# to be differentiated*, while *keyword arguments are never differentiated*. Thus, when using the
# gradient-descent-based :ref:`optimizers <optimization_methods>` included in PennyLane, all
# numerical parameters appearing in non-keyword arguments will be updated, while all numerical
# values included as keyword arguments will not be updated.
#
# .. note::
#
#     When constructing the circuit, keyword arguments are defined by providing a
#     **default value** in the function signature. If you would prefer that the keyword argument
#     value be passed every time the quantum circuit function is called, the default value
#     can be set to ``None``.
#
# For example, let's create a quantum node that accepts two arguments; a differentiable
# circuit parameter ``param``, and a fixed circuit parameter ``fixed``:

import pennylane as qml
from pennylane import numpy as np

dev = qml.device('default.qubit', wires=2)
@qml.qnode(dev)
def circuit3(param, fixed=None):
    qml.RX(fixed, wires=0)
    qml.RX(param, wires=1)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))


##############################################################################
# Calling the circuit, we can feed values to the keyword argument ``fixed``:

print(circuit3(0.1, fixed=-0.2))

print(circuit3(0.1, fixed=1.2))

##############################################################################
# Since keyword arguments do not get considered when computing gradients, the
# Jacobian will still be a 2-dimensional vector.

j3 = qml.jacobian(circuit3, argnum=0)
print(j3(2.5, fixed=3.2))

##############################################################################
# Once defined, keyword arguments must *always* be passed as keyword arguments. PennyLane does
# not support passing keyword argument values as positional arguments.
#
# For example, the following circuit evaluation will correctly update the value of the fixed parameter:

print(circuit3(0.1, fixed=0.4))

##############################################################################
# However, attempting to pass the fixed parameter as a positional argument will
# not work, and PennyLane will attempt to use the default value (``None``) instead:
#
# >>> circuit3(0.1, 0.4)
# ---------------------------------------------------------------------------
# TypeError                                 Traceback (most recent call last)
# <ipython-input-6-949e31911afa> in <module>()
# ----> 1 circuit3(0.1, 0.4)
# ~/pennylane/variable.py in val(self)
#     134
#     135         # The variable is a placeholder for a keyword argument
# --> 136         value = self.kwarg_values[self.name][self.idx] * self.mult
#     137         return value
# TypeError: unsupported operand type(s) for *: 'NoneType' and 'int'
#
# QNodes from different interfaces on one Device
# -----------------------------------------------
#
# PennyLane does not only provide the flexibility of having multiple quantum nodes on one device,
# it also allows these nodes to have different interfaces. Let's look at the following simple
# example:

dev1 = qml.device('default.qubit', wires=1)
def circuit(phi):
    qml.RX(phi, wires=0)
    return qml.expval(qml.PauliZ(0))

##############################################################################
# Now, we construct multiple QNodes on the same device and change the interface of one of them 
# from NumPy to PyTorch: 

qnode1 = qml.QNode(circuit, dev1)
qnode2 = qml.QNode(circuit, dev1)
qnode1_torch = qnode1.to_torch()

##############################################################################
# Let's define the cost function. Notice that we can pass the QNode as an argument too. This 
# avoids duplication of code. 
   
def cost(qnode, phi):
    return qnode(phi)

##############################################################################
# Now we can call the cost function as follows:

print(cost(qnode1, np.pi))

print(cost(qnode2, np.pi))

print(cost(qnode1_torch, np.pi))

