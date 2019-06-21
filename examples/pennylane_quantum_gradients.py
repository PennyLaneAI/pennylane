r"""
.. _quantum_gradients:

Executing Quantum Gradients
===========================

To do machine learning with quantum circuits, we need a gradient descent
strategy. For this, we define a final objective/cost function
:math:`C` which is to be optimized with respect to some free parameters.
As explained in the :ref:`autograd_quantum` section in the key concepts,
PennyLane incorporates both analytic differentiation, as well as
numerical methods (such as the method of finite differences). Both of
these are done automatically.

PennyLane is the first software library to support **automatic
differentiation of quantum circuits**. Internally, it uses *classical
linear combination of unitaries* or “parameter shift” trick to compute
derivatives, i.e. it evaluates derivatives as the difference of two
circuit evaluations with shifted parameters.

.. math:: C(\theta_1, \theta_2) = \texttt{circuit}(\theta_1, \theta_2)

.. math:: \partial_{\theta_1} C = a\big[ \texttt{circuit}(\theta_1+s, \theta_2) - \texttt{circuit}(\theta_1 - s, \theta_2) \big]

**NOTE:** The values of the shift and scale parameters :math:`s` and
:math:`a` typically depend **only** on the *type* of gate, and not its
location.

By using this method, PennyLane provides a hardware-scalable way to
compute gradients and to optimize quantum circuits for Quantum Machine Learning.

Calculating quantum gradients
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let’s initialize a ``device`` and define a quantum function in a
``QNode``
"""

# first we import the essentials
import pennylane as qml
from pennylane import numpy as np

##############################################################################

dev1 = qml.device('default.qubit', wires = 1)
    
@qml.qnode(dev1)
def circuit(params):
    qml.RX(params[0], wires = 0)
    qml.RY(params[1], wires = 0)
    return qml.expval.PauliZ(0)

##############################################################################
#
# The gradient of the function ``circuit`` encapsulated within the
# ``QNode`` can be evaluated by utilizing the same quantum device
# (``dev1``) that we used to evaluate the function itself.
#
# We can differentiate by using the built-in :func:`~.pennylane.grad`
# function.

dcircuit = qml.grad(circuit, argnum = 0)

##############################################################################
#
# The function ``grad()`` itself **returns a function** representing the
# derivative of the ``QNode`` with respect to the argument specified in
# ``argnum``.
#
# In the example above, the function ``circuit`` takes one argument
# ``params``. Hence, we specify ``argnum=0``. As the argument has two
# elements, the returned gradient is two-dimensional. We can then evaluate
# this gradient function at any point in the parameter space.

print(dcircuit([0.54, 0.12]))

##############################################################################
#
# Quantum circuit functions, being a restricted subset of Python
# functions, can also make use of multiple positional arguments and
# keyword arguments.
#
# For example, we could have defined the above quantum circuit function
# using two positional arguments, instead of an array of arguments:

@qml.qnode(dev1)
def circuit2(phi1, phi2):
    qml.RX(phi1, wires = 0)
    qml.RY(phi2, wires = 0)
    return qml.expval.PauliZ(0)

##############################################################################
#
# In this case, ``argnum=0`` will return the gradient with respect to only
# the first parameter ``phi1`` and ``argnum=1`` will give the gradient
# with respect to ``phi2``. To get the gradient with respect to both
# parameters, we can use ``argnum=[0,1]``:

dcircuit2 = qml.grad(circuit2, argnum = [0, 1])
print(dcircuit2(0.54, 0.12))

##############################################################################
# .. note::
# 	PennyLane does **not** differentiate QNodes with respect to keyword
# 	arguments. Hence, they are useful for passing external data to a
# 	quantum node. See ``Keyword arguments`` in the :ref:`advanced_usage` tutorial.
