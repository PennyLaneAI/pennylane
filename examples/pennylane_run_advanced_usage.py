r"""

.. _advanced_features:

Advanced Usage
==============

In the previous three introductory tutorials (:ref:`qubit rotation <qubit_rotation>`,
:ref:`Gaussian transformation <gaussian_transformation>`, and
:ref:`plugins & hybrid computation <plugins_hybrid>`) we explored the basic concepts of
PennyLane, including qubit- and CV-model quantum computations, gradient-based optimization,
and the construction of hybrid classical-quantum computations.

In this tutorial, we will highlight some of the more advanced features of Pennylane.
"""

##############################################################################
# Multiple measurements
# ---------------------
#
# In all the previous examples, we considered quantum functions with only single expectation values.
# In fact, PennyLane supports the return of multiple measurements, up to one per wire.
#
# As usual, we begin by importing PennyLane and the PennyLane-provided version of NumPy, and
# set up a 2-wire qubit device for computations:

import pennylane as qml
from pennylane import numpy as np

dev = qml.device("default.qubit", wires=2)

##############################################################################
# We will start with a simple example circuit, which generates a two-qubit entangled state,
# then evaluates the expectation value of the Pauli Z operator on each wire.


@qml.qnode(dev)
def circuit1(param):
    qml.RX(param, wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))


##############################################################################
# The degree of entanglement of the qubits is determined by the value of ``param``. For a value of
# :math:`\frac{\pi}{2}`, they are maximally entangled. In this case, the reduced states on each
# subsystem are completely mixed, and local expectation values — like those we are measuring —
# will average to zero.

print(circuit1(np.pi / 2))

##############################################################################
# Notice that the output of the circuit is a NumPy array with ``shape=(2,)``, i.e., a two-dimensional
# vector. These two dimensions match the number of expectation values returned in our quantum
# function ``circuit1``.
#
# .. note::
#
#     It is important to emphasize that the expectation values in ``circuit`` are both **local**,
#     i.e., this circuit is evaluating :math:`\left\langle \sigma_z\right\rangle_0` and :math:`\left\langle \sigma_z\right\rangle_1`,
#     not :math:`\left\langle \sigma_z\otimes \sigma_z\right\rangle_{01}` (where the subscript denotes which wires the
#     observable is located on).
#
# We may even mix different return types, for example expectation values and variances:


@qml.qnode(dev)
def circuit1(param):
    qml.RX(param, wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0)), qml.var(qml.PauliZ(1))


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
