r"""
.. _vqe:

Variational quantum eigensolver
===============================

This example demonstrates the principle of a variational quantum
eigensolver (VQE), originally proposed in `Peruzzo et al.
(2014) <https://www.nature.com/articles/ncomms5213>`__. To showcase the
hybrid computational capabilities of PennyLane, we first train a quantum
circuit to minimize the squared energy expectation for a Hamiltonian
:math:`H`,

.. math::

    \langle \psi_v | H | \psi_v \rangle^2  =( 0.1 \langle \psi_{v} | X_2 |
    \psi_v \rangle + 0.5 \langle \psi_v | Y_2 | \psi_v \rangle )^2.

Here, :math:`|\psi_v\rangle` is the state
obtained after applying a quantum circuit to an initial state
:math:`|0\rangle`. The quantum circuit depends on trainable variables
:math:`v = \{v_1, v_2\}`, and :math:`X_2`, :math:`Y_2` denote the
Pauli-X and Pauli-Y operator acting on the second qubit (*Note: We apply
the square to make the optimization landscapes more interesting, but in
common applications the cost is directly the energy expectation value*).

After doing this, we will then turn things around and use a fixed
quantum circuit to prepare a state :math:`|\psi\rangle`, but train the coefficients of
the Hamiltonian to minimize

.. math::

    \langle \psi | H | \psi \rangle^2  = (v_1 \langle \psi | X_2 | \psi
    \rangle + v_2 \langle \psi | Y_2 | \psi \rangle )^2 .
"""

##############################################################################
# 1. Optimizing the quantum circuit
# ---------------------------------
#
# Imports
# ~~~~~~~
#
# We begin by importing PennyLane, the PennyLane-wrapped version of NumPy,
# and the GradientDescentOptimizer.

import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import GradientDescentOptimizer

##############################################################################
# We use the default qubit simulator as a device.

dev = qml.device("default.qubit", wires=2)

##############################################################################
# Quantum nodes
# ~~~~~~~~~~~~~
#
# The quantum circuit of the variational eigensolver is an ansatz that
# defines a manifold of possible quantum states. We use a Hadamard, two
# rotations and a CNOT gate to construct our circuit.


def ansatz(var):
    qml.Rot(0.3, 1.8, 5.4, wires=1)
    qml.RX(var[0], wires=0)
    qml.RY(var[1], wires=1)
    qml.CNOT(wires=[0, 1])


##############################################################################
# A variational eigensolver requires us to evaluate expectations of
# different Pauli operators. In this example, the Hamiltonian is expressed
# by only two single-qubit Pauli operators, namely the X and Y operator
# applied to the first qubit.
#
# Since these operators will be measured on the same wire, we will need to
# create two quantum nodes (one for each operator whose expectation value
# we measure), but we can reuse the same device.
#
# .. note::
#
#     If the Pauli observables were evaluated on different wires, we
#     could use one quantum node and return a tuple of expectations in only
#     one quantum node:
#     ``return qml.expectation.PauliX(0), qml.expectation.PauliY(1)``


@qml.qnode(dev)
def circuit_X(var):
    ansatz(var)
    return qml.expval(qml.PauliX(1))


@qml.qnode(dev)
def circuit_Y(var):
    ansatz(var)
    return qml.expval(qml.PauliY(1))


##############################################################################
# Objective
# ~~~~~~~~~

# The cost function to be optimized in VQE is simply a linear combination
# of the expectations, which defines the expectation of the Hamiltonian we
# are interested in. In our case, we square this cost function to provide
# a more interesting landscape with the same minima.


def cost(var):
    expX = circuit_X(var)
    expY = circuit_Y(var)
    return (0.1 * expX + 0.5 * expY) ** 2


##############################################################################
# This cost defines the following landscape:
#
# *Note: To run the following cell you need the matplotlib library.*

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import MaxNLocator

fig = plt.figure(figsize=(6, 4))
ax = fig.gca(projection="3d")

X = np.linspace(-3.0, 3.0, 20)
Y = np.linspace(-3.0, 3.0, 20)
xx, yy = np.meshgrid(X, Y)
Z = np.array([[cost([x, y]) for x in X] for y in Y]).reshape(len(Y), len(X))
surf = ax.plot_surface(xx, yy, Z, cmap=cm.coolwarm, antialiased=False)

ax.set_xlabel("v1")
ax.set_ylabel("v2")
ax.zaxis.set_major_locator(MaxNLocator(nbins=5, prune="lower"))

plt.show()

##############################################################################
# Optimization
# ~~~~~~~~~~~~
#
# We create a GradientDescentOptimizer and use it to optimize the cost
# function.

opt = GradientDescentOptimizer(0.5)

var = [0.3, 2.5]
var_gd = [var]
for it in range(20):
    var = opt.step(cost, var)
    var_gd.append(var)

    print(
        "Cost after step {:5d}: {: .7f} | Variables: [{: .5f},{: .5f}]".format(
            it + 1, cost(var), var[0], var[1]
        )
    )

##############################################################################
# We can plot the path that the variables took during gradient descent. To
# make the plot more clear, we will shorten the range for :math:`v_2`.

fig = plt.figure(figsize=(6, 4))
ax = fig.gca(projection="3d")

X = np.linspace(-3, np.pi / 2, 20)
Y = np.linspace(-3, 3, 20)
xx, yy = np.meshgrid(X, Y)
Z = np.array([[cost([x, y]) for x in X] for y in Y]).reshape(len(Y), len(X))

surf = ax.plot_surface(xx, yy, Z, cmap=cm.coolwarm, antialiased=False)

path_z = [cost(var) + 1e-8 for var in var_gd]
path_x = [v[0] for v in var_gd]
path_y = [v[1] for v in var_gd]
ax.plot(path_x, path_y, path_z, c="green", marker=".", label="graddesc", zorder=10)

ax.set_xlabel("v1")
ax.set_ylabel("v2")
ax.zaxis.set_major_locator(MaxNLocator(nbins=5, prune="lower"))

plt.legend()
plt.show()


##############################################################################
# 2. Optimizing the Hamiltonian coefficients
# ------------------------------------------
#
# Instead of optimizing the circuit parameters, we can also use a fixed
# circuit,


def ansatz():
    qml.Rot(0.3, 1.8, 5.4, wires=1)
    qml.RX(-0.5, wires=0)
    qml.RY(0.5, wires=1)
    qml.CNOT(wires=[0, 1])


@qml.qnode(dev)
def circuit_X():
    ansatz()
    return qml.expval(qml.PauliX(1))


@qml.qnode(dev)
def circuit_Y():
    ansatz()
    return qml.expval(qml.PauliY(1))


##############################################################################
# and make the classical coefficients that appear in the Hamiltonian the
# trainable variables.


def cost(var):
    expX = circuit_X()
    expY = circuit_Y()
    return (var[0] * expX + var[1] * expY) ** 2


opt = GradientDescentOptimizer(0.5)

var = [0.3, 2.5]
var_gd = [var]
for it in range(20):
    var = opt.step(cost, var)
    var_gd.append(var)

    print(
        "Cost after step {:5d}: {: .7f} | Variables: [{: .5f},{: .5f}]".format(
            it + 1, cost(var), var[0], var[1]
        )
    )

##############################################################################
# The landscape has a quadratic shape.

fig = plt.figure(figsize=(6, 4))
ax = fig.gca(projection="3d")

X = np.linspace(-3, np.pi / 2, 20)
Y = np.linspace(-3, 3, 20)
xx, yy = np.meshgrid(X, Y)
Z = np.array([[cost([x, y]) for x in X] for y in Y]).reshape(len(Y), len(X))
surf = ax.plot_surface(xx, yy, Z, cmap=cm.coolwarm, antialiased=False)

path_z = [cost(var) + 1e-8 for var in var_gd]
path_x = [v[0] for v in var_gd]
path_y = [v[1] for v in var_gd]
ax.plot(path_x, path_y, path_z, c="pink", marker=".", label="graddesc", zorder=10)

ax.set_xlabel("v1")
ax.set_ylabel("v2")
ax.zaxis.set_major_locator(MaxNLocator(nbins=5, prune="lower"))

plt.legend()
plt.show()


##############################################################################
# 3. Optimizing classical and quantum parameters
# ----------------------------------------------

# Finally, we can optimize *classical* and *quantum* weights together by
# combining the two approaches from above.


def ansatz(var):

    qml.Rot(0.3, 1.8, 5.4, wires=1)
    qml.RX(var[0], wires=0)
    qml.RY(var[1], wires=1)
    qml.CNOT(wires=[0, 1])


@qml.qnode(dev)
def circuit_X(var):
    ansatz(var)
    return qml.expval(qml.PauliX(1))


@qml.qnode(dev)
def circuit_Y(var):
    ansatz(var)
    return qml.expval(qml.PauliY(1))


def cost(var):

    expX = circuit_X(var)
    expY = circuit_Y(var)

    return (var[2] * expX + var[3] * expY) ** 2


opt = GradientDescentOptimizer(0.5)
var = [0.3, 2.5, 0.3, 2.5]

for it in range(10):
    var = opt.step(cost, var)
    print("Cost after step {:5d}: {: 0.7f}".format(it + 1, cost(var)))
