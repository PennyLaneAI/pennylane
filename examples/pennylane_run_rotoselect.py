r"""

.. _rotoselect:

Quantum circuit structure learning
==================================

This example shows how to learn a good selection of rotation
gates so as to minimize a cost
function using the Rotoselect algorithm `Ostaszewski et al.
(2019) <https://arxiv.org/abs/1905.09692>`__. We apply the algorithm to minimize the Hamiltonian presented in
a previous tutorial, VQE, and improve upon the circuit structure ansatz
chosen there.
"""
##############################################################################
# Background
# ----------
#
# The effects of noise tend to increase with the depth of a quantum circuit, so
# there is incentive to keep them as shallow as possible. It is often the case that a
# chosen set of gates is suboptimal for the task at hand. Therefore it would be useful to employ an
# algorithm which learns a good structure for the quantum circuit tasked with
# minimizing a certain cost function.
#
# Furthermore, PennyLane's optimizers perform automatic differentiation of quantum nodes by evaluating phase-shifted
# calculations of their constituent operators' expectation values on the quantum circuit itself.
# The output of a calculation, the gradient, is used in optimization methods to minimize a cost. However, there exists
# a technique to find the optimal parameters of a quantum circuit through phase-shifted calculations,
# without the need for calculating the gradient as an intermediate step (i.e. a gradient-free optimization).
# It would be desirable, in some cases, to
# take advantage of this.
#
#
# The Rotoselect algorithm addresses the above two points.
# It works by updating the parameters :math:`\boldsymbol{\theta}=\theta_1...\theta_D` and gate choices
# :math:`\boldsymbol{P}=P_1...P_D`
# one at a time according to a closed-form expression for the optimal parameter value :math:`\theta^{*}_d`
# when the other parameters and gate choices are fixed:
#
# .. math::
#
#   \theta^{*}_d &= \underset{\theta_d}{\text{argmin}} \langle H \rangle (\theta_d) \\
#                &= -\frac{\pi}{2} - \text{arctan}\left(\frac{2\langle H \rangle (0) -
#                \langle H \rangle (\pi/2) - \langle H \rangle (-\pi/2)}{\langle H \rangle (\pi/2) -
#                \langle H \rangle (-\pi/2)}\right)
#
# The calculation makes use of 3 separate evaluations
# of the cost function expectation value :math:`\langle H \rangle (\theta_d)` using the quantum circuit. Although
# :math:`\langle H \rangle` is really a function of all parameters and gate choices
# :math:`\boldsymbol{\theta}, \ \boldsymbol{P}`, we
# are fixing every parameter and gate choice apart from :math:`\theta_d` in this expression so we write it as
# :math:`\langle H \rangle = \langle H \rangle (\theta_d)`.
# For each parameter in the quantum circuit, the algorithm proceeds by evaluating :math:`\theta^{*}_d`
# for each choice of
# gate :math:`P_d \in \{R_x,R_y,R_z\}` and selecting the gate which yields the minimum value.
#
# Thus, one might expect the number of circuit evaluations required to be 9 for each parameter (3 for each gate
# choice). However, since there is a 3-fold
# degeneracy in the expectation value -- :math:`\langle H \rangle (0) = 1` for each of the gates -- the number of
# evaluations reduces to 7.
#
# One cycle of the Rotosolve algorithm constitutes
# iterating through every parameter and performing the calculation above.
# This cycle is repeated for a fixed number of steps or until convergence. In this way, one could learn both
# the optimal parameters and generators for a circuit. Next, we present an example of this algorithm
# applied to a VQE Hamiltonian.
#
# VQE
# ~~~
#
# We choose to focus on the example of a VQE circuit with 2 qubits for simplicity. Here, the Hamiltonian
# is
#
# .. math::
#   H = 0.5Y_2 - 0.8Z_1 - 0.2X_1
#
# where the subscript denotes the qubit upon which the Pauli operator acts. The
# expectation value of this quantity acts as the cost function for our
# optimization.
#
# Rotosolve
# ---------
# As a precursor to implementing Rotoselect we can analyze a version of the algorithm
# which does not optimize the choice of gates, called Rotosolve. Later, we will build on this example
# to implement Rotoselect and vary the circuit structure.
#
# Imports
# ~~~~~~~
# To get started, we import PennyLane and the PennyLane-included version of NumPy. We also
# create a 2-qubit device using the ``default.qubit`` plugin and set the ``analytic`` keyword to ``True``
# in order to obtain exact values for any expectation values calculated. In contrast with real
# devices, simulators have the capability of doing these calculations without sampling.

import pennylane as qml
from pennylane import numpy as np

n_wires = 2

dev = qml.device('default.qubit',analytic=True,wires=2)

##############################################################################
# Creating a fixed quantum circuit
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# .. figure:: ../../examples/figures/original_ansatz.png
#    :scale: 65%
#    :align: center
#    :alt: original_ansatz
#
# |
# Next, we set up a circuit with a fixed ansatz structure -- which will later be subject to change -- and encode
# the Hamiltonian into a cost function. The structure is shown in the figure above.

def ansatz(params):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.CNOT(wires=[0, 1])


@qml.qnode(dev)
def circuit(params):
    ansatz(params)
    return qml.expval(qml.PauliZ(0)),qml.expval(qml.PauliY(1))

@qml.qnode(dev)
def circuit2(params):
    ansatz(params)
    return qml.expval(qml.PauliX(0))

def cost(params):
    Z_1 = circuit(params)[0]
    Y_2 = circuit(params)[1]
    X_1 = circuit2(params)
    return 0.5 * Y_2 + 0.8 * Z_1 - 0.2 * X_1

##############################################################################
# Helper methods for the algorithm
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We define methods to evaluate
# the expression in the previous section. These will serve as the basis for
# our optimization algorithm.

# calculation as described above
def opt_theta_d(d, params, cost):
    params[d] = 0.
    M_0 = cost(params)
    params[d] = np.pi / 2.
    M_0_plus = cost(params)
    params[d] = -np.pi / 2.
    M_0_minus = cost(params)
    a = np.arctan2(2. * M_0 - M_0_plus - M_0_minus, M_0_plus - M_0_minus)
    params[d] = -np.pi / 2. - a
    # restrict output to lie in (-pi,pi], a convention
    # consistent with the Rotosolve paper
    if params[d] > np.pi:
        params[d] -= 2 * np.pi
    if params[d] <= -np.pi:
        params[d] += 2 * np.pi

# one cycle of rotosolve
def rotosolve_cycle(cost, params):
    for d in range(len(params)):
        opt_theta_d(d, params, cost)
    # params object is mutable but return it anyway
    # to match PennyLane optimizer convention
    return params

##############################################################################
# Optimization and comparison with gradient descent
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We set up some initial parameters for the :math:`R_x` and :math:`R_y`
# gates in the ansatz circuit structure and perform an optimization using the
# Rotosolve algorithm.

init_params = [0.3, 0.25]
params = init_params[:]
n_steps = 30

costs_rotosolve = []

for i in range(n_steps):
    costs_rotosolve.append(cost(params))
    params = rotosolve_cycle(cost,params)

##############################################################################
# We then compare the results of Rotosolve to an optimization
# performed with gradient descent and plot
# the cost functions at each step, or cycle in the case of Rotosolve.
# This comparison is fair since the number of circuit
# evaluations involved in a cycle of Rotosolve is similar to those required to calculate
# the gradient of the circuit and step in this direction.

params = init_params[:]
opt = qml.GradientDescentOptimizer(stepsize=0.5)
costs_grad_desc = []
for i in range(n_steps):
    costs_grad_desc.append(cost(params))
    params = opt.step(cost, params)


# plot cost function optimization using the 2 techniques
import matplotlib.pyplot as plt
steps = np.arange(0,n_steps)
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(7,3))
plt.subplot(1,2,1)
plt.plot(steps,costs_grad_desc,'o-')
plt.title("grad. desc.")
plt.xlabel("steps")
plt.ylabel("cost")
plt.subplot(1,2,2)
plt.plot(steps,costs_rotosolve,'o-')
plt.title("rotosolve")
plt.xlabel("cycles")
plt.ylabel("cost")
plt.tight_layout()
plt.show()


##############################################################################
# Cost function surface for circuit ansatz
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Now, we plot the cost function surface for later comparison with the surface generated
# by learning the circuit structure. It is apparent that, based on the structure
# ansatz chosen above, the cost function does not depend on the angle parameter :math:`\theta_2`
# for the rotation gate :math:`R_y`. As we will show in the following sections, this independence is not true
# for alternative gate choices.

from matplotlib import cm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(6, 4))
ax = fig.gca(projection="3d")

X = np.linspace(-4.0, 4.0, 20)
Y = np.linspace(-4.0, 4.0, 20)
xx, yy = np.meshgrid(X, Y)
Z = np.array([[cost([x, y]) for x in X] for y in Y]).reshape(len(Y), len(X))
surf = ax.plot_surface(xx, yy, Z, cmap=cm.coolwarm, antialiased=False)

ax.set_xlabel(r"$\theta_1$")
ax.set_ylabel(r"$\theta_2$")
ax.zaxis.set_major_locator(MaxNLocator(nbins=5, prune="lower"))

plt.show()

##############################################################################
# Rotoselect
# ----------
#
# .. figure:: ../../examples/figures/rotoselect_structure.png
#    :scale: 65%
#    :align: center
#    :alt: rotoselect_structure
#
# |
# We now implement the Rotoselect algorithm to learn a good selection of gates to minimize
# our cost function. The structure is similar to the original ansatz, but the unitaries are
# selected from the set of rotation gates :math:`P_d \in \{R_x,R_y,R_z\}` as shown in the figure above.
#
# Creating a quantum circuit with variable gates
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# First, we setup a quantum circuit with a similar structure to the one above, but
# instead of fixed rotation gates :math:`X` and :math:`Y`, we allow the gates to be specified with the
# ``generators`` keyword. A helper method ``RGen`` returns the correct unitary gate according to the
# rotation specified by an element of ``generators``.

def RGen(param, generator, wires):
    if generator == 'X':
        qml.RX(param, wires=wires)
    elif generator == 'Y':
        qml.RY(param, wires=wires)
    elif generator == 'Z':
        qml.RZ(param, wires=wires)
    else:
        raise Exception("Invalid generator")

def ansatz(params, generators):
    RGen(params[0], generators[0], wires=0)
    RGen(params[1], generators[1], wires=1)
    qml.CNOT(wires=[0,1])

@qml.qnode(dev)
def circuit(params, generators=[]): # generators must be a kwarg in a qnode
    ansatz(params, generators)
    return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))

@qml.qnode(dev)
def circuit2(params, generators=[]): # generators must be a kwarg in a qnode
    ansatz(params, generators)
    return qml.expval(qml.PauliX(0))

def cost(params, generators):
    Z_1 = circuit(params, generators=generators)[0]
    Y_2 = circuit(params, generators=generators)[1]
    X_1 = circuit2(params, generators=generators)
    return 0.5 * Y_2 + 0.8 * Z_1 - 0.2 * X_1



##############################################################################
# Helper methods
# ~~~~~~~~~~~~~~
# We define helper methods in a similar fashion to Rotosolve. In this case,
# we must iterate through the possible gate choices in addition to optimizing each parameter.

def rotosolve_d(d, params, generators, cost, M_0): # M_0 only calculated once
    params[d] = np.pi / 2.
    M_0_plus = cost(params, generators)
    params[d] = -np.pi / 2.
    M_0_minus = cost(params, generators)
    a = np.arctan2(2. * M_0 - M_0_plus - M_0_minus, M_0_plus - M_0_minus)
    params[d] = -np.pi / 2. - a
    if params[d] > np.pi:
        params[d] -= 2 * np.pi
    if params[d] <= -np.pi:
        params[d] += 2 * np.pi
    return cost(params, generators)

def optimal_theta_and_gen_helper(d, params, generators, cost):
    params[d] = 0.
    M_0 = cost(params, generators)  # M_0 independent of generator selection
    for generator in ['X', 'Y', 'Z']:
        generators[d] = generator
        params_cost = rotosolve_d(d, params, generators, cost, M_0)
        if generator == 'X' or params_cost <= params_opt_cost:
            params_opt_d = params[d]
            params_opt_cost = params_cost
            generators_opt_d = generator
    return params_opt_d, generators_opt_d


def rotoselect_cycle(cost, params, generators):
    for d in range(len(params)):
        params[d], generators[d] = optimal_theta_and_gen_helper(d, params, generators, cost)
    return params, generators


##############################################################################
# Optimizing the circuit structure
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We perform the optimization and print the optimal generators. The cost function is
# reduced below the minimal value from gradient descent or Rotosolve on the original
# circuit structure ansatz: the learned circuit structure performs better without
# increasing the depth of the circuit.

costs_rotoselect = []
init_params = [0.3, 0.8]
params = init_params
init_generators = ['X', 'X']
generators = init_generators
for _ in range(n_steps):
    costs_rotoselect.append(cost(params, generators))
    params, generators = rotoselect_cycle(cost, params, generators)

print("Optimal generators are: {}".format(generators))

# plot cost function vs. steps comparison
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(7,3))
plt.subplot(1,2,1)
plt.plot(steps,costs_grad_desc,'o-')
plt.title("grad. desc. on original ansatz")
plt.xlabel("steps")
plt.ylabel("cost")
plt.subplot(1,2,2)
plt.plot(steps,costs_rotoselect,'o-')
plt.title("rotoselect")
plt.xlabel("cycles")
plt.ylabel("cost")
plt.tight_layout()
plt.show()


##############################################################################
# Cost function surface for learned circuit structure
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# .. figure:: ../../examples/figures/learned_structure.png
#    :scale: 65%
#    :align: center
#    :alt: learned_structure
#
# |
# Finally, we plot the cost function surface for the newly discovered optimized
# circuit structure shown in the figure above. It is apparent from the minima in the plot that
# the original ansatz was not expressive enough to arrive at the optimal values
# of the cost function.

fig = plt.figure(figsize=(6, 4))
ax = fig.gca(projection="3d")

X = np.linspace(-4.0, 4.0, 20)
Y = np.linspace(-4.0, 4.0, 20)
xx, yy = np.meshgrid(X, Y)
# plot cost for fixed optimal generators
Z = np.array([[cost([x, y], generators=generators) for x in X] for y in Y]).reshape(len(Y), len(X))
surf = ax.plot_surface(xx, yy, Z, cmap=cm.coolwarm, antialiased=False)

ax.set_xlabel(r"$\theta_1$")
ax.set_ylabel(r"$\theta_2$")
ax.zaxis.set_major_locator(MaxNLocator(nbins=5, prune="lower"))

plt.show()
