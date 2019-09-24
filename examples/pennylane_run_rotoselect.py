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
# Thus, one might expect the number of circuit evaluations required to be 9 (3 for each gate choice). However,
# since there is a 3-fold
# degeneracy in the expectation value -- :math:`\langle H \rangle (0) = 1` for each of the gates -- the number of
# evaluations reduces to 7. In this way, one could learn both the optimal parameters and generators for a circuit. In
# the following sections, we will present an example of this algorithm applied to a VQE Hamiltonian.
#
# VQE
# ~~~
#
# We choose to focus on the example of a VQE circuit with 2 qubits for simplicity. Here, the Hamiltonian
# is
#
# .. math::
#   H = 0.1\sigma_x+0.5\sigma_y
#
# acting on the second qubit. The expectation value of this quantity acts as the cost function for our
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
# Next, we set up a circuit with a fixed ansatz structure which will later be subject to change and encode
# the Hamiltonian into a cost function. We require 2 circuits since we are calculating multiple expectation
# values of Pauli measurements on the same wire.

def ansatz(params):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.CNOT(wires=[0,1])

@qml.qnode(dev)
def circuit_X(params):
    ansatz(params)
    return qml.expval(qml.PauliX(1))

@qml.qnode(dev)
def circuit_Y(params):
    ansatz(params)
    return qml.expval(qml.PauliY(1))

def cost(params):
    X = circuit_X(params)
    Y = circuit_Y(params)
    return 0.1*X + 0.5*Y

##############################################################################
# Helper methods for the algorithm
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We define methods to evaluate
# the expression in the previous section. These will serve as the basis for
# our optimization algorithm.

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


def rotosolve_cycle(cost, params):
    for d in range(len(params)):
        opt_theta_d(d, params, cost)
    # params object is mutable but return it anyway
    # to match PennyLane optimizer convention
    return params

##############################################################################
# Optimization and comparison with gradient descent
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We set up some initial parameters for the :math:`\sigma_x` and :math:`\sigma_y`
# gates in the ansatz circuit structure and perform an optimization using the
# Rotosolve algorithm.


init_params = [0.3, 0.25]
params = init_params[:]
n_steps = 30

#
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
opt = qml.GradientDescentOptimizer(stepsize=2.)
costs_grad_desc = []
for i in range(n_steps):
    costs_grad_desc.append(cost(params))
    params = opt.step(cost, params)


# plot cost function optimization using the 2 techniques
import matplotlib.pyplot as plt
steps = np.arange(0,n_steps)
fig, (ax1,ax2) = plt.subplots(1,2)
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

##############################################################################
# Cost function surface for circuit ansatz
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


##############################################################################
# Rotoselect
# ----------

class Rotoselect:


    def cycle(self, cost, var, gen):
        params = var.copy()
        params_opt = params.copy()
        generators = gen.copy()
        generators_opt = gen.copy()
        params_opt_cost = cost(params_opt, generators_opt)
        for d in range(len(params)):
            params[d] = 0.
            M_phi = cost(params, generators) # M_phi independent of generator selection
            for generator in ['X','Y','Z']:
                generators[d] = generator
                params[d] = np.pi / 2.
                M_phi_plus = cost(params, generators)
                params[d] = -np.pi / 2.
                M_phi_minus = cost(params, generators)
                a = np.arctan2(2. * M_phi - M_phi_plus - M_phi_minus, M_phi_plus - M_phi_minus)
                params[d] = -np.pi / 2. - a
                if params[d] > np.pi:
                    params[d] -= 2 * np.pi
                if params[d] <= -np.pi:
                    params[d] += 2 * np.pi
                params_cost = cost(params, generators)
                if params_cost <= params_opt_cost:
                    params_opt[d] = params[d]
                    params_opt_cost = params_cost
                    generators_opt[d] = generator
            params[d] = params_opt[d]
            generators[d] = generators_opt[d]
        return params, generators


def RGen(theta, generator, wires):
    if generator == 'X':
        qml.RX(theta, wires=wires)
    elif generator == 'Y':
        qml.RY(theta, wires=wires)
    elif generator == 'Z':
        qml.RZ(theta, wires=wires)
    else:
        raise Exception("Invalid generator")


def ansatz(var, generators):
    RGen(var[0], generators[0], wires=0)
    RGen(var[1], generators[1], wires=1)
    qml.CNOT(wires=[0,1])

@qml.qnode(dev)
def circuit_X(params, generators=[]):
    ansatz(params, generators)
    return qml.expval(qml.PauliX(1))

@qml.qnode(dev)
def circuit_Y(params, generators=[]):
    ansatz(params, generators)
    return qml.expval(qml.PauliY(1))


def cost(params, generators):
    X = circuit_X(params, generators=generators)
    Y = circuit_Y(params, generators=generators)
    return 0.1*X + 0.5*Y


obj3 = []
opt = Rotoselect()
init_params = [0.3, 0.8]
params = init_params
init_generators = ['X', 'X']
generators = init_generators
for _ in range(n_steps):
    obj3.append(cost(params, generators))
    params, generators = opt.cycle(cost, params, generators)

print("Optimal generators are: {}".format(generators))


plt.subplot(1,3,3)
plt.title("rotoselect")
plt.xlabel("steps")
plt.ylabel("cost")
plt.plot(steps,obj3,'o-')
plt.tight_layout()
plt.show()

##############################################################################
# Cost function surface for learned circuit structure
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~