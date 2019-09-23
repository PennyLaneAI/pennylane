r"""

.. _rotoselect:

Quantum circuit structure learning
==================================

This example shows how to learn the optimal selection of rotation
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
# there is incentive to keep them as shallow as possible. Furthermore, it is often the case that a
# chosen set of gates is suboptimal for the task at hand. The Rotoselect
# algorithm provides a method for learning a good structure for a quantum circuit tasked with
# minimizing a certain cost function.
#
# The algorithm works by updating the parameters :math:`\boldsymbol{\theta}=\theta_1...\theta_D` and gate choices
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
# where the expression makes use of 3 separate evaluations
# of the cost function expectation value :math:`\langle H \rangle (\theta_d)` using the quantum circuit. Although
# :math:`\langle H \rangle` is really a function of all parameters and gate choices :math:`\boldsymbol{\theta}, \ \boldsymbol{P}`, we
# are fixing every parameter and gate choice apart from :math:`\theta_d` in this expresion.
# For each parameter in the quantum circuit, the algorithm proceeds by evaluating this expression for each choice of
# gate :math:`P_d \in \{R_x,R_y,R_z\}` and selecting the gate which yields the minimum value.
# One might expect the number of circuit evaluations required to be 9 (3 for each gate choice), but
# since there is a 3-fold
# degeneracy in the expectation value :math:`\langle H \rangle (0)` for each of the gates, the number of
# evaluations reduces to 7.
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
# acting on the second qubit.
#
# Rotosolve
# ---------
# As a precursor to implementing Rotoselect we can analyze a version of the algorithm
# which does not optimize the choice of gates, called Rotosolve. Later, we will build on this example
# to implement Rotoselect and vary the circuit structure. 
import pennylane as qml
from pennylane import numpy as np


n_wires = 2

dev = qml.device('default.qubit',analytic=True,wires=2)



def ansatz(var):
    qml.RX(var[0], wires=0)
    qml.RY(var[1], wires=1)
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


class Rotosolve:

    def step(self, cost, thetas):
        params = thetas.copy()
        for d in range(len(params)):
            phi = 0.5
            params[d] = phi
            M_phi = cost(params)
            params[d] = phi + np.pi / 2.
            M_phi_plus = cost(params)
            params[d] = phi - np.pi / 2.
            M_phi_minus = cost(params)
            a = np.arctan2(2. * M_phi - M_phi_plus - M_phi_minus, M_phi_plus - M_phi_minus)
            params[d] = -np.pi / 2. - a + phi
            if params[d] > np.pi:
                params[d] -= 2 * np.pi
            if params[d] <= -np.pi:
                params[d] += 2 * np.pi
        return params


init_params = [0.3,0.25]
params = init_params[:]
n_steps = 30

obj1 = []
opt = Rotosolve()

for i in range(n_steps):
    obj1.append(cost(params))
    params = opt.step(cost,params)

opt = qml.GradientDescentOptimizer(1.8)

params = init_params[:]
obj2 = []
for i in range(n_steps):
    obj2.append(cost(params))
    params = opt.step(cost, params)


import matplotlib.pyplot as plt
steps = np.arange(0,n_steps)
fig, (ax1,ax2,ax3) = plt.subplots(1,3)
plt.subplot(1,3,1)
plt.plot(steps,obj2,'o-')
plt.title("grad. desc.")
plt.xlabel("steps")
plt.ylabel("cost")
plt.subplot(1,3,2)
plt.plot(steps,obj1,'o-')
plt.title("rotosolve")
plt.xlabel("steps")
plt.ylabel("cost")



##############################################################################
# Rotoselect
# ~~~~~~~~~~

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

