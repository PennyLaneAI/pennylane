r"""

.. _rotoselect:

Circuit structure learning with Rotoselect
==========================================

This example demonstrates how to learn the optimal selection of rotation
gates in addition to their parameters so as to minimize a cost
function. We apply the algorithm to VQE (as outlined in a prior
tutorial) and attempt to reproduce the circuit structure for that algorithm
along with the optimal parameters.
"""
##############################################################################
# Background
# ----------
#
# The effects of noise tend to increase with the depth of a quantum circuit, so
# there is incentive to keep circuits as shallow as possible. Generally, the
# chosen set of gates on a circuit are suboptimal for the task at hand. The Rotoselect
# algorithm provides a method for learning the structure of a quantum circuit, in addition
# to those parameters which optimize the cost function.
#
# VQE
# ~~~
#
# We choose to focus on the example of VQE with 2 qubits for simplicity. Here, the Hamiltonian
# is
#
# .. math::
#   H = 0.1*\sigma_x^2+0.5*\sigma_y^2
#
# which act on the second qubit in the circuit. We adopt the ansatz from the previous tutorial and
# and proceed to calculate the ground state using the Rotosolve algorithm. We then attempt to recover
# the ansatz structure (or discover something better) by switching to the Rotoselect algorithm.
#
# Rotosolve
# ~~~~~~~~~
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

    def step(self, cost, params):
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



init_params = [0.3,0.25]
params = init_params[:]
n_steps = 30

obj1 = []
opt = Rotosolve()

for i in range(n_steps):
    obj1.append(cost(params))
    opt.step(cost,params)

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


    def step(self, cost, var, gen):
        params = var[:]
        params_opt = var[:]
        generators = gen[:]
        generators_opt = gen[:]
        params_opt_cost = cost(params_opt, generators_opt)
        for d in range(len(params)):
            generator_opt = 0
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


def rot_selection(theta, generator, wires):
    if generator == 'X':
        qml.RX(theta, wires=wires)
    elif generator == 'Y':
        qml.RY(theta, wires=wires)
    elif generator == 'Z':
        qml.RZ(theta, wires=wires)
    else:
        raise Exception("Invalid generator")


def ansatz(var, generators):
    rot_selection(var[0], generators[0], wires=0)
    rot_selection(var[1], generators[1], wires=1)
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
init_params = [0.3,0.025]
params = init_params
init_generators = ['X', 'X']
generators = init_generators
for _ in range(n_steps):
    obj3.append(cost(params, generators))
    params, generators = opt.step(cost, params, generators)

print("Optimal generators are: {}".format(generators))


plt.subplot(1,3,3)
plt.title("rotoselect")
plt.xlabel("steps")
plt.ylabel("cost")
plt.plot(steps,obj3,'o-')
plt.tight_layout()
plt.show()

