r"""
Doubly stochastic gradient descent
==================================

In this tutorial we investigate and implement the doubly stochastic gradient descent
paper from `Ryan Sweke et al. (2019) <https://arxiv.org/abs/1910.01155>`__. In this paper,
it is shown that quantum gradient descent, where a finite number of measurement samples
(or *shots*) are used to estimate the gradient, is a form of stochastic gradient descent.
Furthermore, if the optimization involves a linear combination of expectation values
(such as VQE), sampling from the terms in this linear combination can further reduce required
resources, allowing for "doubly stochastic gradient descent".

Background
----------

In classical machine learning, `stochastic gradient descent
<https://en.wikipedia.org/wiki/Stochastic_gradient_descent>`__ is a common optimization strategy
where the standard gradient descent parameter update rule,

.. math:: \theta^{(t+1)} = \theta^{(t)} - \eta \nabla \mathcal{L}(\theta^{(t)}),

is modified such that

.. math:: \theta^{(t+1)} = \theta^{(t)} - \eta g^{(t)}(\theta^{(t)})

where :math:`\eta` is the step-size, and :math:`\{g^{(t)}(\theta)\}` is a sequence of random
variables such that

.. math:: \mathbb{E}[g^{(t)}(\theta)] = \nabla\mathcal{L}(\theta).

In general, stochastic gradient descent is preferred over standard gradient
descent for several reasons:

1. Samples of the gradient estimator :math:`g^{(t)}(\theta)` can typically
   be computed much more efficiently than :math:`\mathcal{L}(\theta)`,

2. Stochasticity helps avoid local minima and saddle points,

3. Convergence is guaranteed to be on par with regular gradient descent.

In variational quantum algorithms, a parametrized quantum circuit :math:`U(\theta)`
is optimized by a classical optimization loop in order to minimize a function of the expectation
values. For example, consider the expectation values

.. math:: \langle A_i \rangle = \langle 0 | U(\theta)^\dagger A_i U(\theta) | 0\rangle

for a set of observables :math:`\{A_i\}`, and loss function

.. math:: \mathcal{L}(\theta, \langle A_1 \rangle, \dots, \langle A_M \rangle).

While the expectation values can be calculated analytically in classical simulations,
on quantum hardware we are limited to *sampling* from the expectation values; as the
number of samples (or shots) increase, we converge on the analytic expectation value, but can
never recover the exact expression. Furthermore, the parameter-shift rule
(`Schuld et al., 2018 <https://arxiv.org/abs/1811.11184>`__) allows for analytic
quantum gradients to be computed from a linear combination of the variational circuits'
expectation values.

Putting these two results together, `Sweke et al. (2019) <https://arxiv.org/abs/1910.01155>`__
show that samples of the expectation value fed into the parameter-shift rule provide
unbiased estimators of the quantum gradient---resulting in a form of stochastic gradient descent.
Moreover, they show that convergence of the stochastic gradient descent is guaranteed
to be on par with standard gradient descent, even in the case where the number
of shots is 1!

.. note::

    It is worth noting that the smaller the number of shots used, the larger the
    variance in the estimated expectation value. As a result, it may take
    more optimization steps for convergence than using a larger number of shots,
    or an exact value.

    At the same time, a reduced number of shots may significantly reduce the
    wall time of each optimization step, leading to a reduction in the overall
    optimization time.

Let's consider a simple example in PennyLane, comparing analytic gradient
descent (with exact expectation values) to stochastic gradient descent
using a finite number of shots.

"""

##############################################################################
# A single-shot stochastic gradient descent
# -----------------------------------------
#
# Consider the Hamiltonian
#
# .. math::
#
#     H = \begin{bmatrix}
#           8 & 4 & 0 & -6\\
#           4 & 0 & 4 & 0\\
#           0 & 4 & 8 & 0\\
#           -6 & 0 & 0 & 0
#         \end{bmatrix}.
#
# We can solve for the ground state energy using
# the variational quantum eigensolver (VQE) algorithm.
#
# Let's use the ``default.qubit`` simulator for both the analytic gradient,
# as well as the estimated gradient using number of shots :math:`N\in\{1, 100\}`.

import pennylane as qml
from pennylane import numpy as np

np.random.seed(2)

from pennylane import expval
from pennylane.init import strong_ent_layers_uniform
from pennylane.templates.layers import StronglyEntanglingLayers

num_layers = 2
num_wires = 2
eta = 0.01
steps = 200

dev_GD = qml.device("default.qubit", wires=num_wires, analytic=True)
dev_SGD1 = qml.device("default.qubit", wires=num_wires, analytic=False, shots=1)
dev_SGD100 = qml.device("default.qubit", wires=num_wires, analytic=False, shots=100)

##############################################################################
# We can use ``qml.Hermitian`` to directly specify that we want to measure
# the expectation value of the matrix :math:`H`:

H = np.array([[8, 4, 0, -6],
              [4, 0, 4, 0],
              [0, 4, 8, 0],
              [-6, 0, 0, 0]])

def circuit(params):
    StronglyEntanglingLayers(*params, wires=[0, 1])
    return expval(qml.Hermitian(H, wires=[0, 1]))


##############################################################################
# Now, we create three QNodes, each corresponding to a device above,
# and optimize them using gradient descent via the parameter-shift rule.

qnode_GD = qml.QNode(circuit, dev_GD)
qnode_SGD1 = qml.QNode(circuit, dev_SGD1)
qnode_SGD100 = qml.QNode(circuit, dev_SGD100)

init_params = strong_ent_layers_uniform(num_layers, num_wires)

# Optimizing using exact gradient descent

cost_GD = []
params_GD = init_params
opt = qml.GradientDescentOptimizer(eta)

for _ in range(steps):
    cost_GD.append(qnode_GD(params_GD))
    params_GD = opt.step(qnode_GD, params_GD)

# Optimizing using stochastic gradient descent with shots=1

cost_SGD1 = []
params_SGD1 = init_params
opt = qml.GradientDescentOptimizer(eta)

for _ in range(steps):
    cost_SGD1.append(qnode_SGD1(params_SGD1))
    params_SGD1 = opt.step(qnode_SGD1, params_SGD1)

# Optimizing using stochastic gradient descent with shots=100

cost_SGD100 = []
params_SGD100 = init_params
opt = qml.GradientDescentOptimizer(eta)

for _ in range(steps):
    cost_SGD100.append(qnode_SGD100(params_SGD100))
    params_SGD100 = opt.step(qnode_SGD100, params_SGD100)


##############################################################################
# We can now plot the cost against optimization step for the three cases.

from matplotlib import pyplot as plt

plt.style.use("seaborn")
plt.plot(cost_GD[:100], label="Vanilla gradient descent")
plt.plot(cost_SGD100[:100], label="Stochastic gradient descent, $shots=100$")
plt.plot(cost_SGD1[:100], ".", label="Stochastic gradient descent, $shots=1$")

# analytic ground state
min_energy = min(np.linalg.eigvalsh(H))
plt.hlines(min_energy, 0, 100, linestyles=':', label="Ground-state energy")

plt.ylabel("Cost function value")
plt.xlabel("Optimization steps")
plt.legend()
plt.show()

##############################################################################
# Using the trained parameters from each optimization strategy, we can
# evaluate the analytic quantum device:

print("Vanilla gradient descent min energy = ", qnode_GD(params_GD))
print("Stochastic gradient descent (shots=100) min energy = ", qnode_GD(params_SGD100))
print("Stochastic gradient descent (shots=1) min energy = ", qnode_GD(params_SGD1))

##############################################################################
# Amazingly, we see that even the ``shots=1`` optimization converged
# to a reasonably close approximation of ground-state energy!

##############################################################################
# Doubly stochastic gradient descent for VQE
# ------------------------------------------
#
# As noted in `Sweke et al. (2019) <https://arxiv.org/abs/1910.01155>`__,
# variational quantum algorithms often include terms consisting of linear combinations
# of expectation values. This is true of the parameter-shift rule (where the
# gradient of each parameter is determined by shifting the parameter by macroscopic
# amounts and taking the difference), as well as VQE, where the Hamiltonian
# is usually decomposed into a sum of Pauli expectation values.
#
# Consider the Hamiltonian from the previous section. As this Hamiltonian is an
# Hermitian observable, we can always express it as a sum of Pauli matrices using
# the relation
#
# .. math::
#
#     H = \sum_{i,j=I,x,y,z} a_{i,j} (\sigma_i\otimes \sigma_j),
#         ~~a_{i,j} = \frac{1}{4}\text{tr}[(\sigma_i\otimes \sigma_j )H].
#
# Applying this, we can see that
#
# .. math::
#
#     H = 4  + 2I\otimes X + 4I \otimes Z - X\otimes X + 5 Y\otimes Y + 2Z\otimes X.
#
# To apply "doubly stochastic" gradient descent, we simply apply the stochastic
# gradient descent approach from above, but in addition also uniformly sample
# a subset of the terms for the Hamiltonian expectation at each optimization step.
# This inserts another element of stochasticity into the system---all the while
# convergence continues to be guaranteed!
#
# Let's create a QNode that randomly samples three terms from the above
# Hamiltonian as the observable to be measured.

I = np.identity(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])

terms = np.array([2*np.kron(I, X),
                 4*np.kron(I, Z),
                 -np.kron(X, X),
                 5*np.kron(Y, Y),
                 2*np.kron(Z, X)])

dev = qml.device("default.qubit", wires=num_wires, analytic=False, shots=100)

@qml.qnode(dev)
def sampled_terms(params, A=None):
    StronglyEntanglingLayers(*params, wires=[0, 1])
    return expval(qml.Hermitian(A, wires=[0, 1]))

def loss(params):
    n = np.random.choice(np.arange(5), size=3, replace=False)
    A = np.sum(terms[n], axis=0)
    return 4 + (5/3)*sampled_terms(params, A=A)

##############################################################################
# Optimizing the circuit using gradient descent via the parameter-shift rule:

cost = []
params = init_params
opt = qml.AdamOptimizer(0.01)

for _ in range(250):
    cost.append(loss(params))
    params = opt.step(loss, params)

##############################################################################
# Plotting the cost versus optimization step:

def moving_average(data, n=3) :
    ret = np.cumsum(data, dtype=np.float64)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

average = np.vstack([np.arange(25, 200), moving_average(cost, n=50)[:-26]])

plt.plot(cost_GD, label="Vanilla gradient descent")
plt.plot(cost, ".", label="Doubly stochastic gradient descent")
plt.plot(average[0], average[1], "--", label="Doubly stochastic gradient descent moving average")
plt.hlines(min_energy, 0, 200, linestyles=':', label="Ground state energy")

plt.ylabel("Cost function value")
plt.xlabel("Optimization steps")
plt.xlim(-2, 200)
plt.legend()
plt.show()

##############################################################################
# Finally, verifying that the doubly stochastic gradient descent optimization
# correctly provides the ground state energy when evaluated for a larger
# number of shots:

print("Doubly stochastic gradient descent min energy = ", qnode_GD(params))

##############################################################################
# References
# ----------
#
# 1. Ryan Sweke, Frederik Wilde, Johannes Meyer, Maria Schuld, Paul K. Fährmann,
#    Barthélémy Meynard-Piganeau, Jens Eisert. "Stochastic gradient descent for
#    hybrid quantum-classical optimization." `arXiv:1910.01155
#    <https://arxiv.org/abs/1910.01155>`__, 2019.
