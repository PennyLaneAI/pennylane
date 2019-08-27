"""
.. _quantum_GAN:

Quantum Generative Adversarial Network
======================================

A QGAN is the quantum version of a *Generative Adversarial Network*.
These are a particular kind of deep learning model used to generate
real-looking synthetic data (such as images).

.. figure:: ../../examples/figures/biggan.png
    :align: center
    :width: 60%
    :target: javascript:void(0);

GANs have a unique structure, consisting of two models:

* the generator: its goal is to produce realistic-looking data samples.

* the discriminator: its goal is distinguish fake data produced by the generator from real data.

Training of GANs proceeds as follows:

1. “Real” data is captured in some *training dataset*.

2. The *generator* produces “fake” data by starting with a random input
   vector and transforming it into an output.

3. The *discriminator* is fed samples of both real and fake data and
   must decide which label to assign (‘real’ or ‘fake’).

4. Training consists of alternating steps:
   (i) the generator is frozen and the discriminator is trained to distinguish real from fake. 
   (ii) the discriminator is frozen and the generator is trained to fool the
   discriminator. The gradient of the discriminator’s output provides a
   training signal for the generator to improve its fake generated data.

5. Eventually, this training method should converge to a stage where the
   generator is producing realistic data and the discriminator can’t
   tell real from fake.

.. note:: 
    Training is done via the *gradient descent* algorithm,
    updating only the weights associated with the generator (or vice versa)
    at each step. There is no internal structure imposed on the generator or
    discriminator models except that they are differentiable.

.. figure:: ../../examples/figures/gan.png
    :align: center
    :width: 90%
    :target: javascript:void(0);

This demo constructs a Quantum Generative Adversarial Network (QGAN)
(`Lloyd and Weedbrook
(2018) <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.121.040502>`__,
`Dallaire-Demers and Killoran
(2018) <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.98.012324>`__)
using two subcircuits, a *generator* and a *discriminator*.
"""

##############################################################################
# Imports
# ~~~~~~~
#
# As usual, we import PennyLane, the PennyLane-provided version of NumPy,
# and an optimizer.

import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import GradientDescentOptimizer

##############################################################################
# instantiate a 3-qubit device

dev = qml.device("default.qubit", wires=3)

##############################################################################
# Classical and quantum nodes
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# In classical GANs, the starting point is to draw samples either from
# some “real data” distribution, or from the generator, and feed them to
# the discriminator. In this QGAN example, we will use a quantum circuit
# to generate the real data.
#
# For this simple example, our real data will be a qubit that has been
# rotated (from the starting state :math:`\left|0\right\rangle`) to some
# arbitrary, but fixed, state. We will use the
# PennyLane :mod:`Rot <pennylane.ops.qubit.Rot>` operation.


def real(phi, theta, omega):
    qml.Rot(phi, theta, omega, wires=0)


##############################################################################
# For the generator and discriminator, we will choose the same basic
# circuit structure, but acting on different wires.
#
# Both the real data circuit and the generator will output on wire 0,
# which will be connected as an input to the discriminator. Wire 1 is
# provided as a workspace for the generator, while the discriminator’s
# output will be on wire 2.


def generator(w):
    qml.RX(w[0], wires=0)
    qml.RX(w[1], wires=1)
    qml.RY(w[2], wires=0)
    qml.RY(w[3], wires=1)
    qml.RZ(w[4], wires=0)
    qml.RZ(w[5], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RX(w[6], wires=0)
    qml.RY(w[7], wires=0)
    qml.RZ(w[8], wires=0)


def discriminator(w):
    qml.RX(w[0], wires=0)
    qml.RX(w[1], wires=2)
    qml.RY(w[2], wires=0)
    qml.RY(w[3], wires=2)
    qml.RZ(w[4], wires=0)
    qml.RZ(w[5], wires=2)
    qml.CNOT(wires=[1, 2])
    qml.RX(w[6], wires=2)
    qml.RY(w[7], wires=2)
    qml.RZ(w[8], wires=2)


##############################################################################
# We create two QNodes. One where the real data source is wired up to the
# discriminator, and one where the generator is connected to the
# discriminator. They can both be run on the same device.


@qml.qnode(dev)
def real_disc_circuit(phi, theta, omega, disc_weights):
    real(phi, theta, omega)
    discriminator(disc_weights)
    return qml.expval(qml.PauliZ(2))


@qml.qnode(dev)
def gen_disc_circuit(gen_weights, disc_weights):
    generator(gen_weights)
    discriminator(disc_weights)
    return qml.expval(qml.PauliZ(2))


##############################################################################
# Cost
# ~~~~
#
# There are two ingredients to the cost here:
#
# 1. the probability that the discriminator correctly **classifies real
#    data as real.**
# 2. the probability that the discriminator **classifies fake data (i.e.,
#    a state prepared by the generator) as real.**
#
# The discriminator’s objective is to maximize the probability of
# correctly classifying real data, while minimizing the probability of
# mistakenly classifying fake data.
#
# The generator’s objective is to maximize the probability that the
# discriminator accepts fake data as real.


def prob_real_true(disc_weights):
    true_disc_output = real_disc_circuit(phi, theta, omega, disc_weights)
    # convert from expectation of Pauli-Z [-1,1] to probability [0,1]
    prob_real_true = (true_disc_output + 1) / 2
    return prob_real_true


def prob_fake_true(gen_weights, disc_weights):
    fake_disc_output = gen_disc_circuit(gen_weights, disc_weights)
    # convert from expectation of Pauli-Z [-1,1] to probability [0,1]
    prob_fake_true = (fake_disc_output + 1) / 2
    return prob_fake_true  # generator wants to minimize this prob


def disc_cost(disc_weights):
    cost = prob_fake_true(gen_weights, disc_weights) - prob_real_true(disc_weights)
    return cost 
    # this ensures that prob_fake_true is minimized and prob_real_true is
    # maximized to get the minimum (largest negative 0-1=-1) cost


def gen_cost(gen_weights):
    return -prob_fake_true(gen_weights, disc_weights)
    # -ve sign ensures that gen_cost is minimum when prob_fake_true is close
    # to 1 and the generator was able to fool the discriminator as intended

##############################################################################
# Optimization
# ~~~~~~~~~~~~
#
# We initialize the fixed angles of the “real data” circuit, as well as
# the initial parameters for both generator and discriminator. These are
# chosen so that the generator initially prepares a state on wire 0 that
# is very close to the :math:`\left| 1 \right\rangle` state.

phi = np.pi / 6
theta = np.pi / 2
omega = np.pi / 7
np.random.seed(0)
eps = 1e-2
gen_weights = np.array([np.pi] + [0] * 8) + np.random.normal(scale=eps, size=[9])
disc_weights = np.random.normal(size=[9])

##############################################################################
# We begin by creating the optimizer:

opt = GradientDescentOptimizer(0.1)

##############################################################################
# In the first stage of training, we optimize the discriminator while
# keeping the generator parameters fixed.

for it in range(50):
    disc_weights = opt.step(disc_cost, disc_weights)
    cost = disc_cost(disc_weights)
    if it % 5 == 0:
        print("Step {}: cost = {}".format(it + 1, cost))

##############################################################################
# At the discriminator’s optimum, the probability for the discriminator to
# correctly classify the real data should be close to one.

print(prob_real_true(disc_weights))


##############################################################################
# For comparison, we check how the discriminator classifies the
# generator’s (still unoptimized) fake data:

print(prob_fake_true(gen_weights, disc_weights))


##############################################################################
# In the adverserial game we have to now train the generator to better
# fool the discriminator (we can continue training the models in an
# alternating fashion until we reach the optimum point of the two-player
# adversarial game).

for it in range(200):
    gen_weights = opt.step(gen_cost, gen_weights)
    cost = -gen_cost(gen_weights)
    if it % 5 == 0:
        print("Step {}: cost = {}".format(it, cost))


##############################################################################
# At the optimum of the generator, the probability for the discriminator
# to be fooled should be close to 1.

print(prob_real_true(disc_weights))


##############################################################################
# At the joint optimum the overall cost will be close to zero.

print(disc_cost(disc_weights))


##############################################################################
# The generator has successfully learned how to simulate the real data
# enough to fool the discriminator.
