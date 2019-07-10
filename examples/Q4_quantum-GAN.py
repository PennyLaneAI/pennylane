"""Quantum Generative Adverserial Network.

This demo constructs a Generative Adverserial Network (GAN)
from a quantum circuit that serves as a generator and a second
quantum circuit which takes the role of a discriminator.

Inspired by Dallaire-Demers and Killoran 2018 (arXiv:1804.08641).
"""

import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import GradientDescentOptimizer

dev = qml.device('default.qubit', wires=3)


def real(phi, theta, omega):
    """True distribution from which samples are taken.

    Args:
        phi (float):   first rotation angle
        theta (float): second rotation angle
        omega (float): third rotation angle
    """
    qml.Rot(phi, theta, omega, wires=0)


def generator(w):
    """Circuit that serves as a generator.

    Args:
        w (array[float]): variables of the circuit
    """
    qml.RX(w[0], wires=0)
    qml.RX(w[1], wires=1)
    qml.RY(w[2], wires=0)
    qml.RY(w[3], wires=1)
    qml.RZ(w[4], wires=0)
    qml.RZ(w[5], wires=1)
    qml.CNOT(wires=[0,1])
    qml.RX(w[6], wires=0)
    qml.RY(w[7], wires=0)
    qml.RZ(w[8], wires=0)


def discriminator(w):
    """Circuit that serves as a discriminator. Same structure as generator.

    Args:
        w (array[float]): variables of the circuit
    """
    qml.RX(w[0], wires=0)
    qml.RX(w[1], wires=2)
    qml.RY(w[2], wires=0)
    qml.RY(w[3], wires=2)
    qml.RZ(w[4], wires=0)
    qml.RZ(w[5], wires=2)
    qml.CNOT(wires=[1,2])
    qml.RX(w[6], wires=2)
    qml.RY(w[7], wires=2)
    qml.RZ(w[8], wires=2)


@qml.qnode(dev)
def real_disc_circuit(phi, theta, omega, disc_weights):
    """Feed discriminator with true samples.

    Args:
        phi (float): variable for true circuit
        theta (float): variable for true circuit
        omega (float): variable for true circuit
        disc_weights (array[float]): array of discriminator variables
    """
    real(phi, theta, omega)
    discriminator(disc_weights)
    return qml.expval(qml.PauliZ(2))


@qml.qnode(dev)
def gen_disc_circuit(gen_weights, disc_weights):
    """Feed discriminator with samples produced by the generator.

    Args:
        gen_weights (array[float]): array of generator variables
        disc_weights (array[float]): array of discriminator variables
    """
    generator(gen_weights)
    discriminator(disc_weights)
    return qml.expval(qml.PauliZ(2))


def prob_real_true(disc_weights):
    """Probability that the discriminator guesses correctly on real data.

    Args:
        disc_weights: variables of the discriminator
    """
    true_disc_output = real_disc_circuit(phi, theta, omega, disc_weights)
    # convert to probability
    prob_real_true = (true_disc_output + 1) / 2
    return prob_real_true


def prob_fake_true(gen_weights, disc_weights):
    """Probability that the discriminator guesses wrong on fake data.

    Args:
        gen_weights: variables of the generator
        disc_weights: variables of the discriminator
    """
    fake_disc_output = gen_disc_circuit(gen_weights, disc_weights)
    # convert to probability
    prob_fake_true = (fake_disc_output + 1) / 2
    return prob_fake_true # generator wants to minimize this prob


def disc_cost(disc_weights):
    """Cost for the discriminator. Contains two terms: the probability of classifying
    fake data as real, and the probability of classifying real data correctly.

    Args:
        disc_weights: variables of the discriminator
    """
    cost = prob_fake_true(gen_weights, disc_weights) - prob_real_true(disc_weights) 
    return cost


def gen_cost(gen_weights):
    """Cost for the generator. Contains only the probability of fake data being classified
    as real.

    Args:
        gen_weights: variables of the generator
    """
    return -prob_fake_true(gen_weights, disc_weights)


# Fix the angles with which the true data is generated
phi = np.pi / 6
theta = np.pi / 2
omega = np.pi / 7
# Initialize the other variables
np.random.seed(0)
eps = 1e-2
gen_weights = np.array([np.pi] + [0] * 8) + np.random.normal(scale=eps, size=[9])
disc_weights = np.random.normal(size=[9])

opt = GradientDescentOptimizer(0.1)

print("Training the discriminator.")
for it in range(50):
    disc_weights = opt.step(disc_cost, disc_weights) 
    cost = disc_cost(disc_weights)
    if it % 5 == 0:
        print("Step {}: cost = {}".format(it+1, cost))

print("Probability for the discriminator to classify real data correctly: ", prob_real_true(disc_weights))
print("Probability for the discriminator to classify fake data as real: ", prob_fake_true(gen_weights, disc_weights))

print("Training the generator.")
# train generator
for it in range(200):
    gen_weights = opt.step(gen_cost, gen_weights)
    cost = -gen_cost(gen_weights)
    if it % 5 == 0:
        print("Step {}: cost = {}".format(it, cost))

print("Probability for the discriminator to classify real data correctly: ", prob_real_true(disc_weights))
print("Probability for the discriminator to classify fake data as real: ", prob_fake_true(gen_weights, disc_weights))

# should be close to zero at joint optimum
print("Final cost function value: ", disc_cost(disc_weights))
