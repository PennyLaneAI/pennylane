"""Quantum General Adverserial Network.

This demo constructs a General Adverserial Network (GAN)
from a quantum circuit that serves as a generator and a second
quantum circuit which takes the role of a discriminator.
"""

import pennylane as qml
from pennylane import numpy as np

np.random.seed(0)

dev = qml.device('default.qubit', wires=3)


def true(phi, theta, omega):
    """True distribution from which samples are taken.

    Args:
        phi (float):   x rotation angle
        theta (float): y rotation angle
        omega (float): z rotation angle
    """
    qml.Rot(phi, theta, omega, wires=0)


def routine(w):
    """Base routine used for generator and discriminator.

        Args:
            w (array[float]): variables of the circuit
    """
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


def generator(w):
    """Circuit that serves as a generator.

    Args:
        w (array[float]): variables of the circuit
    """
    routine(w)


def discriminator(w):
    """Circuit that serves as a discriminator. Same as generator.

    Args:
        w (array[float]): variables of the circuit
    """
    routine(w)


@qml.qnode(dev)
def true_disc_circuit(phi, theta, omega, disc_weights):
    """Circuit that serves as a discriminator. Same as generator.

    Args:
        w (array[float]): variables of the circuit
    """
    true(phi, theta, omega)
    discriminator(disc_weights)
    return qml.expval.PauliZ(2)


@qml.qnode(dev)
def gen_disc_circuit(gen_weights, disc_weights):
    generator(gen_weights)
    discriminator(disc_weights)
    return qml.expval.PauliZ(2)


def prob_real_true(disc_weights):
    """Probability that the discriminator guesses correctly.

    Args:
        disc_weights: variables of the discriminator
    """
    true_disc_output = true_disc_circuit(phi, theta, omega, disc_weights)
    # convert to probability
    prob_real_true = (true_disc_output + 1) / 2
    return prob_real_true


def prob_fake_true(gen_weights, disc_weights):
    """Probability that the discriminator guesses wrong.

    Args:
        gen_weights: variables of the generator
        disc_weights: variables of the discriminator
    """
    fake_disc_output = gen_disc_circuit(gen_weights, disc_weights)
    # convert to probability
    prob_fake_true = (fake_disc_output + 1) / 2
    return prob_fake_true  # want to minimize this prob


def disc_cost(disc_weights):
    """Cost for the discriminator is to classify a fake data sample as true.

    Args:
        disc_weights: variables of the discriminator
    """
    cost = prob_fake_true(gen_weights, disc_weights) - prob_real_true(disc_weights)
    return cost


def gen_cost(gen_weights):
    """Cost for the generator is the opposite as the discriminator cost
    in the adversarial game.

    Args:
        gen_weights: variables of the generator
    """
    return -prob_fake_true(gen_weights, disc_weights)

# Fix the angles with which the true data is generated
phi = np.pi / 6
theta = np.pi / 2
omega = np.pi / 7
# Initialize the other variables
gen_weights = np.random.normal(size=[9])
disc_weights = np.random.normal(size=[9])

opt = qml.GradientDescentOptimizer(0.1)

print("Training the discriminator.")
for it in range(10):
    disc_weights = opt.step(disc_cost, disc_weights)
    cost = disc_cost(disc_weights)
    print("Step {}: cost = {}".format(it+1, cost))

print("Probability of the discriminator to be correct: ", prob_real_true(disc_weights))
print("Probability of the discriminator to be wrong: ", prob_fake_true(gen_weights, disc_weights))

print("Training the generator.")
# train generator
for it in range(200):
    gen_weights = opt.step(gen_cost, gen_weights)
    cost = -gen_cost(gen_weights)
    if it % 5 == 0:
        print("Step {}: cost = {}".format(it+1, cost))

print("Probability of the discriminator to be correct: ", prob_real_true(disc_weights))
print("Probability of the discriminator to be wrong: ", prob_fake_true(gen_weights, disc_weights))

# should be close to zero at joint optimum
print("Cost of the discriminator: ", disc_cost(disc_weights))