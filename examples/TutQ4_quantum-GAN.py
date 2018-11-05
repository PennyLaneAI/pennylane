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
    qml.CNOT(wires=[0, 1])
    qml.RX(w[6], wires=0)
    qml.RY(w[7], wires=0)
    qml.RZ(w[8], wires=0)


def discriminator(w):
    """Circuit that serves as a discriminator. Same as generator.

    Args:
        w (array[float]): variables of the circuit
    """
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


@qml.qnode(dev)
def true_disc_circuit(phi, theta, omega, disc_weights):
    """Feed discriminator with true samples.

    Args:
        phi (float): variable for true circuit
        theta (float): variable for true circuit
        omega (float): variable for true circuit
        disc_weights (array[float]): array of discriminator variables
    """
    true(phi, theta, omega)
    discriminator(disc_vars)
    return qml.expval.PauliZ(2)


@qml.qnode(dev)
def gen_disc_circuit(gen_vars, disc_vars):
    """Feed discriminator with samples produced by the generator.

    Args:
        gen_vars (array[float]): array of generator variables
        disc_vars (array[float]): array of discriminator variables
    """
    generator(gen_vars)
    discriminator(disc_vars)
    return qml.expval.PauliZ(2)


def prob_real_true(disc_vars):
    """Probability that the discriminator guesses correctly.

    Args:
        disc_vars: variables of the discriminator
    """
    true_disc_output = true_disc_circuit(phi, theta, omega, disc_vars)
    # convert to probability
    prob_real_true = (true_disc_output + 1) / 2
    return prob_real_true


def prob_fake_true(gen_vars, disc_vars):
    """Probability that the discriminator guesses wrong.

    Args:
        gen_vars: variables of the generator
        disc_vars: variables of the discriminator
    """
    fake_disc_output = gen_disc_circuit(gen_vars, disc_vars)
    # convert to probability
    prob_fake_true = (fake_disc_output + 1) / 2
    return prob_fake_true  # we want to minimize this prob


def disc_cost(disc_vars):
    """Cost for the discriminator is to classify a fake data sample as true.

    Args:
        disc_weights: variables of the discriminator
    """

    cost = prob_fake_true(gen_vars, disc_vars) - prob_real_true(disc_vars)
    return cost


def gen_cost(gen_vars):
    """Cost for the generator is the opposite as the discriminator cost
    in the adversarial game.

    Args:
        gen_weights: variables of the generator
    """
    return -prob_fake_true(gen_vars, disc_vars)


# Fix the angles with which the true data is generated
phi = np.pi / 6
theta = np.pi / 2
omega = np.pi / 7
# Initialize the other variables
eps = 1e-2
gen_vars = np.array([0] + [np.pi] + [0] * 7) + eps * np.random.normal(size=[9])
disc_vars = np.random.normal(size=[9])

opt = qml.GradientDescentOptimizer(0.1)

print("Training the discriminator.")
for it in range(50):
    disc_vars = opt.step(disc_cost, disc_vars)
    cost = disc_cost(disc_vars)
    if it % 5 == 0:
        print("Step {}: cost = {}".format(it, cost))

print("Probability of the discriminator to be correct: ", prob_real_true(disc_vars))
print("Probability of the discriminator to be wrong: ", prob_fake_true(gen_vars, disc_vars))

print("Training the generator.")
# train generator
for it in range(200):
    gen_vars = opt.step(gen_cost, gen_vars)
    cost = -gen_cost(gen_vars)
    if it % 5 == 0:
        print("Step {}: cost = {}".format(it, cost))

print("Probability of the discriminator to be correct: ", prob_real_true(disc_vars))
print("Probability of the discriminator to be wrong: ", prob_fake_true(gen_vars, disc_vars))

# should be close to zero at joint optimum
print("Cost of the discriminator: ", disc_cost(disc_vars))