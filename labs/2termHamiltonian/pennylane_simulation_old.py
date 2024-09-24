# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



# In this script we show how to implement the simulation of time-dependent
# Hamiltonians using CFQMs and PennyLane.

# We will consider the following time-dependent Heisenberg Hamiltonian:
# -iA(t) = \frac{1}{4n}\sum_{i=1}^{n-1} \vec{\sigma}_i\vec{\sigma}_{i+1} 
#        + \frac{1}{4n}\sum_{i=1}^{n} \cos(\phi_i + \omega_i t)\sigma_i^z.

# We will divide the Hamiltonian into
# H_{k,\text{odd}}= \frac{1}{4n}\sum_{k=1}^{\left\lfloor \frac{n}{2}\right\rfloor} 
#                                     \left(\vec{\sigma}_{2k-1}\vec{\sigma}_{2k} 
#                                     + \cos(\phi_{2k-1} + \omega_{2k-1} t_k)\sigma_{2k-1}^z\right),
# H_{k,\text{even}} = \frac{1}{4n}\sum_{k=1}^{\left\lceil \frac{n}{2}\right\rceil -1} 
#                                     \left(\vec{\sigma}_{2k}\vec{\sigma}_{2k+1} 
#                                     + \cos(\phi_{2k} + \omega_{2k} t_k)\sigma_{2k}^z\right),

import pennylane as qml
from pennylane import numpy as np
import json
import os
from magnus_errors import convert_keys_to_float
from scipy.special import roots_legendre

# Select here the parameters of the Hamiltonian and simulation
n = 6
h = 0.1
T = 1
s = 2
m = 2

total_time = np.arange(0, T, h)

# Define the Heisenberg Hamiltonian

## Add the time-dependent term
phase = np.random.uniform(0, 2*np.pi, n)
freq = np.random.uniform(0, 1, n)

obs = []
for i in range(1, n):
    obs.append(qml.PauliX(i-1) @ qml.PauliX(i))
    obs.append(qml.PauliY(i-1) @ qml.PauliY(i))
    obs.append(qml.PauliZ(i-1) @ qml.PauliZ(i))
tdobs = [qml.PauliZ(i) for i in range(n)]

coeffs = list(np.ones(3*(n-1)))
def tdcoeffs(t):
    tdc = list(np.cos(phase+ freq*t))
    return tdc

def H(t, coeffs, obs, tdobs):
    coeffs = coeffs + tdcoeffs(t)
    observables = obs + tdobs
    return qml.Hamiltonian(coeffs = coeffs, 
                        observables= observables)

# Compute the coefficients and times of the commutator-free quasi-Magnus operator

dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
coeff_path = os.path.join(dir_path, 'coefficients')

with open(os.path.join(coeff_path,'zs.json'), 'r') as f:
    zs = json.load(f, object_hook=convert_keys_to_float)

##  Computing the roots of the Gauss Legendre polynomial, where the Hamiltonian is evaluated
##  in each segment
roots, _ = roots_legendre(s)
roots = (roots + 1)/2 * h

# Implement the simulation using PennyLane
## Create device
dev = qml.device('default.qubit', wires=n)

## Define the circuit
@qml.qnode(dev)
def circuit(total_time):

    # Prepare some state
    for i in range(n):
        qml.Hadamard(i)

    for t in total_time:

        # Evolve according to H
        for i in range(m):

            iA = qml.Hamiltonian([], [])
            for k in range(s):
                iA = iA + zs[s][m][i][k] * H(t + h*roots[k], coeffs, obs, tdobs)

            qml.TrotterProduct(iA, time=h, order=2*s)

    # Measure some quantity
    return qml.state()

## Run the simulation
state = circuit(total_time)

print(state)