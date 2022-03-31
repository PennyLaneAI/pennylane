# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This module contains basis set parameters defining Gaussian-type orbitals for a selected number of
atoms. The data are taken from the Basis Set Exchange `library <https://www.basissetexchange.org>`_.
The current data include the STO-3G basis set for atoms with atomic numbers 1-10.
"""

atomic_numbers = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
}

STO3G = {
    "H": {
        "orbitals": ["S"],
        "exponents": [[0.3425250914e01, 0.6239137298e00, 0.1688554040e00]],
        "coefficients": [[0.1543289673e00, 0.5353281423e00, 0.4446345422e00]],
    },
    "He": {
        "orbitals": ["S"],
        "exponents": [[0.6362421394e01, 0.1158922999e01, 0.3136497915e00]],
        "coefficients": [[0.1543289673e00, 0.5353281423e00, 0.4446345422e00]],
    },
    "Li": {
        "orbitals": ["S", "SP"],
        "exponents": [
            [0.1611957475e02, 0.2936200663e01, 0.7946504870e00],
            [0.6362897469e00, 0.1478600533e00, 0.4808867840e-01],
        ],
        "coefficients": [
            [0.1543289673e00, 0.5353281423e00, 0.4446345422e00],
            [-0.9996722919e-01, 0.3995128261e00, 0.7001154689e00],
            [0.1559162750e00, 0.6076837186e00, 0.3919573931e00],
        ],
    },
    "Be": {
        "orbitals": ["S", "SP"],
        "exponents": [
            [0.3016787069e02, 0.5495115306e01, 0.1487192653e01],
            [0.1314833110e01, 0.3055389383e00, 0.9937074560e-01],
        ],
        "coefficients": [
            [0.1543289673e00, 0.5353281423e00, 0.4446345422e00],
            [-0.9996722919e-01, 0.3995128261e00, 0.7001154689e00],
            [0.1559162750e00, 0.6076837186e00, 0.3919573931e00],
        ],
    },
    "B": {
        "orbitals": ["S", "SP"],
        "exponents": [
            [0.4879111318e02, 0.8887362172e01, 0.2405267040e01],
            [0.2236956142e01, 0.5198204999e00, 0.1690617600e00],
        ],
        "coefficients": [
            [0.1543289673e00, 0.5353281423e00, 0.4446345422e00],
            [-0.9996722919e-01, 0.3995128261e00, 0.7001154689e00],
            [0.1559162750e00, 0.6076837186e00, 0.3919573931e00],
        ],
    },
    "C": {
        "orbitals": ["S", "SP"],
        "exponents": [
            [0.7161683735e02, 0.1304509632e02, 0.3530512160e01],
            [0.2941249355e01, 0.6834830964e00, 0.2222899159e00],
        ],
        "coefficients": [
            [0.1543289673e00, 0.5353281423e00, 0.4446345422e00],
            [-0.9996722919e-01, 0.3995128261e00, 0.7001154689e00],
            [0.1559162750e00, 0.6076837186e00, 0.3919573931e00],
        ],
    },
    "N": {
        "orbitals": ["S", "SP"],
        "exponents": [
            [0.9910616896e02, 0.1805231239e02, 0.4885660238e01],
            [0.3780455879e01, 0.8784966449e00, 0.2857143744e00],
        ],
        "coefficients": [
            [0.1543289673e00, 0.5353281423e00, 0.4446345422e00],
            [-0.9996722919e-01, 0.3995128261e00, 0.7001154689e00],
            [0.1559162750e00, 0.6076837186e00, 0.3919573931e00],
        ],
    },
    "O": {
        "orbitals": ["S", "SP"],
        "exponents": [
            [0.1307093214e03, 0.2380886605e02, 0.6443608313e01],
            [0.5033151319e01, 0.1169596125e01, 0.3803889600e00],
        ],
        "coefficients": [
            [0.1543289673e00, 0.5353281423e00, 0.4446345422e00],
            [-0.9996722919e-01, 0.3995128261e00, 0.7001154689e00],
            [0.1559162750e00, 0.6076837186e00, 0.3919573931e00],
        ],
    },
    "F": {
        "orbitals": ["S", "SP"],
        "exponents": [
            [0.1666791340e03, 0.3036081233e02, 0.8216820672e01],
            [0.6464803249e01, 0.1502281245e01, 0.4885884864e00],
        ],
        "coefficients": [
            [0.1543289673e00, 0.5353281423e00, 0.4446345422e00],
            [-0.9996722919e-01, 0.3995128261e00, 0.7001154689e00],
            [0.1559162750e00, 0.6076837186e00, 0.3919573931e00],
        ],
    },
    "Ne": {
        "orbitals": ["S", "SP"],
        "exponents": [
            [0.2070156070e03, 0.3770815124e02, 0.1020529731e02],
            [0.8246315120e01, 0.1916266291e01, 0.6232292721e00],
        ],
        "coefficients": [
            [0.1543289673e00, 0.5353281423e00, 0.4446345422e00],
            [-0.9996722919e-01, 0.3995128261e00, 0.7001154689e00],
            [0.1559162750e00, 0.6076837186e00, 0.3919573931e00],
        ],
    },
}
