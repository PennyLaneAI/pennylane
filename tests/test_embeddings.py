# Copyright 2018 Xanadu Quantum Technologies Inc.

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
Unit tests for the :mod:`pennylane.templates.utils` module.
"""
# pylint: disable=protected-access,cell-var-from-loop
import pytest
from math import pi
import numpy as np
import pennylane as qml
from pennylane.templates.embeddings import (AngleEmbedding,
                                            AmplitudeEmbedding,
                                            DisplacementEmbedding,
                                            SqueezingEmbedding)


@pytest.fixture(scope="session",
                params=[1, 2, 5])
def n_subsystems(request):
    """Number of qubits or modes."""
    return request.param


@pytest.fixture(scope="session")
def qubit_device(n_subsystems):
    """Number of qubits or modes."""
    return qml.device('default.qubit', wires=n_subsystems)


@pytest.fixture(scope="session")
def gaussian_device(n_subsystems):
    """Number of qubits or modes."""
    return qml.device('default.gaussian', wires=n_subsystems)


def test_angle_embedding_state_rotx(qubit_device):
    """Checks the state produced by pennylane.templates.embeddings.AngleEmbedding()
       using the rotation='X' strategy."""

    features = [0, pi, pi/2, 0]

    @qml.qnode(qubit_device)
    def circuit(x=None):
        AngleEmbedding(features=x, n_wires=n_subsystems, rotation='X')
        return [qml.expval.PauliZ(i) for i in range(n_subsystems)]

    res = circuit(x=features)
    target = [1, 0, -1, 1, 1]

    assert np.allclose(res, target[:n_subsystems])

#
# def test_angle_embedding_state_roty(qubit_device):
#     """Checks the state produced by pennylane.templates.embeddings.AngleEmbedding()
#        using the rotation='Y' strategy."""
#
#     features = [0, pi, pi/2, 0]
#
#     @qml.qnode(qubit_device)
#     def circuit(x=None):
#         AngleEmbedding(features=x, n_wires=n_subsystems, rotation='Y')
#         return [qml.expval.PauliZ(i) for i in range(n_subsystems)]
#
#     res = circuit(x=features)
#     target = [1, 0, -1, 1, 1] #TODO
#
#     assert np.allclose(res, target[:n_subsystems])
#
#
# def test_angle_embedding_state_rotz(qubit_device):
#     """Checks the state produced by pennylane.templates.embeddings.AngleEmbedding()
#        using the rotation='Z' strategy."""
#
#     features = [0, pi, pi/2, 0]
#
#     @qml.qnode(qubit_device)
#     def circuit(x=None):
#         AngleEmbedding(features=x, n_wires=n_subsystems, rotation='Z')
#         return [qml.expval.PauliZ(i) for i in range(n_subsystems)]
#
#     res = circuit(x=features)
#     target = [1, 0, -1, 1, 1] #TODO
#
#     assert np.allclose(res, target[:n_subsystems])
#
#
# def test_angle_embedding_state_rotxy(qubit_device):
#     """Checks the state produced by pennylane.templates.embeddings.AngleEmbedding()
#        using the rotation='XY' strategy."""
#
#     features = [0, pi, pi/2, 0]
#
#     @qml.qnode(qubit_device)
#     def circuit(x=None):
#         AngleEmbedding(features=x, n_wires=n_subsystems, rotation='XY')
#         return [qml.expval.PauliZ(i) for i in range(n_subsystems)]
#
#     res = circuit(x=features)
#     target = [1, 0, -1, 1, 1] #TODO
#
#     assert np.allclose(res, target[:n_subsystems])
#
#
# def test_angle_embedding_state_rotxyz(qubit_device):
#     """Checks the state produced by pennylane.templates.embeddings.AngleEmbedding()
#        using the rotation='XYZ' strategy."""
#
#     features = [0, pi, pi/2, 0]
#
#     @qml.qnode(qubit_device)
#     def circuit(x=None):
#         AngleEmbedding(features=x, n_wires=n_subsystems, rotation='XYZ')
#         return [qml.expval.PauliZ(i) for i in range(n_subsystems)]
#
#     res = circuit(x=features)
#     target = [1, 0, -1, 1, 1] #TODO
#
#     assert np.allclose(res, target[:n_subsystems])