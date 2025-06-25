# Copyright 2025 Xanadu Quantum Technologies Inc.

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
Tests for the CotransformCache object.
"""

import pennylane as qml
from pennylane.transforms.core.cotransform_cache import CotransformCache


def test_simple_classical_jacobian():
    """Test the calculation of a simple classical jacobian."""

    @qml.gradients.param_shift
    @qml.transforms.split_non_commuting
    @qml.qnode(qml.device("default.qubit"))
    def c(x, y):
        qml.RX(2 * x, 0)
        qml.RY(x * y, 0)
        return qml.expval(qml.Z(0)), qml.expval(qml.X(0))

    ps_container = c.transform_program[-1]
    x, y = qml.numpy.array(0.5), qml.numpy.array(3.0)

    a = CotransformCache(c, (x, y), {})
    for i in range(2):
        x_jac, y_jac = a.get_classical_jacobian(ps_container, i)
        assert qml.math.allclose(x_jac, qml.numpy.array([2.0, 3.0]))
        assert qml.math.allclose(y_jac, qml.numpy.array([0.0, 0.5]))
