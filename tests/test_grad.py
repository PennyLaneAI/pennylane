# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

import pennylane as qml


@pytest.mark.parametrize("grad_fn", (qml.grad, qml.jacobian))
def test_kwarg_errors_without_qjit(grad_fn):
    """Test that errors are raised with method and h when qjit is not active."""
    def f(x):
        return x**2
    
    x = qml.numpy.array(0.5)

    with pytest.raises(ValueError, match="method = 'fd' unsupported without QJIT."):
        grad_fn(f, method="fd")(x)

    with pytest.raises(ValueError, match="unsupported without QJIT. "):
        grad_fn(f, h=1e-6)(0.5)