# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
Integration tests for odeint (ordinary differential equation integrator)
"""
import sys
import pytest

# cant test this in a file where pennylane is already imported
def test_nojax_ImportError(monkeypatch):
    with monkeypatch.context() as m:
        m.setitem(sys.modules, "jax", None)
        import pennylane as qml

        def fun(y, _):
            return y

        y0 = qml.numpy.array([1.0])
        ts = qml.numpy.array([1.0, 2.0, 3.0])
        with pytest.raises(ImportError, match="Module jax is required"):
            qml.math.odeint(fun, y0, ts)
