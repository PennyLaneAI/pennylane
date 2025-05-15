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

"""Unit test module for pennylane/compiler/python_compiler/impl.py"""

import jax
import pytest

# pylint: disable=wrong-import-position

xdsl = pytest.importorskip("xdsl")

pytestmark = pytest.mark.external

from pennylane.compiler.python_compiler.impl import Compiler
from pennylane.compiler.python_compiler.jax_utils import module


def test_compiler():
    """Test that we can pass a jax module into the compiler.

    In this particular case, the compiler is not doing anything
    because this module does not contain nested modules which is what
    is expected of Catalyst.

    So, it just tests that Compiler.run does not trigger an assertion
    and returns a valid
    """

    @module
    @jax.jit
    def identity(x):
        return x

    input_module = identity(1)
    retval = Compiler.run(input_module)
    assert isinstance(retval) == isinstance(input_module)
    assert str(retval) == str(input_module)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
