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
"""
Unit tests for the CaptureMeta metaclass.
"""
from inspect import signature

# pylint: disable=protected-access, undefined-variable
import pytest

from pennylane.capture.capture_meta import CaptureMeta

jax = pytest.importorskip("jax")

pytestmark = [pytest.mark.jax, pytest.mark.capture]


def test_custom_capture_meta():
    """Test that we can capture custom classes with the CaptureMeta metaclass by defining
    the _primitive_bind_call method."""

    p = jax.extend.core.Primitive("p")

    @p.def_abstract_eval
    def _(a):
        return jax.core.ShapedArray(a.shape, a.dtype)

    # pylint: disable=too-few-public-methods
    class MyObj(metaclass=CaptureMeta):
        """A CaptureMeta class with a _primitive_bind_call class method."""

        @classmethod
        def _primitive_bind_call(cls, *args, **kwargs):
            return p.bind(*args, **kwargs)

        def __init__(self, a: int, b: bool):
            self.a = a
            self.b = b

    def f(a: int, b: bool):
        # similar signature to MyObj but without init
        return a + b

    jaxpr = jax.make_jaxpr(MyObj)(0.5)

    assert len(jaxpr.eqns) == 1
    assert jaxpr.eqns[0].primitive == p

    assert signature(MyObj) == signature(f)


def test_custom_capture_meta_no_bind_primitive_call():
    """Test that an NotImplementedError is raised if the type does not define _primitive_bind_call."""

    # pylint: disable=too-few-public-methods
    class MyObj(metaclass=CaptureMeta):
        """A class that does not define _primitive_bind_call."""

        def __init__(self, a):
            self.a = a

    with pytest.raises(NotImplementedError, match="Types using CaptureMeta must implement"):
        MyObj(0.5)

    def f():
        MyObj(0.5)

    with pytest.raises(NotImplementedError, match="Types using CaptureMeta must implement"):
        jax.make_jaxpr(f)()
