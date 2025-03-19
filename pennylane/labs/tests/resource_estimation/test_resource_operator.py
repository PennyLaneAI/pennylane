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
Test the abstract ResourceOperator class
"""
import pytest

import pennylane.labs.resource_estimation as re

# pylint: disable=abstract-class-instantiated,arguments-differ,missing-function-docstring,too-few-public-methods


def test_abstract_resource_decomp():
    """Test that the _resource_decomp method is abstract."""

    class DummyClass(re.ResourceOperator):
        """Dummy class for testing"""

        @property
        def resource_params(self):
            return

        @staticmethod
        def resource_rep():
            return

    with pytest.raises(
        TypeError,
        match="Can't instantiate abstract class DummyClass with abstract method _resource_decomp",
    ):
        DummyClass()


def test_abstract_resource_params():
    """Test that the resource_params method is abstract"""

    class DummyClass(re.ResourceOperator):
        """Dummy class for testing"""

        @staticmethod
        def _resource_decomp():
            return

        def resource_rep(self):
            return

    with pytest.raises(
        TypeError,
        match="Can't instantiate abstract class DummyClass with abstract method resource_params",
    ):
        DummyClass()


def test_abstract_resource_rep():
    """Test that the resource_rep method is abstract"""

    class DummyClass(re.ResourceOperator):
        """Dummy class for testing"""

        @staticmethod
        def _resource_decomp():
            return

        @property
        def resource_params(self):
            return

    with pytest.raises(
        TypeError,
        match="Can't instantiate abstract class DummyClass with abstract method resource_rep",
    ):
        DummyClass()


def test_set_resources():
    """Test that the resources method can be overriden"""

    class DummyClass(re.ResourceOperator):
        """Dummy class for testing"""

        @property
        def resource_params(self):
            return

        @staticmethod
        def resource_rep():
            return

        @staticmethod
        def _resource_decomp():
            return

    dummy = DummyClass()
    DummyClass.set_resources(lambda _: 5)
    assert DummyClass.resources(10) == 5


def test_resource_rep_from_op():
    """Test that the resource_rep_from_op method is the composition of resource_params and resource_rep"""

    class DummyClass(re.ResourceQFT, re.ResourceOperator):
        """Dummy class for testing"""

        @staticmethod
        def _resource_decomp():
            return

        @property
        def resource_params(self):
            return {"foo": 1, "bar": 2}

        @classmethod
        def resource_rep(cls, foo, bar):
            return re.CompressedResourceOp(cls, {"foo": foo, "bar": bar})

        @staticmethod
        def tracking_name(foo, bar):
            return f"DummyClass({foo}, {bar})"

    op = DummyClass(wires=[1, 2, 3])
    assert op.resource_rep_from_op() == op.__class__.resource_rep(**op.resource_params)
