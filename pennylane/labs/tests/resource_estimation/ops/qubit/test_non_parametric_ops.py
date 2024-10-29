import pytest

import pennylane as qml
from pennylane.labs.resource_estimation import CompressedResourceOp, ResourceHadamard, ResourceS, ResourceT, ResourcesNotDefined

class TestHadamard():
    """Tests for ResourceHadamard"""

    def test_resources(self):
        """Test that ResourceHadamard does not implement a decomposition"""
        op = ResourceHadamard(0)
        with pytest.raises(ResourcesNotDefined):
            op.resources()

    def test_resource_rep(self):
        """Test that the compact representation is correct"""
        op = ResourceHadamard(0)
        expected = CompressedResourceOp(qml.Hadamard, {})
        assert op.resource_rep() == expected

class TestS():
    """Tests for ResourceS"""

    def test_resources(self):
        """Test that S decomposes into two Ts"""
        op = ResourceS(0)
        expected = {CompressedResourceOp(qml.T, {}): 2}
        assert op.resources() == expected

    def test_resource_rep(self):
        """Test that the compressed representation is correct"""
        op = ResourceS(0)
        expected = CompressedResourceOp(qml.S, {})
        assert op.resource_rep() == expected

    def test_resources_from_rep(self):
        """Test that the compressed representation yields the correct resources"""
        op = ResourceS(0)
        expected = {CompressedResourceOp(qml.T, {}): 2}
        assert op.resources(**op.resource_rep().params) == expected

class TestT():
    """Tests for ResourceT"""

    def test_resources(self):
        """Test that ResourceT does not implement a decomposition"""
        op = ResourceT(0)
        with pytest.raises(ResourcesNotDefined):
            op.resources()

    def test_resource_rep(self):
        """Test that the compact representation is correct"""
        op = ResourceT(0)
        expected = CompressedResourceOp(qml.T, {})
        assert op.resource_rep() == expected
