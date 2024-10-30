import pytest

import pennylane as qml
from pennylane.labs.resource_estimation import (
    CompressedResourceOp,
    ResourceHadamard,
    ResourcesNotDefined,
    ResourceT,
)


class TestHadamard:
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


class TestT:
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
