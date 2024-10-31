import pennylane.labs.resource_estimation as re

# pylint: disable=use-implicit-booleaness-not-comparison


class TestIdentity:
    """Test ResourceIdentity"""

    def test_resources(self):
        """ResourceIdentity should have empty resources"""
        op = re.ResourceIdentity()
        assert op.resources() == {}

    def test_resource_rep(self):
        """Test the compressed representation"""
        expected = re.CompressedResourceOp(re.ResourceIdentity, {})
        assert re.ResourceIdentity.resource_rep() == expected

    def test_resources_from_rep(self):
        """Test that the resources can be computed from the compressed representation"""
        op = re.ResourceIdentity()
        assert op.resources(**re.ResourceIdentity.resource_rep().params) == {}
