import pytest

from pennylane.labs.resource_estimation import ResourceConstructor

def test_abstract_resource_decomp():
    class DummyClass(ResourceConstructor):
        def resource_rep(self):
            return

    with pytest.raises(TypeError, match="Can't instantiate abstract class DummyClass with abstract method _resource_decomp"):
        DummyClass()

def test_abstract_resource_rep():
    class DummyClass(ResourceConstructor):
        def _resource_decomp():
            return

    with pytest.raises(TypeError, match="Can't instantiate abstract class DummyClass with abstract method resource_rep"):
        DummyClass()

def test_set_resources():
    class DummyClass(ResourceConstructor):
        def resource_rep(self):
            return

        def _resource_decomp():
            return

    dummy = DummyClass()
    DummyClass.set_resources(lambda _: 5)
    assert DummyClass.resources(10) == 5
