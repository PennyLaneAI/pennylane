import pytest

import pennylane.labs.resource_estimation as re


def test_abstract_resource_decomp():
    class DummyClass(re.ResourceConstructor):
        def resource_params():
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
    class DummyClass(re.ResourceConstructor):
        @staticmethod
        def _resource_decomp():
            return

        def resource_rep():
            return

    with pytest.raises(
        TypeError,
        match="Can't instantiate abstract class DummyClass with abstract method resource_params",
    ):
        DummyClass()


def test_abstract_resource_rep():
    class DummyClass(re.ResourceConstructor):
        @staticmethod
        def _resource_decomp():
            return

        def resource_params():
            return

    with pytest.raises(
        TypeError,
        match="Can't instantiate abstract class DummyClass with abstract method resource_rep",
    ):
        DummyClass()


def test_set_resources():
    class DummyClass(re.ResourceConstructor):
        def resource_params():
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
