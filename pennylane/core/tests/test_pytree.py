import dataclasses
from typing import Generic, TypeVar

import jax
import pytest
from flax import serialization

from simple_pytree import Pytree, field, static_field


class TestPytree:
    def test_immutable_pytree(self):
        class Foo(Pytree):
            x: int = static_field()

            def __init__(self, y) -> None:
                self.x = 2
                self.y = y

        pytree = Foo(y=3)

        leaves = jax.tree_util.tree_leaves(pytree)
        assert leaves == [3]

        pytree = jax.tree_map(lambda x: x * 2, pytree)
        assert pytree.x == 2
        assert pytree.y == 6

        pytree = pytree.replace(x=3)
        assert pytree.x == 3
        assert pytree.y == 6

        with pytest.raises(
            AttributeError, match="is immutable, trying to update field"
        ):
            pytree.x = 4

    def test_immutable_pytree_dataclass(self):
        @dataclasses.dataclass(frozen=True)
        class Foo(Pytree):
            y: int = field()
            x: int = static_field(2)

        pytree = Foo(y=3)

        leaves = jax.tree_util.tree_leaves(pytree)
        assert leaves == [3]

        pytree = jax.tree_map(lambda x: x * 2, pytree)
        assert pytree.x == 2
        assert pytree.y == 6

        pytree = pytree.replace(x=3)
        assert pytree.x == 3
        assert pytree.y == 6

        with pytest.raises(AttributeError, match="cannot assign to field"):
            pytree.x = 4

    def test_jit(self):
        @dataclasses.dataclass
        class Foo(Pytree):
            a: int
            b: int = static_field()

        module = Foo(a=1, b=2)

        @jax.jit
        def f(m: Foo):
            return m.a + m.b

        assert f(module) == 3

    def test_flax_serialization(self):
        class Bar(Pytree):
            a: int = static_field()

            def __init__(self, a, b):
                self.a = a
                self.b = b

        @dataclasses.dataclass
        class Foo(Pytree):
            bar: Bar
            c: int
            d: int = static_field()

        foo: Foo = Foo(bar=Bar(a=1, b=2), c=3, d=4)

        state_dict = serialization.to_state_dict(foo)

        assert state_dict == {
            "bar": {
                "b": 2,
            },
            "c": 3,
        }

        state_dict["bar"]["b"] = 5

        foo = serialization.from_state_dict(foo, state_dict)

        assert foo.bar.b == 5

        del state_dict["bar"]["b"]

        with pytest.raises(ValueError, match="Missing field"):
            serialization.from_state_dict(foo, state_dict)

        state_dict["bar"]["b"] = 5

        # add unknown field
        state_dict["x"] = 6

        with pytest.raises(ValueError, match="Unknown field"):
            serialization.from_state_dict(foo, state_dict)

    def test_generics(self):
        T = TypeVar("T")

        class MyClass(Pytree, Generic[T]):
            def __init__(self, x: T):
                self.x = x

        MyClass[int]

    def test_key_paths(self):
        @dataclasses.dataclass
        class Bar(Pytree):
            a: int = 1
            b: int = static_field(2)

        @dataclasses.dataclass
        class Foo(Pytree):
            x: int = 3
            y: int = static_field(4)
            z: Bar = field(default_factory=Bar)

        foo = Foo()

        path_values, treedef = jax.tree_util.tree_flatten_with_path(foo)
        path_values = [(list(map(str, path)), value) for path, value in path_values]

        assert path_values[0] == ([".x"], 3)
        assert path_values[1] == ([".z", ".a"], 1)

    def test_setter_attribute_allowed(self):
        n = None

        class SetterDescriptor:
            def __set__(self, _, value):
                nonlocal n
                n = value

        class Foo(Pytree):
            x = SetterDescriptor()

        foo = Foo()
        foo.x = 1

        assert n == 1

        with pytest.raises(AttributeError, match=r"<.*> is immutable"):
            foo.y = 2

    def test_replace_unknown_fields_error(self):
        class Foo(Pytree):
            pass

        with pytest.raises(ValueError, match="Trying to replace unknown fields"):
            Foo().replace(y=1)

    def test_dataclass_inheritance(self):
        @dataclasses.dataclass
        class A(Pytree):
            a: int = 1
            b: int = static_field(2)

        @dataclasses.dataclass
        class B(A):
            c: int = 3

        pytree = B()
        leaves = jax.tree_util.tree_leaves(pytree)
        assert leaves == [1, 3]

    def test_pytree_with_new(self):
        class A(Pytree):
            def __init__(self, a):
                self.a = a

            def __new__(cls, a):
                return super().__new__(cls)

        pytree = A(a=1)

        pytree = jax.tree_map(lambda x: x * 2, pytree)


class TestMutablePytree:
    def test_pytree(self):
        class Foo(Pytree, mutable=True):
            x: int = static_field()

            def __init__(self, y) -> None:
                self.x = 2
                self.y = y

        pytree = Foo(y=3)

        leaves = jax.tree_util.tree_leaves(pytree)
        assert leaves == [3]

        pytree = jax.tree_map(lambda x: x * 2, pytree)
        assert pytree.x == 2
        assert pytree.y == 6

        pytree = pytree.replace(x=3)
        assert pytree.x == 3
        assert pytree.y == 6

        # test mutation
        pytree.x = 4
        assert pytree.x == 4

    def test_pytree_dataclass(self):
        @dataclasses.dataclass
        class Foo(Pytree, mutable=True):
            y: int = field()
            x: int = static_field(2)

        pytree: Foo = Foo(y=3)

        leaves = jax.tree_util.tree_leaves(pytree)
        assert leaves == [3]

        pytree = jax.tree_map(lambda x: x * 2, pytree)
        assert pytree.x == 2
        assert pytree.y == 6

        pytree = pytree.replace(x=3)
        assert pytree.x == 3
        assert pytree.y == 6

        # test mutation
        pytree.x = 4
        assert pytree.x == 4
