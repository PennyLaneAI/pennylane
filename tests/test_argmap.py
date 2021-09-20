# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
Unit tests for :mod:`pennylane.argmap`.
"""
from itertools import product, combinations
import pytest
from pennylane.argmap import ArgMap, ArgMapError, _interpret_key

raw_and_interpreted_keys = [
    (0, (0, None)),
    ((0, None), (0, None)),
    ((0, ()), (0, None)),
    ((1, (0, 1, 2)), (1, (0, 1, 2))),
    ((4, 1), (4, 1)),
    (None, (None, None)),
    ((None, None), (None, None)),
    ((), (None, None)),
]
raw_and_interpreted_keys_single_arg = [
    ((None, 0), (None, 0)),
    ((None, (1, 2)), (None, (1, 2))),
    (0, (None, 0)),
    ((1, 2), (None, (1, 2))),
    (None, (None, None)),
    ((None, None), (None, None)),
    ((), (None, None)),
]
uninterpretable_keys = ["Hi", (None, 0, 1), (None, None, None), 1.4]
uninterpretable_keys_single_arg = ["Hi", 1.4]
raw_keys_error = [
    (None, 0),
    (None, (0, 1)),
]


@pytest.mark.parametrize("raw, interpreted", raw_and_interpreted_keys)
def test_interpret_key(raw, interpreted):
    r"""Test the helper functionality ``qml.argmap._interpret_key``."""
    assert _interpret_key(raw, single_arg=False) == interpreted


@pytest.mark.parametrize("raw, interpreted", raw_and_interpreted_keys_single_arg)
def test_interpret_key_single_arg(raw, interpreted):
    r"""Test the helper functionality ``qml.argmap._interpret_key`` with ``single_arg=True``."""
    assert _interpret_key(raw, single_arg=True) == interpreted


@pytest.mark.parametrize("raw", uninterpretable_keys)
def test_interpret_key_uninterpretable(raw):
    r"""Test exception of the helper functionality
    ``qml.argmap._interpret_key`` for unreasonable keys."""
    with pytest.raises(ArgMapError, match="Could not interpret key"):
        _interpret_key(raw, single_arg=False)


@pytest.mark.parametrize("raw", uninterpretable_keys_single_arg)
def test_interpret_key_uninterpretable_single_arg(raw):
    r"""Test exception of the helper functionality
    ``qml.argmap._interpret_key`` for unreasonable keys with ``single_arg=True``"""
    with pytest.raises(ArgMapError, match="Could not interpret key"):
        _interpret_key(raw, single_arg=True)


data_lists = [
    [],
    [(i, i ** 2) for i in range(4)],
    [(i, str(i)) if i % 2 else (i, i ** 2) for i in range(4)],
    [((0, 2), "a"), ((1, 4), 3), ((0, 2), (1, 2))],
    [((0, (0,)), "a"), ((0, (1,)), 3), ((3, (2, 4, 1)), (1, 2))],
    [((0, None), "a"), ((1, (1,)), 3), ((1, (2,)), (1, 2)), ((4, None), "b")],
    [((0, None), "a"), (0, "b"), ((2, ()), "c")],
]

data_garbage = [
    "A string",
    ["a", "list", "of", "strings"],
    {"strings": 0, "as": 1, "keys": 2},
    range(10),
    lambda x: 10,
]

data_objects = [
    None,
    "A string",
    ("A", "tuple", "of", "strings"),
    lambda x: x,
]

data_invalid_arg_index = [
    [((1.2, (0,)), "a"), ((1, (2,)), "b")],
    [((None, (0,)), "a"), ((1, (2,)), "b")],
    [((1, (0,)), "a"), (((2,), (2,)), "b")],
]

data_inconsistent_keys = [
    [((0, None), "a"), ((0, 1), "b")],
    [((0, 0), "a"), ((0, (1,)), "b")],
    [((0, (0, 4, 1)), "a"), ((0, (1,)), "b")],
]

data_inconsistent_keys_single_arg = [
    [((None, (1,)), "a"), ((None, (3, 4)), "b")],
    [((None, (1,)), "a"), ((3, 4), "b")],
    [((2, 6, 1), "a"), ((3, 4), "b")],
]

data_inconsistent_values = [
    [((0, 0), "a"), ((1, 1), "b"), ((2, 2), 1.0)],
    [((0, 0), [1, 3]), ((0, 1), "b")],
    [((0, (0,)), 2), ((0, (1,)), "b")],
    [((0, (0, 4)), lambda x: x), ((0, (1, 0)), "b")],
]

data_consistent_values = [
    [(i, i ** 2) for i in range(4)],
    [(i, str(i)) for i in range(4)],
    [((0, 2), ("a",)), ((1, 4), (3,)), ((0, 2), (1, 2))],
    [((0, (0,)), -1), ((0, (1,)), 3), ((3, (2, 4, 1)), 5)],
    [((0, None), "a"), ((1, (1,)), "b"), ((1, (2,)), "c"), ((4, None), "b")],
    [((0, None), "a"), (0, "b")],
]

new_single_items = [
    ((0, 0), "a"),
    (10, 2),
    (10, "new"),
    ((1, 3), 2),
    ((0, (10,)), 0),
    ((5, None), "new"),
    ((1, None), "new"),
]

existing_single_items = [
    (1, "wrong"),
    (1, "wrong"),
    ((1, 4), "wrong"),
    ((0, (1,)), "wrong"),
    ((0, None), "wrong"),
    (0, "wrong"),
]


class TestArgMap:
    r"""Tests for the ``ArgMap`` class."""

    def test_creation_without_input(self):
        """Test creation without any data input."""
        argmap = ArgMap()
        assert argmap == {}

    @pytest.mark.parametrize(
        "data",
        [{}, {i: i ** 2 for i in range(4)}, {i: str(i) if i % 2 else i ** 2 for i in range(4)}],
    )
    def test_creation_from_dict_of_ints(self, data):
        r"""Test creation from dictionary with int keys.
        This corresponds to no parameter indices being specified and all argument being
        treated as scalars."""
        argmap = ArgMap(data)
        # keys should take the form (arg_key, None)
        assert all((key[0] is not None and key[1] is None) for key in argmap.keys())
        # Values should not be altered
        assert set(argmap.values()) == set(data.values())

    def test_creation_from_dict_of_param_tuples(self):
        r"""Test creation from dictionary with tuple[int] keys.
        This corresponds to pairs of argument and parameter indices being specified."""
        data = {(0, 1): "a", (1, 4): 3, (0, 2): (1, 2)}
        argmap = ArgMap(data)
        assert set(argmap.keys()) == set(data.keys())
        # Values should not be altered
        assert set(argmap.values()) == set(data.values())

    def test_creation_from_dict_of_arg_param_tuples(self):
        r"""Test creation from dictionary with tuple[int, tuple[int]] keys.
        This corresponds to multiple array-valued arguments."""
        data = {(0, (0,)): "a", (0, (1,)): 3, (3, (2, 4, 1)): (1, 2)}
        argmap = ArgMap(data)
        # keys should take the form (None, param_key)
        assert all((key[0] is not None and key[1] is not None) for key in argmap.keys())
        # Values should not be altered
        assert set(argmap.values()) == set(data.values())

    def test_creation_from_dict_of_full_keys(self):
        r"""Test creation from dictionary with keys that do not need to be modified for
        the ArgMap."""
        data = {(0, None): "a", (1, (1, 2, 3)): 3, (1, (2, 4, 1)): (1, 2), (4, None): "b"}
        argmap = ArgMap(data)
        # argmap should be equal to data dictionary
        assert dict(argmap) == data

    def test_creation_from_dict_with_redundant_keys(self):
        r"""Test creation from dictionary with keys that are interpreted to be the same
        by an ArgMap."""
        data = {(0, None): "a", (0, ()): "b", 0: "c"}
        argmap = ArgMap(data)
        # argmap should be equal to data dictionary
        assert dict(argmap) == {(0, None): "c"}

    @pytest.mark.parametrize("data", data_lists)
    def test_creation_from_list_of_tuples(self, data):
        r"""Test creation from dictionary with int keys.
        This corresponds to no parameter indices being specified and all argument being
        treated as scalars."""
        argmap = ArgMap(data)
        argmap_via_dict = ArgMap(dict(data))
        assert argmap == argmap_via_dict

    def test_creation_from_dict_with_single_arg(self):
        r"""Test creation from dictionary with integer keys that are
        interpreted as indices for a single argument by using ``single_arg``."""
        data = [(i, [i]) for i in range(100)]
        argmap = ArgMap(data, single_arg=True)
        # argmap should be equal to data dictionary
        assert dict(argmap) == {(None, i): [i] for i in range(100)}

        data = [((i, i + 1), str(i)) for i in range(100)]
        argmap = ArgMap(data, single_arg=True)
        # argmap should be equal to data dictionary
        assert dict(argmap) == {(None, (i, i + 1)): str(i) for i in range(100)}

    @pytest.mark.parametrize("data", data_objects)
    def test_creation_from_object_with_single_entry(self, data):
        r"""Test creation from a single object, using ``single_entry``."""
        argmap = ArgMap(data, single_entry=True)
        if data is None:
            data = {}
        # the only element of the argmap can be retrieved by
        # ``(None, None)``, ``None``, and ``()``
        assert argmap[(None, None)] == data
        assert argmap[None] == data
        assert argmap[()] == data

    @pytest.mark.parametrize("data", data_lists)
    def test_value_retrieval(self, data):
        """Test obtaining a value by key via ``__getitem__`` and ``get``."""
        argmap = ArgMap(data)
        for key in argmap.keys():
            assert argmap.get(key) == argmap.__getitem__(key)
            assert argmap.get(key) == argmap.__getitem__(key)
            assert argmap.get(key) == argmap[key]
            assert argmap.get(key) == argmap[key]

    @pytest.mark.parametrize("data, new_item", zip(data_lists, new_single_items))
    def test_item_setting_new_item(self, data, new_item):
        """Test setting a new item by key via ``__setitem__`` and indexing as well
        as the functionality of ``setdefault``."""
        argmap = ArgMap(data)
        argmap.__setitem__(new_item[0], new_item[1])
        assert argmap[new_item[0]] == new_item[1]
        argmap = ArgMap(data)
        setdefault_out = argmap.setdefault(new_item[0], new_item[1])
        assert argmap[new_item[0]] == new_item[1]
        assert setdefault_out == new_item[1]

    @pytest.mark.parametrize("data, existing_item", zip(data_lists[1:], existing_single_items))
    def test_item_setting_existing_item(self, data, existing_item):
        """Test setting an item by an existing key via ``__setitem__`` and indexing as well
        as the functionality of ``setdefault``."""
        argmap = ArgMap(data)
        argmap.__setitem__(existing_item[0], existing_item[1])
        assert argmap[existing_item[0]] == "wrong"
        argmap = ArgMap(data)
        old_value = argmap[existing_item[0]]
        setdefault_out = argmap.setdefault(existing_item[0], existing_item[1])
        assert argmap[existing_item[0]] == old_value
        assert setdefault_out == old_value

    @pytest.mark.parametrize("data", data_lists)
    def test_consistency_check_True(self, data):
        """Test that `consistency_check` is successful if it should."""
        argmap = ArgMap(data)
        argmap.consistency_check()

    @pytest.mark.parametrize("data", data_consistent_values)
    def test_consistency_check_True_values(self, data):
        """Test that `consistency_check` is successful if it should with
        usage of ``check_values=True``."""
        argmap = ArgMap(data)
        argmap.consistency_check(check_values=True)

    def test_consistency_check_True_values_single_entry(self):
        """Test that `consistency_check` is successful for ``single_entry=True``
        if it should, with usage of ``check_values=True``."""
        argmap = ArgMap([(None, "single item")], single_entry=True)
        argmap.consistency_check(check_values=True)

    @pytest.mark.parametrize("data", data_invalid_arg_index)
    def test_consistency_check_invalid_arg_index(self, data):
        """Test that `consistency_check` fails for invalid argument indices."""
        with pytest.raises(ArgMapError, match="Invalid argument index"):
            argmap = ArgMap(data)

    @pytest.mark.parametrize("data", data_inconsistent_keys)
    def test_consistency_check_inconsistent(self, data):
        """Test that `consistency_check` fails for inconsistent keys."""
        with pytest.raises(ArgMapError, match="Inconsistent keys"):
            argmap = ArgMap(data)

    @pytest.mark.parametrize("data", data_inconsistent_keys_single_arg)
    def test_consistency_check_inconsistent_single_arg(self, data):
        """Test that `consistency_check` fails for inconsistent keys for ``single_arg=True``."""
        with pytest.raises(ArgMapError, match="Inconsistent keys .* only argument index\."):
            argmap = ArgMap(data, single_arg=True)

    def test_consistency_check_None_key_wo_single_entry(self):
        """Test that `consistency_check` fails for ``None`` as key in an ArgMap
        with ``single_entry=False``."""
        with pytest.raises(ArgMapError, match="The key \(None, None\) indicates"):
            argmap = ArgMap([(None, "single object")])

    @pytest.mark.parametrize("data", data_inconsistent_values)
    def test_consistency_check_False_values(self, data):
        """Test that `consistency_check` fails for differing types of the values
        in an ArgMap, when using ``check_values=True``."""
        argmap = ArgMap(data)
        with pytest.raises(ArgMapError, match="Inconsistent value types"):
            argmap.consistency_check(check_values=True)

    def test_consistency_check_error_single_arg_not_set(self):
        """Test that `consistency_check` fails for argument index ``None`` if
        ``single_arg=False``."""
        data = {(None, 0): "a", (None, 1): "b"}
        with pytest.raises(ArgMapError, match="Invalid argument index None;"):
            argmap = ArgMap(data)

    def test_consistency_check_error_single_arg_set(self):
        """Test that `consistency_check` fails for argument index unequal ``None`` if
        ``single_arg=True``."""
        data = {(0, (0, 1)): "a", (5, (1, 2)): "b"}
        with pytest.raises(ArgMapError, match="Invalid entries in parameter"):
            argmap = ArgMap(data, single_arg=True)

    @pytest.mark.parametrize("data", data_lists)
    def test_error_unknown_key(self, data):
        """Test that ``get`` returns the default for an unknown key and that ``__getitem__``
        raises an ``ArgMapError``."""
        argmap = ArgMap(data)
        assert argmap.get(1000, "default") == "default"
        with pytest.raises(KeyError):
            argmap.__getitem__(1000)

    @pytest.mark.parametrize("data", data_lists)
    def test_error_incomprehensible_key(self, data):
        """Test that ``get`` and ``__getitem__`` raise an ``ArgMapError`` for
        uninterpretable keys."""
        argmap = ArgMap(data)
        with pytest.raises(ArgMapError, match="Could not interpret"):
            argmap.get("new_key")
        with pytest.raises(ArgMapError, match="Could not interpret"):
            argmap.__getitem__("new_key")

    @pytest.mark.parametrize("data", data_garbage)
    def test_error_garbage_data(self, data):
        """Test that instantiating an ``ArgMap`` with unreasonable objects as data
        raises an ``ArgMapError`` if ``single_entry=False``."""
        with pytest.raises(ArgMapError):
            ArgMap(data)

    def test_error_single_entry_multiple(self):
        """Test that instantiating an ``ArgMap`` and adding items raises an
        ``ArgMapError`` in the ``consistency_check`` if ``single_entry=True``."""
        argmap = ArgMap([(None, "a")], single_entry=True)
        argmap.__setitem__(1, "b")
        with pytest.raises(ArgMapError, match="ArgMap.single_entry=True but len"):
            argmap.consistency_check()

    def test_error_single_entry_wrong_key(self):
        """Test that accessing an item in an ``ArgMap`` with the wrong key
        raises an ``ArgMapError`` in the ``consistency_check`` if ``single_entry=True``."""
        argmap = ArgMap({1: "b"})
        argmap.single_entry = True
        with pytest.raises(ArgMapError, match="ArgMap.single_entry=True but key="):
            argmap.consistency_check()

    def test_error_single_arg_wrong_key(self):
        """Test that accessing an item in an ``ArgMap`` with the wrong key
        raises an ``ArgMapError`` in the ``consistency_check`` if ``single_arg=True``."""
        argmap = ArgMap({1: "b"})
        argmap.single_arg = True
        with pytest.raises(ArgMapError, match="Invalid key \(1, None\)"):
            argmap.consistency_check()

    def test_error_wrong_param_key(self):
        """Test that creating an ``ArgMap`` with an invalid param_idx
        raises an ``ArgMapError``."""
        with pytest.raises(ArgMapError, match="Invalid key \(1, 'not a tuple'\)"):
            argmap = ArgMap({(1, "not a tuple"): "b"})
        with pytest.raises(ArgMapError, match="Invalid entries in parameter index"):
            argmap = ArgMap({(1, (4, "not an int", 7)): "b"})

    def test_creation_like(self):
        """Test the usage of inheriting single_arg and single_entry
        via the keyword argument ``like``."""
        data = {(0, 3): "a", (4, 2): "b"}
        argmap = ArgMap(data)
        argmap_like = ArgMap(data, like=argmap)
        assert not argmap_like.single_arg and not argmap_like.single_entry

        data = {(0, 3): "a", (4, 2): "b"}
        argmap = ArgMap(data, single_arg=True)
        argmap_like = ArgMap(data, like=argmap)
        assert argmap_like.single_arg and not argmap_like.single_entry

        data = [1, 2, 3]
        argmap = ArgMap(data, single_entry=True)
        argmap_like = ArgMap(data, like=argmap)
        assert not argmap_like.single_arg and argmap_like.single_entry

        data = [1, 2, 3]
        argmap = ArgMap(data, single_entry=True, single_arg=True)
        argmap_like = ArgMap(data, like=argmap)
        assert argmap_like.single_arg and argmap_like.single_entry

    def test_creation_like_error_non_argmap(self):
        """Test that inheriting single_arg and single_entry from non ArgMap instance fails."""
        data = {(0, 3): "a", (4, 2): "b"}
        with pytest.raises(ArgMapError, match="Trying to inherit properties"):
            argmap_like = ArgMap(data, like="Not an ArgMap")

    def test_comparison_argmaps(self):
        """Test the comparison operations __eq__ and __neq__ between two ArgMaps."""
        data = [((1, (2,)), "a"), ((3, None), "b")]
        argmap1 = ArgMap(data)
        argmap2 = ArgMap(data)
        assert argmap1 == argmap2 and argmap2 == argmap1
        assert not argmap1 != argmap2 and not argmap2 != argmap1
        data = {(0, 3): "a", (4, 2): "b"}
        argmaps = [ArgMap(data)]
        for single_arg, single_entry in product([False, True], repeat=2):
            argmaps.append(ArgMap(data, single_arg=single_arg, single_entry=single_entry))
            argmaps[-1].consistency_check()
        assert argmaps[0] == argmaps[1] and argmaps[1] == argmaps[0]
        assert not argmaps[0] != argmaps[1] and not argmaps[1] != argmaps[0]
        for am1, am2 in combinations(argmaps[1:], r=2):
            assert am1 != am2 and am2 != am1
            assert not am1 == am2 and not am2 == am1

    def test_comparison_to_dict(self):
        """Test the comparison operations __eq__ and __neq__
        between an ArgMaps and a ``dict``."""
        data = {(1, (2,)): "a", (3, None): "b"}
        argmap = ArgMap(data)
        assert data == argmap and argmap == data
        assert not data != argmap and not argmap != data
