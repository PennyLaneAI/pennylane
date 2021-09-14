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
This module contains the :class:`ArgMap` class, which is a flexible-access container.
"""
from functools import lru_cache
from pennylane import numpy as np


class ArgMapError(Exception):
    r"""Exception raised by an :class:`~.pennylane.argmap.ArgMap` instance
    when it is unable to create an instance, or access or write a requested item."""


@lru_cache(maxsize=None)
def _interpret_key(key, single_arg):
    r"""Interpret ArgMap key into argument and parameter index."""
    if single_arg:
        if isinstance(key, tuple):
            assert all((np.issubdtype(type(k), int) for k in key))
        return None, key
    if isinstance(key, tuple):
        if len(key) == 2:
            return key
        if len(key) == 0:
            return None, None
        return None, key
    if np.issubdtype(type(key), int):
        return key, None
    if key is None:
        return None, None
    raise ArgMapError(f"Could not interpret key {key}.")


class ArgMap(dict):
    r"""Argument index map for storing objects per QNode argument and parameter.

    Args:
        data (list[tuple] or dict or object): Data to store in the ArgMap.
            If not a dict and ``single_object=False``, ``data`` must be possible
            to be parsed via ``dict(data)``.
        single_arg (bool): Whether the ``ArgMap`` stores objects for a single argument.
        single_object (bool): Whether the ``ArgMap`` stores only a single object.

    An ``ArgMap`` acts as a dictionary with a restricted type of keys in the context of
    a given ``QNode``. Each key takes the form ``tuple[int, tuple[int] or int]``.
    The first ``int`` in the tuple describes the argument index of the ``QNode``,
    while the second entry describes the index within that argument.
    That is ``(2, (0, 3))`` describes the parameter in the ``0``th row and ``3``rd
    column of the ``2``nd argument of a QNode. The second entry of the tuple may be an
    ``int`` instead, indexing a one-dimensional array.

    If the argument index (first entry) of a key is ``None``, this indicates that there is
    only one argument in the QNode, and all keys of the ``ArgMap`` should have the form
    ``(None, tuple[int] or int)``.
    If the parameter index (second entry) of a key is ``None``, the respective argument
    is a scalar and no other key with the same argument index can exist in the ``ArgMap``.
    If both entries are ``None``, i.e. ``key=(None, None)``, the QNode only takes a single
    scalar argument.

    The main feature of an ``ArgMap`` is its flexible access: Users can specify only an
    argument index or only a parameter index if this fully uniquely describes the
    parameter.

    **Example**
    Suppose we have the following ``QNode`` instance:

    .. code-block:: python

        dev = qml.device('default.qubit', wires=2)
        @qml.qnode(dev)
        def circuit(arg1, arg2, arg3):
            qml.Rot(arg1, wires=0)
            qml.RX(arg2, wires=0)
            qml.Rot(arg3, wires=1)
            return qml.expval(qml.PauliY(0) @ qml.PauliY(1))

    Then we might want to set up an ``ArgMap`` to store the generators of the rotations in
    this circuit, depending on the parameter that controls them:

    .. code-block:: python

        generators = qml.ArgMap({
            (0, 0): qml.PauliZ(0),
            (0, 1): qml.PauliY(0),
            (0, 2): qml.PauliZ(0),
            1: qml.PauliX(0),
            (1, 0): qml.PauliZ(1),
            (1, 1): qml.PauliY(1),
            (1, 2): qml.PauliZ(1),
        })

    Then the generators can be accessed by using the tuples ``(arg_index, par_index)`` as key
    with ``par_index=None`` for the ``PauliX(0)`` generator. The latter may alternatively
    be accessed directly via ``generators[1]``.
    """

    def __init__(self, data, single_arg=False, single_object=False):
        self.single_arg = single_arg
        self.single_object = single_object
        _data = self._preprocess_data(data)
        super().__init__(_data)

    def _preprocess_data(self, data):
        if self.single_object:
            return {(None, None): data}

        if not isinstance(data, dict):
            try:
                data = dict(data)
            except (ValueError, TypeError) as e:
                raise ArgMapError(
                    f"The input could not be interpreted as dictionary; input:\n{data}"
                ) from e

        return {_interpret_key(key, self.single_arg): val for key, val in data.items()}

    def __getitem__(self, key):
        key = _interpret_key(key, self.single_arg)
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        key = _interpret_key(key, self.single_arg)
        return super().__setitem__(key, value)

    def setdefault(self, key, default=None):
        key = _interpret_key(key, self.single_arg)
        return super().setdefault(key, default)

    def get(self, key, default=None):
        key = _interpret_key(key, self.single_arg)
        return super().get(key, default)

    def consistency_check(self, check_values=False):
        r"""Check the stored keys and optionally values for consistency."""
        scalar_variables = {key[0] for key in self if key[1] is None}
        _single_arg_by_key = (key[0] is None for key in self)
        if any(_single_arg_by_key) and not all(_single_arg_by_key):
            raise ArgMapError("Inconsistent keys in ArgMap for single argument with index.")
        shapes = {}
        for key in self:
            # Check for scalar variables
            if key[0] in scalar_variables:
                if key[1] is not None:
                    raise ArgMapError(
                        f"Inconsistent keys in ArgMap for argument with index {key[0]}"
                    )
            else:
                _type = int if np.issubdtype(type(key[1]), int) else len(key[1])
                if shapes.setdefault(key[0], _type) != _type:
                    raise ArgMapError(
                        f"Inconsistent keys in ArgMap for argument with index {key[0]}"
                    )
        if check_values:
            types = [(key, type(val)) for key, val in self.items()]
            if not all((_type[1] == types[0][1] for _type in types)):
                raise ArgMapError(
                    "\n".join(
                        (
                            ["Inconsistent value types in ArgMap"]
                            + [f"{_type[0]}: {_type[1]}" for _type in types]
                        )
                    )
                )


# Todo: figure out behaviour for keys = {(0, 1), (2, 3), (1, 2)}
#       - with single_arg=True -> single array-valued argument
#       - with single_arg=False -> pairs of argument and param indices
