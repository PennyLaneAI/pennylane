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
    if key in (None, (None, None), ()):
        return None, None
    if single_arg:
        # Only single argument, so only param_key is given
        if np.issubdtype(type(key), int):
            return None, key
        if isinstance(key, tuple):
            sub_key = key[1] if key[0] is None else key
            return None, sub_key
    else:
        # Only arg_key given
        if np.issubdtype(type(key), int):
            return key, None
        # (arg_key, param_key) given
        if isinstance(key, tuple) and len(key) == 2:
            if key[1] == ():
                key = (key[0], None)
            return key
    raise ArgMapError(f"Could not interpret key {key}.")


class ArgMap(dict):
    r"""Argument index map for storing objects per QNode argument and parameter.

    Args:
        data (list[tuple] or dict or object): Data to store in the ArgMap.
            If not a dict and ``single_entry=False``, ``data`` must be possible
            to be parsed via ``dict(data)``.
        single_arg (bool): Whether the ``ArgMap`` stores entries for a single argument.
        single_entry (bool): Whether the ``ArgMap`` stores only a single entry.
        like (ArgMap): ArgMap instance to inherit ``single_arg`` and ``single_entry`` from.

    An ``ArgMap`` acts as a dictionary with a restricted type of keys in the context of
    a given ``QNode``. Each key takes the form ``tuple[int, tuple[int] or int]``.
    The first ``int`` in the tuple describes the argument index of the ``QNode``,
    while the second entry describes the index within that argument.
    That is ``(2, (0, 3))`` describes the element in the ``0``th row and ``3``rd
    column of the ``2``nd argument of a QNode. The second entry of the tuple may be an
    ``int`` instead, indexing a one-dimensional array.

    If the argument index (first entry) of a key is ``None``, this indicates that there is
    only one argument in the QNode, and all keys of the ``ArgMap`` should have the form
    ``(None, tuple[int] or int)``. In this case, ``single_arg`` must be set to ``True``.
    If the parameter index (second entry) of a key is ``None``, the respective argument
    is a scalar and no other key with the same argument index can exist in the ``ArgMap``.
    If both entries are ``None``, i.e. ``key=(None, None)``, the QNode only takes a single
    scalar argument and the ArgMap stores a single entry. In this case, ``single_entry``
    must be set to ``True``.

    The main feature of an ``ArgMap`` is its flexible access: Users can specify only an
    argument index or only a parameter index if this uniquely describes the parameter.

    **Example**
    Suppose we have the following ``QNode`` instance:

    .. code-block:: python

        dev = qml.device('default.qubit', wires=2)
        @qml.qnode(dev)
        def circuit(arg1, arg2, arg3):
            qml.Rot(*arg1, wires=0)
            qml.RX(arg2, wires=0)
            qml.Rot(*arg3, wires=1)
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

    **Interpretation table**
    The key interpretation has some subtleties, in particular for keys that have a different
    meaning for activated and deactivated ``single_arg``. This is how the interpretation
    maps input keys:

    .. list-table:: Interpretation table for ``ArgMap``.
       :widths: 25 25 25 25
       :header-rows: 1

       * - Input:
         - default
         - ``single_arg=True``
         - ``single_entry=True``
         - ``argnum=Sequence[int]``
       * - ``(int, tuple[int] or None)``
         - unchanged
         - *invalid*
         - *invalid*
       * - ``(int, int)``
         - unchanged
         - ``(None, (int, int))``
         - *invalid*
       * - ``int``
         - ``(int, None)``
         - ``(None, int)``
         - *invalid*
       * - ``(None, tuple[int] or int)``
         - *invalid*
         - unchanged
         - *invalid*
       * - ``tuple[int]``
         - *invalid* if ``len(key)!=2``
         - ``(None, tuple[int])``
         - *invalid*
       * - ``(None, None) or None or ()``
         - *invalid*
         - *invalid*
         - ``(None, None)``

    Note that not all cases marked as *invalid* will necessarily raise an error during the
    interpretation. However, when creating an ``ArgMap``, the keys are checked for these
    invalid entries via ``ArgMap.consistency_check``.
    """
    def __init__(self, data=None, single_arg=False, single_entry=False, like=None):
        if like is not None:
            if not isinstance(like, ArgMap):
                raise ArgMapError("Trying to inherit properties from non-ArgMap instance.")
            single_arg, single_entry = like.single_arg, like.single_entry
        self.single_arg = single_arg
        self.single_entry = single_entry
        _data = self._preprocess_data(data)
        super().__init__(_data)
        self.consistency_check()

    def _preprocess_data(self, data):
        if data is None:
            data = {}
        if self.single_entry:
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

    def __eq__(self, other):
        if isinstance(other, ArgMap):
            if (self.single_arg + other.single_arg) % 2:
                return False
            if (self.single_entry + other.single_entry) % 2:
                return False
        return super().__eq__(other)

    def __ne__(self, other):
        if isinstance(other, ArgMap):
            if (self.single_arg + other.single_arg) % 2:
                return True
            if (self.single_entry + other.single_entry) % 2:
                return True
        return super().__ne__(other)

    def get(self, key, default=None):
        key = _interpret_key(key, self.single_arg)
        return super().get(key, default)

    def consistency_check(self, check_values=False):
        r"""Check the stored keys and optionally values of the ArgMap for consistency.

        Args:
            check_values (bool): Whether to check the data types of the ArgMap values
                for consistency.

        **Details**
        The ArgMap is checked for invalid entries that do not contain valid indices in the keys
        like ``(0, ("a", "string", "tuple!"))``, and for consistency between the keys.
        Which keys are accepted as consistent depends on the properties ``single_arg``
        and ``single_entry``.
        As an example, the keys ``{(0, (3, 2)), (2, 1)}`` would be valid for 
        ``single_arg=False`` but not for ``single_arg=True``, as they explicitly specify
        the argument index, which does not make sense for a single argument.
        The keys ``{(None, (2, 3)), (None, (4, 1))}`` on the other hand only are valid
        for ``single_arg=True``, as they do not specify an argument index.

        For ``check_values=True``, the types of the values in the ArgMap are checked
        to be identical.

        **Valid examples**
        Some examples that would pass the ``consistency_check`` are

        .. code-block:: python

            ArgMap({(0, (3, 2)): "first value", (2, 1): "second value"})
            ArgMap({(1, None): "first value", (2, None): "second value"})
            ArgMap({(None, 1): "first value", (None, 2): "second value"}, single_arg=True)
            ArgMap({(None, None): "only value"}, single_entry=True)

        whereas cases that do not pass the ``consistency_check`` include

        .. code-block:: python

            # Keys do not fit a single_arg ArgMap
            ArgMap({(0, (3, 2)): "first value", (2, 1): "second value"}, single_arg=True)
            # The two keys for the first argument are incompatible
            ArgMap({(1, None): "first value", (1, (2,)): "second value"})
            # Keys only fit a single_arg ArgMap
            ArgMap({(None, 1): "first value", (None, 2): "second value"})
            # There are multiple entries in a single_entry ArgMap
            ArgMap({(None, None): "only value", (0, None): "excess entry"}, single_entry=True)
            # The key (None, None) is invalid in a non-single_entry ArgMap
            ArgMap({(None, None): "invalid key", (0, 3): "standard entry"})

        Here, all of the examples are determined to be inconsistent based on their keys.
        If we activate ``check_values`` in addition, all values are tested to have the same
        type, so that the first of the following ``ArgMap``s passes the ``consistency_check``
        but the second does not:

        .. code-block:: python
            
            argmap = ArgMap({(0, (3, 2)): "first value", (2, 1): "second value"})
            argmap.consistency_check(check_values=True) # Passes
            ArgMap({(0, (3, 2)): "a string", (2, 1): ["a", "list"]})
            argmap.consistency_check(check_values=True) # Fails
        """
        if self.single_entry:
            self._check_single_entry()
            if check_values:
                self._check_values()
            return
        if any(key == (None, None) for key in self):
            raise ArgMapError(
                "The key (None, None) indicates a single entry but ArgMap.single_entry=False."
            )
        if self.single_arg:
            self._check_single_arg()
        else:
            self._check_multiple_args()
        if check_values:
            self._check_values()

    def _check_values(self):
        """Check the values to have the same type."""
        types = [(key, type(val)) for key, val in self.items()]
        first_type = types[0][1]
        if not all(_type[1] is first_type for _type in types):
            raise ArgMapError(
                "\n".join(
                    (
                        ["Inconsistent value types in ArgMap"]
                        + [f"{_type[0]}: {_type[1]}" for _type in types]
                    )
                )
            )

    def _check_single_entry(self):
        """Check that there is only one entry and that the key is ``(None, None)``."""
        if not len(self) == 1:
            raise ArgMapError(f"ArgMap.single_entry=True but len(ArgMap)={len(self)}.")
        key = list(self)[0]
        if key != (None, None):
            raise ArgMapError(f"ArgMap.single_entry=True but key={key}; expected (None, None).")

    def _check_single_arg(self):
        """Check that the keys for a single argument are consistent and valid array indices."""
        shapes = {}
        for key in self:
            if np.issubdtype(type(key[1]), int):
                par_key_type = int
            else:
                if not isinstance(key[1], tuple):
                    raise ArgMapError(f"Invalid key {key} in ArgMap; expected (None, tuple[int]).")
                if not all(np.issubdtype(type(k), int) for k in key[1]):
                    raise ArgMapError(
                        f"Invalid entries in parameter index in ArgMap: {key[1]}; "
                        "expected integers."
                    )
                par_key_type = len(key[1])
            if shapes.setdefault(None, par_key_type) != par_key_type:
                raise ArgMapError("Inconsistent keys in ArgMap for the only argument index.")

    def _check_multiple_args(self):
        """Check that the keys per argument are valid, consistent array indices."""
        shapes = {}
        for key in self:
            if not np.issubdtype(type(key[0]), int):
                raise ArgMapError(f"Invalid argument index {key[0]}; expected integer.")
            if np.issubdtype(type(key[1]), int):
                par_key_type = int
            elif key[1] is None:
                par_key_type = None
            else:
                if not isinstance(key[1], tuple):
                    raise ArgMapError(f"Invalid key {key} in ArgMap; expected (int, tuple[int]).")
                if not all(np.issubdtype(type(k), int) for k in key[1]):
                    raise ArgMapError(
                        f"Invalid entries in parameter index in ArgMap: {key[1]}; "
                        "expected integers."
                    )
                par_key_type = len(key[1])
            if shapes.setdefault(key[0], par_key_type) != par_key_type:
                raise ArgMapError(f"Inconsistent keys in ArgMap for argument with index {key[0]}")
