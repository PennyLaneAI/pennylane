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
    when it is unable to access or write a requested item."""

@lru_cache(maxsize=None)
def _interpret_key(key, single_arg=False):
    if isinstance(key, tuple):
        if len(key)==2:
            if isinstance(key[1], tuple):
                return key
            if key[1] is None:
                return key
        if len(key)==0:
            return None, None
        return None, key
    if np.issubdtype(type(key), int):
        if single_arg:
            return None, key
        return key, None
    if key is None:
        return None, None
    raise ArgMapError(f"Could not interpret key {key}.")

class ArgMap(dict):

    def __init__(self, data, single_arg=False, single_object=False):
        self.single_arg = single_arg
        _data = self._preprocess_data(data, single_object)
        super().__init__(_data)


    def _preprocess_data(self, data, single_object):
        if single_object:
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

    def get(self, key, default=None):
        key = _interpret_key(key, self.single_arg)
        return super().get(key, default)
        
