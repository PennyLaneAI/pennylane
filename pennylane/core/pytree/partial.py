# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys

from functools import partial


class HashablePartial(partial):
    """
    A class behaving like functools.partial, but that retains it's hash
    if it's created with a lexically equivalent (the same) function and
    with the same partially applied arguments and keywords.

    It also stores the computed hash for faster hashing.
    """

    # TODO remove when dropping support for Python < 3.10
    def __new__(cls, func, *args, **keywords):
        # In Python 3.10+ if func is itself a functools.partial instance,
        # functools.partial.__new__ would merge the arguments of this HashablePartial
        # instance with the arguments of the func
        # Pre 3.10 this does not happen, so here we emulate this behaviour recursively
        # This is necessary since functools.partial objects do not have a __code__
        # property which we use for the hash
        # For python 3.10+ we still need to take care of merging with another HashablePartial
        while isinstance(
            func, partial if sys.version_info < (3, 10) else HashablePartial
        ):
            original_func = func
            func = original_func.func
            args = original_func.args + args
            keywords = {**original_func.keywords, **keywords}
        return super(HashablePartial, cls).__new__(cls, func, *args, **keywords)

    def __init__(self, *args, **kwargs):
        self._hash = None

    def __eq__(self, other):
        return (
            type(other) is HashablePartial
            and self.func.__code__ == other.func.__code__
            and self.args == other.args
            and self.keywords == other.keywords
        )

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(
                (self.func.__code__, self.args, frozenset(self.keywords.items()))
            )

        return self._hash

    def __repr__(self):
        return f"<hashable partial {self.func.__name__} with args={self.args} and kwargs={self.keywords}, hash={hash(self)}>"
