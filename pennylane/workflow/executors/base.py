# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

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
Contains concurrent executor abstractions for task-based workloads.
"""

import abc
from collections.abc import Callable, Sequence


class RemoteExecABC(abc.ABC):
    """
    Abstract base class for defining a task-based parallel executor backend for running python functions
    """

    def __init__(self, *args, **kwargs): ...
    @abc.abstractmethod
    def __call__(self, fn: Callable, data: Sequence):
        """
        fn:     the callable function to run on the executor backend
        data:   is a sequence where each work-item is a packaged chunk for execution
        """
        ...

    @property
    def size(self):
        return self._size


class IntExecABC(RemoteExecABC, abc.ABC):
    """
    Executor class for native Python library concurrency support
    """

    pass


class ExtExecABC(RemoteExecABC, abc.ABC):
    """
    Executor class for external package provided concurrency support
    """

    pass
