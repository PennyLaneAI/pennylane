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
r"""
.. currentmodule:: pennylane.concurrency.executors

Provides abstractions for task-based parallel workloads within PennyLane using a simplified `concurrent.futures <https://docs.python.org/3/library/concurrent.futures.html>`_ executor-like interface.
"""

from .backends import ExecBackends, create_executor, get_executor, get_supported_backends
from .base import RemoteExec, IntExec, ExtExec, ExecBackendConfig
from .native import PyNativeExec, SerialExec, MPPoolExec, ProcPoolExec, ThreadPoolExec

__all__ = [
    "ExecBackends",
    "create_executor",
    "get_executor",
    "get_supported_backends",
    "ExecBackendConfig",
    "RemoteExec",
    "IntExec",
    "ExtExec",
    "PyNativeExec",
    "SerialExec",
    "MPPoolExec",
    "ProcPoolExec",
    "ThreadPoolExec",
]
