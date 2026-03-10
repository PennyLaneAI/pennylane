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
Submodule for concurrent executors relying on the Python standard library.

.. currentmodule:: pennylane.concurrency.executor

All executor functionality in this module is implemented directly using native Python abstractions.

.. currentmodule:: pennylane.concurrency.executors.native

.. autosummary::
    :toctree: api

    ~api.PyNativeExec
    ~conc_futures.ProcPoolExec
    ~conc_futures.ThreadPoolExec
    ~multiproc.MPPoolExec
    ~serial.SerialExec

"""

from .api import PyNativeExec
from .multiproc import MPPoolExec
from .conc_futures import ProcPoolExec, ThreadPoolExec
from .serial import SerialExec

__all__ = ["MPPoolExec", "PyNativeExec", "ProcPoolExec", "ThreadPoolExec", "SerialExec"]
