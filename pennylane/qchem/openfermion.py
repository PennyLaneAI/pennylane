# Copyright 2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Helper module for lazy importing of openfermion."""
import importlib


def __getattr__(name):
    """Allow for lazy importing of openfermion"""
    try:
        return importlib.import_module("openfermion." + name)
    except ModuleNotFoundError:
        mod = importlib.import_module("openfermion")
        return getattr(mod, name)
