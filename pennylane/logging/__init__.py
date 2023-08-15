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
"""This module enables support for log-level messaging throughout PennyLane, following the native Python logging framework interface. Please see the :doc:`PennyLane logging development guidelines</development/guide/logging>`, and the official Python documentation for details on usage https://docs.python.org/3/library/logging.html"""

from .configuration import enable_logging
from .configuration import config_path
from .configuration import TRACE
from .formatters.formatter import DefaultFormatter, SimpleFormatter
from .filter import LocalProcessFilter, DebugOnlyFilter
