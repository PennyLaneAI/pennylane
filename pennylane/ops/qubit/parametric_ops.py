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
# pylint: disable=too-many-arguments
"""
This file exists to support backwards compatibility for datasets pickled with
pennylane.ops.qubit.parametric_ops (i.e. from before the file splitting).

All new parametric operators should go into the more precisely named files.
"""

# pylint:disable=wildcard-import,unused-wildcard-import

from .parametric_ops_controlled import *
from .parametric_ops_multi_qubit import *
from .parametric_ops_single_qubit import *
