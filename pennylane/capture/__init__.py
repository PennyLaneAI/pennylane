# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

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
Tools for enabling the capture of pennylane objects into JaxPR.
"""
from .switches import enable_plxpr, disable_plxpr, plxpr_enabled
from .meta_type import create_operator_primitive, PLXPRObj
from .measure import measure
from .nested_jaxpr import bind_nested_jaxpr
