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

# pylint: disable=too-few-public-methods,function-redefined

"""
This module contains core quantum operations supported by PennyLane -
such as gates, state preparations and observables.
"""
from .cv import *
from .qubit import *
from .channel import *

from .cv import __all__ as _cv__all__
from .cv import ops as _cv__ops__
from .cv import obs as _cv__obs__

from .qubit import __all__ as _qubit__all__
from .qubit import ops as _qubit__ops__
from .qubit import obs as _qubit__obs__

from .channel import __all__ as _channel__ops__


class AdjointError(Exception):
    """Exception for non-adjointable operations."""

    pass


__all__ = _cv__all__ + _qubit__all__ + _channel__ops__
__all_ops__ = list(_cv__ops__ | _qubit__ops__)
__all_obs__ = list(_cv__obs__ | _qubit__obs__)
