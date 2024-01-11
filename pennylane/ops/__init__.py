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
from .op_math import *
from .qutrit import *

from .cv import __all__ as _cv__all__
from .cv import __ops__ as _cv__ops__
from .cv import __obs__ as _cv__obs__

from .qutrit import __all__ as _qutrit__all__
from .qutrit import __ops__ as _qutrit__ops__
from .qutrit import __obs__ as _qutrit__obs__

from .channel import __all__ as _channel__ops__

from .qubit import __all__ as _qubit__all__
from .qubit import __ops__ as _qubit__ops__
from .qubit import __obs__ as _qubit__obs__


# we would like these to just live in .qubit, but can't because of circular imports
from .op_math import controlled_qubit_ops as _controlled_qubit__ops__

_qubit__ops__ = _qubit__ops__ | _controlled_qubit__ops__
_qubit__all__ = _qubit__all__ + list(_controlled_qubit__ops__)


__all__ = _cv__all__ + _qubit__all__ + _qutrit__all__ + _channel__ops__
__all_ops__ = list(_cv__ops__ | _qubit__ops__ | _qutrit__ops__)
__all_obs__ = list(_cv__obs__ | _qubit__obs__ | _qutrit__obs__)
