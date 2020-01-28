# Copyright 2018-2019 Xanadu Quantum Technologies Inc.

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
This module contains templates, which are pre-coded routines that can be used in a quantum node.
"""

from .decorator import *
from .layers import *
from .embeddings import *
from .subroutines import *
from .state_preparations import *

from .layers import __all__ as _layers__all__
from .embeddings import __all__ as _embeddings__all__
from .subroutines import __all__ as _subroutines__all__
from .state_preparations import __all__ as _stateprep__all__


__all__ = _layers__all__ + _embeddings__all__ + _subroutines__all__ + _stateprep__all__
