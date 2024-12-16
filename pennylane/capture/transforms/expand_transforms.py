# Copyright 2024 Xanadu Quantum Technologies Inc.

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
Helper function for expanding transforms with program capture
"""
from functools import wraps
from typing import Callable

from pennylane.capture.base_interpreter import PlxprInterpreter


class ExpandTransformsInterpreter(PlxprInterpreter):
    """Interpreter for expanding transform primitives that are applied to plxpr.

    This interpreter does not do anything special by itself. Instead, it is used
    by the PennyLane transforms to expand transform primitives in plxpr by
    applying the respective transform to the inner plxpr. When a transform is created
    using :func:`~pennylane.transform`, a custom primitive interpretation rule for
    that transform is automatically registered for ``ExpandTransformsInterpreter``.
    """


def expand_plxpr_transforms(f: Callable) -> Callable:
    """Function for expanding plxpr transforms.

    This function wraps the input callable. The returned function

    Args:
        f (Callable): The callable for which we want to expand transforms.

    Returns:
        Callable: Callable with expanded transforms
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        transformed_f = ExpandTransformsInterpreter()(f)
        return transformed_f(*args, **kwargs)

    return wrapper
