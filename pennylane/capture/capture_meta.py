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
Defines a metaclass for automatic integration of any ``Operator`` with plxpr program capture.

See ``explanations.md`` for technical explanations of how this works.
"""
from inspect import signature

from .switches import enabled


# pylint: disable=no-self-argument, too-few-public-methods
class CaptureMeta(type):
    """A metatype that dispatches class creation to ``cls._primitve_bind_call`` instead
    of normal class creation.

    See ``pennylane/capture/explanations.md`` for more detailed information on how this technically
    works.
    """

    @property
    def __signature__(cls):
        return signature(cls.__init__)

    def _primitive_bind_call(cls, *args, **kwargs):
        raise NotImplementedError(
            "Types using CaptureMeta must implement cls._primitive_bind_call to"
            " gain integration with plxpr program capture."
        )

    def __call__(cls, *args, **kwargs):
        # this method is called everytime we want to create an instance of the class.
        # default behavior uses __new__ then __init__

        if enabled():
            # when tracing is enabled, we want to
            # use bind to construct the class if we want class construction to add it to the jaxpr
            return cls._primitive_bind_call(*args, **kwargs)
        return type.__call__(cls, *args, **kwargs)
