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
"""This module contains functions to analyze the Fourier representation
of quantum circuits."""
import warnings
from .coefficients import coefficients
from .simple_spectrum import simple_spectrum
from .advanced_spectrum import advanced_spectrum


def spectrum(*args, **kwargs):
    """Alias for ``simple_spectrum``. To be removed soon."""
    warnings.warn(
        "qml.fourier.spectrum has been renamed to qml.fourier.simple_spectrum. "
        "The alias qml.fourier.spectrum is deprecated and will be removed soon.",
        UserWarning
    )
    return simple_spectrum(*args, **kwargs)
