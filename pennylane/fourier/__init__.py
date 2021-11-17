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
from .circuit_spectrum import circuit_spectrum
from .qnode_spectrum import qnode_spectrum
from .reconstruct import reconstruct
from .utils import join_spectra, get_spectrum


def spectrum(*args, **kwargs):
    """Alias for :func:`~.circuit_spectrum`. To be removed soon."""
    warnings.warn(
        "qml.fourier.spectrum has been renamed to qml.fourier.circuit_spectrum. "
        "The alias qml.fourier.spectrum is deprecated and will be removed soon.",
        UserWarning,
    )
    return circuit_spectrum(*args, **kwargs)
