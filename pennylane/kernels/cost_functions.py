# Copyright

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
This file contains functionalities for kernel related costs.
"""
import pennylane as qml
import itertools
import math

def kernel_polarization(X, Y, kernel, assume_normalized_kernel=False):
    polarization = 0

    for (x1, y1), (x2, y2) in itertools.combinations(zip(X, Y), 2):
        # Factor 2 accounts for symmetry of the kernel
        polarization += 2 * kernel(x1, x2) * y1 * y2

    if assume_normalized_kernel:
        polarization += len(X)
    else:
        for x in X:
            polarization += kernel(x, x)

    return polarization

def kernel_target_alignment(X, Y, kernel, assume_normalized_kernel=False):
    alignment = 0
    normalization = 0

    for (x1, y1), (x2, y2) in itertools.combinations(zip(X, Y), 2):
        k = kernel(x1, x2) 

        # Factor 2 accounts for symmetry of the kernel
        alignment += 2 * k * y1 * y2
        normalization += 2 * k**2

    if assume_normalized_kernel:
        alignment += len(X)
        normalization += len(X)
    else:
        for x in X:
            k = kernel(x, x)
            alignment += k
            normalization += k**2

    return alignment / math.sqrt(len(X) * normalization)
