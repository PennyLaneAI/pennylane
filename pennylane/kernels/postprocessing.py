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
This file contains functionalities for postprocessing of kernel matrices.
"""
import numpy as np


def threshold_matrix(K):
    """Remove negative eigenvalues from the given kernel matrix.

    Args:
        K (array[float]): Kernel matrix assumed to be symmetric

    Returns:
        array[float]: Kernel matrix with negative eigenvalues cropped.
    """
    w, v = np.linalg.eigh(K)

    print("K = ", K)
    print("w = ", w)
    print("v = ", v)
    print("vwvt = ", v @ np.diag(w) @ np.transpose(v))
    print("vtwv = ", np.transpose(v) @ np.diag(w) @ v)

    w0 = np.clip(w, 0, None)

    print("vw0vt = ", v @ np.diag(w0) @ np.transpose(v))
    print("vtw0v = ", np.transpose(v) @ np.diag(w0) @ v)

    return v @ np.diag(w0) @ np.transpose(v)
