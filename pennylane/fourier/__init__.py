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
r"""Tools to compute and visualise the Fourier series representation of quantum circuits."""
from .coefficients import fourier_coefficients
from .spectrum import spectrum

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    print("Module matplotlib is required for visualization in the Fourier module.")
else:
    from .visualization import (
        coefficients_violin_plot,
        coefficients_bar_plot,
        coefficients_box_plot,
        coefficients_panel_plot,
        coefficients_radial_box_plot,
        reconstruct_function_1D_plot,
        reconstruct_function_2D_plot,
    )
