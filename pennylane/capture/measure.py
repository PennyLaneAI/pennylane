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
This submodule defines the measure primitive for performing measurements in plxpr.
"""
from functools import lru_cache

import pennylane as qml  # need shots class without circular dependency issues


has_jax = True
try:
    import jax
except ImportError:
    has_jax = False


@lru_cache
def _get_measure_primitive():
    if not has_jax:
        raise ImportError("jax is required to create the measure primitive.")

    if jax.config.jax_enable_x64:
        dtype_map = {
            float: jax.numpy.float64,
            int: jax.numpy.int64,
            complex: jax.numpy.complex128,
        }
    else:
        dtype_map = {
            float: jax.numpy.float32,
            int: jax.numpy.int32,
            complex: jax.numpy.complex64,
        }

    measure_prim = jax.core.Primitive("measure")
    measure_prim.multiple_results = True

    def trivial_processing(results):
        return results

    # pylint: disable=unused-argument
    @measure_prim.def_impl
    def _(*measurements, shots, num_device_wires):
        # depends on the jax interpreter
        if not all(isinstance(m, qml.measurements.MidMeasureMP) for m in measurements):
            raise NotImplementedError("requires an interpreter to perform a measurement.")
        return qml.measurements.MeasurementValue(measurements, trivial_processing)

    # pylint: disable=unused-argument
    @measure_prim.def_abstract_eval
    def _(*measurements, shots, num_device_wires):

        shapes = []
        if not shots:
            for m in measurements:
                shape, dtype = m.abstract_eval(
                    n_wires=m.n_wires, shots=None, num_device_wires=num_device_wires
                )
                shapes.append(jax.core.ShapedArray(shape, dtype_map[dtype]))
            return shapes

        for s in shots:
            for m in measurements:
                shape, dtype = m.abstract_eval(
                    n_wires=m.n_wires, shots=s, num_device_wires=num_device_wires
                )
                shapes.append(jax.core.ShapedArray(shape, dtype_map[dtype]))
        return shapes

    return measure_prim


def measure(*measurements, shots=None, num_device_wires=0):
    """Perform a measurement."""
    shots = qml.measurements.Shots(shots)
    measure_prim = _get_measure_primitive()
    return measure_prim.bind(*measurements, shots=shots, num_device_wires=num_device_wires)
