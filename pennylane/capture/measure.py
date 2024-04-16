# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
Defines a primitive for performing measurements
"""
import jax


measure_prim = jax.core.Primitive("measure")
measure_prim.multiple_results = True


@measure_prim.def_impl
def _(*measurements, shots):
    # depends on the jax interpreter
    raise NotImplementedError


@measure_prim.def_abstract_eval
def _(*measurements, shots):
    # later extend to more than just float measurements
    return tuple(jax.core.ShapedArray((shots.total_shots,), jax.numpy.int32) for _ in measurements)


def measure(*measurements, shots=0):
    """Perform a measurement."""
    return measure_prim.bind(*measurements, shots=shots)
