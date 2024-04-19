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
import jax


import pennylane as qml

### All the abstract types


def _register_abstract_class(cls):
    jax.core.raise_to_shaped_mappings[cls] = lambda aval, _: aval
    return cls


@_register_abstract_class
class AbstractMeasurement(jax.core.AbstractValue):
    def __eq__(self, other):
        return isinstance(other, type(self))

    def __hash__(self):
        return hash(type(self).__name__)

    def abstract_measurement(self, shots: int, num_device_wires: int):
        dtype = jax.numpy.float64 if jax.config.jax_enable_x64 else jax.numpy.float32
        return jax.core.ShapedArray((), dtype)


@_register_abstract_class
class AbstractState(AbstractMeasurement):
    def abstract_measurement(self, shots, num_device_wires):
        dtype = jax.numpy.complex128 if jax.config.jax_enable_x64 else jax.numpy.complex64
        shape = (2**num_device_wires,)
        return jax.core.ShapedArray(shape, dtype)


@_register_abstract_class
class AbstractSample(AbstractMeasurement):
    def __init__(self, n_wires: int):
        self.n_wires = n_wires

    def abstract_measurement(self, shots, num_device_wires):
        dtype = jax.numpy.int64 if jax.config.jax_enable_x64 else jax.numpy.int32
        n_wires = num_device_wires if self.n_wires == 0 else self.n_wires
        shape = (n_wires, shots)
        return jax.core.ShapedArray(shape, dtype)


@_register_abstract_class
class AbstractObsSample(AbstractMeasurement):
    def abstract_measurement(self, shots, num_device_wires):
        dtype = jax.numpy.float64 if jax.config.jax_enable_x64 else jax.numpy.float32
        shape = (shots,)
        return jax.core.ShapedArray(shape, dtype)


@_register_abstract_class
class AbstractProbs(AbstractMeasurement):
    def __init__(self, n_wires: int):
        self.n_wires = n_wires

    def abstract_measurement(self, shots, num_device_wires):
        dtype = jax.numpy.float64 if jax.config.jax_enable_x64 else jax.numpy.float32
        n_wires = num_device_wires if self.n_wires == 0 else self.n_wires
        shape = (2**n_wires,)
        return jax.core.ShapedArray(shape, dtype)


#### Primitives #####


### The measure primitive ###############

measure_prim = jax.core.Primitive("measure")
measure_prim.multiple_results = True


@measure_prim.def_impl
def _(*measurements, shots, num_device_wires):
    # depends on the jax interpreter
    raise NotImplementedError


@measure_prim.def_abstract_eval
def _(*measurements, shots, num_device_wires):
    if not shots.has_partitioned_shots:
        return tuple(
            m.abstract_measurement(shots.total_shots, num_device_wires) for m in measurements
        )
    vals = []
    for s in shots:
        v = tuple(m.abstract_measurement(s, num_device_wires) for m in measurements)
        vals.extend(v)
    return vals


def measure(*measurements, shots=None, num_device_wires=0):
    """Perform a measurement."""
    shots = qml.measurements.Shots(shots)
    return measure_prim.bind(*measurements, shots=shots, num_device_wires=num_device_wires)
