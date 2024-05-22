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

    # def trivial_processing(*results):
    #    return results

    # pylint: disable=unused-argument
    @measure_prim.def_impl
    def _(*measurements, shots, num_device_wires):
        # depends on the jax interpreter
        if not all(isinstance(m, qml.measurements.MidMeasureMP) for m in measurements):
            raise NotImplementedError("requires an interpreter to perform a measurement.")
        raise NotImplementedError("currently causes jax to enter an infinite loop")
        # TODO: figure out how to make this return a measurement value
        # return qml.measurements.MeasurementValue(measurements, trivial_processing)

    # pylint: disable=unused-argument
    @measure_prim.def_abstract_eval
    def _(*measurements, shots, num_device_wires):

        shapes = []
        if not shots:
            shots = [None]

        for s in shots:
            for m in measurements:
                shape, dtype = m.abstract_eval(shots=s, num_device_wires=num_device_wires)
                shapes.append(jax.core.ShapedArray(shape, dtype_map.get(dtype, dtype)))
        return shapes

    return measure_prim


def measure(*measurements, shots=None, num_device_wires=0):
    """An instruction to perform measurements.

    .. warning::
        Note that this function does not provide concrete implementations.
        It is strictly used to capture the quantum/ classical
        boundary into jaxpr for later interpretation by a jaxpr interpreter.

    Args:
        *measurements (.MeasurementProcess): any number of simultaneous measurements

    Keyword Args:
        shots (Optional[int, Sequence[int], Shots]): the number of shots used to perform the execution
        num_device_wires (int): the number of device wires. Used to determine shape information when
            measurements are broadcasted across all wires.

    >>> qml.capture.enable()
    >>> def f():
    ...     mp1 = qml.expval(qml.Z(0))
    ...     mp2 = qml.sample()
    ...     return qml.capture.measure(mp1, mp2, shots=50, num_device_wires=4)
    >>> jax.make_jaxpr(f)()
    { lambda ; . let
        a:AbstractOperator() = PauliZ[n_wires=1] 0
        b:AbstractMeasurement(n_wires=None) = expval a
        c:AbstractMeasurement(n_wires=0) = sample
        d:f32[] e:i32[50,4] = measure[num_device_wires=4 shots=Shots(total=50)] b c
    in (d, e) }

    Here ``measure`` takes the number of shots and number of device wires, and converts the
    measurement processes into shaped arrays. ``measure`` can also be used with shot vectors.
    In the case of a shot vector, the results are flattened out, and will need to be repacked
    into the pytree structure before getting returned to a user.

    >>> def f():
    ...     mp1 = qml.expval(qml.Z(0))
    ...     mp2 = qml.sample()
    ...     return qml.capture.measure(mp1, mp2, shots=(50, 100), num_device_wires=4)
    >>> jax.make_jaxpr(f)()
    { lambda ; . let
        a:AbstractOperator() = PauliZ[n_wires=1] 0
        b:AbstractMeasurement(n_wires=None) = expval a
        c:AbstractMeasurement(n_wires=0) = sample
        d:f32[] e:i32[50,4] f:f32[] g:i32[100,4] = measure[
        num_device_wires=4
        shots=Shots(total=150, vector=[50 shots, 100 shots])
        ] b c
    in (d, e, f, g) }

    """
    shots = qml.measurements.Shots(shots)
    measure_prim = _get_measure_primitive()
    return measure_prim.bind(*measurements, shots=shots, num_device_wires=num_device_wires)
