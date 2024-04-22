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

from functools import lru_cache
from typing import Callable, Optional

import pennylane as qml

has_jax = True
try:
    import jax
except ImportError:
    has_jax = False


@lru_cache
def _get_abstract_measurement():
    if not has_jax:
        raise ImportError("Jax is required for plxpr.")

    class AbstractMeasurement(jax.core.AbstractValue):
        def __init__(self, abstract_eval: Callable, n_wires: Optional[int] = None):
            self.abstract_eval = abstract_eval
            self.n_wires = n_wires

        def __repr__(self):
            return f"AbstractMeasurement(n_wires={self.n_wires})"

        # pylint: disable=missing-function-docstring
        def at_least_vspace(self):
            # TODO: investigate the proper definition of this method
            raise NotImplementedError

        # pylint: disable=missing-function-docstring
        def join(self, other):
            # TODO: investigate the proper definition of this method
            raise NotImplementedError

        # pylint: disable=missing-function-docstring
        def update(self, **kwargs):
            # TODO: investigate the proper definition of this method
            raise NotImplementedError

        def __eq__(self, other):
            return isinstance(other, AbstractMeasurement)

        def __hash__(self):
            return hash("AbstractMeasurement")

    jax.core.raise_to_shaped_mappings[AbstractMeasurement] = lambda aval, _: aval

    return AbstractMeasurement


def create_measurement_obs_primitive(
    measurement_type: type, name: str
) -> Optional["jax.core.Primitive"]:
    if not has_jax:
        return None

    primitive = jax.core.Primitive(name)

    @primitive.def_impl
    def _(obs):
        return measurement_type(obs=obs)

    abstract_type = _get_abstract_measurement()

    @primitive.def_abstract_eval
    def _(obs):
        abstract_eval = measurement_type._abstract_eval
        return abstract_type(abstract_eval, n_wires=None)

    return primitive


def create_measurement_wires_primitive(
    measurement_type: type, name: str
) -> Optional["jax.core.Primitive"]:
    if not has_jax:
        return None

    primitive = jax.core.Primitive(name)

    @primitive.def_impl
    def _(*wires, **kwargs):
        wires = qml.wires.Wires(wires)
        return measurement_type(wires=wires)

    abstract_type = _get_abstract_measurement()

    @primitive.def_abstract_eval
    def _(*wires, **kwargs):
        abstract_eval = measurement_type._abstract_eval
        return abstract_type(abstract_eval, n_wires=len(wires))

    return primitive


### The measure primitive ###############

measure_prim = jax.core.Primitive("measure")
measure_prim.multiple_results = True


def trivial_processing(results):
    return results


@measure_prim.def_impl
def _(*measurements, shots, num_device_wires):
    # depends on the jax interpreter
    if not all(isinstance(m, qml.measurements.MidMeasureMP) for m in measurements):
        raise NotImplementedError("requires an interpreter to perform a measurement.")
    return qml.measurements.MeasurementValue(measurements, trivial_processing)


@measure_prim.def_abstract_eval
def _(*measurements, shots, num_device_wires):

    if not shots.has_partitioned_shots:
        kwargs = {"shots": shots.total_shots, "num_device_wires": num_device_wires}
        return tuple(m.abstract_eval(n_wires=m.n_wires, **kwargs) for m in measurements)
    vals = []
    for s in shots:
        v = tuple(
            m.abstract_eval(n_wires=m.n_wires, shots=s, num_device_wires=num_device_wires)
            for m in measurements
        )
        vals.extend(v)
    return vals


def measure(*measurements, shots=None, num_device_wires=0):
    """Perform a measurement."""
    shots = qml.measurements.Shots(shots)
    return measure_prim.bind(*measurements, shots=shots, num_device_wires=num_device_wires)
