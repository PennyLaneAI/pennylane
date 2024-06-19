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
This submodule defines the abstract classes and primitives for capture.
"""

from functools import lru_cache
from typing import Callable, Optional, Tuple, Type

import pennylane as qml

has_jax = True
try:
    import jax
except ImportError:
    has_jax = False


@lru_cache  # construct the first time lazily
def _get_abstract_operator() -> type:
    """Create an AbstractOperator once in a way protected from lack of a jax install."""
    if not has_jax:  # pragma: no cover
        raise ImportError("Jax is required for plxpr.")  # pragma: no cover

    class AbstractOperator(jax.core.AbstractValue):
        """An operator captured into plxpr."""

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
            return isinstance(other, AbstractOperator)

        def __hash__(self):
            return hash("AbstractOperator")

        @staticmethod
        def _matmul(*args):
            return qml.prod(*args)

        @staticmethod
        def _mul(a, b):
            return qml.s_prod(b, a)

        @staticmethod
        def _rmul(a, b):
            return qml.s_prod(b, a)

        @staticmethod
        def _add(a, b):
            return qml.sum(a, b)

        @staticmethod
        def _pow(a, b):
            return qml.pow(a, b)

    jax.core.raise_to_shaped_mappings[AbstractOperator] = lambda aval, _: aval

    return AbstractOperator


@lru_cache
def _get_abstract_measurement():
    if not has_jax:  # pragma: no cover
        raise ImportError("Jax is required for plxpr.")  # pragma: no cover

    class AbstractMeasurement(jax.core.AbstractValue):
        """An abstract measurement.

        Args:
            abstract_eval (Callable): See :meth:`~.MeasurementProcess._abstract_eval`.  A function of
               ``n_wires``, ``has_eigvals``, ``num_device_wires`` and ``shots`` that returns a shape
               and numeric type.
            n_wires=None (Optional[int]): the number of wires
            has_eigvals=False (bool): Whether or not the measurement contains eigenvalues in a wires+eigvals
               diagonal representation.

        """

        def __init__(
            self, abstract_eval: Callable, n_wires: Optional[int] = None, has_eigvals: bool = False
        ):
            self._abstract_eval = abstract_eval
            self._n_wires = n_wires
            self.has_eigvals: bool = has_eigvals

        def abstract_eval(self, num_device_wires: int, shots: int) -> Tuple[Tuple, type]:
            """Calculate the shape and dtype for an evaluation with specified number of device
            wires and shots.

            """
            return self._abstract_eval(
                n_wires=self._n_wires,
                has_eigvals=self.has_eigvals,
                num_device_wires=num_device_wires,
                shots=shots,
            )

        @property
        def n_wires(self) -> Optional[int]:
            """The number of wires for a wire based measurement.

            Options are:
            * ``None``:  The measurement is observable based or single mcm based
            * ``0``: The measurement is broadcasted across all available devices wires
            * ``int>0``: A wire or mcm based measurement with specified wires or mid circuit measurements.

            """
            return self._n_wires

        def __repr__(self):
            if self.has_eigvals:
                return f"AbstractMeasurement(n_wires={self.n_wires}, has_eigvals=True)"
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


def create_operator_primitive(
    operator_type: Type["qml.operation.Operator"],
) -> Optional["jax.core.Primitive"]:
    """Create a primitive corresponding to an operator type.

    Called when defining any :class:`~.Operator` subclass, and is used to set the
    ``Operator._primitive`` class property.

    Args:
        operator_type (type): a subclass of qml.operation.Operator

    Returns:
        Optional[jax.core.Primitive]: A new jax primitive with the same name as the operator subclass.
        ``None`` is returned if jax is not available.

    """
    if not has_jax:
        return None

    primitive = jax.core.Primitive(operator_type.__name__)

    @primitive.def_impl
    def _(*args, **kwargs):
        if "n_wires" not in kwargs:
            return type.__call__(operator_type, *args, **kwargs)
        n_wires = kwargs.pop("n_wires")

        # need to convert array values into integers
        # for plxpr, all wires must be integers
        wires = tuple(int(w) for w in args[-n_wires:])
        args = args[:-n_wires]
        return type.__call__(operator_type, *args, wires=wires, **kwargs)

    abstract_type = _get_abstract_operator()

    @primitive.def_abstract_eval
    def _(*_, **__):
        return abstract_type()

    return primitive


def create_measurement_obs_primitive(
    measurement_type: Type["qml.measurements.MeasurementProcess"], name: str
) -> Optional["jax.core.Primitive"]:
    """Create a primitive corresponding to the input type where the abstract inputs are an operator.

    Called by default when defining any class inheriting from :class:`~.MeasurementProcess`, and is used to
    set the ``MeasurementProcesss._obs_primitive`` property.

    Args:
        measurement_type (type): a subclass of :class:`~.MeasurementProcess`
        name (str): the preferred string name for the class. For example, ``"expval"``.
            ``"_obs"`` is appended to this name for the name of the primitive.

    Returns:
        Optional[jax.core.Primitive]: A new jax primitive. ``None`` is returned if jax is not available.

    """
    if not has_jax:
        return None

    primitive = jax.core.Primitive(name + "_obs")

    @primitive.def_impl
    def _(obs, **kwargs):
        return type.__call__(measurement_type, obs=obs, **kwargs)

    abstract_type = _get_abstract_measurement()

    @primitive.def_abstract_eval
    def _(*_, **__):
        abstract_eval = measurement_type._abstract_eval  # pylint: disable=protected-access
        return abstract_type(abstract_eval, n_wires=None)

    return primitive


def create_measurement_mcm_primitive(
    measurement_type: Type["qml.measurements.MeasurementProcess"], name: str
) -> Optional["jax.core.Primitive"]:
    """Create a primitive corresponding to the input type where the abstract inputs are classical
    mid circuit measurement results.

    Called by default when defining any class inheriting from :class:`~.MeasurementProcess`, and is used to
    set the ``MeasurementProcesss._mcm_primitive`` property.

    Args:
        measurement_type (type): a subclass of :class:`~.MeasurementProcess`
        name (str): the preferred string name for the class. For example, ``"expval"``.
            ``"_mcm"`` is appended to this name for the name of the primitive.

    Returns:
        Optional[jax.core.Primitive]: A new jax primitive. ``None`` is returned if jax is not available.
    """

    if not has_jax:
        return None

    primitive = jax.core.Primitive(name + "_mcm")

    @primitive.def_impl
    def _(*mcms, **kwargs):
        raise NotImplementedError(
            "mcm measurements do not currently have a concrete implementation"
        )
        # need to figure out how to convert a jaxpr int into a measurement value, and pass
        # that measurment value here.
        # return type.__call__(measurement_type, obs=mcms, **kwargs)

    abstract_type = _get_abstract_measurement()

    @primitive.def_abstract_eval
    def _(*mcms, **__):
        abstract_eval = measurement_type._abstract_eval  # pylint: disable=protected-access
        return abstract_type(abstract_eval, n_wires=len(mcms))

    return primitive


def create_measurement_wires_primitive(
    measurement_type: type, name: str
) -> Optional["jax.core.Primitive"]:
    """Create a primitive corresponding to the input type where the abstract inputs are the wires.

    Called by default when defining any class inheriting from :class:`~.MeasurementProcess`, and is used to
    set the ``MeasurementProcesss._wires_primitive`` property.

    Args:
        measurement_type (type): a subclass of :class:`~.MeasurementProcess`
        name (str): the preferred string name for the class. For example, ``"expval"``.
            ``"_wires"`` is appended to this name for the name of the primitive.

    Returns:
        Optional[jax.core.Primitive]: A new jax primitive. ``None`` is returned if jax is not available.
    """
    if not has_jax:
        return None

    primitive = jax.core.Primitive(name + "_wires")

    @primitive.def_impl
    def _(*args, has_eigvals=False, **kwargs):
        if has_eigvals:
            wires = qml.wires.Wires(args[:-1])
            kwargs["eigvals"] = args[-1]
        else:
            wires = qml.wires.Wires(args)
        return type.__call__(measurement_type, wires=wires, **kwargs)

    abstract_type = _get_abstract_measurement()

    @primitive.def_abstract_eval
    def _(*args, has_eigvals=False, **_):
        abstract_eval = measurement_type._abstract_eval  # pylint: disable=protected-access
        n_wires = len(args) - 1 if has_eigvals else len(args)
        return abstract_type(abstract_eval, n_wires=n_wires, has_eigvals=has_eigvals)

    return primitive
