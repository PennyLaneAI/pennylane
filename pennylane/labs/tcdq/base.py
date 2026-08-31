# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Abstract base class for TCDQ (Train Classical, Deploy Quantum) simulators.

This module defines the contract that lets a loss function consume an
estimator from any simulator, including ones defined outside PennyLane, as
long as the estimator declares a compatible :class:`ObservableAlgebra`.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum


class ObservableAlgebra(Enum):
    """Encoding used to pass observables to an estimator, and the shape of its output.

    Each member fixes a complete calling contract. An estimator declaring an
    algebra promises to accept observables in that encoding and to return
    ``(values, uncertainty)`` with the stated dtypes and shapes.

    ``PAULI_Z`` uses the same integer encoding as ``PAULI``, restricted to the
    entries ``I`` and ``Z``. A ``PAULI_Z`` observable array is therefore also a
    valid ``PAULI`` observable array, so a consumer that only needs diagonal
    observables can accept estimators declaring either member.
    """

    PAULI_Z = "pauli_z"
    """Observables are an integer array of shape ``(n_obs, n_wires)`` with
    entries drawn from ``I=0, Z=3``. Returns real ``values`` of shape
    ``(n_obs,)`` and real ``uncertainty`` of shape ``(n_obs,)``, the variance
    of the mean."""

    PAULI = "pauli"
    """As ``PAULI_Z``, but entries may be any of ``I=0, X=1, Y=2, Z=3``."""

    HEISENBERG_WEYL = "heisenberg_weyl"
    """Observables are a pair ``(l_vecs, m_vecs)`` of integer arrays of shape
    ``(n_obs, n_wires)``. Returns complex ``values`` of shape ``(n_obs,)`` and
    real ``uncertainty`` of shape ``(n_obs, 2, 2)``, the real-imaginary
    covariance of the mean."""


@dataclass(frozen=True)
class EstimatorSpec:
    """Description of what a built estimator measures.

    Args:
        name (str): Registered name of the estimator on its simulator.
        algebra (ObservableAlgebra): Observable encoding and return contract.
        local_dims (tuple[int, ...]): Local Hilbert space dimension of each wire.

    **Example**

    >>> from pennylane.labs.tcdq import EstimatorSpec, ObservableAlgebra
    >>> spec = EstimatorSpec("expval", ObservableAlgebra.PAULI, (2, 2, 2))
    >>> spec.n_wires
    3
    """

    #: Registered name of the estimator on its simulator.
    name: str
    #: Observable encoding and return contract.
    algebra: ObservableAlgebra
    #: Local Hilbert space dimension of each wire.
    local_dims: tuple[int, ...]

    @property
    def n_wires(self) -> int:
        """int: Number of wires, inferred from :attr:`local_dims`."""
        return len(self.local_dims)


@dataclass(frozen=True)
class Estimator:
    """A pure estimation function bundled with its specification.

    Instances are frozen and hashable, so they can be passed as a static
    argument to :func:`jax.jit`. Two estimators compare equal only when they
    wrap the same underlying closure, so each rebuild triggers one compilation.

    Args:
        spec (EstimatorSpec): What this estimator measures.
        fn (Callable): Pure callable with signature
            ``fn(params, observables, *, key=None, n_samples=None, phase_params=None)``
            returning ``(values, uncertainty)`` as fixed by ``spec.algebra``.
            Implementations may accept further keyword-only runtime overrides;
            both built-in simulators also accept ``init_state``.

    .. seealso:: :meth:`TCDQSimulator.build_estimator`
    """

    #: What this estimator measures.
    spec: EstimatorSpec
    #: The underlying pure estimation function.
    fn: Callable

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


def estimator(name: str, *, algebra: ObservableAlgebra) -> Callable:
    """Register a :class:`TCDQSimulator` method as an estimator factory.

    The decorated method takes only build-time options and returns a pure,
    JAX-traceable closure. Precomputation that does not depend on the trainable
    parameters or the observables belongs in the method body, so that the
    returned closure is cheap to call repeatedly under ``jit`` and ``grad``.

    Args:
        name (str): Name used to look the estimator up via
            :meth:`TCDQSimulator.build_estimator`.
        algebra (ObservableAlgebra): Observable encoding and return contract
            the returned closure honours.

    Returns:
        Callable: A decorator that tags the method for registration.

    **Example**

    .. code-block:: python

        class MySimulator(TCDQSimulator):

            @estimator("pauli_expval", algebra=ObservableAlgebra.PAULI)
            def _build_pauli_expval(self):
                precomputed = self._expensive_setup()

                def expval(params, observables, *, key=None, n_samples=None,
                           phase_params=None):
                    return _core(params, observables, precomputed)

                return expval
    """

    def decorate(method):
        method._tcdq_estimator = (name, algebra)  # pylint: disable=protected-access
        return method

    return decorate


class TCDQSimulator(ABC):
    r"""Base class for classically trainable, quantum deployable circuit models.

    A TCDQ simulator describes a family of circuits that can be *trained*
    classically, by estimating circuit properties without simulating the full
    quantum state, and later *deployed* on quantum hardware. Subclasses supply
    the circuit structure and one or more estimators.

    To define a simulator, implement :attr:`local_dims` and decorate one or
    more factory methods with :func:`estimator`. Each factory returns a pure
    closure with the signature::

        fn(params, observables, *, key=None, n_samples=None, phase_params=None)
            -> (values, uncertainty)

    Observables are always supplied by the caller. Their encoding, and the
    dtypes and shapes of the two return values, are fixed by the estimator's
    declared :class:`ObservableAlgebra`. The keyword arguments are runtime
    overrides of build-time defaults; an implementation may accept further ones
    of its own, since consumers only rely on the signature above.

    **Example**

    >>> import jax
    >>> from pennylane.labs.tcdq import IQPSimulator, create_local_gates
    >>> sim = IQPSimulator(
    ...     gates=create_local_gates(4, max_weight=2),
    ...     n_qubits=4,
    ...     n_samples=1000,
    ...     key=jax.random.PRNGKey(0),
    ... )
    >>> sim.available_estimators()
    ('pauli_expval',)
    >>> sim.n_wires
    4

    .. seealso:: :class:`~pennylane.labs.tcdq.IQPSimulator`,
        :class:`~pennylane.labs.tcdq.QuditIQPSimulator`
    """

    _estimators: dict[str, tuple[str, ObservableAlgebra]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        registry = dict(cls._estimators)
        for attr_name, attr in vars(cls).items():
            tag = getattr(attr, "_tcdq_estimator", None)
            if tag is not None:
                name, algebra = tag
                registry[name] = (attr_name, algebra)
        cls._estimators = registry

    @property
    @abstractmethod
    def local_dims(self) -> tuple[int, ...]:
        """tuple[int, ...]: Local Hilbert space dimension of each wire."""

    @property
    def n_wires(self) -> int:
        """int: Number of wires, inferred from :attr:`local_dims`."""
        return len(self.local_dims)

    @classmethod
    def available_estimators(cls) -> tuple[str, ...]:
        """Names of every estimator this simulator provides.

        Returns:
            tuple[str, ...]: Sorted estimator names accepted by
            :meth:`build_estimator`.
        """
        return tuple(sorted(cls._estimators))

    def build_estimator(self, name: str, /, **kwargs) -> Estimator:
        """Build a named estimator as a pure function plus its specification.

        Args:
            name (str): One of :meth:`available_estimators`.
            **kwargs: Build-time options forwarded to the factory method.

        Returns:
            Estimator: The built estimator, ready to pass to a loss function.

        Raises:
            ValueError: If ``name`` is not a registered estimator.
        """
        try:
            attr_name, algebra = self._estimators[name]
        except KeyError:
            raise ValueError(
                f"{type(self).__name__} has no estimator {name!r}. "
                f"Available estimators: {list(self.available_estimators())}"
            ) from None

        spec = EstimatorSpec(name=name, algebra=algebra, local_dims=self.local_dims)
        return Estimator(spec=spec, fn=getattr(self, attr_name)(**kwargs))
