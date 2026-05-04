# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module contains special logic of decomposing symbolic operations."""

from __future__ import annotations

import copy
from textwrap import dedent

import numpy as np

import pennylane as qp
from pennylane import allocation, math

from .decomposition_rule import DecompositionRule, register_condition, register_resources
from .reconstruct import reconstruct
from .resources import adjoint_resource_rep, controlled_resource_rep, pow_resource_rep, resource_rep


def make_adjoint_decomp(base_decomposition: DecompositionRule, use_reconstructor=False):
    """Create a decomposition rule for the adjoint of a decomposition rule."""

    def _condition_fn(base_class, base_params):  # pylint: disable=unused-argument
        return base_decomposition.is_applicable(**base_params)

    def _resource_fn(base_class, base_params):  # pylint: disable=unused-argument
        base_resources = base_decomposition.compute_resources(**base_params)
        return {
            adjoint_resource_rep(decomp_op.op_type, decomp_op.params): count
            for decomp_op, count in base_resources.gate_counts.items()
        }

    base_source = base_decomposition._source

    if not use_reconstructor:

        # pylint: disable=protected-access
        @register_condition(_condition_fn)
        @register_resources(
            _resource_fn,
            work_wires=base_decomposition._work_wire_spec,
            exact=base_decomposition.exact_resources,
            name=f"adjoint({base_decomposition.name})",
        )
        def _impl(*params, wires, base):
            # pylint: disable=protected-access
            qp.adjoint(base_decomposition._impl)(*params, wires=wires, **base.hyperparameters)

        _impl._source = (
            dedent(_impl._source).strip()
            + "\n\nwhere base_decomposition is defined as:\n\n"
            + dedent(base_source).strip()
        )

        return _impl

    # pylint: disable=protected-access
    @register_condition(_condition_fn)
    @register_resources(
        _resource_fn,
        work_wires=base_decomposition._work_wire_spec,
        exact=base_decomposition.exact_resources,
        name=f"adjoint({base_decomposition.name})",
    )
    def _impl_using_reconstructor(*params, wires, base_params, **_):
        # pylint: disable=protected-access
        qp.adjoint(base_decomposition._impl)(*params, wires=wires, **base_params)

    _impl_using_reconstructor._source = (
        dedent(_impl_using_reconstructor._source).strip()
        + "\n\nwhere base_decomposition is defined as:\n\n"
        + dedent(base_source).strip()
    )
    return _impl_using_reconstructor


def _cancel_adjoint_resource(base_class, base_params):  # pylint:disable=unused-argument
    # The base of a nested adjoint is an adjoint, so the base of the base is the original operator,
    # and the "base_params" of base_params are the resource params of the original operator.
    base_class, base_params = base_params["base_class"], base_params["base_params"]
    return {resource_rep(base_class, **base_params): 1}


# pylint: disable=protected-access,unused-argument
@register_resources(_cancel_adjoint_resource)
def cancel_adjoint(*params, wires, base):
    """Decompose the adjoint of the adjoint of an operator."""
    qp.pytrees.unflatten(*qp.pytrees.flatten(base.base))


@register_resources(_cancel_adjoint_resource)
def qjit_compatible_cancel_adjoint(*params, wires, base_class, base_params):
    """A catalyst-compatible decomposition rule that cancels nested adjoints.

    Precondition
    - has_reconstructor(base_class, base_params)

    """
    base_class, base_params = base_params["base_class"], base_params["base_params"]
    reconstruct(params, wires, base_class, base_params)


def _adjoint_rotation_resource(base_class, base_params):
    return {resource_rep(base_class, **base_params): 1}


# pylint: disable=protected-access,unused-argument
@register_resources(_adjoint_rotation_resource)
def adjoint_rotation(phi, wires, base):
    """Decompose the adjoint of a rotation operator by inverting the angle."""
    _, struct = qp.pytrees.flatten(base)
    qp.pytrees.unflatten((-phi,), struct)


# pylint: disable=protected-access,unused-argument
@register_resources(_adjoint_rotation_resource)
def qjit_compatible_adjoint_rotation(phi, wires, base_class, base_params):
    """A catalyst-compatible decomposition rule for single-angle rotations."""
    reconstruct((-phi,), wires, base_class, base_params)


def is_integer(x):
    """Checks if x is an integer."""
    return isinstance(x, int) or np.issubdtype(getattr(x, "dtype", None), np.integer)


# pylint: disable=protected-access,unused-argument
@register_condition(lambda z, **__: is_integer(z) and z >= 0)
@register_resources(lambda base_class, base_params, z: {resource_rep(base_class, **base_params): z})
def repeat_pow_base(*params, wires, base, z, **__):
    """Decompose the power of an operator by repeating the base operator. Assumes z
    is a non-negative integer."""

    @qp.for_loop(0, z)
    def _loop(i):
        qp.pytrees.unflatten(*qp.pytrees.flatten(base))

    _loop()  # pylint: disable=no-value-for-parameter


# pylint: disable=protected-access,unused-argument
@register_condition(lambda z, **__: is_integer(z) and z >= 0)
@register_resources(lambda base_class, base_params, z: {resource_rep(base_class, **base_params): z})
def qjit_compatible_repeat_pow_base(*params, wires, base_class, base_params, z, **__):
    """Decompose the power of an operator by repeating the base operator, in a qjit compatible way."""

    @qp.for_loop(0, z)
    def _loop(i):
        reconstruct(params, wires, base_class, base_params)

    _loop()  # pylint: disable=no-value-for-parameter


def _merge_powers_resource(base_class, base_params, z):  # pylint: disable=unused-argument
    return {
        pow_resource_rep(
            base_params["base_class"],
            base_params["base_params"],
            z * base_params["z"],
        ): 1
    }


# pylint: disable=protected-access,unused-argument
@register_resources(_merge_powers_resource)
def merge_powers(*params, wires, base, z, **__):
    """Decompose nested powers by combining them."""
    base_op = qp.pytrees.unflatten(*qp.pytrees.flatten(base.base))
    qp.pow(base_op, z * base.z)


@register_resources(_merge_powers_resource)
def qjit_compatible_merge_powers(*params, wires, base_class, base_params, z, **__):
    """Decompose nested powers by combining them in a qjit compatible way."""
    new_params = copy.copy(base_params)
    new_params["z"] = z * base_params["z"]
    return reconstruct(params, wires, base_class, new_params)


def _flip_pow_adjoint_resource(base_class, base_params, z):  # pylint: disable=unused-argument
    # base class is adjoint, and the base of the base is the target class
    target_class, target_params = base_params["base_class"], base_params["base_params"]
    return {
        adjoint_resource_rep(
            qp.ops.Pow, {"base_class": target_class, "base_params": target_params, "z": z}
        ): 1
    }


# pylint: disable=protected-access,unused-argument
@register_resources(_flip_pow_adjoint_resource)
def flip_pow_adjoint(*params, wires, base, z, **__):
    """Decompose the power of an adjoint by power to the base of the adjoint and
    then taking the adjoint of the power."""
    base_op = qp.pytrees.unflatten(*qp.pytrees.flatten(base.base))
    qp.adjoint(qp.pow(base_op, z))


@register_resources(_flip_pow_adjoint_resource)
def qjit_compatible_flip_pow_adjoint(*params, wires, base_class, base_params, z, **__):
    """Decompose the power of an adjoint in a qjit compatible way."""
    base = reconstruct(params, wires, base_params["base_class"], base_params["base_params"])
    qp.adjoint(qp.pow(base, z))


def make_pow_decomp_with_period(period, use_reconstructor=False) -> DecompositionRule:
    """Make a decomposition rule for the power of an op that has a period."""

    def _condition_fn(base_class, base_params, z):  # pylint: disable=unused-argument
        return math.shape(z) == () and z % period != z

    def _resource_fn(base_class, base_params, z):
        z_mod_period = z % period
        if z_mod_period == 0:
            return {}
        if z_mod_period == 1:
            return {resource_rep(base_class, **base_params): 1}
        return {pow_resource_rep(base_class, base_params, z_mod_period): 1}

    if not use_reconstructor:

        @register_condition(_condition_fn)
        @register_resources(_resource_fn)
        def _impl(*params, wires, base, z, **__):  # pylint: disable=unused-argument
            z_mod_period = z % period
            if z_mod_period == 1:
                qp.pytrees.unflatten(*qp.pytrees.flatten(base))
            elif z_mod_period > 0 and z_mod_period != period:
                qp.pow(base, z_mod_period)

        return _impl

    @register_condition(_condition_fn)
    @register_resources(_resource_fn)
    def _impl_using_reconstructor(
        *params, wires, base_class, base_params, z, **__
    ):  # pylint: disable=unused-argument
        z_mod_period = z % period
        if z_mod_period == 1:
            reconstruct(params, wires, base_class, base_params)
        elif z_mod_period > 0 and z_mod_period != period:
            qp.pow(reconstruct(params, wires, base_class, base_params), z_mod_period)

    return _impl_using_reconstructor


pow_involutory = make_pow_decomp_with_period(2, True)
pow_involutory_no_reconstructor = make_pow_decomp_with_period(2, False)


def _pow_rotation_resource(base_class, base_params, z):  # pylint: disable=unused-argument
    return {resource_rep(base_class, **base_params): 1}


# pylint: disable=protected-access,unused-argument
@register_resources(_pow_rotation_resource)
def pow_rotation(phi, wires, base, z, **__):
    """Decompose the power of a general rotation operator by multiplying the power by the angle."""
    _, struct = base._flatten()
    base._unflatten((phi * z,), struct)


@register_resources(_pow_rotation_resource)
def qjit_compatible_pow_rotation(phi, wires, base_class, base_params, z, **__):
    """Decompose the power of a general rotation operator by multiplying the power by the angle in a qjit compatible way."""
    reconstruct([phi * z], wires, base_class, base_params)


def _decompose_to_base_resource(base_class, base_params, **__):
    return {resource_rep(base_class, **base_params): 1}


# pylint: disable=protected-access,unused-argument
@register_resources(_decompose_to_base_resource)
def decompose_to_base(*params, wires, base, **__):
    """Decompose a symbolic operator to its base."""
    qp.pytrees.unflatten(*qp.pytrees.flatten(base))


@register_resources(_decompose_to_base_resource)
def qjit_compatible_decompose_to_base(*params, wires, base_class, base_params, **__):
    """Decompose a symbolic operator to its base in a qjit compatible way."""
    reconstruct(params, wires, base_class, base_params)


self_adjoint: DecompositionRule = decompose_to_base
qjit_compatible_self_adjoint: DecompositionRule = qjit_compatible_decompose_to_base


def make_controlled_decomp(base_decomposition: DecompositionRule):
    """Create a decomposition rule for the control of a decomposition rule."""

    def _condition_fn(base_params, **_):
        return base_decomposition.is_applicable(**base_params)

    def _resource_fn(
        base_params, num_control_wires, num_zero_control_values, num_work_wires, work_wire_type, **_
    ):
        base_resources = base_decomposition.compute_resources(**base_params)
        gate_counts = {
            controlled_resource_rep(
                base_class=base_op_rep.op_type,
                base_params=base_op_rep.params,
                num_control_wires=num_control_wires,
                num_zero_control_values=0,
                num_work_wires=num_work_wires,
                work_wire_type=work_wire_type,
            ): count
            for base_op_rep, count in base_resources.gate_counts.items()
        }
        # None of the other gates in gate_counts will be X, because they are all
        # controlled operations. So we can safely set the X gate counts here.
        gate_counts[resource_rep(qp.PauliX)] = num_zero_control_values * 2
        return gate_counts

    # pylint: disable=protected-access,too-many-arguments
    @register_condition(_condition_fn)
    @register_resources(
        _resource_fn,
        work_wires=base_decomposition._work_wire_spec,
        exact=base_decomposition.exact_resources,
        name=f"controlled({base_decomposition.name})",
    )
    def _impl(*params, wires, control_wires, control_values, work_wires, work_wire_type, base, **_):
        zero_control_wires = [w for w, val in zip(control_wires, control_values) if not val]
        for w in zero_control_wires:
            qp.PauliX(w)
        # We're extracting control wires and base wires from the wires argument instead
        # of directly using control_wires and base.wires, `wires` is properly traced, but
        # `control_wires` and `base.wires` are not.
        qp.ctrl(
            base_decomposition._impl,  # pylint: disable=protected-access
            control=wires[: len(control_wires)],
            work_wires=work_wires,
            work_wire_type=work_wire_type,
        )(*params, wires=wires[-len(base.wires) :], **base.hyperparameters)
        for w in zero_control_wires:
            qp.PauliX(w)

    _impl._source = (
        dedent(_impl._source).strip()
        + "\n\nwhere base_decomposition is defined as:\n\n"
        + dedent(base_decomposition._source).strip()
    )
    return _impl


def flip_zero_control(inner_decomp: DecompositionRule, name: str = "") -> DecompositionRule:
    """Wraps a decomposition for a controlled operator with X gates to flip zero control wires."""

    def _condition_fn(**resource_params):
        new_params = resource_params.copy()
        new_params["num_zero_control_values"] = 0
        return inner_decomp.is_applicable(**new_params)

    def _resource_fn(**resource_params):
        new_params = resource_params.copy()
        new_params["num_zero_control_values"] = 0
        inner_resource = inner_decomp.compute_resources(**new_params)
        num_x = resource_params["num_zero_control_values"]
        gate_counts = inner_resource.gate_counts.copy()
        # Add the counts of the flipping X gates to the gate count
        gate_counts[resource_rep(qp.X)] = gate_counts.get(resource_rep(qp.X), 0) + num_x * 2
        return gate_counts

    # pylint: disable=protected-access
    @register_condition(_condition_fn)
    @register_resources(
        _resource_fn,
        work_wires=inner_decomp._work_wire_spec,
        exact=inner_decomp.exact_resources,
        name=name or f"flip_zero_ctrl_values({inner_decomp.name})",
    )
    def _impl(*params, wires, control_wires, control_values, **kwargs):
        zero_control_wires = [w for w, val in zip(control_wires, control_values) if not val]
        for w in zero_control_wires:
            qp.PauliX(w)
        inner_decomp(
            *params,
            wires=wires,
            control_wires=control_wires,
            control_values=[1] * len(control_wires),  # all control values are 1 now
            **kwargs,
        )
        for w in zero_control_wires:
            qp.PauliX(w)

    base_source = inner_decomp._source
    _impl._source = (
        dedent(_impl._source).strip()
        + "\n\nwhere inner_decomp is defined as:\n\n"
        + dedent(base_source).strip()
    )
    return _impl


# Map from work-wire type (as stored in ``WorkWireSpec``) to the ``(state, restored)``
# arguments that ``qp.allocate`` expects for that type of wire.
_WORK_WIRE_ALLOCATE_KWARGS = {
    "zeroed": {"state": "zero", "restored": True},
    "borrowed": {"state": "any", "restored": True},
    "burnable": {"state": "zero", "restored": False},
    "garbage": {"state": "any", "restored": False},
}


def make_dynamic_work_wires_rule(
    inner_decomp: DecompositionRule,
    work_wires,
    name: str | None = None,
) -> DecompositionRule:
    """Wrap a decomposition rule that consumes explicit ``work_wires`` into one that
    dynamically allocates them via :func:`~pennylane.allocate`.

    The wrapped rule calls ``inner_decomp`` inside a :func:`~pennylane.allocate` context
    so that the caller does not need to provide work wires up front. The resource and
    condition functions of the inner rule are automatically re-evaluated with the
    allocated wire counts substituted in, so the wrapped rule reports identical gate
    counts to the inner rule and can be registered alongside it.

    Args:
        inner_decomp (DecompositionRule): a decomposition rule that takes
            ``work_wires=...`` as a keyword argument (i.e. it expects work wires to
            be supplied explicitly). Its resource function is expected to depend on
            ``num_control_wires`` / ``base_params`` but not on ``num_work_wires``
            or ``work_wire_type`` (the wrapper substitutes those for you).
        work_wires (dict or Callable): the work-wire requirements to allocate, using
            the same format accepted by :func:`~pennylane.register_resources`'s
            ``work_wires`` argument — either a static ``dict`` like
            ``{"zeroed": 1}`` or a callable ``f(**resource_params) -> dict`` that
            returns one. Only a single work-wire type is currently supported
            (``"zeroed"``, ``"borrowed"``, ``"burnable"``, or ``"garbage"``).
        name (str, optional): custom name for the wrapped rule. If not provided,
            defaults to ``f"allocate_work_wires({inner_decomp.name})"``.

    Returns:
        DecompositionRule: a new decomposition rule that dynamically allocates the
        requested work wires and delegates the body to ``inner_decomp``.

    **Example**

    .. code-block:: python

        def _resources(num_control_wires, base_params, **_):
            ...

        @qp.register_condition(
            lambda num_control_wires, num_work_wires, work_wire_type, **_: (
                num_control_wires >= 2
                and num_work_wires >= num_control_wires - 1
                and work_wire_type == "zeroed"
            )
        )
        @qp.register_resources(_resources)
        def _decomp_with_work_wires(*_, control_wires, work_wires, base, **__):
            ...

        _decomp_with_allocated_work_wires = make_dynamic_work_wires_rule(
            _decomp_with_work_wires,
            work_wires=lambda num_control_wires, **_: {"zeroed": num_control_wires - 1},
        )

        qp.add_decomps(
            "C(Prod)",
            _decomp_with_work_wires,
            _decomp_with_allocated_work_wires,
        )
    """
    # pylint: disable=protected-access

    def _resolve_spec(resource_params) -> dict:
        spec = work_wires(**resource_params) if callable(work_wires) else work_wires
        if not isinstance(spec, dict):
            raise TypeError(
                "work_wires must be a dict or a callable returning a dict. "
                f"Got {type(spec).__name__}."
            )
        # Filter out zero-count entries so ``_resolve_type`` does not consider them.
        return {k: v for k, v in spec.items() if v}

    def _resolve_type(spec) -> tuple[str, int]:
        """Return the single (work_wire_type, count) pair described by ``spec``.

        ``make_dynamic_work_wires_rule`` currently only supports rules that require
        a single kind of work wire — the common case. Mixed allocation would require
        substituting different ``work_wire_type`` values into the inner resource
        function which is not well-defined.
        """
        total = sum(spec.values())
        if total == 0:
            # No work wires required — delegate directly without allocating.
            return "zeroed", 0
        nonzero = [k for k, v in spec.items() if v]
        if len(nonzero) > 1:
            raise NotImplementedError(
                "make_dynamic_work_wires_rule currently only supports a single "
                f"work-wire type at a time. Got {spec}."
            )
        (ww_type,) = nonzero
        if ww_type not in _WORK_WIRE_ALLOCATE_KWARGS:
            raise ValueError(
                f"Unknown work wire type '{ww_type}'. "
                f"Expected one of {sorted(_WORK_WIRE_ALLOCATE_KWARGS)}."
            )
        return ww_type, total

    def _patched_params(resource_params):
        """Patch ``num_work_wires`` / ``work_wire_type`` so the inner rule sees the
        wires the wrapper is about to allocate."""
        spec = _resolve_spec(resource_params)
        ww_type, total = _resolve_type(spec)
        new_params = resource_params.copy()
        new_params["num_work_wires"] = total
        new_params["work_wire_type"] = ww_type
        return new_params

    def _condition_fn(**resource_params):
        try:
            return inner_decomp.is_applicable(**_patched_params(resource_params))
        except (KeyError, TypeError):  # pragma: no cover
            return False

    def _resource_fn(**resource_params):
        return inner_decomp.compute_resources(**_patched_params(resource_params)).gate_counts

    @register_condition(_condition_fn)
    @register_resources(
        _resource_fn,
        work_wires=work_wires,
        exact=inner_decomp.exact_resources,
        name=name or f"allocate_work_wires({inner_decomp.name})",
    )
    def _impl(*params, **kwargs):
        # ``kwargs`` may contain ``work_wires`` / ``work_wire_type`` from the outer
        # ``Controlled.hyperparameters`` — drop them so we can substitute the ones
        # we're about to allocate.
        resource_params = _infer_resource_params(kwargs)
        kwargs.pop("work_wires", None)
        kwargs.pop("work_wire_type", None)
        spec = _resolve_spec(resource_params)
        ww_type, total = _resolve_type(spec)
        if total == 0:
            inner_decomp(*params, work_wires=(), **kwargs)
            return
        allocate_kwargs = _WORK_WIRE_ALLOCATE_KWARGS[ww_type]
        with allocation.allocate(total, **allocate_kwargs) as allocated_wires:
            inner_decomp(*params, work_wires=allocated_wires, **kwargs)

    _impl._source = (
        dedent(_impl._source).strip()
        + "\n\nwhere inner_decomp is defined as:\n\n"
        + dedent(inner_decomp._source).strip()
    )
    return _impl


def _infer_resource_params(qfunc_kwargs: dict) -> dict:
    """Derive the ``resource_params`` subset needed by the work-wire spec callable
    from the kwargs the decomposition qfunc was invoked with.

    When a decomposition qfunc is executed, it receives ``op.hyperparameters``-style
    kwargs (e.g. ``control_wires``, ``control_values``, ``base``) rather than
    ``op.resource_params``. We translate the subset that work-wire spec callables
    typically depend on so users can write a single callable that works for both
    ``register_resources`` (which receives ``resource_params``) and the wrapper.
    """
    params = {}
    control_wires = qfunc_kwargs.get("control_wires")
    if control_wires is not None:
        params["num_control_wires"] = len(control_wires)
    control_values = qfunc_kwargs.get("control_values")
    if control_values is not None:
        params["num_zero_control_values"] = sum(1 for v in control_values if not v)
    base = qfunc_kwargs.get("base")
    if base is not None:
        params["base_class"] = type(base)
        params["base_params"] = base.resource_params
    return params


def _flip_control_adjoint_resource(
    base_class,
    base_params,
    num_control_wires,
    num_zero_control_values,
    num_work_wires,
    work_wire_type,
):  # pylint: disable=too-many-arguments
    # base class is adjoint, and the base of the base is the target class
    target_class, target_params = base_params["base_class"], base_params["base_params"]
    inner_rep = controlled_resource_rep(
        base_class=target_class,
        base_params=target_params,
        num_control_wires=num_control_wires,
        num_zero_control_values=num_zero_control_values,
        num_work_wires=num_work_wires,
        work_wire_type=work_wire_type,
    )
    return {adjoint_resource_rep(inner_rep.op_type, inner_rep.params): 1}


# pylint: disable=too-many-arguments
@register_resources(_flip_control_adjoint_resource)
def flip_control_adjoint(
    *_, wires, control_wires, control_values, work_wires, work_wire_type, base, **__
):
    """Decompose the control of an adjoint by applying control to the base of the adjoint
    and taking the adjoint of the control."""
    base_op = qp.pytrees.unflatten(*qp.pytrees.flatten(base.base))
    qp.adjoint(
        qp.ctrl(
            base_op,
            control=wires[: len(control_wires)],
            control_values=control_values,
            work_wires=work_wires,
            work_wire_type=work_wire_type,
        )
    )


def _ctrl_single_work_wire_resource(base_class, base_params, num_control_wires, **__):
    return {
        controlled_resource_rep(qp.X, {}, num_control_wires): 2,
        controlled_resource_rep(base_class, base_params, 1): 1,
    }


# pylint: disable=protected-access,unused-argument
@register_condition(lambda num_control_wires, **_: num_control_wires > 2)
@register_resources(_ctrl_single_work_wire_resource, work_wires={"zeroed": 1})
def _ctrl_single_work_wire(*params, wires, control_wires, base, **__):
    """Implements Lemma 7.11 from https://arxiv.org/abs/quant-ph/9503016."""
    base_op = qp.pytrees.unflatten(*qp.pytrees.flatten(base))
    with allocation.allocate(1, state="zero", restored=True) as work_wires:
        qp.ctrl(qp.X(work_wires[0]), control=control_wires)
        qp.ctrl(base_op, control=work_wires[0])
        qp.ctrl(qp.X(work_wires[0]), control=control_wires)


ctrl_single_work_wire = flip_zero_control(_ctrl_single_work_wire)


def _to_controlled_qu_condition(base_class, **__):
    return base_class.has_matrix and base_class.num_wires == 1


def _to_controlled_qu_resource(
    num_control_wires, num_zero_control_values, num_work_wires, work_wire_type, **__
):
    return {
        resource_rep(
            qp.ControlledQubitUnitary,
            num_target_wires=1,
            num_control_wires=num_control_wires,
            num_zero_control_values=num_zero_control_values,
            num_work_wires=num_work_wires,
            work_wire_type=work_wire_type,
        ): 1
    }


@register_condition(_to_controlled_qu_condition)
@register_resources(_to_controlled_qu_resource)
def to_controlled_qubit_unitary(*_, wires, control_values, work_wires, work_wire_type, base, **__):
    """Convert a controlled operator to a controlled qubit unitary."""
    matrix = base.matrix()
    qp.ControlledQubitUnitary(
        matrix,
        wires,
        control_values=control_values,
        work_wires=work_wires,
        work_wire_type=work_wire_type,
    )
