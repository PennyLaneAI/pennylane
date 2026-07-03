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
r"""Resource operators for symbolic operations."""

from functools import wraps
from inspect import Signature, signature
from typing import Callable, Iterable

import pennylane.labs.estimator_beta as qre
from pennylane.core.operator import Operator
from pennylane.labs.estimator_beta import (
    CompressedResourceOp,
    GateCount,
    ResourceOperator,
)
from pennylane.queuing import AnnotatedQueue, QueuingManager
from pennylane.wires import Wires

# pylint: disable=arguments-differ


def _generate_name(
    func_name: str,
    func_sig: Signature,
    include_params: Iterable[str] | None = None,
    *args,
    **kwargs,
):  # pylint: disable=keyword-arg-before-vararg
    r"""Format a string representing the name of a function from its signature.

    Args:
        func_name (str): the name of the function
        func_sig (Signature): the function signature
        include_params (Iterable[str] | None): An optional iterable of strings listing the
            parameters to include in the name.

    Returns:
        str: formatted string representing the name of the function

    **Example**

    The resources for this operation are computed using:

    >>> from inspect import signature
    >>> def my_func(arg1, arg2, kwarg1 = "a"):
    ...     return
    ...
    >>> f_name = my_func.__name__
    >>> f_sig = signature(my_func)
    >>>
    >>> _generate_name(f_name, f_sig)
    'my_func'
    >>> _generate_name(
    ...     f_name,
    ...     f_sig,
    ...     include_params = ["arg1", "kwarg1"],
    ...     arg1 = 10,
    ...     arg2 = True,
    ...     kwarg1 = "b",
    ... )
    'my_func(arg1=10, kwarg1=b)'

    """
    if include_params is None:
        return func_name

    param_strs = []
    for param_name in include_params:

        param_kind = str(func_sig.parameters[param_name].kind)
        param_index = list(func_sig.parameters.keys()).index(param_name)
        if param_kind in ("KEYWORD_ONLY", "POSITIONAL_OR_KEYWORD"):
            param_value = kwargs.get(param_name, args[param_index])

        if param_kind == "VAR_POSITIONAL":
            param_value = args[param_index:]

        else:  # param_kind == "VAR_KEYWORD"
            param_value = {k: v for k, v in kwargs.items() if k not in func_sig.parameters}

        param_str = param_name + "=" + str(param_value)
        param_strs.append(param_str)

    name = func_name + "(" + ", ".join(param_strs) + ")"
    return name


def mark_subroutine(qfunc: Callable, include_params: Iterable[str] | None = None):
    r"""A decorator that can be used to promote a quantum function to a resource operator.
    This allows users to explicitly track counts to subroutines which have been implented as
    quantum functions.

    Args:
        qfunc (Callable): the quantum function representing the subroutine we wish to capture
        include_params (Iterable[str] | None): An optional iterable of strings listing the
            parameters to include in the name.

    Returns:
        Callable: A function which, when executed, returns a :class:``~.estimator.ResourceOperator``
            that represents the product of the operators defined within the input ``qfunc`` function.

    **Example**

    This decorator can be used to help track resources at higher, user-defined levels of abstraction.

    .. code-block:: python

        import pennylane.labs.estimator_beta as qre

        @qre.mark_subroutine
        def SubroutineA(num_iter, op_type="Z"):
            for i in range(num_iter):
                if op_type == "Z":
                    qre.Z()
                else:
                    qre.X()

        def circuit():
            SubroutineA(5)

            qre.CNOT()

            SubroutineA(3)

            qre.QROM(3, 2)

            SubroutineA(5)

    >>> gate_set = {"CNOT", "QROM", "SubroutineA"}
    >>> print(qre.estimate(circuit, gate_set)())
    --- Resources: ---
     Total wires: 4
       algorithmic wires: 4
       allocated wires: 0
         zero state: 0
         any state: 0
     Total gates : 5
       'SubroutineA': 3,
       'QROM': 1,
       'CNOT': 1

    The name that is used to track the counts of the function can be further configured using the
    ``include_params`` keyword argument.

    .. code-block:: python

        from functools import partial
        import pennylane.labs.estimator_beta as qre

        @partial(qre.mark_subroutine, include_params=["num_iter"])
        def SubroutineA(num_iter, op_type="Z"):
            for i in range(num_iter):
                if op_type == "Z":
                    qre.Z()
                else:
                    qre.X()

    >>> gate_set = {"CNOT", "QROM", "SubroutineA(num_iter=5)", "SubroutineA(num_iter=3)"}
    >>> print(qre.estimate(circuit, gate_set)())
    --- Resources: ---
     Total wires: 4
       algorithmic wires: 4
       allocated wires: 0
         zero state: 0
         any state: 0
     Total gates : 5
       'SubroutineA(num_iter=5)': 2,
       'SubroutineA(num_iter=3)': 1,
       'QROM': 1,
       'CNOT': 1

    """

    @wraps(qfunc)
    def wrapper(*args, **kwargs):
        func_name = qfunc.__name__
        func_sig = signature(qfunc)

        name = _generate_name(func_name, func_sig, include_params, *args, **kwargs)
        return ResourceQfunc(name, qfunc, *args, **kwargs)

    return wrapper


class ResourceQfunc(ResourceOperator):
    r"""Resource class for a quantum function (qfunc) promoted to a resource operator.

    This symbolic class can be used to represent a subroutine defined as a quantum function. All of the
    operators defined in the quantum function represent the resources of the subroutine. This allows users
    to quickly define and track the resources at higher levels of abstraction.

    Args:
        name (str): the name used to track the counts of the subroutine
        resource_decomp_fn (Callable): the quantum function representing the subroutine
        *resource_args: positional arguments of the ``resource_decomp_fn``
        **resource_kwargs: keyword arguments of the ``resource_decomp_fn``

    Resources:
        The resources are defined trivially as each operation called within the quantum function.

    Raises:
        TypeError: if the quantum function calls a :class:`~.pennylane.labs.estimator_beta.MarkQubits`
            instance, which is not currently supported inside a ``ResourceQfunc``.

    .. seealso:: The decorator :func:`~.pennylane.labs.estimator_beta.mark_subroutine`.

    **Example**

    .. code-block:: python

        import pennylane.labs.estimator_beta as qre

        def SubroutineA(num_iter, op_type="Z"):
            for _ in range(num_iter):
                if op_type == "Z":
                    qre.Z()
                else:
                    qre.X()

    We obtain the expected resources when a suitable gate set is chosen:

    >>> op = qre.ResourceQfunc(
    ...     name = "SubA",
    ...     resource_decomp_fn = SubroutineA,
    ...     num_iter = 3,
    ...     op_type = "X",
    ... )
    >>> print(qre.estimate(op, {"X", "Z"}))
    --- Resources: ---
     Total wires: 1
       algorithmic wires: 1
       allocated wires: 0
         zero state: 0
         any state: 0
     Total gates : 3
       'X': 3
    >>> print(qre.estimate(op, {"SubA"}))
    --- Resources: ---
     Total wires: 1
       algorithmic wires: 1
       allocated wires: 0
         zero state: 0
         any state: 0
     Total gates : 1
       'SubA': 1

    """

    resource_keys = {"name", "num_wires", "cmpr_ops"}

    def __init__(
        self, name, resource_decomp_fn, *resource_args, **resource_kwargs
    ):  # pylint: disable=super-init-not-called
        self.name = name

        with QueuingManager.stop_recording():
            with AnnotatedQueue() as q:
                resource_decomp_fn(*resource_args, **resource_kwargs)

            decomp = []
            for op in q.queue:  # Filter the queue and only pull the operators
                if isinstance(op, ResourceOperator):
                    decomp.append(op)
                elif isinstance(op, Operator):
                    decomp.append(qre._map_to_resource_op(op))
                elif isinstance(
                    op, qre.MarkQubits
                ):  # TODO: @Jaybsoni to add support for this eventually
                    raise TypeError(
                        "Marking qubits is currently not supported with mark_subroutine and ResourceQfunc. "
                        "Instead, instantiate MarkQubit instances within the main Qnode or qfunc directly."
                    )
                else:
                    continue

        self.cmpr_ops = tuple(op.resource_rep_from_op() for op in decomp)

        ops_wires = Wires.all_wires([op.wires for op in decomp if op.wires is not None])
        num_unique_wires_required = max(op.num_wires for op in self.cmpr_ops)

        if (
            len(ops_wires) < num_unique_wires_required
        ):  # If factors didn't provide enough wire labels
            self.wires = None  # we assume they all act on the same set
            self.num_wires = num_unique_wires_required

        else:  # If there are more wire labels, use that as the operator wires
            self.wires = ops_wires
            self.num_wires = len(self.wires)

        self.queue()

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * name (str): the name used to track the counts of the subroutine
                * num_wires (int): the number of wires the subroutine acts upon
                * cmpr_ops (tuple[:class:`~.pennylane.labs.estimator_beta.CompressedResourceOp`]): A
                  tuple containing the operations queued by the subroutine, in the compressed
                  representation, corresponding to the factors of the product.

        """
        return {
            "name": self.name,
            "num_wires": self.num_wires,
            "cmpr_ops": self.cmpr_ops,
        }

    @classmethod
    def resource_rep(cls, name, num_wires, cmpr_ops):
        r"""Returns a compressed representation containing only the parameters of
        the operator that are needed to compute a resource estimation.

        Args:
            name (str): the name used to track the counts of the subroutine
            num_wires (int): the number of wires the subroutine acts upon
            cmpr_ops (tuple[:class:`~.pennylane.labs.estimator_beta.CompressedResourceOp`]): A
                tuple containing the operations queued by the subroutine, in the compressed
                representation, corresponding to the factors of the product.

        Returns:
            :class:`~.pennylane.labs.estimator_beta.CompressedResourceOp`: the operator in a compressed representation
        """
        params = {
            "name": name,
            "num_wires": num_wires,
            "cmpr_ops": cmpr_ops,
        }
        return CompressedResourceOp(cls, num_wires, params, name=name)

    @classmethod
    def resource_decomp(cls, name, num_wires, cmpr_ops):  # pylint: disable=unused-argument
        r"""Returns a list representing the resources of the operator. Each object represents a
        quantum gate and the number of times it occurs in the decomposition.

        Args:
            name (str): the name used to track the counts of the subroutine
            num_wires (int): the number of wires the subroutine acts upon
            cmpr_ops (tuple[:class:`~.pennylane.labs.estimator_beta.CompressedResourceOp`]): A
                tuple containing the operations queued by the subroutine, in the compressed
                representation, corresponding to the factors of the product.

        Resources:
            The resources are defined trivially as each operation called within the quantum function.

        Returns:
            list[:class:`~.pennylane.labs.estimator_beta.GateCount`]: A list of ``GateCount`` objects,
            where each object represents a specific quantum gate and the number of times it appears
            in the decomposition.

        """
        return [GateCount(cmpr_op) for cmpr_op in cmpr_ops]

    @staticmethod
    def tracking_name(name, num_wires, cmpr_ops) -> str:  # pylint: disable=unused-argument
        r"""Returns the tracking name built with the operator's parameters."""
        return name
