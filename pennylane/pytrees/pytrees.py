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
An internal module for working with pytrees.
"""
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import autograd

import pennylane.queuing

has_jax = True
try:
    import jax.tree_util as jax_tree_util
except ImportError:
    has_jax = False

Leaves = Any
Metadata = Any

FlattenFn = Callable[[Any], tuple[Leaves, Metadata]]
UnflattenFn = Callable[[Leaves, Metadata], Any]


def flatten_list(obj: list):
    """Flatten a list."""
    return obj, None


def flatten_tuple(obj: tuple):
    """Flatten a tuple."""
    return obj, None


def flatten_dict(obj: dict):
    """Flatten a dictionary."""
    return obj.values(), tuple(obj.keys())


flatten_registrations: dict[type, FlattenFn] = {
    list: flatten_list,
    tuple: flatten_tuple,
    dict: flatten_dict,
}


def unflatten_list(data, _) -> list:
    """Unflatten a list."""
    return data if isinstance(data, list) else list(data)


def unflatten_tuple(data, _) -> tuple:
    """Unflatten a tuple."""
    return tuple(data)


def unflatten_dict(data, metadata) -> dict:
    """Unflatten a dictionary."""
    return dict(zip(metadata, data, strict=True))


unflatten_registrations: dict[type, UnflattenFn] = {
    list: unflatten_list,
    tuple: unflatten_tuple,
    dict: unflatten_dict,
}

type_to_typename: dict[type, str] = {
    list: "builtins.list",
    dict: "builtins.dict",
    tuple: "builtins.tuple",
}

typename_to_type: dict[str, type] = {name: type_ for type_, name in type_to_typename.items()}


def _register_pytree_with_pennylane(
    pytree_type: type, typename: str, flatten_fn: FlattenFn, unflatten_fn: UnflattenFn
):
    type_to_typename[pytree_type] = typename
    typename_to_type[typename] = pytree_type

    flatten_registrations[pytree_type] = flatten_fn
    unflatten_registrations[pytree_type] = unflatten_fn


def _register_pytree_with_jax(pytree_type: type, flatten_fn: FlattenFn, unflatten_fn: UnflattenFn):
    def jax_unflatten(aux, parameters):
        return unflatten_fn(parameters, aux)

    jax_tree_util.register_pytree_node(pytree_type, flatten_fn, jax_unflatten)


def register_pytree(
    pytree_type: type, flatten_fn: FlattenFn, unflatten_fn: UnflattenFn, *, namespace: str = "qml"
):
    """Register a type with all available pytree backends.

    Current backends are jax and pennylane.

    Args:
        pytree_type (type): the type to register, such as ``qml.RX``
        flatten_fn (Callable): a function that splits an object into trainable leaves and hashable metadata.
        unflatten_fn (Callable): a function that reconstructs an object from its leaves and metadata.
        namespace (str): A prefix for the name under which this type will be registered.

    Returns:
        None

    Side Effects:
        ``pytree`` type becomes registered with available backends.

    .. seealso:: :func:`~.flatten`, :func:`~.unflatten`.

    """

    typename = f"{namespace}.{pytree_type.__qualname__}"
    _register_pytree_with_pennylane(pytree_type, typename, flatten_fn, unflatten_fn)

    if has_jax:
        _register_pytree_with_jax(pytree_type, flatten_fn, unflatten_fn)


def is_pytree(type_: type[Any]) -> bool:
    """Returns True if ``type_`` is a registered Pytree."""
    return type_ in type_to_typename


def get_typename(pytree_type: type[Any]) -> str:
    """Return the typename under which ``pytree_type``
    was registered.

    Raises:
        TypeError: If ``pytree_type`` is not a Pytree.

    >>> get_typename(list)
    'builtins.list'
    >>> import pennylane
    >>> get_typename(pennylane.PauliX)
    'qml.PauliX'
    """

    try:
        return type_to_typename[pytree_type]
    except KeyError as exc:
        raise TypeError(f"{repr(pytree_type)} is not a Pytree type") from exc


def get_typename_type(typename: str) -> type[Any]:
    """Return the Pytree type with given ``typename``.

    Raises:
        ValueError: If ``typename`` is not the name of a
            pytree type.

    >>> import pennylane
    >>> get_typename_type("builtins.list")
    <class 'list'>
    >>> get_typename_type("qml.PauliX")
    <class 'pennylane.ops.qubit.non_parametric_ops.PauliX'>
    """
    try:
        return typename_to_type[typename]
    except KeyError as exc:
        raise ValueError(f"{repr(typename)} is not the name of a Pytree type.") from exc


@dataclass(repr=False)
class PyTreeStructure:
    """A pytree data structure, holding the type, metadata, and child pytree structures.

    >>> op = qml.adjoint(qml.RX(0.1, 0))
    >>> data, structure = qml.pytrees.flatten(op)
    >>> structure
    PyTreeStructure(AdjointOperation, (), [PyTreeStructure(RX, (Wires([0]), ()), [PyTreeStructure()])])

    A leaf is defined as just a ``PyTreeStructure`` with ``type_=None``.
    """

    type_: type[Any] | None = None
    """The type corresponding to the node. If ``None``, then the structure is a leaf."""

    metadata: Metadata = ()
    """Any metadata needed to reproduce the original object."""

    children: list["PyTreeStructure"] = field(default_factory=list)
    """The children of the pytree node.  Can be either other structures or terminal leaves."""

    @property
    def is_leaf(self) -> bool:
        """Whether or not the structure is a leaf."""
        return self.type_ is None

    def __repr__(self):
        if self.is_leaf:
            return "PyTreeStructure()"
        return f"PyTreeStructure({self.type_.__name__}, {self.metadata}, {self.children})"

    def __str__(self):
        if self.is_leaf:
            return "Leaf"
        children_string = ", ".join(str(c) for c in self.children)
        return f"PyTree({self.type_.__name__}, {self.metadata}, [{children_string}])"


leaf = PyTreeStructure(None, (), [])


def flatten(
    obj: Any, is_leaf: Callable[[Any], bool] | None = None
) -> tuple[list[Any], PyTreeStructure]:
    """Flattens a pytree into leaves and a structure.

    Args:
        obj (Any): any object.
        is_leaf (Callable[[Any], bool] | None = None): an optionally specified
            function that will be called at each flattening step. It should return
            a boolean, with ``True`` stopping the traversal and the whole subtree being
            treated as a leaf, and ``False`` indicating the flattening should traverse
            the current object.

    Returns:
        List[Any], Union[Structure, Leaf]: a list of leaves and a structure representing the object

    See also :func:`~.unflatten`.

    **Example**

    >>> op = qml.adjoint(qml.Rot(1.2, 2.3, 3.4, wires=0))
    >>> data, structure = flatten(op)
    >>> data
    [1.2, 2.3, 3.4]

    >>> structure
    PyTreeStructure(AdjointOperation, (), [PyTreeStructure(Rot, (Wires([0]), ()), [PyTreeStructure(), PyTreeStructure(), PyTreeStructure()])])
    """
    flatten_fn = flatten_registrations.get(type(obj), None)
    # set the flag is_leaf_node if is_leaf argument is provided and returns true
    is_leaf_node = is_leaf(obj) if is_leaf is not None else False
    if flatten_fn is None or is_leaf_node:
        return [obj], leaf
    leaves, metadata = flatten_fn(obj)

    flattened_leaves = []
    child_structures = []
    for l in leaves:
        child_leaves, child_structure = flatten(l, is_leaf)
        flattened_leaves += child_leaves
        child_structures.append(child_structure)

    structure = PyTreeStructure(type(obj), metadata, child_structures)
    return flattened_leaves, structure


def unflatten(data: list[Any], structure: PyTreeStructure) -> Any:
    """Bind data to a structure to reconstruct a pytree object.

    Args:
        data (Iterable): iterable of numbers and numeric arrays
        structure (Structure, Leaf): The pytree structure object

    Returns:
        A repacked pytree.

    .. seealso:: :func:`~.flatten`

    **Example**

    >>> op = qml.adjoint(qml.Rot(1.2, 2.3, 3.4, wires=0))
    >>> data, structure = flatten(op)
    >>> unflatten([-2, -3, -4], structure)
    Adjoint(Rot(-2, -3, -4, wires=[0]))

    """
    with pennylane.queuing.QueuingManager.stop_recording():
        return _unflatten(iter(data), structure)


def _unflatten(new_data, structure):
    if structure.is_leaf:
        return next(new_data)
    children = tuple(_unflatten(new_data, s) for s in structure.children)
    return unflatten_registrations[structure.type_](children, structure.metadata)


register_pytree(
    autograd.builtins.list,
    lambda obj: (list(obj), ()),
    lambda data, _: autograd.builtins.list(data),
)
register_pytree(
    autograd.builtins.tuple,
    lambda obj: (list(obj), ()),
    lambda data, _: autograd.builtins.tuple(data),
)
register_pytree(
    autograd.builtins.SequenceBox,
    lambda obj: (list(obj), ()),
    lambda data, _: autograd.builtins.SequenceBox(data),
)
