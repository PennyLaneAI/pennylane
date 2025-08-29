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
"""
This module contains the commands for allocating and deallocating wires dynamically.
"""
import uuid
from collections.abc import Sequence

from pennylane.capture import enabled as capture_enabled
from pennylane.operation import Operator
from pennylane.wires import Wires

has_jax = True
try:
    import jax

    # pylint: disable=ungrouped-imports
    from pennylane.capture import QmlPrimitive
except ImportError:
    jax = None
    has_jax = False


if not has_jax:
    allocate_prim = None
    deallocate_prim = None
else:
    allocate_prim = QmlPrimitive("allocate")
    allocate_prim.multiple_results = True

    # pylint: disable=unused-argument
    @allocate_prim.def_impl
    def _(*, num_wires, require_zeros=True, restored=False):
        raise NotImplementedError("jaxpr containing qubit allocation cannot be executed.")

    # pylint: disable=unused-argument
    @allocate_prim.def_abstract_eval
    def _(*, num_wires, require_zeros=True, restored=False):
        return [jax.core.ShapedArray((), dtype=int) for _ in range(num_wires)]

    deallocate_prim = QmlPrimitive("deallocate")
    deallocate_prim.multiple_results = True

    # pylint: disable=unused-argument
    @deallocate_prim.def_impl
    def _(*wires):
        raise NotImplementedError("jaxpr containing qubit deallocation cannot be executed.")

    # pylint: disable=unused-argument
    @deallocate_prim.def_abstract_eval
    def _(*wires):
        return []


class DynamicWire:
    """A wire whose concrete value will be determined later during a compilation step or execution.

    Multiple dynamic wires can correspond to the same device wire as long as they are properly allocated and
    deallocated.

    Args:
        key (Optional[str]): a ``uuid4`` string to uniquely identify the dynamic wire.
    """

    def __init__(self, key: uuid.UUID | None = None):
        self.key = key or uuid.uuid4()

    def __eq__(self, other):
        if not isinstance(other, DynamicWire):
            return False
        return self.key == other.key

    def __hash__(self):
        return hash(("DynamicWire", self.key))

    def __repr__(self):
        return "<DynamicWire>"


class Allocate(Operator):
    """An instruction to request dynamic wires.

    Args:
        wires (list[DynamicWire]): a list of dynamic wire values.

    Keyword Args:
        require_zeros (bool): Whether or not the wire must start in a ``0`` state.
        restored (bool): Whether or not the qubit will be restored to the original state before being deallocated.

    ..see-also:: :func:`~.allocate`.

    """

    def __init__(self, wires, require_zeros=True, restored=False):
        super().__init__(wires=wires)
        self._hyperparameters = {"require_zeros": require_zeros, "restored": restored}

    @property
    def require_zeros(self):
        """Whether or not the allocated wires are required to be in the zero state."""
        return self.hyperparameters["require_zeros"]

    @property
    def restored(self):
        """Whether the allocated wires will be restored to their original state before deallocation."""
        return self.hyperparameters["restored"]

    @classmethod
    def from_num_wires(cls, num_wires: int, require_zeros=True, restored=False) -> "Allocate":
        """Initialize an ``Allocate`` op from a number of wires instead of already constructed dynamic wires."""
        wires = tuple(DynamicWire() for _ in range(num_wires))
        return cls(wires=wires, require_zeros=require_zeros, restored=restored)


class Deallocate(Operator):
    """An instruction to deallocate the provided ``DynamicWire``'s.

    Args:
        wires (DynamicWire, Sequence[DynamicWire]): one or more dynamic wires to deallocate.

    """

    def __init__(self, wires: DynamicWire | Sequence[DynamicWire]):
        super().__init__(wires=wires)


def deallocate(wires: DynamicWire | Wires | Sequence[DynamicWire]) -> Deallocate:
    """Deallocates wires that were previously allocated with :func:`~.allocate`. Upon deallocating,
    the wires are available to be allocated thereafter for efficient resource usage.

    Args:
        wires (DynamicWire, Wires, Sequence[DynamicWire]): one or more dynamic wires.

    .. seealso:: :func:`~.allocate`

    .. note::
        The :func:`~.allocate` function can be used as a context manager with automatic deallocation 
        (recommended for most cases) upon exiting the scope.

    **Example**

    .. code-block:: python

        import pennylane as qml
    
        @qml.qnode(qml.device("default.qubit"))
        def c():
            qml.H(0)

            wires = qml.allocate(1, state="zero", restored=True)
            qml.CNOT((0, wire[0]))
            qml.CNOT((0, wire[0]))
            qml.deallocate(wire)

            new_wire = qml.allocate(2, state="zero", restored=True)
            qml.SWAP((new_wire[1], new_wire[0]))
            qml.deallocate(new_wire)
            
            return qml.expval(qml.Z(0))

    >>> print(qml.draw(c)())
                0: ──H────────╭●────╭●──────────────────────┤  <Z>
    <DynamicWire>: ──Allocate─╰X────╰X───────────Deallocate─┤     
    <DynamicWire>: ─╭Allocate─╭SWAP─╭Deallocate─────────────┤     
    <DynamicWire>: ─╰Allocate─╰SWAP─╰Deallocate─────────────┤     

    Here, three dynamic wires were allocated in the circuit originally. When PennyLane determines
    what concrete values to use for dynamic wires to send to the device for execution, we can see 
    that the first dynamic wire is already deallocated back into the zero state. This allows us to 
    use it for one of the wires requested in the second allocation, resulting in three wires total
    being required from the device:

    >>> print(qml.draw(c, level="device")())
    0: ──H─╭●─╭●───────┤  <Z>
    1: ────╰X─╰X─╭SWAP─┤     
    2: ──────────╰SWAP─┤    
    """
    if capture_enabled():
        if not isinstance(wires, Sequence):
            wires = (wires,)
        return deallocate_prim.bind(*wires)
    wires = Wires(wires)
    if not_dynamic_wires := [w for w in wires if not isinstance(w, DynamicWire)]:
        raise ValueError(f"deallocate only accepts DynamicWire wires. Got {not_dynamic_wires}")
    return Deallocate(wires)


# pylint: disable=too-many-ancestors
class DynamicRegister(Wires):
    """A specialized ``Wires`` class for dynamic wires with a context manager for automatic deallocation."""

    def __repr__(self):
        return f"<DynamicRegister: size={len(self._labels)}>"

    def __enter__(self):
        return self

    def __exit__(self, *_, **__):
        deallocate(self)

    def __hash__(self):
        raise TypeError("unhashable type 'DynamicRegister'")


def allocate(num_wires: int, require_zeros: bool = True, restored: bool = False) -> DynamicRegister:
    """Dynamically allocates new wires in-line, or as a context manager which also safely 
    deallocates the new wires upon exiting the context.

    Args:
        num_wires (int): 
            The number of wires to dynamically allocate.

    Keyword Args:
        state (str):
            Specifies whether to allocate ``num_wires`` in the all-zeros state (``"zero"``) or in 
            any arbitrary state (``"any"``). The default value is ``state="zero"``.

        restored (bool):
            Whether or not the dynamically allocated wires are returned to the same state they 
            started in. ``restored=True`` indicates that the user promises to restore the 
            dynamically allocated wires to their original state before being deallocated. 
            ``restored=False`` indicates that the user does not promise to restore the dynamically 
            allocated wires before before being deallocated. The default value is ``False``.

    Returns:
        DynamicRegister: an object, behaving similarly to ``Wires``, that represents the dynamically
        allocated wires.

    .. note::
        The ``allocate`` function can be used as a context manager with automatic deallocation 
        (recommended for most cases) or with manual deallocation via :func:`~.deallocate`.

    .. seealso::
        :func:`~.deallocate`

    **Example**
  
    Using ``allocate`` to dynamically request wires returns an array of wires 
    (``DynamicRegister``) that can be indexed into:

    >>> wires = qml.allocate(3)
    >>> wires
    <DynamicRegister: size=3>
    >>> wires[1]
    <DynamicWire>

    Note that allocating just one wire still requires indexing into:

    >>> wire = qml.allocate(1)
    >>> wire
    <DynamicRegister: size=1>
    >>> wire[0]
    <DynamicWire>
        
    Most use cases for ``allocate`` are covered by using it as a context manager, which ensures 
    that allocation and safe deallocation are controlled within a localized scope.

    .. code-block:: python

        import pennylane as qml

        @qml.qnode(qml.device("default.qubit")) 
        def circuit():
            qml.H(0)
            qml.H(1)

            with qml.allocate(2, state="zero", restored=True) as new_wires:
                qml.H(new_wires[0])
                qml.H(new_wires[1])
                
            return qml.expval(qml.Z(0))
            
    >>> print(qml.draw(circuit)())
                0: ──H───────────────────────┤  <Z>
                1: ──H───────────────────────┤
    <DynamicWire>: ─╭Allocate──H─╭Deallocate─┤
    <DynamicWire>: ─╰Allocate──H─╰Deallocate─┤

    Equivalenty, ``allocate`` can be used in-line along with :func:`~.deallocate` for manual 
    handling: 

    .. code-block:: python

        new_wires = qml.allocate(2, state="zero", restored=True)
        qml.H(new_wires[0])
        qml.H(new_wires[1])
        qml.deallocate(new_wires)
    
        
    .. details:: 
        :title: Usage details 

        For more complex dynamic allocation in circuits, PennyLane will resolve the dynamic 
        allocation calls in the most resource-efficient manner before sending the program to the 
        device. Consider the following circuit, which contains two dynamic allocations within a 
        ``for`` loop.

        .. code-block:: python

            @qml.qnode(qml.device("default.qubit"), mcm_method="tree-traversal") 
            def circuit():
                qml.H(0)

                for i in range(2):
                    with qml.allocate(1, state="zero", restored=True) as new_qubit1:
                        with qml.allocate(1, state="any", restored=False) as new_qubit2:
                            m0 = qml.measure(new_qubit1[0], reset=True)
                            qml.cond(m0 == 1, qml.Z)(new_qubit2[0])
                            qml.CNOT((0, new_qubit2[0]))

                return qml.expval(qml.Z(0))

        >>> print(qml.draw(circuit)())
                    0: ──H─────────────────────╭●───────────────────────╭●─────────────┤  <Z>
        <DynamicWire>: ──Allocate──┤↗│  │0⟩────│──────────Deallocate────│──────────────┤     
        <DynamicWire>: ──Allocate───║────────Z─╰X─────────Deallocate────│──────────────┤     
        <DynamicWire>: ─────────────║────────║──Allocate──┤↗│  │0⟩──────│───Deallocate─┤     
        <DynamicWire>: ─────────────║────────║──Allocate───║──────────Z─╰X──Deallocate─┤     
                                    ╚════════╝             ╚══════════╝                      

        The user-level circuit drawing shows four separate allocations and deallocations (two per 
        loop iteration). However, the circuit that the device receives gets automatically compiled 
        to only use **two** additional wires (wires labelled ``1`` and ``2`` in the diagram below). The 
        is due to the fact that ``new_qubit1`` and ``new_qubit2`` can both be reused after they've been 
        deallocated in the first iteration of the ``for`` loop:

        >>> print(qml.draw(circuit, level="device")())
        0: ──H───────────╭●──────────────╭●─┤  <Z>
        1: ──┤↗│  │0⟩────│───┤↗│  │0⟩────│──┤     
        2: ───║────────Z─╰X───║────────Z─╰X─┤     
                ╚════════╝      ╚════════╝          
    """
    if capture_enabled():
        wires = allocate_prim.bind(
            num_wires=num_wires, require_zeros=require_zeros, restored=restored
        )
    else:
        wires = [DynamicWire() for _ in range(num_wires)]
    reg = DynamicRegister(wires)
    if not capture_enabled():
        Allocate(reg, require_zeros=require_zeros, restored=restored)
    return reg
