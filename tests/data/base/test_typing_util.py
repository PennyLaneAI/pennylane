from pennylane.data.base.typing_util import get_type_str, get_type
from pennylane.qchem import Molecule
import pytest
import typing
import pennylane as qml


@pytest.mark.parametrize(
    "type_, expect",
    [
        (list, "list"),
        (typing.List, "list"),
        (Molecule, "pennylane.qchem.molecule.Molecule"),
        ("nonsense", "nonsense"),
        (typing.List[int], "list[int]"),
        (typing.List[typing.Tuple[int, "str"]], "list[tuple[int, str]]"),
        (typing.Optional[int], "Union[int, None]"),
        (typing.Union[int, "str", Molecule], "Union[int, str, pennylane.qchem.molecule.Molecule]"),
        (str, "str"),
        (typing.Type[str], "type[str]"),
    ],
)
def test_get_type_str(type_, expect):
    """Test that ``get_type_str()`` returns the expected value for various
    typing forms."""
    assert get_type_str(type_) == expect


@pytest.mark.parametrize(
    "obj, expect",
    [
        (list, list),
        ([1, 2], list),
        (typing.List, list),
        (typing.List[int], list),
        (qml.RX, qml.RX),
        (qml.RX(1, [1]), qml.RX),
    ],
)
def test_get_type(obj, expect):
    """Test that ``get_type()`` returns the expected value for various objects
    and types."""
    assert get_type(obj) is expect
