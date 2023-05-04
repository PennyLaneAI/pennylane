try:
    import zarr
except ImportError as Error:
    raise ImportError(
        "This feature requires the 'zarr' package. "
        "It can be installed with:\n\n pip install zarr."
    ) from Error


from .attributes import (
    DatasetArray,
    DatasetDict,
    DatasetHamiltonian,
    DatasetList,
    DatasetScalar,
    DatasetString,
)
from .base.attribute import AttributeInfo, AttributeType
from .dataset import Attribute, Dataset, attribute


__all__ = (
    "Attribute",
    "AttributeType",
    "AttributeInfo",
    "attribute",
    "Dataset",
    "DatasetArray",
    "DatasetDict",
    "DatasetList",
    "DatasetScalar",
    "DatasetString",
    "DatasetHamiltonian",
)
