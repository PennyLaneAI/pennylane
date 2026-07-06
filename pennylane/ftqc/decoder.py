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

"""The Decoder contract for real-time quantum error correction.

A decoder declares the syndrome input and correction I/O schema, and registers the compiled kernel the device
dispatches to. Decoders are selectable by name through the registry.
"""

from dataclasses import dataclass

_DECODERS = {}


@dataclass(frozen=True)
class DecoderSchema:
    """The syndrome to correction I/O contract a decoder declares, which is implement on device.

    Args:
        syndrome (str): Type of the syndrome shipped in, e.g. ``"uint8[N]"``.
        correction (str): Type of the correction shipped out, e.g. ``"uint8[N]"``.
    """

    syndrome: str
    correction: str


@dataclass(frozen=True)
class Decoder:
    """A named decoder: its I/O schema and the compiled kernel the device dispatches to.

    Args:
        name (str): The name used for registration and selection.
        schema (DecoderSchema): The syndrome -> correction I/O contract.
        kernel (object | None): The compiled GPU artifact (e.g. a builder or a prebuilt library
            handle). Defaults to ``None``.
    """

    name: str
    schema: DecoderSchema
    kernel: object | None = None


def register_decoder(name):
    """Register a decoder factory under ``name``.

    Args:
        name (str): The name the decoder is selected by.

    Returns:
        Callable: A decorator that registers the factory and returns it.
    """

    def decorator(factory):
        _DECODERS[name] = factory
        return factory

    return decorator


def get_decoder(name):
    """Return the :class:`Decoder` registered under ``name``.

    Args:
        name (str): The decoder name.

    Returns:
        Decoder: The decoder produced by the registered factory.

    Raises:
        ValueError: If ``name`` is not registered.
    """
    try:
        factory = _DECODERS[name]
    except KeyError:
        raise ValueError(
            f"unknown decoder {name!r}; registered decoders: {sorted(_DECODERS)}"
        ) from None
    return factory()
