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

"""The ONNX coprocessor function: :func:`onnx_decoder`, which runs an ONNX model on each message
with onnxruntime."""

import importlib
import importlib.util
import sys
from pathlib import Path

from .functions import CoprocessorFunction

_ONNX_FUNCTION = "catalyst_onnx_coprocessor"
_ONNX_PROVIDERS = ("auto", "cpu", "migraphx", "cuda", "tensorrt")

# Bytes per element of each onnxruntime tensor type an ONNX model may take or give.
_ONNX_ELEMENT_BYTES = {
    "tensor(bool)": 1,
    "tensor(int8)": 1,
    "tensor(uint8)": 1,
    "tensor(int16)": 2,
    "tensor(uint16)": 2,
    "tensor(float16)": 2,
    "tensor(bfloat16)": 2,
    "tensor(int32)": 4,
    "tensor(uint32)": 4,
    "tensor(float)": 4,
    "tensor(int64)": 8,
    "tensor(uint64)": 8,
    "tensor(double)": 8,
}


def _onnx_message_bytes(model: Path) -> tuple[int, int]:
    """The ``(in_bytes, out_bytes)`` of an ONNX model's single input and output tensors, with each
    dynamic dimension taken as 1, read with the installed onnxruntime's CPU provider."""
    onnxruntime = importlib.import_module("onnxruntime")
    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = onnxruntime.InferenceSession(
        str(model), sess_options=options, providers=["CPUExecutionProvider"]
    )
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError(
            f"onnx_decoder: the model must have one input and one output, it has {len(inputs)} "
            f"and {len(outputs)}"
        )

    def tensor_bytes(arg) -> int:
        if arg.type not in _ONNX_ELEMENT_BYTES:
            raise ValueError(f"onnx_decoder: unsupported tensor type {arg.type} for {arg.name!r}")
        elements = 1
        for dim in arg.shape:
            elements *= dim if isinstance(dim, int) and dim >= 0 else 1
        return elements * _ONNX_ELEMENT_BYTES[arg.type]

    return tensor_bytes(inputs[0]), tensor_bytes(outputs[0])


def _onnxruntime_library() -> str:
    """The onnxruntime shared library of the installed ``onnxruntime`` package."""
    spec = importlib.util.find_spec("onnxruntime")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError(
            "onnx_decoder needs an onnxruntime package, such as onnxruntime for the CPU or "
            "onnxruntime-migraphx for AMD GPUs"
        )
    capi = Path(list(spec.submodule_search_locations)[0]) / "capi"
    pattern = "libonnxruntime.*dylib" if sys.platform == "darwin" else "libonnxruntime.so*"
    libraries = sorted(capi.glob(pattern))
    if not libraries:
        raise ImportError(f"no onnxruntime shared library found in {capi}")
    return str(libraries[0])


def _check_count(name: str, value, minimum: int) -> None:
    """Raise if ``value`` is not an int of at least ``minimum``."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"onnx_decoder: {name} must be an int, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"onnx_decoder: {name} must be at least {minimum}, got {value}")


def onnx_decoder(
    model, *, provider: str = "auto", device: int = 0, threads: int = 1
) -> CoprocessorFunction:
    """Create a coprocessor function that runs an ONNX model on each message, for use with
    :mod:`~.backline`.

    Each message the controller sends is the model's input tensor as raw bytes, and each reply is
    its output tensor. The tensors' sizes become the controller's message sizes, so the controller
    needs no ``in_bytes`` or ``out_bytes`` of its own. The model runs on a GPU when the installed
    onnxruntime has a GPU provider, and on the CPU otherwise.

    .. warning::

        :mod:`Backline <.backline>` is experimental and only usable through the Catalyst
        compiler.

    Args:
        model (str or os.PathLike): Path to the ``.onnx`` model, which must have one input and one
            output.
        provider (str): The onnxruntime execution provider: ``"auto"``, ``"cpu"``, ``"migraphx"``
            for AMD GPUs, or ``"cuda"`` or ``"tensorrt"`` for NVIDIA GPUs. A GPU provider needs an
            onnxruntime build that contains it. ``"auto"`` uses the first of ``"migraphx"`` and
            ``"cuda"`` that the installed onnxruntime has, or the CPU when it has neither. A GPU
            provider that is present but cannot be attached stops the coprocessor from starting,
            rather than falling back to the CPU.
        device (int): the GPU a GPU provider runs on, from ``0``
        threads (int): The number of onnxruntime intra-op threads, at least ``1``. One thread keeps
            onnxruntime's thread pool from competing with the transport's threads.

    Returns:
        CoprocessorFunction: The function, ready to pass as :attr:`~.Coprocessor.coprocessor_fn`.

    Raises:
        FileNotFoundError: If ``model`` does not exist.
        ValueError: If ``provider`` is unknown, ``device`` or ``threads`` is out of range, or the
            model does not have one input and one output of a supported tensor type.
        TypeError: If ``device`` or ``threads`` is not an int.
        ImportError: If no onnxruntime package is installed, or it has no shared library.

    .. seealso:: :class:`~.CoprocessorFunction`, :class:`~.Coprocessor`

    **Example**

    A model taking 120 bytes and returning 121 sets both message sizes:

    >>> fn = qp.backline.onnx_decoder("predecoder.onnx")  # doctest: +SKIP
    >>> coproc = qp.Coprocessor(hardware="gpu", coprocessor_fn=fn)  # doctest: +SKIP
    >>> dev = qp.Backline(  # doctest: +SKIP
    ...     controller=qp.Controller(), coprocessors=[coproc], transport="memcpy"
    ... )
    >>> dev.placement.in_bytes, dev.placement.out_bytes  # doctest: +SKIP
    (120, 121)

    .. details::
        :title: Usage Details

        **Message sizes.** Each payload is the input tensor as raw bytes in row-major order, and
        each reply is the output tensor. A dynamic dimension, such as a batch dimension, is taken
        as 1. A controller that sets :attr:`~.Controller.in_bytes` or
        :attr:`~.Controller.out_bytes` to a size the model does not match is rejected when the
        :class:`~pennylane.Backline` is built.

        **Choosing the device.** With ``provider="auto"``, the same program runs on an AMD GPU with
        the ``onnxruntime-migraphx`` package, on an NVIDIA GPU with ``onnxruntime-gpu``, and on the
        CPU with plain ``onnxruntime``. ``"auto"`` never picks ``"tensorrt"``, so pass it
        explicitly to use it. The provider in use is printed when the coprocessor starts (or when
        a provider fails to attach). ``"cpu"`` and ``"migraphx"`` have been tested to work, whereas
        ``"cuda"`` and ``"tensorrt"`` are experimental and currently untested.

        **In-process only.** The model and onnxruntime paths are resolved on the compiling
        machine, so the coprocessor runs in the compiling process: it takes neither
        :attr:`~.Node.executor_options` nor :attr:`~.Node.executor`, and so cannot set
        :attr:`~.Node.remote`. Catalyst rejects a coprocessor running this function that has
        either.
    """
    model = Path(model).resolve()
    if not model.is_file():
        raise FileNotFoundError(f"onnx_decoder: no model at {model}")
    if provider not in _ONNX_PROVIDERS:
        raise ValueError(
            f"onnx_decoder: provider must be one of {list(_ONNX_PROVIDERS)}, got {provider!r}"
        )
    _check_count("device", device, 0)
    _check_count("threads", threads, 1)
    entries = [
        f"model={model}",
        f"ort_lib={_onnxruntime_library()}",
        f"provider={provider}",
        f"device={device}",
        f"threads={threads}",
    ]
    if any(";" in entry for entry in entries):
        raise ValueError("onnx_decoder: paths must not contain ';', which separates config entries")
    return CoprocessorFunction(
        name=_ONNX_FUNCTION,
        config=";".join(entries),
        per_message=True,
        message_bytes=_onnx_message_bytes(model),
    )
