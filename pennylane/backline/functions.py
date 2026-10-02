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

"""Coprocessor functions for backline placement."""

import importlib
import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path

from numpy.typing import ArrayLike


@dataclass(frozen=True)
class CoprocessorFunction:
    """A precompiled function run on a :class:`~.Coprocessor` to process messages received from the
    :class:`~.Controller`.

    This is a thin handle over a precompiled library symbol. It contains the information needed to
    locate and dispatch the function - its symbol name, and the library it lives in. The compiled
    artifact is produced separately (cross-compiled, or built on the same host, e.g., via Triton) and
    loaded by the runtime.

    .. warning::

        :mod:`Backline <.backline>` is experimental and only usable through the Catalyst
        compiler.

    Args:
        name (str): The name the function is known by, used to resolve the precompiled symbol.
        lib_path (str, None): Path to the shared library providing the symbol. Defaults to
            ``None``, in which case the runtime resolves :attr:`name` from the symbols already
            loaded on the host.
        config (str): The function's own configuration, as ``key=value`` entries separated by
            ``;``. Defaults to ``""``. A function whose library exports lifecycle hooks under
            ``<name>_info`` (for example ``catalyst_onnx_coprocessor_info``) receives these entries
            in its ``init`` hook once, before its first message, and is then called with the
            context that hook returns.
        per_message (bool): Whether the function is a host function called once per message,
            rather than a launcher that starts a persistent GPU kernel. Defaults to ``False``. It
            matters only on a GPU coprocessor, which then runs the function per message. On a CPU
            coprocessor every function is called per message.
        message_bytes (tuple[int, int], None): The ``(in_bytes, out_bytes)`` message sizes the
            function expects. Defaults to ``None``, declaring none.

    .. seealso:: :class:`~.Coprocessor`, :func:`~.css_bp_decoder`, :func:`~.triton_decoder`

    **Example**

    A coprocessor function is usually named rather than constructed --- passing a string to
    :class:`~.Coprocessor` resolves it:

    >>> coproc = qp.Coprocessor(coprocessor_fn="decoder")
    >>> coproc.coprocessor_fn
    CoprocessorFunction(name='decoder', lib_path=None, config='', per_message=False)

    Construct one directly to point at a symbol in a specific shared library. The path is what the
    coprocessor's node passes to the runtime as its backend library:

    >>> fn = qp.CoprocessorFunction(
    ...     name="decode_syndrome", lib_path="/opt/backline/libdecoder.so"
    ... )
    >>> fn.symbol_name
    'decode_syndrome'
    """

    name: str
    """The name the function is known by; used to resolve the precompiled symbol."""

    lib_path: str | None = None
    """Path to the shared library that provides the symbol. Defaults to ``None``, in which case the
    runtime resolves :attr:`name` from the symbols already loaded on the host."""

    config: str = ""
    """The function's own ``key=value;...`` configuration, handed to the ``init`` hook its library
    exports under ``<name>_info``."""

    per_message: bool = False
    """Whether the function is called once per message, rather than launching a persistent GPU
    kernel."""

    message_bytes: tuple[int, int] | None = field(default=None, repr=False)
    """The ``(in_bytes, out_bytes)`` message sizes the function expects, or ``None`` when it does
    not declare them. A :class:`~.Placement` takes the controller's unset sizes from it, and
    rejects a controller size that differs."""

    def __post_init__(self):
        if self.message_bytes is not None:
            object.__setattr__(self, "message_bytes", tuple(self.message_bytes))

    @property
    def symbol_name(self) -> str:
        """The symbol the runtime resolves and invokes for this function."""
        return self.name


def triton_decoder(
    decoder_fns: tuple[object, ...],
    **build_options,
) -> CoprocessorFunction:
    """Compile Triton quantum error correction decoder functions into a coprocessor function
    for use with :mod:`~.backline`.

    This function accepts a tuple of un-jitted Triton decoder functions, and compiles them into a
    shared library that can be used as a :class:`~.CoprocessorFunction`.

    .. warning::

        :mod:`Backline <.backline>` is experimental and only usable through the Catalyst
        compiler.

    Args:
        decoder_fns (tuple[object, ...]): Un-jitted Triton decoder functions. Each entry is
            jit compiled internally, and ``decoder_id`` selects the tuple index at runtime.

    Keyword Args:
        platform (str): Required Triton platform string of the form ``"backend:arch:warp_size"``.
            For example, ``"hip:gfx942:64"`` or ``"cuda:80:32"``.
        grid (tuple[int, int, int]): Triton kernel launch grid dimensions.
            Defaults to ``(1, 1, 1)``.
        num_warps (int): Triton kernel launch warp count. Defaults to ``1``.
        num_stages (int): Triton pipeline stage count. Defaults to ``1``.
        compiler (str): Optional compiler executable override. Defaults to ``""``.
        cflags (tuple[str, ...]): Extra compiler flags. Defaults to ``()``.

    Returns:
        CoprocessorFunction: The compiled decode function, ready to pass as
            :attr:`~.Coprocessor.coprocessor_fn`.

    Raises:
        ImportError: If Triton decoder support is unavailable.
        TypeError: If ``decoder_fns`` contains already jit compiled Triton functions.
        ValueError: If the decoder build options are invalid.

    .. seealso:: :class:`~.CoprocessorFunction`, :class:`~.Coprocessor`,
        :func:`~.css_bp_decoder`

    **Example**

    >>> import pennylane as qp
    >>> import triton.language as tl
    >>> def steane_lookup(syndrome):
    ...     return tl.where(syndrome != 0, 1 << (syndrome - 1), 0)
    >>> decoder = qp.backline.triton_decoder(  # doctest: +SKIP
    ...     (steane_lookup, steane_lookup),
    ...     platform="hip:gfx942:64",
    ... )
    """
    try:
        from pennylane.backline.decoders.triton.decoder_frontend import (  # pylint: disable=import-outside-toplevel
            _build_triton_decoder,
        )
    except ImportError as exc:
        raise ImportError("Triton decoders require installed `triton` Python package.") from exc

    so_path, symbol_name = _build_triton_decoder(decoder_fns, **build_options)  # pragma: no cover
    return CoprocessorFunction(name=symbol_name, lib_path=str(so_path))  # pragma: no cover


def css_bp_decoder(
    Hx: ArrayLike,
    Hz: ArrayLike,
    *,
    postprocess: str = "osd",
    num_iters: int = 10,
    prob: float = 0.1,
    **build_options,
) -> CoprocessorFunction:
    """Compile a CSS code's Tanner graph into a coprocessor decode function for use with
    :mod:`~.backline`.

    Accepts the X- and Z-type parity-check matrices of a CSS code and compiles a decoder down to a
    shared library that can be used as a :class:`~.CoprocessorFunction`.

    .. warning::

        :mod:`Backline <.backline>` is experimental and only usable through the Catalyst
        compiler.

    Args:
        Hx (ArrayLike): X parity-check matrix.
        Hz (ArrayLike): Z parity-check matrix.

    Keyword Args:
        postprocess (str): Postprocessing step applied after belief propagation. Use
            ``"hard"`` for hard-decision output or ``"osd"`` for ordered-statistics decoding.
        num_iters (int): Number of belief-propagation iterations.
        prob (float): Uniform prior error probability across qubits.
        platform (str): Required Triton platform string of the form ``"backend:arch:warp_size"``.
            For example, ``"hip:gfx942:64"`` or ``"cuda:80:32"``.
        grid (tuple[int, int, int]): Triton kernel launch grid dimensions.
            Defaults to ``(1, 1, 1)``.
        num_warps (int): Triton kernel launch warp count. Defaults to ``1``.
        num_stages (int): Triton pipeline stage count. Defaults to ``1``.
        compiler (str): Optional compiler executable override. Defaults to ``""``.
        cflags (tuple[str, ...]): Extra compiler flags. Defaults to ``()``.

    Returns:
        CoprocessorFunction: The compiled decode function, ready to pass as
            :attr:`~.Coprocessor.coprocessor_fn`.

    Raises:
        ImportError: If Triton decoder support is unavailable.
        ValueError: If the decoder options or parity-check matrices are invalid.

    .. seealso:: :class:`~.CoprocessorFunction`, :class:`~.Coprocessor`,
        :func:`~.triton_decoder`

    **Example**

    >>> import numpy as np
    >>> import pennylane as qp
    >>> Hz = Hx = np.array([
    ...     [1, 0, 1, 0, 1, 0, 1],
    ...     [0, 1, 1, 0, 0, 1, 1],
    ...     [0, 0, 0, 1, 1, 1, 1],
    ... ])
    >>> decoder = qp.backline.css_bp_decoder(  # doctest: +SKIP
    ...     Hx,
    ...     Hz,
    ...     postprocess="hard",
    ...     num_iters=5,
    ...     platform="hip:gfx942:64",
    ... )
    """
    try:
        from pennylane.backline.decoders.triton.decoder_frontend import (  # pylint: disable=import-outside-toplevel
            _build_css_bp_decoder,
        )
    except ImportError as exc:
        raise ImportError("Triton decoders require installed `triton` Python package.") from exc

    so_path, symbol_name = _build_css_bp_decoder(  # pragma: no cover
        Hx, Hz, postprocess=postprocess, num_iters=num_iters, prob=prob, **build_options
    )
    return CoprocessorFunction(name=symbol_name, lib_path=str(so_path))  # pragma: no cover


_ONNX_FUNCTION = "catalyst_onnx_coprocessor"
_ONNX_PROVIDERS = ("auto", "cpu", "migraphx", "cuda", "tensorrt", "rocm")

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
        model (str | os.PathLike): Path to the ``.onnx`` model, which must have one input and one
            output.

    Keyword Args:
        provider (str): The onnxruntime execution provider: ``"auto"``, ``"cpu"``, ``"migraphx"``
            for AMD GPUs, ``"cuda"`` or ``"tensorrt"`` for NVIDIA GPUs, or ``"rocm"``. ``"auto"``
            uses the first of ``"migraphx"``, ``"cuda"`` and ``"rocm"`` that the installed
            onnxruntime has, or the CPU when it has none. A GPU provider that is present but cannot
            be attached stops the coprocessor from starting, rather than falling back to the CPU.
            Defaults to ``"auto"``.
        device (int): The GPU a GPU provider runs on, from ``0``. Defaults to ``0``.
        threads (int): onnxruntime's intra-op threads, at least ``1``. Defaults to ``1``, which
            keeps onnxruntime's thread pool from competing with the transport's threads.

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
        explicitly to use it. The provider in use is printed when the coprocessor starts, which is
        also when a provider that cannot be attached fails. ``"cpu"`` and ``"migraphx"`` have been
        tested, and the NVIDIA providers and ``"rocm"`` have not.

        **In-process only.** The model and onnxruntime paths are resolved on the compiling
        machine, so the coprocessor must run in the same process, not on an executor.
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
