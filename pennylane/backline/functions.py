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

import importlib.util
from dataclasses import dataclass
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
            ``;``. Defaults to ``""``. A function that exports ``<name>_init`` is handed these
            entries once, before its first message, and is then called with what that returns.
        per_message (bool): Whether the function is a host function called once per message,
            rather than a launcher that starts a persistent GPU kernel. Defaults to ``False``. It
            matters only on a GPU coprocessor, which then runs the function per message. On a CPU
            coprocessor every function is called per message.

    .. seealso:: :class:`~.Coprocessor`, :func:`~.css_bp_decoder`, :func:`~.triton_decoder`

    **Example**

    A coprocessor function is usually named rather than constructed --- passing a string to
    :class:`~.Coprocessor` resolves it:

    >>> coproc = qp.Coprocessor(coprocessor_fn="decoder")
    >>> coproc.coprocessor_fn
    CoprocessorFunction(name='decoder', lib_path=None)

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
    """The function's own ``key=value;...`` configuration, handed to its ``<name>_init``."""

    per_message: bool = False
    """Whether the function is called once per message, rather than launching a persistent GPU
    kernel."""

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


def _onnxruntime_library() -> str:
    """The onnxruntime shared library of the installed ``onnxruntime`` package."""
    spec = importlib.util.find_spec("onnxruntime")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError(
            "onnx_decoder needs an onnxruntime package, such as onnxruntime for the CPU or "
            "onnxruntime-migraphx for AMD GPUs"
        )
    capi = Path(list(spec.submodule_search_locations)[0]) / "capi"
    libraries = sorted(capi.glob("libonnxruntime.so*"))
    if not libraries:
        raise ImportError(f"no onnxruntime shared library found in {capi}")
    return str(libraries[0])


def onnx_decoder(model, *, provider: str = "auto", device: int = 0) -> CoprocessorFunction:
    """A coprocessor function that runs an ONNX model on each message, for use with
    :mod:`~.backline`.

    The model must have one input and one output. Each message's payload is the input tensor as raw
    bytes in row-major order, and the reply is the output tensor, likewise. A dynamic dimension of
    the input, such as a batch dimension, is taken as 1. So the controller's
    :attr:`~.Controller.in_bytes` must hold the input tensor and its
    :attr:`~.Controller.out_bytes` the output tensor.

    The function is one Catalyst ships, so nothing is compiled. It loads onnxruntime and the model
    when its coprocessor starts, so a model file is all a decoder needs.
    It is a per-message function, so a GPU coprocessor calls it once per message from the host,
    and the model's own GPU work runs from there.

    The device the model runs on is chosen by onnxruntime. With ``provider="auto"`` it is the first
    GPU provider the installed onnxruntime has, so one program runs on an AMD GPU with the
    ``onnxruntime-migraphx`` package, on an NVIDIA GPU with ``onnxruntime-gpu``, and on the CPU with
    plain ``onnxruntime``.

    .. warning::

        :mod:`Backline <.backline>` is experimental and only usable through the Catalyst
        compiler.

    Args:
        model (str | os.PathLike): Path to the ``.onnx`` model.

    Keyword Args:
        provider (str): The onnxruntime execution provider: ``"auto"`` (the default), ``"cpu"``,
            ``"migraphx"`` for AMD GPUs, ``"cuda"`` or ``"tensorrt"`` for NVIDIA GPUs, or
            ``"rocm"``. A GPU provider needs an onnxruntime build that contains it.
        device (int): The GPU a GPU provider runs on. Defaults to ``0``.

    Returns:
        CoprocessorFunction: The function, ready to pass as :attr:`~.Coprocessor.coprocessor_fn`.

    Raises:
        FileNotFoundError: If ``model`` does not exist.
        ValueError: If ``provider`` is unknown.
        ImportError: If no onnxruntime package is installed.

    .. seealso:: :class:`~.CoprocessorFunction`, :class:`~.Coprocessor`

    **Example**

    >>> fn = qp.backline.onnx_decoder("predecoder.onnx")  # doctest: +SKIP
    >>> coproc = qp.Coprocessor(name="gpu-coproc", hardware="gpu", coprocessor_fn=fn)  # doctest: +SKIP
    """
    model = Path(model).resolve()
    if not model.is_file():
        raise FileNotFoundError(f"onnx_decoder: no model at {model}")
    if provider not in _ONNX_PROVIDERS:
        raise ValueError(f"provider must be one of {list(_ONNX_PROVIDERS)}, got {provider!r}")
    entries = [
        f"model={model}",
        f"ort_lib={_onnxruntime_library()}",
        f"provider={provider}",
        f"device={device}",
    ]
    if any(";" in entry for entry in entries):
        raise ValueError("onnx_decoder: paths must not contain ';', which separates config entries")
    return CoprocessorFunction(name=_ONNX_FUNCTION, config=";".join(entries), per_message=True)
