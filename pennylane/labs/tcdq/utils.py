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
"""Helpers for building TCDQ gate dictionaries, gate arrays and observable lists.

These functions generate the ``gates`` and ``observables`` inputs expected by
:class:`~pennylane.labs.tcdq.CircuitConfig` and :class:`~pennylane.labs.tcdq.QuditCircuitConfig`.
"""

from collections.abc import Sequence
from functools import partial
from itertools import combinations

import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np

from .qudit_expval_functions import QuditGateArrays


def create_local_gates(n_qubits: int, max_weight: int = 2) -> dict[int, list[list[int]]]:
    """Generate a gate dictionary containing all qubit subsets up to a given weight.

    Creates one gate (and one trainable parameter) for every combination of
    qubits whose size is between 1 and ``max_weight``. For example, with
    ``max_weight=2`` on 3 qubits, this produces single-qubit Z gates
    ``[0], [1], [2]`` and two-qubit ZZ gates ``[0,1], [0,2], [1,2]``.

    Args:
        n_qubits (int): Total number of qubits.
        max_weight (int): Maximum number of qubits per gate (Pauli weight).
            Defaults to ``2``.

    Returns:
        dict[int, list[list[int]]]: A gate dictionary suitable for
        :class:`~pennylane.labs.tcdq.CircuitConfig`. Keys are parameter
        indices (starting from 0) and values are single-element lists
        containing the list of qubit indices for that gate.

    **Example**

    >>> from pennylane.labs.tcdq import create_local_gates
    >>> gates = create_local_gates(n_qubits=3, max_weight=2)
    >>> gates
    {0: [[0]], 1: [[1]], 2: [[2]], 3: [[0, 1]], 4: [[0, 2]], 5: [[1, 2]]}
    """
    gates = {}
    param_idx = 0
    for weight in range(1, max_weight + 1):
        for gate in combinations(range(n_qubits), weight):
            gates[param_idx] = [list(gate)]
            param_idx += 1
    return gates


def create_lattice_gates(
    rows: int, cols: int, distance: int = 1, max_weight: int = 2, periodic: bool = False
) -> dict[int, list[list[int]]]:
    """Generate gates based on spatial locality on a 2-D qubit lattice.

    Qubits are arranged on a ``rows × cols`` grid and numbered in row-major
    order (qubit index = row * cols + col). For each qubit, gates are created
    involving that qubit and up to ``max_weight - 1`` neighbours within graph
    distance ``distance``.

    Args:
        rows (int): Number of rows in the lattice.
        cols (int): Number of columns in the lattice.
        distance (int): Maximum graph distance (Manhattan steps on the grid)
            for two qubits to be considered neighbours. Defaults to ``1``
            (nearest neighbours only).
        max_weight (int): Maximum number of qubits per gate. Defaults to ``2``.
        periodic (bool): If ``True``, use periodic (toroidal) boundary
            conditions. Defaults to ``False``.

    Returns:
        dict[int, list[list[int]]]: A gate dictionary suitable for
        :class:`~pennylane.labs.tcdq.CircuitConfig`.

    **Example**

    >>> from pennylane.labs.tcdq import create_lattice_gates
    >>> gates = create_lattice_gates(rows=2, cols=2, distance=1, max_weight=2)
    >>> len(gates)  # 4 single-qubit + 4 nearest-neighbour pairs
    8
    """
    G = nx.grid_2d_graph(rows, cols, periodic=periodic)
    mapping = {(i, j): i * cols + j for i in range(rows) for j in range(cols)}
    G = nx.relabel_nodes(G, mapping)

    gates_list = []

    for source in list(G.nodes):
        lengths = nx.single_source_shortest_path_length(G, source, cutoff=distance)
        neighbors = [n for n in lengths if n != source]

        for weight in range(max_weight):
            for combo in combinations(neighbors, weight):
                new_gate = sorted(list(combo) + [source])
                if new_gate not in gates_list:
                    gates_list.append(new_gate)

    gates_dict = {}
    for i, g in enumerate(gates_list):
        gates_dict[i] = [g]

    return gates_dict


def create_random_gates(
    n_qubits: int, n_gates: int, min_weight: int = 1, max_weight: int = 2, seed: int = None
) -> dict[int, list[list[int]]]:
    """Generate a gate dictionary with randomly chosen qubit subsets.

    Each gate acts on a uniformly random subset of qubits whose size is
    drawn uniformly from ``[min_weight, max_weight]``.

    Args:
        n_qubits (int): Total number of qubits.
        n_gates (int): Number of gates (and trainable parameters) to generate.
        min_weight (int): Minimum number of qubits per gate. Defaults to ``1``.
        max_weight (int): Maximum number of qubits per gate. Defaults to ``2``.
        seed (int): Random seed for reproducibility. Defaults to ``None``.

    Returns:
        dict[int, list[list[int]]]: A gate dictionary suitable for
        :class:`~pennylane.labs.tcdq.CircuitConfig`.

    **Example**

    >>> from pennylane.labs.tcdq import create_random_gates
    >>> gates = create_random_gates(n_qubits=6, n_gates=10, max_weight=3, seed=0)
    >>> len(gates)
    10
    """
    rng = np.random.default_rng(seed)
    gates_dict = {}

    for i in range(n_gates):
        weight = rng.integers(min_weight, max_weight + 1)
        gate = sorted(list(rng.choice(range(n_qubits), size=weight, replace=False)))
        gates_dict[i] = [[int(q) for q in gate]]

    return gates_dict


def generate_pauli_observables(
    n_qubits: int, orders: list[int] = (1,), bases: list[str] = ("Z",)
) -> list[list[int]]:
    """Generate Pauli observables as integer-encoded arrays.

    Produces all tensor-product observables of the requested Pauli bases at
    the given interaction orders. Each observable is a list of length
    ``n_qubits`` with integer entries: I=0, X=1, Y=2, Z=3.

    For example, with ``orders=[2]`` and ``bases=["Z"]`` on 3 qubits, this
    generates all two-body ZZ operators: ``[3,3,0]``, ``[3,0,3]``,
    ``[0,3,3]``.

    Args:
        n_qubits (int): Number of qubits.
        orders (list[int]): List of interaction orders (number of non-identity
            sites). For instance, ``[1, 2]`` generates all single-qubit and
            two-qubit observables.
        bases (list[str]): Pauli bases to use. Any subset of
            ``["X", "Y", "Z"]``.

    Returns:
        list[list[int]]: A list of observables, each of length ``n_qubits``,
        suitable for use as the ``observables`` argument of
        :class:`~pennylane.labs.tcdq.CircuitConfig`.

    **Example**

    >>> from pennylane.labs.tcdq import generate_pauli_observables
    >>> obs = generate_pauli_observables(n_qubits=3, orders=[1], bases=["Z"])
    >>> obs
    [[3, 0, 0], [0, 3, 0], [0, 0, 3]]
    >>> obs_2body = generate_pauli_observables(3, orders=[2], bases=["Z"])
    >>> obs_2body
    [[3, 3, 0], [3, 0, 3], [0, 3, 3]]
    """
    observables = []
    base_map = {"I": 0, "X": 1, "Y": 2, "Z": 3}

    for order in orders:
        if order > n_qubits:
            continue
        for base in bases:
            base_int = base_map[base.upper()]
            for positions in combinations(range(n_qubits), order):
                obs_row = [0] * n_qubits
                for pos in positions:
                    obs_row[pos] = base_int
                observables.append(obs_row)

    return observables


def create_random_qudit_gates(
    n_gates: int,
    dims: Sequence[int],
    min_weight: int = 1,
    max_weight: int = 2,
    seed: int = None,
) -> dict[int, list[dict[int, int]]]:
    r"""Generate a sparse qudit gate dictionary with randomly chosen supports and powers.

    This is the qudit analogue of :func:`~pennylane.labs.tcdq.create_random_gates`.
    Each gate acts on a uniformly random subset of qudits whose size (the gate
    *weight*) is drawn uniformly from ``[min_weight, max_weight]``. On every
    active qudit :math:`j` the power of :math:`Z` is drawn uniformly from
    :math:`\{1, \ldots, d_j - 1\}`, so each gate has exactly the drawn weight.

    Args:
        n_gates (int): Number of gates (and trainable parameters) to generate.
        dims (Sequence[int]): Local dimension :math:`d_j \geq 2` of each qudit. The
            number of qudits is ``len(dims)``.
        min_weight (int): Minimum number of active qudits per gate. Defaults to ``1``.
        max_weight (int): Maximum number of active qudits per gate. Defaults to ``2``.
        seed (int): Random seed for reproducibility. Defaults to ``None``.

    Returns:
        dict[int, list[dict[int, int]]]: A gate dictionary suitable for
        :class:`~pennylane.labs.tcdq.QuditCircuitConfig`. Parameter index ``i``
        maps to a single sparse gate ``{qudit_index: z_power}`` with sorted keys.

    Raises:
        ValueError: If ``n_gates`` is negative, if any dimension is smaller than 2, or if
            the weight bounds do not satisfy ``1 <= min_weight <= max_weight <= len(dims)``.

    **Example**

    >>> from pennylane.labs.tcdq import create_random_qudit_gates
    >>> gates = create_random_qudit_gates(n_gates=4, dims=[3, 3, 4, 2], max_weight=3, seed=0)
    >>> len(gates)
    4
    >>> all(1 <= len(gate) <= 3 for (gate,) in gates.values())
    True
    """
    dims = [int(d) for d in dims]
    n_qudits = len(dims)

    if n_gates < 0:
        raise ValueError(f"n_gates must be non-negative, got {n_gates}.")
    if any(d < 2 for d in dims):
        raise ValueError(f"All qudit dimensions must be at least 2, got dims={dims}.")
    if not 1 <= min_weight <= max_weight <= n_qudits:
        raise ValueError(
            "Weights must satisfy 1 <= min_weight <= max_weight <= len(dims); "
            f"got min_weight={min_weight}, max_weight={max_weight}, len(dims)={n_qudits}."
        )

    rng = np.random.default_rng(seed)
    gates_dict = {}

    for i in range(n_gates):
        weight = int(rng.integers(min_weight, max_weight + 1))
        support = sorted(int(q) for q in rng.choice(n_qudits, size=weight, replace=False))
        gates_dict[i] = [{q: int(rng.integers(1, dims[q])) for q in support}]

    return gates_dict


def _smallest_int_dtype(max_value: int):
    """Smallest signed integer dtype able to hold ``max_value``."""
    if max_value <= 127:
        return jnp.int8
    if max_value <= 32767:
        return jnp.int16
    return jnp.int32


@partial(jax.jit, static_argnums=(1, 2, 3))
def _random_weight_counts(key, n_gates: int, min_weight: int, max_weight: int) -> jnp.ndarray:
    """Histogram of ``n_gates`` weights drawn uniformly from ``[min_weight, max_weight]``."""
    weights = jax.random.randint(key, (n_gates,), min_weight, max_weight + 1, dtype=jnp.int32)
    return jnp.bincount(weights, length=max_weight + 1)


@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def _random_fixed_weight_gates(
    key, n_gates: int, dims: tuple[int, ...], weight: int, max_weight: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Draw ``n_gates`` gates of exactly ``weight`` active qudits in canonical padded form.

    Supports are uniform ``weight``-subsets of the qudits, obtained by sequential sampling
    without replacement: the ``j``-th draw is uniform over the ``n_qudits - j`` remaining
    qudits, realised by drawing an index into the remaining qudits and shifting it past the
    (sorted) qudits drawn before. Powers are uniform in ``{1, ..., d_q - 1}``.
    """
    n_qudits = len(dims)
    dims_j = jnp.asarray(dims, dtype=jnp.int32)
    key_support, key_power = jax.random.split(key)
    keys = jax.random.split(key_support, weight)

    chosen = [jax.random.randint(keys[0], (n_gates,), 0, n_qudits, dtype=jnp.int32)]
    for j in range(1, weight):
        r = jax.random.randint(keys[j], (n_gates,), 0, n_qudits - j, dtype=jnp.int32)
        prev = jnp.sort(jnp.stack(chosen, axis=1), axis=1)  # (n_gates, j)
        for t in range(j):
            r = r + (r >= prev[:, t])
        chosen.append(r)
    supports = jnp.sort(jnp.stack(chosen, axis=1), axis=1)  # (n_gates, weight), sorted rows
    powers = jax.random.randint(
        key_power, (n_gates, weight), 1, jnp.take(dims_j, supports), dtype=jnp.int32
    )
    if weight < max_weight:
        pad = jnp.zeros((n_gates, max_weight - weight), dtype=jnp.int32)
        supports = jnp.concatenate([supports, pad], axis=1)
        powers = jnp.concatenate([powers, pad], axis=1)
    return (
        supports.astype(_smallest_int_dtype(n_qudits - 1)),
        powers.astype(_smallest_int_dtype(max(dims) - 1)),
    )


@partial(jax.jit, static_argnums=(4, 5, 6, 7, 8), donate_argnums=(0, 1))
def _fill_fixed_weight_gates(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    sup_buf, pow_buf, key_weight, offset, n_weight: int, chunk_size: int, dims, weight, max_weight
):
    """Write ``n_weight`` random gates of one weight into rows ``offset:offset + n_weight``.

    The buffers are donated and updated in place chunk by chunk, so the peak memory is the
    output plus one chunk of temporaries (no concatenation copy). Chunk ``c`` is drawn with
    ``fold_in(key_weight, c)``.
    """
    idx = offset.dtype
    zero = jnp.zeros((), dtype=idx)
    n_full = n_weight // chunk_size

    def write(bufs, c, size):
        sup, pw = _random_fixed_weight_gates(
            jax.random.fold_in(key_weight, c), size, dims, weight, max_weight
        )
        start = (offset + jnp.asarray(c, dtype=idx) * chunk_size).astype(idx)
        return (
            jax.lax.dynamic_update_slice(bufs[0], sup, (start, zero)),
            jax.lax.dynamic_update_slice(bufs[1], pw, (start, zero)),
        )

    bufs = (sup_buf, pow_buf)
    if n_full:
        bufs = jax.lax.fori_loop(0, n_full, lambda c, b: write(b, c, chunk_size), bufs)
    if n_weight > n_full * chunk_size:
        bufs = write(bufs, n_full, n_weight - n_full * chunk_size)
    return bufs


def create_random_qudit_gates_jax(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    key,
    n_gates: int,
    dims: Sequence[int],
    min_weight: int = 1,
    max_weight: int = 2,
    chunk_size: int = 2**24,
) -> QuditGateArrays:
    r"""Generate random sparse qudit gates as arrays, using JAX only.

    This is the array-based counterpart of :func:`~pennylane.labs.tcdq.create_random_qudit_gates`
    and draws from the same distribution: each gate acts on a uniformly random subset of qudits
    whose size (the gate *weight*) is uniform on ``[min_weight, max_weight]``, and on every
    active qudit :math:`j` the power of :math:`Z` is uniform on :math:`\{1, \ldots, d_j - 1\}`.
    Gate ``g`` is controlled by parameter ``g``.

    Unlike the dictionary version, which costs several hundred bytes and tens of microseconds
    per gate in Python, the output is a :class:`~pennylane.labs.tcdq.QuditGateArrays` in
    canonical layout (rows grouped by weight, sorted active prefix, smallest sufficient integer
    dtypes) that :class:`~pennylane.labs.tcdq.QuditCircuitConfig` consumes without copying. A
    two-qudit gate on fewer than 32768 qudits of dimension below 128 costs 6 bytes, so
    :math:`10^9` gates occupy 6 GB (9 GB for three-qudit gates). The output buffers are
    allocated once and filled in place, ``chunk_size`` gates per step, so the peak memory is the
    output plus temporaries proportional to ``chunk_size`` (about 2.7 GB for three-qudit gates
    at the default ``2**24``; lower ``chunk_size`` to reduce it). Chunks are drawn with keys
    derived from ``key`` via :func:`jax.random.fold_in`, so the result is reproducible but
    depends on ``chunk_size`` (fix ``chunk_size`` to reproduce a circuit).

    Args:
        key (ArrayLike): JAX PRNG key.
        n_gates (int): Number of gates (and trainable parameters) to generate.
        dims (Sequence[int]): Local dimension :math:`d_j \geq 2` of each qudit. The
            number of qudits is ``len(dims)``.
        min_weight (int): Minimum number of active qudits per gate. Defaults to ``1``.
        max_weight (int): Maximum number of active qudits per gate. Defaults to ``2``.
        chunk_size (int): Number of gates generated per compiled call. Defaults to ``2**24``.

    Returns:
        QuditGateArrays: Gate arrays with ``supports`` and ``powers`` of shape
        ``(n_gates, max_weight)`` and ``param_indices=None``.

    Raises:
        ValueError: If ``n_gates`` is negative, if any dimension is smaller than 2, if the
            weight bounds do not satisfy ``1 <= min_weight <= max_weight <= len(dims)``, or if
            ``chunk_size`` is not positive.

    **Example**

    >>> import jax
    >>> from pennylane.labs.tcdq import create_random_qudit_gates_jax
    >>> gates = create_random_qudit_gates_jax(jax.random.PRNGKey(0), n_gates=5, dims=[3] * 4, max_weight=3)
    >>> gates.supports.shape, gates.powers.shape
    ((5, 3), (5, 3))
    >>> gates.param_indices is None
    True
    """
    dims = tuple(int(d) for d in dims)
    n_qudits = len(dims)

    if n_gates < 0:
        raise ValueError(f"n_gates must be non-negative, got {n_gates}.")
    if any(d < 2 for d in dims):
        raise ValueError(f"All qudit dimensions must be at least 2, got dims={list(dims)}.")
    if not 1 <= min_weight <= max_weight <= n_qudits:
        raise ValueError(
            "Weights must satisfy 1 <= min_weight <= max_weight <= len(dims); "
            f"got min_weight={min_weight}, max_weight={max_weight}, len(dims)={n_qudits}."
        )
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}.")

    support_dtype = _smallest_int_dtype(n_qudits - 1)
    power_dtype = _smallest_int_dtype(max(dims) - 1)
    if n_gates == 0:
        empty = jnp.zeros((0, max_weight), dtype=jnp.int32)
        return QuditGateArrays(empty.astype(support_dtype), empty.astype(power_dtype), None)

    key_weights, key_gates = jax.random.split(key)

    # How many gates of each weight (drawn chunk by chunk to bound the temporaries).
    counts = np.zeros(max_weight + 1, dtype=np.int64)
    for chunk, start in enumerate(range(0, n_gates, chunk_size)):
        size = min(chunk_size, n_gates - start)
        counts += np.asarray(
            _random_weight_counts(
                jax.random.fold_in(key_weights, chunk), size, min_weight, max_weight
            )
        )

    # Fill preallocated output buffers in place, one weight group after the other.
    idx_dtype = jnp.int64 if jax.config.jax_enable_x64 else jnp.int32
    supports = jnp.zeros((n_gates, max_weight), dtype=support_dtype)
    powers = jnp.zeros((n_gates, max_weight), dtype=power_dtype)
    offset = 0
    for weight in range(min_weight, max_weight + 1):
        n_weight = int(counts[weight])
        if n_weight:
            supports, powers = _fill_fixed_weight_gates(
                supports,
                powers,
                jax.random.fold_in(key_gates, weight),
                jnp.asarray(offset, dtype=idx_dtype),
                n_weight,
                min(chunk_size, n_weight),
                dims,
                weight,
                max_weight,
            )
        offset += n_weight
    return QuditGateArrays(supports, powers, None)
