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
"""This module contains functions to perform VSCF calculation."""


import logging
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, lax

logger = logging.getLogger(__name__)
jax.config.update("jax_enable_x64", True)

# pylint: disable=too-many-arguments, too-many-positional-arguments


# ---------------------------------------------------------------------
# Data types used across the module
# ---------------------------------------------------------------------
class HostBatches(NamedTuple):
    """
    Batched data container: used to move compactly data from the generator to the user (CPU, numpy version)
    """

    term_map: np.ndarray  # Host-side: (L,) int32
    target_mode: np.ndarray  # Host-side: (L,) int32
    i: np.ndarray  # Host-side: (L,) int32
    j: np.ndarray  # Host-side: (L,) int32
    coeffs: np.ndarray  # Host-side: (L,) float64
    partners: np.ndarray  # Host-side: (L, P) int32 padded with -1
    valid: (
        np.ndarray
    )  # Host-side: (L,) bool  --> this is needed, because otherwise 1body terms would be treated equally to padded data (unvalid)


class DeviceBatches(NamedTuple):
    """
    Batched data container: used to move compactly data from the generator to the user (Device, Jax version)
    """

    # stacked device arrays: leading axis = n_batches, second = batch size (B)
    term_map: jnp.ndarray  # (n_batches, B) int32
    target_mode: jnp.ndarray  # (n_batches, B) int32
    i: jnp.ndarray  # (n_batches, B) int32
    j: jnp.ndarray  # (n_batches, B) int32
    coeffs: jnp.ndarray  # (n_batches, B) float64
    partners: jnp.ndarray  # (n_batches, B, P) int32
    valid: jnp.ndarray  # (n_batches, B) bool


def _integral_proc_worker(args):
    """
    Worker function to process a raw slice of the Hamiltonian.

    Identifies the active terms in the Hamiltonian, following the equations 20-22
    in `J. Chem. Theory Comput. 2010, 6, 235-248 <https://pubs.acs.org/doi/10.1021/ct9004454>`_.
    The equation suggests that if mode m is not contained in a Hamiltonian term, it evaluates to zero.

    Args:
        chunk_ham_vals (TensorLike[float]): a flattened view of the integrals terms to be processed by this worker.
        start_global_idx (int):             the index of the first element wrt the original flattened array.
        original_shape (tuple):             the shape of the original tensor.
        num_modes_in_term (int):            how many different modes coupled by the same hamiltonian term?
        modals_limits (TensorLike[int]):    max number of modals to be used foreach mode. This can be used to reduce the calculation space.
        cutoff (float):                     under this value, integrals terms are treated as null. Increasing this parameter greatly improves sparsity.
        max_partners (int):                 the max number of interations with different modes, in the whole Hamiltonian.

    Returns exploded arrays for this chunk or None.
    """
    (
        chunk_integral_vals,
        start_global_idx,
        original_shape,
        num_modes_in_term,
        modals_limits,
        cutoff,
        max_partners,
    ) = args

    # cutoff --> we aim to collect sparse, nonzero entries.
    local_nonzero = np.nonzero(np.abs(chunk_integral_vals) >= cutoff)[0]
    if len(local_nonzero) == 0:
        return None

    vals = chunk_integral_vals[local_nonzero]
    global_indices_flat = local_nonzero + start_global_idx
    subs = np.unravel_index(global_indices_flat, original_shape)
    indices = np.stack(subs, axis=1)

    # the original integral shape was something like (3,3,5,5,5,5) (2bodies interactions here), where the first indices are for the modes, the latters for the modals.
    mode_cols = indices[:, :num_modes_in_term]
    exc_cols = indices[:, num_modes_in_term:]

    # canonical filter for multi-body: we leverage the fact that integrals are symmetric. There is no need to read duplicated values.
    if num_modes_in_term > 1:
        is_canonical = np.all(mode_cols[:, :-1] > mode_cols[:, 1:], axis=1)
        if not np.all(is_canonical):
            indices = indices[is_canonical]
            vals = vals[is_canonical]
            mode_cols = mode_cols[is_canonical]
            exc_cols = exc_cols[is_canonical]

    if len(vals) == 0:
        return None

    # truncate to active space limits
    # (if the user defined less modals than the ones provided by the integrals).
    i_cols = exc_cols[:, :num_modes_in_term]
    j_cols = exc_cols[:, num_modes_in_term:]
    limits = modals_limits[mode_cols]
    valid_dims = (i_cols < limits) & (j_cols < limits)
    keep_mask = np.all(valid_dims, axis=1)

    if not np.all(keep_mask):
        mode_cols = mode_cols[keep_mask]
        i_cols = i_cols[keep_mask]
        j_cols = j_cols[keep_mask]
        vals = vals[keep_mask]

    num_valid = len(vals)
    if num_valid == 0:
        return None

    # explosion: repeat per the modes involved in this term
    exploded_coeffs = np.repeat(vals, num_modes_in_term)
    exploded_targets = mode_cols.flatten()
    exploded_i = i_cols.flatten()
    exploded_j = j_cols.flatten()

    # partner generation: fixed-size padded partners array
    # NOTE: a partner is a mode which is involved in the same integral term (for >=2bodies couplings)
    repeated_modes = np.repeat(mode_cols, num_modes_in_term, axis=0)
    mask_self = repeated_modes == exploded_targets[:, None]
    raw_partners = repeated_modes[~mask_self]  # flattened partners for all rows

    current_n_partners = num_modes_in_term - 1
    exploded_partners = np.full((len(exploded_targets), max_partners), -1, dtype=np.int32)
    if current_n_partners > 0:
        raw_partners = raw_partners.reshape(-1, current_n_partners)
        # copy into padded array
        exploded_partners[:, :current_n_partners] = raw_partners

    # NOTE: validity flags are added, because otherwise 1body terms would be treated equally to padded data (unvalid)
    exploded_validity = np.ones(len(exploded_targets), dtype=bool)

    return (
        num_valid,
        exploded_targets.astype(np.int32),
        exploded_i.astype(np.int32),
        exploded_j.astype(np.int32),
        exploded_coeffs.astype(np.float64),
        exploded_partners.astype(np.int32),
        exploded_validity.astype(bool),
    )


# ---------------------------------------------------------------------
# Parallelized generator that yields exploded numpy chunks
# ---------------------------------------------------------------------
def _generate_jacobi_batches(
    h_integrals, modals, cutoff, max_partners, n_workers=4, chunk_size=50_000
):
    """
    Parallelized memory-resilient generator. Yields tuples: (exploded_term_map, tgt, i, j, coeffs, partners, valid)
    where exploded arrays are numpy arrays ready to be buffered. They can be moved on the device memory on the fly.

    Args:
        h_integrals (list(TensorLike[float])):  list containing Hamiltonian integral matrices [1b_terms, 2b_terms, ...]
        modals (TensorLike[int]):               number of modals to be used foreach mode. This can be used to reduce the calculation space.
        cutoff (float):                         under this value, integral terms are treated as null. Increasing this parameter greatly improves sparsity.
        max_partners (int):                     the max number of interations with different modes, in the whole Hamiltonian.
        n_workers (int):                        how many workers (threads) to employ to subdivide the computation?
        chunk_size (int):                       how many hamiltonian terms to provide (each time) to each worker?

    """
    modals_arr = np.array(modals)
    tasks = []
    # build tasks (views)
    for ham_n in h_integrals:

        # infer number of modes per term assuming each term adds 3 dims per body
        # in fact, 1b integrals have shape (M, m, m), 2b integrals have shape (M, M, m, m, m, m) and so on (M=modes, m=modals)
        num_modes_in_term = len(ham_n.shape) // 3

        original_shape = np.shape(ham_n)
        flat_ham = np.reshape(ham_n, -1)
        total_elements = flat_ham.size
        for start_idx in range(0, total_elements, chunk_size):
            end_idx = min(start_idx + chunk_size, total_elements)
            chunk_vals = flat_ham[start_idx:end_idx]
            args = (
                chunk_vals,
                start_idx,
                original_shape,
                num_modes_in_term,
                modals_arr,
                cutoff,
                max_partners,
            )  # a task for a worker
            tasks.append(args)

    active_num = 0
    with ThreadPoolExecutor(max_workers=n_workers) as exe:
        for res in exe.map(_integral_proc_worker, tasks):  # here we create (and start the worker)
            if res is None:
                continue
            num_valid, tgt, i, j, c, p, v = res
            # compute exploded term ids
            n_modes_in_this_chunk = len(tgt) // num_valid
            new_ids = np.arange(active_num, active_num + num_valid, dtype=np.int32)
            exploded_tm = np.repeat(new_ids, n_modes_in_this_chunk)
            active_num += num_valid
            yield (exploded_tm, tgt, i, j, c, p, v)


# ---------------------------------------------------------------------
# Efficient growable NumPy buffer (preallocate & double growth)
# ---------------------------------------------------------------------
class GrowableBuffer:
    """
    Simple growable 1D NumPy buffer for appending without frequent reallocs.
    Appending many small NumPy arrays using Python lists + np.concatenate becomes O(N^2) and destroys throughput when you have millions/billions of entries.
    To improve performances for this operation, here we:
    - keep a large preallocated array
    - track how many elements are actually used
    - when full, it doubles its capacity
    - copying happens logarithmically, not every append
    - amortized O(1) append time
    """

    def __init__(self, dtype, initial_capacity=1 << 16):
        self.dtype = dtype
        self.capacity = max(1, initial_capacity)
        self.size = 0
        self.arr = np.empty(self.capacity, dtype=self.dtype)

    def append(self, data: np.ndarray):
        """
        Add new (possibly irregular) data to this container.
        This is equivalent to numpy concatenation, but made efficient.

        Args:
            data (np.ndarray): The data you aim to collect.
        """
        n = len(data)
        needed = self.size + n
        if needed > self.capacity:
            # grow by doubling until fit
            new_cap = self.capacity
            while new_cap < needed:
                new_cap *= 2
            new_arr = np.empty(new_cap, dtype=self.dtype)
            new_arr[: self.size] = self.arr[: self.size]
            self.arr = new_arr
            self.capacity = new_cap
        self.arr[self.size : self.size + n] = data
        self.size += n

    def take(self, n: int) -> np.ndarray:
        """
        Take first n elements, shift buffer left.
        """
        assert n <= self.size
        out = self.arr[:n].copy()
        # shift remainder left
        rem = self.size - n
        if rem > 0:
            self.arr[:rem] = self.arr[n : self.size]
        self.size = rem
        return out

    def pad_and_take_all(self, total_size: int, pad_value=0):
        """
        Return array of length total_size with padding (used for last partial batch).
        """
        assert self.size <= total_size
        out = np.full(total_size, pad_value, dtype=self.dtype)
        if self.size > 0:
            out[: self.size] = self.arr[: self.size]
        self.size = 0
        return out

    def __len__(self):
        return self.size


class Growable2DBuffer:
    """
    Growable 2D buffer where rows have fixed width (width).
    Same logic as before, but optimized for 2D (with one fixed dim).
    """

    def __init__(self, width: int, dtype, initial_rows=1 << 12):
        self.width = width
        self.dtype = dtype
        self.capacity_rows = max(1, initial_rows)
        self.rows = 0
        self.arr = np.empty((self.capacity_rows, width), dtype=self.dtype)

    def append_rows(self, rows: np.ndarray):
        """
        Add new rows to this container.
        This is equivalent to numpy concatenation, but made efficient.

        Args:
            rows (np.ndarray): the rows you want to collect.
        """
        n = rows.shape[0]
        needed = self.rows + n
        if needed > self.capacity_rows:
            new_cap = self.capacity_rows
            while new_cap < needed:
                new_cap *= 2
            new_arr = np.empty((new_cap, self.width), dtype=self.dtype)
            new_arr[: self.rows, :] = self.arr[: self.rows, :]
            self.arr = new_arr
            self.capacity_rows = new_cap
        self.arr[self.rows : self.rows + n, :] = rows
        self.rows += n

    def take_rows(self, n: int) -> np.ndarray:
        """
        Extract n rows from the buffer,
        """
        assert n <= self.rows
        out = self.arr[:n, :].copy()
        rem = self.rows - n
        if rem > 0:
            self.arr[:rem, :] = self.arr[n : self.rows, :]
        self.rows = rem
        return out

    def pad_and_take_all(self, total_rows: int, pad_value=-1):
        """
        To be invoked at the end, when you aim to collect the latest batch, even if it doesn't contain enough data for a batch.
        In this case, it will be padded with unvalid data. This is better for GPUs.
        """
        assert self.rows <= total_rows
        out = np.full((total_rows, self.width), pad_value, dtype=self.dtype)
        if self.rows > 0:
            out[: self.rows, :] = self.arr[: self.rows, :]
        self.rows = 0
        return out

    def __len__(self):
        return self.rows


# ---------------------------------------------------------------------
# High-performance BatchRegularizer that uses growable buffers
# ---------------------------------------------------------------------
class BatchRegularizer:
    """
    Efficient fixed-size batch normalizer that avoids per-element Python lists.
    Accepts raw generator that yields unregular (term_map, tgt, i, j, coeffs, partners, valid)
    and yields fixed-size NumPy VSCFData batches (way more confortable for GPUs.)

    It basically collects and aggregates elements until the target batch size is reached, so to release the same data but differently aggregated (more regular).
    The last batch will be padded, so to reach the target size.
    This allows GPUs to work on the same shapes, while still exploiting hamiltonian sparsity.
    It also allows JAX jitted functions to reuse the same compilation (since shapes are the same).

    """

    # pylint: disable=too-many-instance-attributes, too-few-public-methods

    def __init__(
        self, raw_generator: Iterator[tuple[np.ndarray, ...]], target_size: int, max_partners: int
    ):
        self.raw_gen = raw_generator
        self.target_size = int(target_size)
        self.max_partners = int(max_partners)

        # Preallocate growable buffers
        init_rows = max(1, self.target_size // 8)
        self.buf_term_map = GrowableBuffer(
            np.int32, initial_capacity=max(1 << 12, self.target_size)
        )
        self.buf_target_mode = GrowableBuffer(
            np.int32, initial_capacity=max(1 << 12, self.target_size)
        )
        self.buf_i = GrowableBuffer(np.int32, initial_capacity=max(1 << 12, self.target_size))
        self.buf_j = GrowableBuffer(np.int32, initial_capacity=max(1 << 12, self.target_size))
        self.buf_coeffs = GrowableBuffer(
            np.float64, initial_capacity=max(1 << 12, self.target_size)
        )
        self.buf_partners = Growable2DBuffer(
            width=self.max_partners, dtype=np.int32, initial_rows=init_rows
        )
        self.buf_valid = GrowableBuffer(np.bool_, initial_capacity=max(1 << 12, self.target_size))

    def __iter__(self):
        return self._generator()

    def _emit_full_batch(self):
        B = self.target_size
        tm = self.buf_term_map.take(B)
        tgt = self.buf_target_mode.take(B)
        ii = self.buf_i.take(B)
        jj = self.buf_j.take(B)
        cc = self.buf_coeffs.take(B)
        pp = self.buf_partners.take_rows(B)
        vv = self.buf_valid.take(B)

        return HostBatches(tm, tgt, ii, jj, cc, pp, vv)

    def _emit_padded_last_batch(self):
        B = self.target_size
        L = len(self.buf_term_map)
        if L == 0:
            return None
        tm = self.buf_term_map.pad_and_take_all(B, pad_value=0)
        tgt = self.buf_target_mode.pad_and_take_all(B, pad_value=0)
        ii = self.buf_i.pad_and_take_all(B, pad_value=0)
        jj = self.buf_j.pad_and_take_all(B, pad_value=0)
        cc = self.buf_coeffs.pad_and_take_all(B, pad_value=0.0)
        pp = self.buf_partners.pad_and_take_all(B, pad_value=-1)
        vv = self.buf_valid.pad_and_take_all(B, pad_value=False)
        return HostBatches(tm, tgt, ii, jj, cc, pp, vv)

    def _generator(self):
        for term_map, target_mode, i, j, coeffs, partners, valid in self.raw_gen:
            # Append arrays efficiently to growable buffers
            self.buf_term_map.append(np.asarray(term_map, dtype=np.int32))
            self.buf_target_mode.append(np.asarray(target_mode, dtype=np.int32))
            self.buf_i.append(np.asarray(i, dtype=np.int32))
            self.buf_j.append(np.asarray(j, dtype=np.int32))
            self.buf_coeffs.append(np.asarray(coeffs, dtype=np.float64))
            partners_arr = np.asarray(partners, dtype=np.int32)
            # partners_arr shape (L, P)
            self.buf_partners.append_rows(partners_arr)
            self.buf_valid.append(np.asarray(valid, dtype=bool))

            # Emit as many full batches as possible
            while len(self.buf_term_map) >= self.target_size:
                yield self._emit_full_batch()

        # final padded batch
        last = self._emit_padded_last_batch()
        if last is not None:
            yield last


# ---------------------------------------------------------------------
# Fused JAX kernel: takes stacked device batches and compiles/performs one big large GPU kernel
# ---------------------------------------------------------------------
@partial(
    jit, static_argnames=["modals_tuple", "nmodes", "max_modals", "P", "B", "n_batches", "damping"]
)
def scf_step_batched(
    h_mat,
    mode_rots,
    device_batches: DeviceBatches,
    modals_tuple,
    nmodes,
    max_modals,
    P,
    B,
    n_batches,
    damping=0.0,
):
    """
    Fused SCF step that operates on stacked batches.
    device_batches fields shapes:
        term_map:    (n_batches, B)
        target_mode: (n_batches, B)
        i, j:        (n_batches, B)
        coeffs:      (n_batches, B)
        partners:    (n_batches, B, P)
        valid:       (n_batches, B)
    The function flattens all batches and performs one large vectorized computation
    then uses lax.segment_sum once to build the Fock storage and one segment to update h_mat.
    """

    # flatten everything: shape -> (n_batches * B, ...)
    nb = n_batches
    total_entries = nb * B

    tm_flat = device_batches.term_map.reshape((total_entries,))  # int32
    tgt_flat = device_batches.target_mode.reshape((total_entries,))
    i_flat = device_batches.i.reshape((total_entries,))
    j_flat = device_batches.j.reshape((total_entries,))
    coeff_flat = device_batches.coeffs.reshape((total_entries,))
    partners_flat = device_batches.partners.reshape((total_entries, P))
    valid_flat = device_batches.valid.reshape((total_entries,))

    # we need fo perform a lookup into the hamiltonian matrix.
    # clamp partners for safe indexing
    lookup_idx = jnp.maximum(partners_flat, 0)  # (total_entries, P)

    # gather h values per partner: h_mat[tm_flat[:,None], lookup_idx] -> (total_entries, P)
    rows = tm_flat[:, None]
    raw_potentials = h_mat[rows, lookup_idx]  # (total_entries, P)

    # mask padded partners -> use 1.0 for padded (partners==-1)
    is_real_partner = partners_flat != -1
    factors = jnp.where(is_real_partner, raw_potentials, 1.0)  # Hartree product factors!

    # compute tensorially the mean field effect for each mode
    mean_fields = jnp.prod(factors, axis=1)  # (total_entries,)

    # compute contribution (zero for padded rows)
    vals = coeff_flat * mean_fields * valid_flat

    # compute flat indices into fock storage
    flat_indices = (
        (tgt_flat * max_modals * max_modals) + (i_flat * max_modals) + j_flat
    )  # (total_entries,)

    total_segments = nmodes * max_modals * max_modals

    # single large reduction to build fock_flat
    # NOTE: we use segment sum because it is very efficient. However, its behaviour is:
    # for i: out[segment_ids[i]] += data[i]
    # Here, we use it to generate contributions per exploded integral
    fock_flat = jax.ops.segment_sum(vals, flat_indices, total_segments)  # (total_segments,)
    all_focks = fock_flat.reshape((nmodes, max_modals, max_modals))

    # rotate and diagonalize (same as before)
    u_t = jnp.transpose(mode_rots, (0, 2, 1))
    all_focks_rot = jnp.matmul(u_t, jnp.matmul(all_focks, mode_rots))

    # NOTE: here we introduce a penalty, since we pad each mode to max_modals, even tho actual modes have (or can have) smaller dimension.
    # if for e.g. mode 0 has dim=4, and max_modals=7, indices 4–6 are non-physical.
    # but Fock is computed as a full (max_modals x max_modals) matrix.
    # we must ensure the SCF does NOT rotate physical modal space into padded junk. We do so by introducing a penalty mechanism,
    # which adds a huge number (1e9) to the padded diagonal entries
    # --> eigenvalue solver makes the padded states extremely costly
    # --> they never mix with physical modal subspace
    # --> rotations stay inside physical dimensions
    huge_number = 1e9
    modals_tensor = jnp.array(modals_tuple)
    idx_range = jnp.arange(max_modals)
    padding_mask = idx_range[None, :] >= modals_tensor[:, None]
    penalty_diag = jax.vmap(jnp.diag)(padding_mask.astype(float) * huge_number)
    all_focks_rot = all_focks_rot + penalty_diag

    eigvals, all_eigvecs = jax.scipy.linalg.eigh(all_focks_rot)
    new_rots = jnp.matmul(mode_rots, all_eigvecs)

    # -----------------------------------------------------------------
    # Update h_mat in a vectorized fashion:
    # new_h_val for each flattened entry = gs[tgt_flat, i_flat] * gs[tgt_flat, j_flat]
    # Then scatter into h_mat rows using segment_sum over linear index (term * nmodes + target)
    # -----------------------------------------------------------------
    gs_vectors = new_rots[:, :, 0]  # (nmodes, max_modals)
    # Gather in vectorized form
    gs_i = gs_vectors[tgt_flat, i_flat]
    gs_j = gs_vectors[tgt_flat, j_flat]
    new_h_vals_flat = jnp.where(valid_flat, gs_i * gs_j, 0.0)

    linear_h_idx = tm_flat * nmodes + tgt_flat
    total_h_slots = h_mat.shape[0] * nmodes
    # accumulate (sum) into flat h vector and reshape
    h_flat_updates = jax.ops.segment_sum(new_h_vals_flat, linear_h_idx, total_h_slots)
    h_mat_new = h_flat_updates.reshape(h_mat.shape)

    # damping, if it can help to converge
    h_mat_next = (1.0 - damping) * h_mat_new + damping * h_mat

    total_energy = jnp.sum(eigvals[:, 0])
    return h_mat_next, new_rots, total_energy


def vscf(
    h_integrals,
    modals,
    cutoff,
    tol=1e-8,
    max_iters=10000,
    batch_size=50000,
    n_workers=5,
    chunk_size=None,
    damping=0.0,
):
    r"""
    Perform a high-performance Vibrational Self-Consistent Field (VSCF) calculation using
    fixed-size batch processing, sparse Hamiltonian pruning, and a fused JAX/XLA SCF kernel.

    This routine evaluates the VSCF effective Hamiltonian and obtains the optimal
    one-mode modal functions through a fully self-consistent procedure.  The method is
    implemented here with modern sparse-tensor techniques and accelerators.

    The implementation uses:

    * sparse filtering of anharmonic PES tensor elements via a ``cutoff``
    * canonical ordering of ``n``-body Hamiltonian terms
    * explosion of each multi-mode term into ``n`` single-target contributions
    * fixed-size sub-batches for JAX's static-shape requirements
    * one **fused** XLA GPU kernel performing:
      - evaluation of all mean-field potentials
      - construction of the full Fock operator via ``segment_sum``
      - modal rotations and diagonalization
      - update of the effective Hamiltonian
    * a stabilized SCF loop (optionally with ``damping``)

    This allows the VSCF procedure to scale efficiently to very sparse
    high-dimensional PES representations for polyatomic vibrational problems.

    Args:
        h_integrals (list[TensorLike[float]]):
            List of Hamiltonian integral tensors encoding 1-mode, 2-mode, … ``n``-mode coupled
            anharmonic vibrational terms.
            Each tensor must follow the Christiansen factorized integral format:

            * 1-body terms: ``(n, m, m)``
            * 2-body terms: ``(n, n, m, m, m, m)``
            * 3-body terms: ``(n, n, n, m, m, m, m, m, m)``

            where ``n`` is the number of vibrational modes and ``m`` the modal dimension.

        modals (TensorLike[int]):
            List specifying the number of modals retained for each vibrational mode.
            This allows truncating the basis for expensive high-dimensional problems.
            The returned modal rotation matrices obey these mode-dependent dimensions.

        cutoff (float):
            Threshold below which Hamiltonian elements are discarded.
            Increasing ``cutoff`` dramatically increases sparsity and reduces runtime.

        tol (float):
            Convergence tolerance for the VSCF energy.
            The SCF loop stops when ``|E_i - E_(i-1)| < tol``.

        max_iters (int):
            Maximum number of SCF iterations.

        batch_size (int):
            Number ``B`` of exploded PES terms to pack into each GPU batch.
            Larger batches improve GPU throughput but increase memory usage.

        n_workers (int):
            Number of CPU threads used to process and explode Hamiltonian slices in parallel.

        chunk_size (int):
            CPU chunk size for scanning integral tensors.
            ``None`` defaults to a heuristic based on ``2.0 x batch_size``.

        damping (float):
            Linear damping factor for SCF mixing.
            Increase if convergence is slow or oscillatory.
            ``damping=0.0`` corresponds to pure Jacobi iteration.

    Returns:
        TensorLike[float]:
            Array of shape ``(n_modes, max(modals), max(modals))`` containing the final
            modal rotation matrices for each vibrational mode.

    **Example**

    >>> # simple 1-mode quartic potential
    >>> h1 = np.zeros((1, 4, 4))
    >>> h1[0] = np.diag([0.1, 0.2, 0.5, 0.9])
    >>> rots = vscf(h_integrals=[h1], modals=[4], cutoff=1e-12)
    >>> rots[0]
    Array([[ 1.,  0.,  0.,  0.],
           [ 0.,  1.,  0.,  0.],
           [ 0.,  0.,  1.,  0.],
           [ 0.,  0.,  0.,  1.]])

    .. details::
        :title: Usage Details

        **Tensor Dimensions**

        The function accepts vibrational Hamiltonian integrals in Christiansen format:

        - 1-mode terms:
          ``(n, m, m)``
        - 2-mode terms:
          ``(n, n, m, m, m, m)``
        - 3-mode terms:
          ``(n, n, n, m, m, m, m, m, m)``

        where
        ``n`` = number of vibrational modes,
        ``m`` = number of harmonic-oscillator modals.

        **Sparse Filtering**

        Entries with ``|h| < cutoff`` are omitted before SCF processing.
        This can eliminate **99%+** of anharmonic PES tensor elements in realistic molecules.

        **PES Explosion into Single-Target Terms**

        Each ``n``-body term is expanded into ``n`` single-target VSCF contributions:

        .. math::
            V(q_1, \dots, q_n)
            \;\rightarrow\;
            \sum_{k=1}^{n} V^{(k)}(q_k)

        This yields a list of exploded rows, each containing:

        * the target mode index
        * the excitation indices ``(i, j)``
        * the coefficient
        * the list of partner modes

        **Batch Regularization**

        Exploded terms are collected CPU-side into fixed-size blocks of length ``B``:

        * irregular explosion sizes are handled by growable buffers
        * final batches are padded to length ``B``

        This ensures **static tensor shapes**, which is mandatory for efficient JAX compilation.

        **Fused SCF Kernel**

        A single jitted JAX/XLA kernel performs:

        1. gather of required PES elements
        2. product of partner potentials
        3. construction of the full Fock operator (via ``segment_sum``)
        4. modal rotations
        5. diagonalization for each mode
        6. update of the effective Hamiltonian

        All operations for all batches are fused into a **single GPU kernel**,
        maximizing throughput and minimizing memory traffic.

        **SCF Convergence**

        The loop:

        .. math::
            E^{(k+1)}, \; \phi^{(k+1)}
            = \text{VSCF-Step}\bigl(E^{(k)}, \phi^{(k)}\bigr)

        is repeated until:

        .. math::
            | E^{(k+1)} - E^{(k)} | < \text{tol}

        Optional damping may be applied:

        .. math::
            H_{\text{eff}}^{(k+1)}
            \leftarrow
            (1 - \lambda) H_{\text{new}}
            + \lambda H^{(k)}

    """

    # pylint: disable=too-many-statements

    t0 = time.time()
    nmodes = len(modals)

    # figure out max bodies involved in the integrals
    max_bodies_involved = 0
    for h in h_integrals:
        # infer number of modes per term assuming each term adds 3 dims per body
        # in fact, 1b integrals have shape (M, m, m), 2b integrals have shape (M, M, m, m, m, m) and so on (M=modes, m=modals)
        num_modes_in_term = len(h.shape) // 3
        max_bodies_involved = max(max_bodies_involved, num_modes_in_term)
    max_partners = max(1, max_bodies_involved - 1)
    P = max_partners
    B = int(batch_size)

    if chunk_size is None:
        chunk_size = max(1000, int(B * 2.0))  # heuristic

    # prepare raw generator
    raw_gen = _generate_jacobi_batches(
        h_integrals, modals, cutoff, max_partners, n_workers=n_workers, chunk_size=chunk_size
    )

    # pack into fixed-size batches using preallocated growable buffers and regularization of the flow
    packer = BatchRegularizer(raw_gen, target_size=B, max_partners=max_partners)
    packed_batches: list[HostBatches] = []
    active_num_final = 0
    for pb in packer:
        packed_batches.append(pb)
        active_num_final += B  # each packed batch has exactly B entries (last one padded)

    n_batches = len(packed_batches)
    if n_batches == 0:
        # nothing to do
        max_modals = max(modals)
        mode_rots = jnp.eye(max_modals)[None, :, :].repeat(nmodes, axis=0)
        return 0.0, mode_rots

    # move stacked batches to device
    term_map_stack = jnp.stack(
        [jnp.array(pb.term_map, dtype=jnp.int32) for pb in packed_batches], axis=0
    )  # (n_batches, B)
    target_stack = jnp.stack(
        [jnp.array(pb.target_mode, dtype=jnp.int32) for pb in packed_batches], axis=0
    )
    i_stack = jnp.stack([jnp.array(pb.i, dtype=jnp.int32) for pb in packed_batches], axis=0)
    j_stack = jnp.stack([jnp.array(pb.j, dtype=jnp.int32) for pb in packed_batches], axis=0)
    coeff_stack = jnp.stack(
        [jnp.array(pb.coeffs, dtype=jnp.float64) for pb in packed_batches], axis=0
    )
    partners_stack = jnp.stack(
        [jnp.array(pb.partners, dtype=jnp.int32) for pb in packed_batches], axis=0
    )  # (n_batches, B, P)
    valid_stack = jnp.stack([jnp.array(pb.valid, dtype=bool) for pb in packed_batches], axis=0)

    device_batches = DeviceBatches(
        term_map_stack, target_stack, i_stack, j_stack, coeff_stack, partners_stack, valid_stack
    )

    # initialize h_mat and rotations
    max_modals = max(modals)
    mode_rots_np = np.zeros((nmodes, max_modals, max_modals), dtype=np.float64)
    for idx, dim in enumerate(modals):
        mode_rots_np[idx, :dim, :dim] = np.eye(dim)
        if dim < max_modals:
            mode_rots_np[idx, dim:, dim:] = np.eye(max_modals - dim)
    mode_rots = jnp.array(mode_rots_np)

    # h_mat rows = number of unique term IDs discovered in generator ~ active_num_final/B *? but we tracked active_num_final = n_batches*B
    # max_term_id estimation: we can scan packed_batches to find max term id used
    max_term = 0
    for pb in packed_batches:
        if pb.term_map.size > 0:
            max_term = max(max_term, int(pb.term_map.max()))

    # if the maximum term ID (integer) that appears in term_map is K, then valid row indices are: 0, 1, ..., K (= K + 1)
    active_terms = max_term + 1
    h_mat = jnp.zeros((active_terms, nmodes), dtype=jnp.float64)

    logger.debug(
        "Hamiltonian has %i active terms, packed in %i batches of size %i (consider explosion due to n-body integrals)",
        active_terms,
        n_batches,
        B,
    )

    # JAX jitted step function
    step_fn = partial(
        scf_step_batched,
        device_batches=device_batches,
        modals_tuple=tuple(modals),
        nmodes=nmodes,
        max_modals=max_modals,
        P=P,
        B=B,
        n_batches=n_batches,
        damping=damping,
    )

    # SCF helpers: condition and body of the cycle
    def cond_fn(state):
        _, _, e_cur, e_prev, idx = state
        return (jnp.abs(e_cur - e_prev) > tol) & (idx < max_iters)

    def body_fn(state):
        h_old, rots_old, e_old, _, it_idx = state
        h_new, rots_new, e_new = step_fn(h_old, rots_old)
        return (h_new, rots_new, e_new, e_old, it_idx + 1)

    # initial run to populate starting vars
    init_h, init_rots, init_e = step_fn(h_mat, mode_rots)
    loop_state = (init_h, init_rots, init_e, jnp.inf, 0)
    t1 = time.time()

    # run SCF loop using lax.while_loop (jit compiled)
    _, final_rots, final_e, _, n_iters = lax.while_loop(cond_fn, body_fn, loop_state)
    t2 = time.time()

    logger.debug(
        "Converged in %i iterations to energy %f:.8f\
                 \nPreparation time: %f:.3 s; SCF time: %f:.3f s --> total: %f:.3f s",
        int(n_iters),
        float(final_e),
        t1 - t0,
        t2 - t1,
        t2 - t0,
    )
    # hard copy, so that arrays get back to be mutable (in jax they are immutable)
    final_rots = np.array(final_rots)
    return final_rots


def _rotate_one_body(h1, nmodes, mode_rots, modals):
    r"""Rotates one body integrals.

    Args:
        h1 (TensorLike[float]): one-body integrals
        nmodes (int): number of vibrational modes
        mode_rots (list[TensorLike[float]]): list of rotation matrices for all vibrational modes
        modals (list[int]): list containing the maximum number of modals to consider for each mode

    Returns:
        TensorLike[float]: rotated one-body integrals

    """
    imax = np.max(modals)
    h1_rot = np.zeros((nmodes, imax, imax))
    for m in range(nmodes):
        h1_rot[m, : modals[m], : modals[m]] = np.einsum(
            "ij,ia,jb->ab", h1[m, :, :], mode_rots[m][:, : modals[m]], mode_rots[m][:, : modals[m]]
        )

    return h1_rot


def _rotate_two_body(h2, nmodes, mode_rots, modals):
    r"""Rotates two body integrals.

    Args:
        h2 (TensorLike[float]): two-body integrals
        nmodes (int): number of vibrational modes
        mode_rots (list[TensorLike[float]]): list of rotation matrices for all vibrational modes
        modals (list[int]): list containing the maximum number of modals to consider for each mode

    Returns:
        TensorLike[float]: rotated two-body integrals

    """
    imax = np.max(modals)

    U_mats = [mode_rots[m] for m in range(nmodes)]
    h2_rot = np.zeros((nmodes, nmodes, imax, imax, imax, imax))
    for m1 in range(nmodes):
        for m2 in range(nmodes):
            h2_rot[m1, m2, : modals[m1], : modals[m2], : modals[m1], : modals[m2]] = np.einsum(
                "ijkl,ia,jb,kc,ld->abcd",
                h2[m1, m2, :, :, :, :],
                U_mats[m1][:, : modals[m1]],
                U_mats[m2][:, : modals[m2]],
                U_mats[m1][:, : modals[m1]],
                U_mats[m2][:, : modals[m2]],
            )

    return h2_rot


def _rotate_three_body(h3, nmodes, mode_rots, modals):
    r"""Rotates three body integrals.

    Args:
        h3 (TensorLike[float]): three-body integrals
        nmodes (int): number of vibrational modes
        mode_rots (list[TensorLike[float]]): list of rotation matrices for all vibrational modes
        modals (list[int]): list containing the maximum number of modals to consider for each mode

    Returns:
        TensorLike[float]: rotated three-body integrals

    """
    imax = np.max(modals)

    h3_rot = np.zeros((nmodes, nmodes, nmodes, imax, imax, imax, imax, imax, imax))
    for m1 in range(nmodes):
        for m2 in range(nmodes):
            for m3 in range(nmodes):
                h3_rot[
                    m1,
                    m2,
                    m3,
                    : modals[m1],
                    : modals[m2],
                    : modals[m3],
                    : modals[m1],
                    : modals[m2],
                    : modals[m3],
                ] = np.einsum(
                    "ijklmn,ia,jb,kc,ld,me,nf->abcdef",
                    h3[m1, m2, m3, :, :, :, :, :, :],
                    mode_rots[m1][:, : modals[m1]],
                    mode_rots[m2][:, : modals[m2]],
                    mode_rots[m3][:, : modals[m3]],
                    mode_rots[m1][:, : modals[m1]],
                    mode_rots[m2][:, : modals[m2]],
                    mode_rots[m3][:, : modals[m3]],
                )

    return h3_rot


def _rotate_dipole(d_integrals, mode_rots, modals):
    r"""Generates VSCF rotated dipole.

    Args:
        d_integrals (list[TensorLike[float]]): list of n-mode expansion of dipole integrals
        mode_rots (list[TensorLike[float]]): list of rotation matrices for all vibrational modes
        modals (list[int]): list containing the maximum number of modals to consider for each mode

    Returns:
        tuple(TensorLike[float]): a tuple of rotated dipole integrals

    """
    n = len(d_integrals)

    nmodes = np.shape(d_integrals[0])[0]
    imax = np.max(modals)
    d1_rot = np.zeros((3, nmodes, imax, imax))

    for alpha in range(3):
        d1_rot[alpha, ::] = _rotate_one_body(
            d_integrals[0][alpha, ::], nmodes, mode_rots, modals=modals
        )
    dip_data = [d1_rot]

    if n > 1:
        d2_rot = np.zeros((3, nmodes, nmodes, imax, imax, imax, imax))
        for alpha in range(3):
            d2_rot[alpha, ::] = _rotate_two_body(
                d_integrals[1][alpha, ::], nmodes, mode_rots, modals=modals
            )
        dip_data = [d1_rot, d2_rot]

    if n > 2:
        d3_rot = np.zeros((3, nmodes, nmodes, nmodes, imax, imax, imax, imax, imax, imax))
        for alpha in range(3):
            d3_rot[alpha, ::] = _rotate_three_body(
                d_integrals[2][alpha, ::], nmodes, mode_rots, modals=modals
            )
        dip_data = [d1_rot, d2_rot, d3_rot]

    return dip_data


def _rotate_hamiltonian(h_integrals, mode_rots, modals):
    r"""Generates VSCF rotated Hamiltonian.

    Args:
        h_integrals (list[TensorLike[float]]): list of n-mode expansion of Hamiltonian integrals
        mode_rots (list[TensorLike[float]]): list of rotation matrices for all vibrational modes
        modals (list[int]): list containing the maximum number of modals to consider for each mode

    Returns:
        tuple(TensorLike[float]): tuple of rotated Hamiltonian integrals

    """

    n = len(h_integrals)
    nmodes = np.shape(h_integrals[0])[0]

    h1_rot = _rotate_one_body(h_integrals[0], nmodes, mode_rots, modals)
    h_data = [h1_rot]

    if n > 1:
        h2_rot = _rotate_two_body(h_integrals[1], nmodes, mode_rots, modals)
        h_data = [h1_rot, h2_rot]

    if n > 2:
        h3_rot = _rotate_three_body(h_integrals[2], nmodes, mode_rots, modals)
        h_data = [h1_rot, h2_rot, h3_rot]

    return h_data


def vscf_integrals(h_integrals, d_integrals=None, modals=None, cutoff=None, cutoff_ratio=1e-6):
    r"""Generates vibrational self-consistent field rotated integrals.

    This function converts the Christiansen vibrational Hamiltonian integrals obtained in the harmonic
    oscillator basis to integrals in the vibrational self-consistent field (VSCF) basis.
    The implementation is based on the method described in
    `J. Chem. Theory Comput. 2010, 6, 235–248 <https://pubs.acs.org/doi/10.1021/ct9004454>`_.

    Args:
        h_integrals (list[TensorLike[float]]): List of Hamiltonian integrals for up to 3 coupled vibrational modes.
            Look at the Usage Details for more information.
        d_integrals (list[TensorLike[float]]): List of dipole integrals for up to 3 coupled vibrational modes.
            Look at the Usage Details for more information.
        modals (list[int]): list containing the maximum number of modals to consider for each vibrational mode.
            Default value is the maximum number of modals.
        cutoff (float): threshold value for including matrix elements into operator
        cutoff_ratio (float): ratio for discarding elements with respect to biggest element in the integrals.
            Default value is ``1e-6``.

    Returns:
        tuple: a tuple containing:
            - list[TensorLike[float]]: List of Hamiltonian integrals in VSCF basis for up to 3 coupled vibrational modes.
            - list[TensorLike[float]]: List of dipole integrals in VSCF basis for up to 3 coupled vibrational modes.

        ``None`` is returned if ``d_integrals`` is ``None``.

    **Example**

    >>> h1 = np.array([[[0.00968289, 0.00233724, 0.0007408,  0.00199125],
    ...                 [0.00233724, 0.02958449, 0.00675431, 0.0021936],
    ...                 [0.0007408,  0.00675431, 0.0506012,  0.01280986],
    ...                 [0.00199125, 0.0021936,  0.01280986, 0.07282307]]])
    >>> qml.qchem.vscf_integrals(h_integrals=[h1], modals=[4,4,4])
    ([array([[[ 9.36124041e-03, -4.20128342e-19,  3.25260652e-19,
            1.08420217e-18],
            [-9.21571847e-19,  2.77803512e-02, -3.46944695e-18,
            5.63785130e-18],
            [-3.25260652e-19, -8.67361738e-19,  4.63297357e-02,
            -1.04083409e-17],
            [ 1.30104261e-18,  5.20417043e-18, -1.38777878e-17,
            7.92203227e-02]]])],
    None)


    .. details::
        :title: Usage Details

        The ``h_integral`` tensor must have one of these dimensions:

        - 1-mode coupled integrals: `(n, m, m)`
        - 2-mode coupled integrals: `(n, n, m, m, m, m)`
        - 3-mode coupled integrals: `(n, n, n, m, m, m, m, m, m)`

        where ``n`` is the number of vibrational modes in the molecule and ``m`` represents the number
        of modals.

        The ``d_integral`` tensor must have one of these dimensions:

        - 1-mode coupled integrals: `(3, n, m)`
        - 2-mode coupled integrals: `(3, n, n, m, m, m, m)`
        - 3-mode coupled integrals: `(3, n, n, n, m, m, m, m, m, m)`

        where ``n`` is the number of vibrational modes in the molecule, ``m`` represents the number
        of modals and the first axis represents the ``x, y, z`` component of the dipole. Default is ``None``.

    """

    if len(h_integrals) > 3:
        raise ValueError(
            f"Building n-mode Hamiltonian is not implemented for n equal to {len(h_integrals)}."
        )

    if d_integrals is not None:
        if len(d_integrals) > 3:
            raise ValueError(
                f"Building n-mode dipole is not implemented for n equal to {len(d_integrals)}."
            )

    nmodes = np.shape(h_integrals[0])[0]

    imax = np.shape(h_integrals[0])[1]
    max_modals = nmodes * [imax]
    if modals is None:
        modals = max_modals
    else:
        if np.max(modals) > imax:
            raise ValueError(
                "Number of maximum modals cannot be greater than the modals for unrotated integrals."
            )
        imax = np.max(modals)

    if cutoff is None:
        max_val = np.max([np.max(np.abs(H)) for H in h_integrals])
        cutoff = max_val * cutoff_ratio

    mode_rots = vscf(h_integrals, modals=max_modals, cutoff=cutoff)

    h_data = _rotate_hamiltonian(h_integrals, mode_rots, modals)

    if d_integrals is not None:
        dip_data = _rotate_dipole(d_integrals, mode_rots, modals)
        return h_data, dip_data

    return h_data, None
