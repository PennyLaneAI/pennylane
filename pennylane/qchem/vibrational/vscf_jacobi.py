# vscf_jacobi_refactored.py
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, lax
from functools import partial
from typing import NamedTuple, Iterator, Tuple, List
import logging
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)
jax.config.update("jax_enable_x64", True)  # keep f64 as requested

# ---------------------------------------------------------------------
# Data types used across the module
# ---------------------------------------------------------------------
class VSCFData(NamedTuple):
    term_map: np.ndarray     # Host-side: (L,) int32
    target_mode: np.ndarray  # Host-side: (L,) int32
    i: np.ndarray            # Host-side: (L,) int32
    j: np.ndarray            # Host-side: (L,) int32
    coeffs: np.ndarray       # Host-side: (L,) float64
    partners: np.ndarray     # Host-side: (L, P) int32 padded with -1
    valid: np.ndarray        # Host-side: (L,) bool


class DeviceBatches(NamedTuple):
    # stacked device arrays: leading axis = n_batches, second = B
    term_map: jnp.ndarray    # (n_batches, B) int32
    target_mode: jnp.ndarray # (n_batches, B) int32
    i: jnp.ndarray           # (n_batches, B) int32
    j: jnp.ndarray           # (n_batches, B) int32
    coeffs: jnp.ndarray      # (n_batches, B) float64
    partners: jnp.ndarray    # (n_batches, B, P) int32
    valid: jnp.ndarray       # (n_batches, B) bool


# ---------------------------------------------------------------------
# Your existing per-chunk static processing function (unchanged semantics)
# Keep this identical to what you had, small adjustments to return proper shapes
# ---------------------------------------------------------------------
def _process_chunk_static(args):
    """
    Static worker function to process a raw slice of the Hamiltonian.
    Returns exploded arrays for this chunk or None.
    (This function is essentially your original CPU worker; kept mostly unchanged.)
    """
    chunk_vals, start_global_idx, original_shape, n, modals_arr, cutoff, max_partners = args

    # find nonzero entries in chunk
    local_nonzero = np.nonzero(np.abs(chunk_vals) >= cutoff)[0]
    if len(local_nonzero) == 0:
        return None

    vals = chunk_vals[local_nonzero]
    global_indices_flat = local_nonzero + start_global_idx
    subs = np.unravel_index(global_indices_flat, original_shape)
    indices = np.stack(subs, axis=1)

    num_modes_in_term = n + 1
    mode_cols = indices[:, :num_modes_in_term]
    exc_cols = indices[:, num_modes_in_term:]

    # canonical filter for multi-body
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
    i_cols = exc_cols[:, :num_modes_in_term]
    j_cols = exc_cols[:, num_modes_in_term:]
    limits = modals_arr[mode_cols]
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

    # explosion: repeat per target
    exploded_c = np.repeat(vals, num_modes_in_term)
    exploded_tgt = mode_cols.flatten()
    exploded_i = i_cols.flatten()
    exploded_j = j_cols.flatten()

    # partner generation: fixed-size padded partners array
    repeated_modes = np.repeat(mode_cols, num_modes_in_term, axis=0)
    mask_self = (repeated_modes == exploded_tgt[:, None])
    raw_partners = repeated_modes[~mask_self]  # flattened partners for all rows

    current_n_partners = num_modes_in_term - 1
    exploded_p = np.full((len(exploded_tgt), max_partners), -1, dtype=np.int32)
    if current_n_partners > 0:
        raw_partners = raw_partners.reshape(-1, current_n_partners)
        # copy into padded array
        exploded_p[:, :current_n_partners] = raw_partners

    exploded_v = np.ones(len(exploded_tgt), dtype=bool)

    return (num_valid, exploded_tgt.astype(np.int32), exploded_i.astype(np.int32),
            exploded_j.astype(np.int32), exploded_c.astype(np.float64),
            exploded_p.astype(np.int32), exploded_v.astype(bool))


# ---------------------------------------------------------------------
# Parallelized generator that yields exploded numpy chunks
# ---------------------------------------------------------------------
def _generate_jacobi_batches(h_integrals, modals, cutoff, max_partners, n_workers=4, chunk_size=50_000):
    """
    Parallelized memory-resilient generator. Yields tuples:
    (exploded_term_map, tgt, i, j, coeffs, partners, valid)
    where exploded arrays are numpy arrays ready to be buffered.
    """
    modals_arr = np.array(modals)
    tasks = []
    # build tasks (views)
    for n, ham_n in enumerate(h_integrals):
        original_shape = np.shape(ham_n)
        flat_ham = np.reshape(ham_n, -1)
        total_elements = flat_ham.size
        for start_idx in range(0, total_elements, chunk_size):
            end_idx = min(start_idx + chunk_size, total_elements)
            chunk_vals = flat_ham[start_idx:end_idx]
            args = (chunk_vals, start_idx, original_shape, n, modals_arr, cutoff, max_partners)
            tasks.append(args)

    active_num = 0
    # use ThreadPool to avoid copy overhead; NumPy ops here release GIL generally
    with ThreadPoolExecutor(max_workers=n_workers) as exe:
        for res in exe.map(_process_chunk_static, tasks):
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
    """Simple growable 1D NumPy buffer for appending without frequent reallocs."""

    def __init__(self, dtype, initial_capacity=1 << 16):
        self.dtype = dtype
        self.capacity = max(1, initial_capacity)
        self.size = 0
        self.arr = np.empty(self.capacity, dtype=self.dtype)

    def append(self, data: np.ndarray):
        n = len(data)
        needed = self.size + n
        if needed > self.capacity:
            # grow by doubling until fit
            new_cap = self.capacity
            while new_cap < needed:
                new_cap *= 2
            new_arr = np.empty(new_cap, dtype=self.dtype)
            new_arr[:self.size] = self.arr[:self.size]
            self.arr = new_arr
            self.capacity = new_cap
        self.arr[self.size:self.size + n] = data
        self.size += n

    def take(self, n: int) -> np.ndarray:
        """Take first n elements, shift buffer left."""
        assert n <= self.size
        out = self.arr[:n].copy()
        # shift remainder left
        rem = self.size - n
        if rem > 0:
            self.arr[:rem] = self.arr[n:self.size]
        self.size = rem
        return out

    def pad_and_take_all(self, total_size: int, pad_value=0):
        """Return array of length total_size with padding (used for last partial batch)."""
        assert self.size <= total_size
        out = np.full(total_size, pad_value, dtype=self.dtype)
        if self.size > 0:
            out[:self.size] = self.arr[:self.size]
        self.size = 0
        return out

    def __len__(self):
        return self.size


class Growable2DBuffer:
    """Growable 2D buffer where rows have fixed width (width)."""

    def __init__(self, width: int, dtype, initial_rows=1 << 12):
        self.width = width
        self.dtype = dtype
        self.capacity_rows = max(1, initial_rows)
        self.rows = 0
        self.arr = np.empty((self.capacity_rows, width), dtype=self.dtype)

    def append_rows(self, rows: np.ndarray):
        n = rows.shape[0]
        needed = self.rows + n
        if needed > self.capacity_rows:
            new_cap = self.capacity_rows
            while new_cap < needed:
                new_cap *= 2
            new_arr = np.empty((new_cap, self.width), dtype=self.dtype)
            new_arr[:self.rows, :] = self.arr[:self.rows, :]
            self.arr = new_arr
            self.capacity_rows = new_cap
        self.arr[self.rows:self.rows + n, :] = rows
        self.rows += n

    def take_rows(self, n: int) -> np.ndarray:
        assert n <= self.rows
        out = self.arr[:n, :].copy()
        rem = self.rows - n
        if rem > 0:
            self.arr[:rem, :] = self.arr[n:self.rows, :]
        self.rows = rem
        return out

    def pad_and_take_all(self, total_rows: int, pad_value=-1):
        assert self.rows <= total_rows
        out = np.full((total_rows, self.width), pad_value, dtype=self.dtype)
        if self.rows > 0:
            out[:self.rows, :] = self.arr[:self.rows, :]
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
    Accepts raw generator that yields (term_map, tgt, i, j, coeffs, partners, valid)
    and yields fixed-size NumPy VSCFData batches.
    """

    def __init__(self, raw_generator: Iterator[Tuple[np.ndarray, ...]],
                 target_size: int, max_partners: int):
        self.raw_gen = raw_generator
        self.target_size = int(target_size)
        self.max_partners = int(max_partners)

        # Preallocate growable buffers
        init_rows = max(1, self.target_size // 8)
        self.buf_term_map = GrowableBuffer(np.int32, initial_capacity=max(1 << 12, self.target_size))
        self.buf_target_mode = GrowableBuffer(np.int32, initial_capacity=max(1 << 12, self.target_size))
        self.buf_i = GrowableBuffer(np.int32, initial_capacity=max(1 << 12, self.target_size))
        self.buf_j = GrowableBuffer(np.int32, initial_capacity=max(1 << 12, self.target_size))
        self.buf_coeffs = GrowableBuffer(np.float64, initial_capacity=max(1 << 12, self.target_size))
        self.buf_partners = Growable2DBuffer(width=self.max_partners, dtype=np.int32, initial_rows=init_rows)
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

        return VSCFData(tm, tgt, ii, jj, cc, pp, vv)

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
        return VSCFData(tm, tgt, ii, jj, cc, pp, vv)

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
# New fused JAX kernel: takes stacked device batches and performs one big accumulation
# ---------------------------------------------------------------------
@partial(jit, static_argnames=['modals_tuple', 'nmodes', 'max_modals', 'P', 'B', 'n_batches', 'damping'])
def scf_step_batched(h_mat, mode_rots, device_batches: DeviceBatches,
                     modals_tuple, nmodes, max_modals, P, B, n_batches, damping=0.0):
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

    tm_flat = device_batches.term_map.reshape((total_entries,))     # int32
    tgt_flat = device_batches.target_mode.reshape((total_entries,))
    i_flat = device_batches.i.reshape((total_entries,))
    j_flat = device_batches.j.reshape((total_entries,))
    coeff_flat = device_batches.coeffs.reshape((total_entries,))
    partners_flat = device_batches.partners.reshape((total_entries, P))
    valid_flat = device_batches.valid.reshape((total_entries,))

    # clamp partners for safe indexing
    lookup_idx = jnp.maximum(partners_flat, 0)  # (total_entries, P)

    # gather h values per partner: h_mat[tm_flat[:,None], lookup_idx] -> (total_entries, P)
    rows = tm_flat[:, None]
    raw_potentials = h_mat[rows, lookup_idx]  # (total_entries, P)

    # mask padded partners -> use 1.0 for padded (partners==-1)
    is_real_partner = (partners_flat != -1)
    factors = jnp.where(is_real_partner, raw_potentials, 1.0)

    # product across partners
    mean_fields = jnp.prod(factors, axis=1)  # (total_entries,)

    # compute contribution (zero for padded rows via valid_flat)
    vals = coeff_flat * mean_fields * valid_flat  # (total_entries,)

    # compute flat indices into fock storage
    flat_indices = (tgt_flat * max_modals * max_modals) + (i_flat * max_modals) + j_flat  # (total_entries,)

    total_segments = nmodes * max_modals * max_modals

    # single large reduction to build fock_flat
    fock_flat = jax.ops.segment_sum(vals, flat_indices, total_segments)  # (total_segments,)
    all_focks = fock_flat.reshape((nmodes, max_modals, max_modals))

    # rotate and diagonalize (same as before)
    u_t = jnp.transpose(mode_rots, (0, 2, 1))
    all_focks_rot = jnp.matmul(u_t, jnp.matmul(all_focks, mode_rots))

    # penalty
    modals_tensor = jnp.array(modals_tuple)
    idx_range = jnp.arange(max_modals)
    padding_mask = idx_range[None, :] >= modals_tensor[:, None]
    penalty_diag = jax.vmap(jnp.diag)(padding_mask.astype(float) * 1e9)
    all_focks_rot = all_focks_rot + penalty_diag

    eigvals, all_eigvecs = jax.scipy.linalg.eigh(all_focks_rot)
    new_rots = jnp.matmul(mode_rots, all_eigvecs)

    # -----------------------------------------------------------------
    # Update h_mat in one vectorized pass:
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

    # damping
    h_mat_next = (1.0 - damping) * h_mat_new + damping * h_mat

    total_energy = jnp.sum(eigvals[:, 0])
    return h_mat_next, new_rots, total_energy


# ---------------------------------------------------------------------
# Top-level vscf integrating everything: CPU packing -> device stacking -> SCF loop
# ---------------------------------------------------------------------
def vscf(h_integrals, modals, cutoff, tol=1e-8, max_iters=10000,
         batch_size=20000, n_workers=4, chunk_size=None):
    """
    High-performance VSCF using fixed-size sub-batches and a fused SCF kernel.
    - h_integrals: iterable of numpy arrays (your integrals)
    - modals: list of modal counts per mode
    - cutoff: float threshold
    - batch_size: B (entries per packed batch)
    - n_workers: threads for CPU chunk processing
    - chunk_size: size for scanning integrals; default tuned from batch_size
    """
    nmodes = len(modals)

    # figure out max bodies involved
    max_bodies_involved = 0
    for h in h_integrals:
        # infer number of modes per term assuming each term adds 3 dims per body (your previous heuristic)
        num_modes_in_term = len(h.shape) // 3
        max_bodies_involved = max(max_bodies_involved, num_modes_in_term)
    max_partners = max(1, max_bodies_involved - 1)
    P = max_partners
    B = int(batch_size)

    if chunk_size is None:
        chunk_size = max(1, int(B * 1.5))

    # 1) Prepare raw generator (parallel explosion)
    raw_gen = _generate_jacobi_batches(h_integrals, modals, cutoff, max_partners, n_workers=n_workers, chunk_size=chunk_size)

    # 2) Pack into fixed-size batches using preallocated growable buffers
    packer = BatchRegularizer(raw_gen, target_size=B, max_partners=max_partners)
    packed_batches: List[VSCFData] = []
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

    # 3) Move stacked batches to device (single py-tree)
    term_map_stack = jnp.stack([jnp.array(pb.term_map, dtype=jnp.int32) for pb in packed_batches], axis=0)  # (n_batches, B)
    target_stack = jnp.stack([jnp.array(pb.target_mode, dtype=jnp.int32) for pb in packed_batches], axis=0)
    i_stack = jnp.stack([jnp.array(pb.i, dtype=jnp.int32) for pb in packed_batches], axis=0)
    j_stack = jnp.stack([jnp.array(pb.j, dtype=jnp.int32) for pb in packed_batches], axis=0)
    coeff_stack = jnp.stack([jnp.array(pb.coeffs, dtype=jnp.float64) for pb in packed_batches], axis=0)
    partners_stack = jnp.stack([jnp.array(pb.partners, dtype=jnp.int32) for pb in packed_batches], axis=0)  # (n_batches, B, P)
    valid_stack = jnp.stack([jnp.array(pb.valid, dtype=bool) for pb in packed_batches], axis=0)

    device_batches = DeviceBatches(term_map_stack, target_stack, i_stack, j_stack, coeff_stack, partners_stack, valid_stack)

    # 4) initialize h_mat and rotations
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
    h_mat = jnp.zeros((max_term + 1, nmodes), dtype=jnp.float64)

    # 5) define step fn
    step_fn = partial(scf_step_batched, device_batches=device_batches,
                      modals_tuple=tuple(modals),
                      nmodes=nmodes, max_modals=max_modals, P=P, B=B, n_batches=n_batches, damping=0.0)

    # 6) run SCF loop using lax.while_loop (jit compiled)
    def cond_fn(state):
        _, _, e_cur, e_prev, idx = state
        return (jnp.abs(e_cur - e_prev) > tol) & (idx < max_iters)

    def body_fn(state):
        h_old, rots_old, e_old, e_prev, it_idx = state
        h_new, rots_new, e_new = step_fn(h_old, rots_old)
        return (h_new, rots_new, e_new, e_old, it_idx + 1)

    # initial run
    init_h, init_rots, init_e = step_fn(h_mat, mode_rots)
    loop_state = (init_h, init_rots, init_e, jnp.inf, 0)
    final_h, final_rots, final_e, _, n_iters = lax.while_loop(cond_fn, body_fn, loop_state)

    logger.debug(f"Converged in {int(n_iters)} iterations to energy {float(final_e):.10f}")
    return float(final_e), final_rots
