import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, lax

from functools import partial
from typing import NamedTuple, Iterator, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor


from .timers import Timer

logger = logging.getLogger(__name__)

# Enable 64-bit precision for SCF stability
jax.config.update("jax_enable_x64", True)


class VSCFData(NamedTuple):
    """
    Global Flattened Data Structure.
    'target_mode' maps every term to the specific mode (Fock matrix) it belongs to.
    """
    term_map: jnp.ndarray     # Index into h_mat (row)
    target_mode: jnp.ndarray  # Index into h_mat (col) / Mode Index
    i: jnp.ndarray            # Bra index
    j: jnp.ndarray            # Ket index
    coeffs: jnp.ndarray       # Raw hamiltonian coefficient
    partners: jnp.ndarray     # indices of modes this terms interacts with
    valid: jnp.ndarray        # boolean enabling partners --> needed bc otherwise buffering padding creates troubles


def _process_chunk_static(args):
    """
    Static worker function to process a raw slice of the Hamiltonian.
    Returns the exploded, sparse arrays for this chunk.
    """
    # Unpack arguments
    chunk_vals, start_global_idx, original_shape, n, modals_arr, cutoff, max_partners = args
    
    # 1. Sparsity Check
    local_nonzero = np.nonzero(np.abs(chunk_vals) >= cutoff)[0]
    if len(local_nonzero) == 0:
        return None

    vals = chunk_vals[local_nonzero]
    
    # 2. Recover Indices
    global_indices_flat = local_nonzero + start_global_idx
    subs = np.unravel_index(global_indices_flat, original_shape)
    indices = np.stack(subs, axis=1)

    num_modes_in_term = n + 1
    mode_cols = indices[:, :num_modes_in_term]
    exc_cols = indices[:, num_modes_in_term:]

    # 3. Canonical Filter
    if num_modes_in_term > 1:
        is_canonical = np.all(mode_cols[:, :-1] > mode_cols[:, 1:], axis=1)
        indices = indices[is_canonical]
        vals = vals[is_canonical]
        mode_cols = mode_cols[is_canonical]
        exc_cols = exc_cols[is_canonical]
    
    if len(vals) == 0: return None

    # 4. Truncation Filter
    keep_mask = np.ones(len(vals), dtype=bool)
    i_cols = exc_cols[:, :num_modes_in_term]
    j_cols = exc_cols[:, num_modes_in_term:]
    limits = modals_arr[mode_cols]
    valid_dims = (i_cols < limits) & (j_cols < limits)
    keep_mask = np.all(valid_dims, axis=1)

    mode_cols = mode_cols[keep_mask]
    i_cols = i_cols[keep_mask]
    j_cols = j_cols[keep_mask]
    vals = vals[keep_mask]
    
    num_valid = len(vals)
    if num_valid == 0: return None

    # 5. Explosion (Target + Partners)
    
    # Repeat values
    exploded_c = np.repeat(vals, num_modes_in_term)
    exploded_tgt = mode_cols.flatten()
    exploded_i = i_cols.flatten()
    exploded_j = j_cols.flatten()
    
    # Partner Generation (The new sparse logic)
    repeated_modes = np.repeat(mode_cols, num_modes_in_term, axis=0)
    mask_self = (repeated_modes == exploded_tgt[:, None])
    raw_partners = repeated_modes[~mask_self]
    
    current_n_partners = num_modes_in_term - 1
    
    # Create fixed-size partner array filled with -1
    exploded_p = np.full((len(exploded_tgt), max_partners), -1, dtype=np.int32)
    
    if current_n_partners > 0:
        raw_partners = raw_partners.reshape(-1, current_n_partners)
        exploded_p[:, :current_n_partners] = raw_partners

    # Valid mask (always True for real data)
    exploded_v = np.ones(len(exploded_tgt), dtype=bool)

    # Note: We return raw count, main process handles active_num offset
    return (num_valid, exploded_tgt, exploded_i, exploded_j, exploded_c, exploded_p, exploded_v)



# we perform data preparation on CPU with numpy, since:
# most of the integrals are zero --> we would clog PCIe bus transferring empty data.
# + GPU memory (VRAM) is usally more limited than system RAM
# + JAX loves static shapes.
def _generate_jacobi_batches(h_integrals, modals, cutoff, max_bodies_involved, n_workers=4, chunk_size=50_000):
    """
    Parallelized Memory-Resilient Generator (Thread-Based).
    Uses Threads to exploit NumPy's GIL-releasing operations without memory overhead.
    """
    max_partners = max(1, max_bodies_involved - 1)
    modals_arr = np.array(modals)
    active_num = 0
    
    # 1. Prepare Tasks
    tasks = []
    
    # Optimization: Pre-calculate total size to avoid reallocation issues? 
    # No, lists are fine for task storage.
    
    for n, ham_n in enumerate(h_integrals):
        original_shape = np.shape(ham_n)
        
        # Flattening creates a view (fast)
        flat_ham = np.reshape(ham_n, -1)
        total_elements = flat_ham.size
        
        for start_idx in range(0, total_elements, chunk_size):
            end_idx = min(start_idx + chunk_size, total_elements)
            
            # Slicing a numpy array creates a VIEW, not a copy.
            # Since we use threads, this view is valid and shared instantly.
            chunk_vals = flat_ham[start_idx:end_idx]
            
            args = (chunk_vals, start_idx, original_shape, n, modals_arr, cutoff, max_partners)
            tasks.append(args)

    # 2. Execute with ThreadPool
    # We don't need 'spawn' context because threads live in the same process.
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        
        # map preserves order
        results = executor.map(_process_chunk_static, tasks)
        
        for res in results:
            if res is None:
                continue
                
            (num_valid_terms, tgt, i, j, c, p, v) = res
            
            # 3. Post-Process (Offset term_ids)
            # This is fast scalar math, keep it in the main thread.
            n_modes_in_this_chunk = len(tgt) // num_valid_terms
            
            new_ids = np.arange(active_num, active_num + num_valid_terms)
            exploded_tm = np.repeat(new_ids, n_modes_in_this_chunk)
            
            active_num += num_valid_terms
            
            yield (
                exploded_tm.astype(np.int32),
                tgt.astype(np.int32),
                i.astype(np.int32),
                j.astype(np.int32),
                c.astype(np.float64),
                p.astype(np.int32),
                v.astype(bool),
                #active_num
            )


class BatchRegularizer:
    """
    Efficient fixed-size batch normalizer.
    Maintains NumPy buffers and outputs batches of uniform shape.

    raw_generator: yields VSCFData with arbitrary (variable) length
    target_size: desired batch size (int)
    """

    def __init__(self, raw_generator: Iterator[VSCFData],
                 target_size: int,  max_partners:int):
        
        self.raw_gen = raw_generator
        self.target_size = target_size
        self.max_partners = max_partners

        # Initialize empty buffers (NumPy arrays)
        self.buf_term_map = np.empty((0,), dtype=np.int32)
        self.buf_target_mode = np.empty((0,), dtype=np.int32)
        self.buf_i = np.empty((0,), dtype=np.int32)
        self.buf_j = np.empty((0,), dtype=np.int32)
        self.buf_coeffs = np.empty((0,), dtype=np.float64)
        self.buf_partners = np.empty((0, max_partners), dtype=np.int32)
        self.buf_valid = np.empty((0,), dtype=bool)

    def __iter__(self):
        return self._generator()

    def _emit_full_batch(self):
        """Emit the first target_size entries from the buffer."""
        B = self.target_size

        batch = VSCFData(
            term_map=self.buf_term_map[:B],
            target_mode=self.buf_target_mode[:B],
            i=self.buf_i[:B],
            j=self.buf_j[:B],
            coeffs=self.buf_coeffs[:B],
            partners=self.buf_partners[:B],
            valid=self.buf_valid[:B]
        )

        # Retain remaining part in buffer by slicing
        self.buf_term_map = self.buf_term_map[B:]
        self.buf_target_mode = self.buf_target_mode[B:]
        self.buf_i = self.buf_i[B:]
        self.buf_j = self.buf_j[B:]
        self.buf_coeffs = self.buf_coeffs[B:]
        self.buf_partners = self.buf_partners[B:]
        self.buf_valid = self.buf_valid[B:]

        return batch

    def _emit_padded_last_batch(self):
        """Pad the last partial batch."""
        B = self.target_size
        L = len(self.buf_term_map)
        pad = B - L

        if L == 0:
            return None  # nothing left

        batch = VSCFData(
            term_map=np.pad(self.buf_term_map, (0, pad), mode="constant"),
            target_mode=np.pad(self.buf_target_mode, (0, pad), mode="constant"),
            i=np.pad(self.buf_i, (0, pad), mode="constant"),
            j=np.pad(self.buf_j, (0, pad), mode="constant"),
            coeffs=np.pad(self.buf_coeffs, (0, pad), mode="constant"),
            partners=np.pad(self.buf_partners, ((0, pad), (0, 0)), mode="constant", constant_values=-1),
            valid=np.pad(self.buf_valid, (0, pad), mode="constant", constant_values=False)
        )

        # clear buffer
        self.buf_term_map = self.buf_target_mode = self.buf_i = self.buf_j = self.buf_coeffs = np.empty((0,), dtype=np.int32)
        self.buf_valid = np.empty((0, self.max_partners), dtype=bool)

        return batch

    def _generator(self):
        for term_map, target_mode, i, j, coeffs, partners, valid in self.raw_gen:
            # Concatenate new chunk
            self.buf_term_map  = np.concatenate([self.buf_term_map, term_map])
            self.buf_target_mode = np.concatenate([self.buf_target_mode, target_mode])
            self.buf_i = np.concatenate([self.buf_i, i])
            self.buf_j = np.concatenate([self.buf_j, j])
            self.buf_coeffs = np.concatenate([self.buf_coeffs, coeffs])
            self.buf_partners = np.concatenate([self.buf_partners, partners])
            self.buf_valid = np.concatenate([self.buf_valid, valid])

            # Emit full batches as long as possible
            while len(self.buf_term_map) >= self.target_size:
                yield self._emit_full_batch()

        # End-of-stream → emit final padded batch
        last = self._emit_padded_last_batch()
        if last is not None:
            yield last


@partial(jit, static_argnames=['modals_tuple', 'nmodes', 'max_modals', 'damping'])
def scf_step_batched(h_mat, mode_rots, batch_list, modals_tuple, nmodes, max_modals, damping=0.0):
    """
    Batched JAX Kernel.
    Iterates over a list of data batches to construct the Fock matrix incrementally.
    """
    
    # Initialize global accumulators
    fock_storage = jnp.zeros(nmodes * max_modals * max_modals, dtype=h_mat.dtype)
    
    # -----------------------------------------------------------------------
    # 1. Loop over Batches (Unrolled by JAX)
    # -----------------------------------------------------------------------
    for batch in batch_list:
        tm = batch.term_map       # [B]
        tgt = batch.target_mode   # [B]
        i = batch.i               # [B]
        j = batch.j               # [B]
        c = batch.coeffs          # [B]
        partners = batch.partners # [B, P] (Contains indices 0..M, or -1)
        valid = batch.valid       # [B]
        
        # sparse mean field logic
        # subset_h = h_mat[tm]
        # factors = subset_h * m + (~m)
        # mean_fields = jnp.prod(factors, axis=1)
        # vals = c * mean_fields
        
        # 1. Safe Indexing for Lookup
        # We need to look up h_mat[tm, partner_idx].
        # -1 is invalid. We clamp it to 0 so JAX doesn't crash on lookup.
        # We will mask the result later.
        lookup_idx = jnp.maximum(partners, 0) 
        
        # 2. Gather Mean Field Values
        # h_mat shape: (N_active_terms, nmodes)
        # We use advanced indexing:
        # Rows: tm (broadcasted to [B, P])
        # Cols: lookup_idx [B, P]
        # Result shape: [B, P]
        raw_potentials = h_mat[tm[:, None], lookup_idx]
        
        # 3. Apply Mask for Padding (-1)
        # If partner was -1, the potential should be treated as 1.0 (identity for product)
        is_real_partner = (partners != -1)
        
        # factors: Real potential if valid, else 1.0
        factors = jnp.where(is_real_partner, raw_potentials, 1.0)
        
        # 4. Product across partners
        # Shape: [B, P] -> [B]
        mean_fields = jnp.prod(factors, axis=1)
        
        # 5. Calculate Final Value
        # Apply Coeffs * MeanFields * Validity (handle buffer padding)
        # If valid is False, val becomes 0.0 and adds nothing to Fock matrix
        vals = c * mean_fields * valid
        
        
        # Accumulate into global storage
        flat_indices = (tgt * max_modals * max_modals) + (i * max_modals) + j
        fock_storage = fock_storage.at[flat_indices].add(vals)
        
        # Note: We do NOT update h_mat here. We only read from it.
        # h_mat update happens after we have the full new rotation.

    # -----------------------------------------------------------------------
    # 2. Diagonalization & Updates (Same as before)
    # -----------------------------------------------------------------------
    all_focks = fock_storage.reshape(nmodes, max_modals, max_modals)
    
    # Rotate: U.T @ F @ U
    u_t = jnp.transpose(mode_rots, (0, 2, 1))
    all_focks_rot = jnp.matmul(u_t, jnp.matmul(all_focks, mode_rots))
    
    # Penalty
    modals_tensor = jnp.array(modals_tuple)
    idx_range = jnp.arange(max_modals)
    padding_mask = idx_range[None, :] >= modals_tensor[:, None]
    penalty_diag = jax.vmap(jnp.diag)(padding_mask.astype(float) * 1e9)
    all_focks_rot = all_focks_rot + penalty_diag

    # Diagonalize
    eigvals, all_eigvecs = jax.scipy.linalg.eigh(all_focks_rot)

    # Update Rotations
    new_rots = jnp.matmul(mode_rots, all_eigvecs)
    
    # -----------------------------------------------------------------------
    # 3. Update Hamiltonian (Batched)
    # -----------------------------------------------------------------------
    # We must iterate batches again to update h_mat.
    # JAX handles this fine; it's just more nodes in the graph.
    
    # Get ground state vectors
    gs_vectors = new_rots[:, :, 0]
    
    # We can create a buffer for h_mat updates or use lax.fori_loop if batches are uniform.
    # Since h_mat update is scatter, we can apply updates sequentially.
    
    h_mat_new = h_mat # Start with current (copy)
    
    for batch in batch_list:
        tm = batch.term_map
        tgt = batch.target_mode
        i = batch.i
        j = batch.j
        
        val_i = gs_vectors[tgt, i]
        val_j = gs_vectors[tgt, j]
        new_h_vals = val_i * val_j
        
        h_mat_new = h_mat_new.at[tm, tgt].set(new_h_vals)

    # Damping
    h_mat_next = (1.0 - damping) * h_mat_new + damping * h_mat
    
    # Energy (Sum of eigenvalues of occupied states)
    total_energy = jnp.sum(eigvals[:, 0])

    return h_mat_next, new_rots, total_energy


def vscf(h_integrals, modals, cutoff, tol=1e-8, max_iters=10000, batch_size=20000, n_workers=4):
    nmodes = len(modals)
    
    max_bodies_involved = 0
    for h_data in h_integrals:
        # 1body terms have len(shape) = 3, 2body = 6 etc
        num_modes_in_term = len(h_data.shape)//3 # this number = #bodies 
        max_bodies_involved = max(max_bodies_involved, num_modes_in_term)
    
    # we store at most (max_bodies - 1) partners per term.
    # e.g., for 3-body terms, each target sees 2 partners.
    max_partners = max(1, max_bodies_involved - 1)
    
    with Timer("prep", verbose=True):
        
        
        gpu_batches = []
        active_num_final = 0
        
        # rough estimate
        chunk_size = int(batch_size * 1.5)
        
        raw_gen = _generate_jacobi_batches(h_integrals, modals, cutoff, max_partners, n_workers, chunk_size)
        buffered_gen = BatchRegularizer(raw_gen, target_size=batch_size, max_partners=max_partners)
        
        # consume data from the generator, and move them to device
        # NOTE: if data preparation becomes a bottleneck, it should be possible to subdivide this step into 
        # parallel CPU computations.
        for tm, tgt, i, j, c, p, v in buffered_gen:
            # move numpy arrays to JAX (GPU if available) immediately
            batch = VSCFData(
                term_map = jnp.array(tm),
                target_mode = jnp.array(tgt),
                i = jnp.array(i),
                j = jnp.array(j),
                coeffs = jnp.array(c),
                partners= jnp.array(p),
                valid= jnp.array(v)
            )
            #print(f"{tm.shape}, {tgt.shape}, {i.shape}, {j.shape}, {c.shape}, {p.shape}")
            gpu_batches.append(batch)
            
            active_num_final += batch_size
            
            # explicitly delete cpu arrays to free memory
            del tm, tgt, i, j, c, p, v
    #print(len(gpu_batches))
    logger.debug(f"Data ready. {len(gpu_batches)} batches on device. Total active terms: {active_num_final}")
    
    # initialize h_mat and mode_rots
    max_modals = max(modals)
    mode_rots_np = np.zeros((nmodes, max_modals, max_modals))
    for idx, dim in enumerate(modals):
        mode_rots_np[idx, :dim, :dim] = np.eye(dim) # eye rotation at the beginning for each mode
        
        # padding with identity for unused modals (since number of modals can differ per mode.)
        if dim < max_modals:
            pad_idx = np.arange(dim, max_modals)
            mode_rots_np[idx, pad_idx, pad_idx] = 1.0
            
    mode_rots = jnp.array(mode_rots_np)
    h_mat = jnp.zeros((active_num_final, nmodes))

    with Timer("SCF loop", verbose=True):
        step_fn = partial(scf_step_batched, 
                        batch_list=gpu_batches, # Pass list of batches
                        modals_tuple=tuple(modals), 
                        nmodes=nmodes, 
                        max_modals=max_modals,
                        damping=0.0)

        
        # helper functions for lax.while_loop
        def condition_function(val):
            _, _, e_curr, e_prev, iter_num = val
            return (jnp.abs(e_curr - e_prev) > tol) & (iter_num < max_iters)

        def body_function(val):
            h_mat_old, rots_old, e_old, _, iter_num = val
            h_mat_new, rots_new, e_new = step_fn(h_mat_old, rots_old)
            return (h_mat_new, rots_new, e_new, e_old, iter_num + 1)

        # initialization run
        init_h, init_rots, init_e = step_fn(h_mat, mode_rots)
        loop_data = (init_h, init_rots, init_e, jnp.inf, 0)
        
        #SCF loop:
        final_vals = lax.while_loop(condition_function, body_function, loop_data)
        
    energy = final_vals[2]
    rots = final_vals[1]
    n_iters = final_vals[4]
    
    logger.debug(f"VSCF Jacobi converged in {n_iters} iterations to energy {energy:.10f}")
    Timer.print_statistics()
    return energy, rots


