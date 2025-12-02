import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, lax

from functools import partial
from typing import NamedTuple, Iterator, Tuple
import logging



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
    mask: jnp.ndarray         # Boolean mask for mean-field product



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


# we perform data preparation on CPU with numpy, since:
# most of the integrals are zero --> we would clog PCIe bus transferring empty data.
# + GPU memory (VRAM) is usally more limited than system RAM
# + JAX loves static shapes.

def _generate_jacobi_batches(h_integrals, modals, cutoff, max_partners, chunk_size=500_000):
    r"""
    Memory-Resilient Generator.
    Yields VSCFData batches instead of returning one huge object.
    Prevents CPU RAM explosion during data preparation.
    """
    active_num = 0
    modals_arr = np.array(modals)

    for ham_n in h_integrals:
        
        # 1body terms have len(shape) = 3, 2body = 6 etc
        num_modes_in_term = len(ham_n.shape)//3 # this number = #bodies 
        
        total_elements = np.prod(np.shape(ham_n))
        flat_ham = np.reshape(ham_n, -1)
        
        # we process the tensor in flattened chunks to avoid loading/exploding everything at once.
        # iterate over the tensor in chunks
        for start_idx in range(0, total_elements, chunk_size):
            
            end_idx = min(start_idx + chunk_size, total_elements)
            chunk_vals = flat_ham[start_idx:end_idx]
            
            # exploit sparsity!
            local_nonzero = np.nonzero(np.abs(chunk_vals) >= cutoff)[0]
            
            if len(local_nonzero) == 0:
                continue
                
            vals = chunk_vals[local_nonzero]
            
            # calculate global flat indices
            global_indices_flat = local_nonzero + start_idx
            
            # for each element, we store its original indices
            multidim_indices = np.unravel_index(global_indices_flat, np.shape(ham_n))
            indices = np.stack(multidim_indices, axis=1)

        
            mode_cols = indices[:, :num_modes_in_term]
            exc_cols = indices[:, num_modes_in_term:]

            # since the Hamiltonian is symmetric, we can reduce the number of terms to consider
            if num_modes_in_term > 1: 
                #if we are in multi-body term, let's keep only the canonical ordering:
                # e.g. 2-bodies: if a term is (2,1), we keep it. if (1,2), we discard it.
                is_canonical = np.all(mode_cols[:, :-1] > mode_cols[:, 1:], axis=1) 
                indices = indices[is_canonical]
                vals = vals[is_canonical]
                mode_cols = mode_cols[is_canonical]
                exc_cols = exc_cols[is_canonical]
            if len(vals) == 0: 
                continue


            # truncate to active space:
            # modals_arr -> we may want to use less modals than those calculated in the integral (to speedup)
            # if integrals contain data for 20 modals, but we only want ot use [4,4,..] then let's filter them.
            
            keep_mask = np.ones(len(vals), dtype=bool)
            i_cols = exc_cols[:, :num_modes_in_term]
            j_cols = exc_cols[:, num_modes_in_term:]
            limits = modals_arr[mode_cols]
            valid_dims = (i_cols < limits) & (j_cols < limits) # let's keep only indices inside limits (active space)
            keep_mask = np.all(valid_dims, axis=1)

            mode_cols = mode_cols[keep_mask]
            i_cols = i_cols[keep_mask]
            j_cols = j_cols[keep_mask]
            vals = vals[keep_mask]
            
            num_valid = len(vals)
            if num_valid == 0: 
                continue

            
            # Data expansion: (safe because chunk_size is limited)
            # if we had one entry: 
            #   Modes: [A, B], Val: V
            # we expand to two entries:
            #   Entry 1: Target: A, Partners: [A, B], Val: V
            #   Entry 2: Target: B, Partners: [A, B], Val: V
            # (recall that num_modes_in_term = #bodies involved in this integral)
            
            term_ids = np.arange(active_num, active_num + num_valid)
            exploded_tm = np.repeat(term_ids, num_modes_in_term)
            exploded_coeff = np.repeat(vals, num_modes_in_term)
            exploded_target = mode_cols.flatten()
            exploded_i = i_cols.flatten()
            exploded_j = j_cols.flatten()
            
            
            # 2. Partner Index Generation (Vectorized)
            
            # Step A: Repeat the mode columns to match explosion
            # shape: (N * num_modes_in_term, num_modes_in_term)
            # e.g. Row 0: [A, B], Row 1: [A, B]
            repeated_modes = np.repeat(mode_cols, num_modes_in_term, axis=0)
            
            # Step B: Identify self-interaction (where mode == target)
            # exploded_target has shape (N * num_modes_in_term,)
            # We broadcast to identify the diagonal elements
            mask_self = (repeated_modes == exploded_target[:, None])
            
            # Step C: Select only non-self modes (the partners)
            # This flattens the array to (N * num_modes * (num_modes-1))
            raw_partners = repeated_modes[~mask_self]
            
            # Step D: Reshape back to rows
            # Each exploded term has exactly (num_modes_in_term - 1) partners
            current_n_partners = num_modes_in_term - 1
            
            if current_n_partners > 0:
                raw_partners = raw_partners.reshape(-1, current_n_partners)
                
                # Step E: Pad with -1 to match 'max_partners' width
                # Create a buffer filled with -1
                exploded_partners = np.full((len(exploded_tm), max_partners), -1, dtype=np.int32)
                
                # Fill the valid slots
                exploded_partners[:, :current_n_partners] = raw_partners
            else:
                # 1-body terms have 0 partners. Fill with -1.
                exploded_partners = np.full((len(exploded_tm), max_partners), -1, dtype=np.int32)

            # 3. Valid Mask (Always True for real data)
            exploded_valid = np.ones(len(exploded_tm), dtype=bool)
            
            active_num += num_valid
            
            # yield a batch:
            # the yield keyword turns a function into a function generator. The function generator returns an iterator.
            yield (
                np.array(exploded_tm, dtype=np.int32),
                np.array(exploded_target, dtype=np.int32),
                np.array(exploded_i, dtype=np.int32),
                np.array(exploded_j, dtype=np.int32),
                np.array(exploded_coeff, dtype=np.float64),
                np.array(exploded_partners, dtype=np.int32),
                np.array(exploded_valid, dtype=bool),
                #active_num # Return current count so we know total active
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


def vscf(h_integrals, modals, cutoff, tol=1e-8, max_iters=10000, batch_size=20000):
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
        
        raw_gen = _generate_jacobi_batches(h_integrals, modals, cutoff, max_partners, chunk_size)
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


