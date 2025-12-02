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


# we perform data preparation on CPU with numpy, since:
# most of the integrals are zero --> we would clog PCIe bus transferring empty data.
# + GPU memory (VRAM) is usally more limited than system RAM
# + JAX loves static shapes.

def _generate_jacobi_batches(h_integrals, modals, cutoff, chunk_size=500_000):
    r"""
    Memory-Resilient Generator.
    Yields VSCFData batches instead of returning one huge object.
    Prevents CPU RAM explosion during data preparation.
    """
    nmodes = np.shape(h_integrals[0])[0]
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
            
            
            # mask generation: this will be used to compute mean-field products efficiently.
            # for each exploded entry, we create a boolean mask that is True for all partner modes (False for the current mode)
            # NOTE this boolean mask could be stored as integer bitmask for memory efficiency, but let's keep it simple for now.
            # (idk if jax can handle integer bitmasks easily)
            
            base_mask = np.zeros((num_valid, nmodes), dtype=bool)
            rows = np.arange(num_valid)[:, None]
            base_mask[rows, mode_cols] = True
            exploded_mask = np.repeat(base_mask, num_modes_in_term, axis=0)
            
            # logic: set diagonal to false (no self-interaction in mean-field product)
            # Entry 1 (target A): mask is [False, True]
            # Entry 2 (target B): mask is [True, False]
            exploded_mask[np.arange(len(exploded_target)), exploded_target] = False
            
            active_num += num_valid
            
            # yield a batch:
            # the yield keyword turns a function into a function generator. The function generator returns an iterator.
            yield (
                np.array(exploded_tm, dtype=np.int32),
                np.array(exploded_target, dtype=np.int32),
                np.array(exploded_i, dtype=np.int32),
                np.array(exploded_j, dtype=np.int32),
                np.array(exploded_coeff, dtype=np.float64),
                np.array(exploded_mask, dtype=bool),
                active_num # Return current count so we know total active
            )


class BatchRegularizer:
    """
    Efficient fixed-size batch normalizer.
    Maintains NumPy buffers and outputs batches of uniform shape.

    raw_generator: yields VSCFData with arbitrary (variable) length
    target_size: desired batch size (int)
    """

    def __init__(self, raw_generator: Iterator[VSCFData],
                 target_size: int, nmodes: int):
        self.raw_gen = raw_generator
        self.target_size = target_size
        self.nmodes = nmodes

        # Initialize empty buffers (NumPy arrays)
        self.tm_buf = np.empty((0,), dtype=np.int32)
        self.tgt_buf = np.empty((0,), dtype=np.int32)
        self.i_buf = np.empty((0,), dtype=np.int32)
        self.j_buf = np.empty((0,), dtype=np.int32)
        self.c_buf = np.empty((0,), dtype=np.float64)
        self.m_buf = np.empty((0, nmodes), dtype=bool)

    def __iter__(self):
        return self._generator()

    def _emit_full_batch(self):
        """Emit the first target_size entries from the buffer."""
        B = self.target_size

        batch = VSCFData(
            term_map=self.tm_buf[:B],
            target_mode=self.tgt_buf[:B],
            i=self.i_buf[:B],
            j=self.j_buf[:B],
            coeffs=self.c_buf[:B],
            mask=self.m_buf[:B],
        )

        # Retain remaining part in buffer by slicing
        self.tm_buf   = self.tm_buf[B:]
        self.tgt_buf  = self.tgt_buf[B:]
        self.i_buf    = self.i_buf[B:]
        self.j_buf    = self.j_buf[B:]
        self.c_buf    = self.c_buf[B:]
        self.m_buf    = self.m_buf[B:]

        return batch

    def _emit_padded_last_batch(self):
        """Pad the last partial batch."""
        B = self.target_size
        L = len(self.tm_buf)
        pad = B - L

        if L == 0:
            return None  # nothing left

        batch = VSCFData(
            term_map=np.pad(self.tm_buf, (0, pad), mode="constant"),
            target_mode=np.pad(self.tgt_buf, (0, pad), mode="constant"),
            i=np.pad(self.i_buf, (0, pad), mode="constant"),
            j=np.pad(self.j_buf, (0, pad), mode="constant"),
            coeffs=np.pad(self.c_buf, (0, pad), mode="constant"),
            mask=np.pad(self.m_buf, ((0, pad), (0, 0)), mode="constant"),
        )

        # clear buffer
        self.tm_buf = self.tgt_buf = self.i_buf = self.j_buf = self.c_buf = np.empty((0,), dtype=np.int32)
        self.m_buf = np.empty((0, self.nmodes), dtype=bool)

        return batch

    def _generator(self):
        for term_map, target_mode, i, j, coeffs, mask, _ in self.raw_gen:
            # Concatenate new chunk
            self.tm_buf  = np.concatenate([self.tm_buf, term_map])
            self.tgt_buf = np.concatenate([self.tgt_buf, target_mode])
            self.i_buf   = np.concatenate([self.i_buf, i])
            self.j_buf   = np.concatenate([self.j_buf, j])
            self.c_buf   = np.concatenate([self.c_buf, coeffs])
            self.m_buf   = np.concatenate([self.m_buf, mask])

            # Emit full batches as long as possible
            while len(self.tm_buf) >= self.target_size:
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
        tm = batch.term_map
        tgt = batch.target_mode
        i = batch.i
        j = batch.j
        c = batch.coeffs
        m = batch.mask
        
        # Standard Mean Field Logic
        subset_h = h_mat[tm]
        factors = subset_h * m + (~m)
        mean_fields = jnp.prod(factors, axis=1)
        vals = c * mean_fields
        
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
    
    with Timer("prep", verbose=True):
        
        
        gpu_batches = []
        active_num_final = 0
        
        # rough estimate
        chunk_size = int(batch_size * 1.5)
        
        raw_gen = _generate_jacobi_batches(h_integrals, modals, cutoff, chunk_size)
        buffered_gen = BatchRegularizer(raw_gen, target_size=batch_size, nmodes=nmodes)
        
        # consume data from the generator, and move them to device
        # NOTE: if data preparation becomes a bottleneck, it should be possible to subdivide this step into 
        # parallel CPU computations.
        for tm, tgt, i, j, c, m in buffered_gen:
            # move numpy arrays to JAX (GPU if available) immediately
            batch = VSCFData(
                term_map = jnp.array(tm),
                target_mode = jnp.array(tgt),
                i = jnp.array(i),
                j = jnp.array(j),
                coeffs = jnp.array(c),
                mask = jnp.array(m)
            )
            print(f"{tm.shape}, {tgt.shape}, {i.shape}, {j.shape}, {c.shape}, {m.shape}")
            gpu_batches.append(batch)
            
            active_num_final += batch_size
            
            # explicitly delete cpu arrays to free memory
            del tm, tgt, i, j, c, m
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


