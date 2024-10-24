# Contraction Methods for Applying a Generic Channel to a Density Matrix

## Background

1. **Density Matrix (DM)**: Also known as a mixed state, a density matrix is characterized by:
   - Trace equal to 1
   - Positive semi-definite
   - Symmetric

   The trace of its square represents the purity of the quantum state.

2. **Channels**: A generalization of quantum operations, typically:
   - Completely positive
   - Trace preserving

   Channels can be represented by a set of Kraus operators. A single operator acting on the states can be viewed as a special case of a channel.

3. **Channel Action**: The effect of a channel on a density matrix is given by:

   $$
   \mathcal{E}(\rho) = \sum_{i} A_i \rho A_i^{\dagger}
   $$

   Where:
   - $\rho$ is the density matrix
   - $A_i$ are the Kraus operators

## Implementation Methods

### Einsum Contraction

Our `einsum` implementation is visualized as follows:

![Einsum Contraction Diagram](kraus.drawio.svg)

Where:
- $\rho$ represents the density matrix
- $O$ denotes a Kraus operator
- $O^\dagger$ is the conjugate of the Kraus operator

The order of contraction is automatically determined by `numpy.einsum` itself.

### Tensordot Contraction

The `tensordot` implementation follows a similar approach, with one key difference:
- The summation over the indices $i$ of the Kraus operators is performed outside the contraction operation.

This method maintains the core functionality while potentially offering performance benefits in certain scenarios.

## Conclusion

Both implementations effectively apply a generic channel to a density matrix, with slight variations in how the contractions and summations are performed. The choice between `einsum` and `tensordot` may depend on specific performance requirements or computational framework preferences.

!! Note that further discusion on the performance of these methods is awaiting.