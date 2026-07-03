# TCDQ Qudit Module: Technical Notes

These notes develop the theory behind the classically trainable qudit
circuit family implemented in this module.  The presentation follows
the mathematical narrative — circuit ↦ classical estimator ↦ training
loss — and points to the implementing functions where helpful.

Throughout, $d$ is the qudit dimension, $n$ is the number of qudits,
$\omega = e^{2\pi i/d}$ is the primitive $d$-th root of unity, and
$\oplus$ denotes componentwise addition modulo $d$ on $\mathbb{Z}_d^n$.

---

## 1. Circuit Architecture

### 1.1 From IQP to Fourier Circuits over $\mathbb{Z}_d$

The qubit IQP circuit has the form $F D(\boldsymbol{\theta}) F^\dagger |0\rangle$,
where $F$ is the $n$-fold Hadamard (the QFT over $\mathbb{Z}_2^n$) and
$D(\boldsymbol{\theta})$ is a diagonal unitary built from Pauli $Z$ tensor
products.

The key property that makes classical training possible is that
the Fourier-conjugated observable $F Z_{\mathbf{a}} F^\dagger = X_{\mathbf{a}}$
acts as a *permutation* in the computational basis, collapsing a double sum
over $4^n$ terms into a single sum over $2^n$ terms — an expectation value
that can be estimated by Monte Carlo sampling.

This structure generalises from $\mathbb{Z}_2$ to any abelian group.  For
the cyclic group $\mathbb{Z}_d$, the QFT is the $d$-point discrete Fourier
transform with characters $\chi_k(x) = \omega^{kx}$, and the construction
yields circuits on $n$ qudits whose expectation values are again classically
estimable via Monte Carlo.

### 1.2 Heisenberg–Weyl Observables

On a single qudit, define the shift and clock operators:

$$X|j\rangle = |j + 1 \bmod d\rangle, \qquad Z|j\rangle = \omega^j |j\rangle.$$

The **displacement operator** is

$$\mathcal{O}(l, m) = Z^l X^m  e^{-i\pi l m / d},$$

and its action on a computational basis state is:

$$\mathcal{O}(l, m)|z\rangle = \exp\left(\frac{i\pi}{d} l(2z + m)\right) |z \oplus m\rangle.$$

Under the QFT, displacement operators conjugate as
$F\mathcal{O}(l,m)F^\dagger = \mathcal{O}(m, -l)$, so the Fourier
transform swaps the $l$ and $m$ indices (up to a sign).

On $n$ qudits, we use the tensor product

$$\mathcal{O}(\mathbf{l}, \mathbf{m}) = \bigotimes_{i=1}^n \mathcal{O}(l_i, m_i).$$

Observables are specified as `(l_vecs, m_vecs)` arrays in
`QuditCircuitConfig.observables`.

### 1.3 The Diagonal Unitary

The Hermitian generator for a single qudit is

$$\mathcal{Q}(l, 0) = \chi\mathcal{O}(l, 0) + \chi^*\mathcal{O}^\dagger(l, 0), \qquad \chi = (1+i)/\sqrt{2}.$$

For commutativity, we restrict to $m = 0$ (pure $Z$-power gates).  On $n$
qudits, a generator vector $\mathbf{g} = (g_1, \dots, g_n) \in \mathbb{Z}_d^n$
defines the tensor product

$$\mathcal{Q}_{\mathbf{g}} = \bigotimes_{i=1}^n \mathcal{Q}(g_i, 0),$$

and the full diagonal unitary is

$$D(\boldsymbol{\theta}) = \prod_{\mathbf{g} \in \mathcal{G}} \exp(i\theta_{\mathbf{g}}\mathcal{Q}_{\mathbf{g}}).$$

Its eigenvalues on a computational basis state $|\mathbf{z}\rangle$ are
$D(\boldsymbol{\theta})|\mathbf{z}\rangle = \gamma_{\mathbf{z}}(\boldsymbol{\theta})|\mathbf{z}\rangle$,
where

$$\gamma_{\mathbf{z}}(\boldsymbol{\theta}) = \exp\left(i \sum_{\mathbf{g} \in \mathcal{G}} \theta_{\mathbf{g}}\Phi_{\mathbf{g}}(\mathbf{z})\right)$$

with the **gate eigenvalue function**

$$\Phi_{\mathbf{g}}(\mathbf{z}) = \prod_{j=1}^n \sqrt{2}\cos\left(\frac{2\pi g_j z_j}{d} + \frac{\pi}{4}\right).$$

The gate set $\mathcal{G}$ is specified via `QuditCircuitConfig.gates`;
`_parse_qudit_generator_dict` flattens this into the generator and
parameter-map arrays used by the rest of the module.

---

## 2. Classically Estimating Expectation Values

### 2.1 The Core Identity

The circuit $U(\boldsymbol{\theta}) = F^{\otimes n\dagger} D(\boldsymbol{\theta}) F^{\otimes n}$
acts on the input state $|0\rangle$.  By conjugating through the QFT and
inserting resolutions of the identity, the expectation value of
$\mathcal{O}(\mathbf{l}, \mathbf{m})$ reduces to a single sum:

$$\langle \mathcal{O}(\mathbf{l}, \mathbf{m}) \rangle = \mathbb{E}_{\mathbf{z} \sim U}\left[\exp\left(\frac{i\pi}{d}\mathbf{m} \cdot (2\mathbf{z} - \mathbf{l})\right) \exp\left(i \sum_{\mathbf{g}} \theta_{\mathbf{g}}\big(\Phi_{\mathbf{g}}(\mathbf{z}) - \Phi_{\mathbf{g}}(\mathbf{z} \ominus \mathbf{l})\big)\right)\right]$$

where $\mathbf{z}$ is drawn uniformly from $\mathbb{Z}_d^n$ and
$\ominus$ denotes componentwise subtraction modulo $d$.  The two
exponential factors are:

1. **The observable phase** $J = \exp\big(\frac{i\pi}{d}\mathbf{m} \cdot (2\mathbf{z} - \mathbf{l})\big)$, determined by the choice of observable (computed by `_obs_phase_matrix`).
2. **The accumulated phase difference** $e^{iE}$ where $E = \sum_{\mathbf{g}} \theta_{\mathbf{g}}(\Phi_{\mathbf{g}}(\mathbf{z}) - \Phi_{\mathbf{g}}(\mathbf{z} \ominus \mathbf{l}))$, determined by the circuit parameters (assembled by `_accumulate_phase_diffs`).

Since this is an expectation of a bounded integrand, it can be estimated to
inverse polynomial precision via a sample mean over $s$ uniformly random
dit-strings.

### 2.2 Batching over Observables

To estimate a batch of $|\mathcal{O}|$ observables simultaneously, we
construct $|\mathcal{O}| \times s$ matrices $J$ and $E$ and combine them:

$$\{\langle \mathcal{O}(\mathbf{l}_i, \mathbf{m}_i) \rangle\}_i = \text{mean}_1\big[J \odot e^{iE}\big]$$

where $\odot$ is elementwise multiplication and $\text{mean}_1$ averages
over the sample axis (columns).

### 2.3 The Angle-Addition Expansion

The computational bottleneck is building $E$.  A naïve approach would
evaluate $\Phi_{\mathbf{g}}(\mathbf{z} \ominus \mathbf{l})$ for every
gate–sample–observable triple, but the **angle-addition identity** enables
a factored matrix formulation.

#### Trigonometric building blocks

Define the cos/sin factors:

$$c(v, u) = \sqrt{2}\cos\left(\frac{2\pi v u}{d} + \frac{\pi}{4}\right), \qquad \bar{s}(v, u) = \sqrt{2}\sin\left(\frac{2\pi v u}{d} + \frac{\pi}{4}\right)$$

so that $\Phi_{\mathbf{g}}(\mathbf{z}) = \prod_{j \in S_{\mathbf{g}}} c(g_j, z_j)$ where $S_{\mathbf{g}} = \{j : g_j \neq 0\}$ is the **support** (set of active qudits) of gate $\mathbf{g}$.  These are computed by `_compute_trigonometric_building_blocks`.

#### Expanding the shifted eigenvalue

Applying the angle-addition formula $c(v, u-l) = c(v,u)\cos\alpha + \bar{s}(v,u)\sin\alpha$ (with $\alpha = 2\pi v l/d$) to each factor and expanding the product gives $2^\omega$ terms:

$$\Phi_{\mathbf{g}}(\mathbf{z} \ominus \mathbf{l}) = \sum_{\boldsymbol{\sigma} \in \{0,1\}^\omega} \prod_{k=1}^\omega T_{\sigma_k}(g_k, z_k) \cdot A_{\sigma_k}(g_k, l_k)$$

where $\omega = |S_{\mathbf{g}}|$ is the **weight** of the gate (number of active qudits), and

$$T_0 = c, \quad T_1 = \bar{s}, \qquad A_0 = \cos(\tfrac{2\pi vl}{d}), \quad A_1 = \sin(\tfrac{2\pi vl}{d}).$$

The enumeration of all $2^\omega$ terms is done by `_expand_angle_addition`.

#### Weight groups and matrix factorisation

Each term in the expansion factors into a product of sample-dependent
quantities times observable-dependent quantities.  Grouping gates by
weight $\omega$, define the **factor matrices**:

$$[\mathbf{B}^\omega_{\boldsymbol{\sigma}}]_{\mathbf{g},\mathbf{z}} = \prod_{k=1}^\omega T_{\sigma_k}(g_k, z_k), \qquad [\mathbf{C}^\omega_{\boldsymbol{\sigma}}]_{\mathbf{g},\mathbf{l}} = \prod_{k=1}^\omega A_{\sigma_k}(g_k, l_k)$$

These are stored in the `WeightGroupData` named tuple — `samples_matrices`
for the sample-side factors and `obs_matrices` for the observable-side
factors.  The accumulated phase-difference matrix for weight group $\omega$
is then:

$$E_\omega = \boldsymbol{\theta}_\omega \cdot \tilde{\mathbf{B}}_\omega - \sum_{\boldsymbol{\sigma}}\mathbf{C}_{\boldsymbol{\sigma}}^T\text{diag}(\boldsymbol{\theta}_\omega)\mathbf{B}_{\boldsymbol{\sigma}}$$

The first term uses the all-cos entry
($\boldsymbol{\sigma} = \mathbf{0}$), which equals the gate eigenvalue
itself. The total is $E = \sum_\omega E_\omega$.

For $\omega = 1$ (single-qudit gates) this is 2 matrix products; for $\omega = 2$ (two-qudit gates) it is 4; in general $2^\omega$ per weight group.  The per-group pipeline is orchestrated by `_build_weight_group`; `_accumulate_phase_diffs` sums $E_\omega$ over all weight groups.

---

## 3. General Input States

The default input state is $|0\rangle$, but the estimator supports arbitrary
sparse input states

$$|\psi_{\text{in}}\rangle = \sum_{\mathbf{x} \in \mathcal{X}} \psi_{\mathbf{x}} |\mathbf{x}\rangle$$

where $\mathcal{X} \subset \mathbb{Z}_d^n$ is the support and $N = |\mathcal{X}|$
must not be exponentially large.  Following the same derivation (inserting
resolutions of the identity and using the character homomorphism
$\omega^{(\mathbf{z} \oplus \mathbf{l}) \cdot \mathbf{x}} = \omega^{\mathbf{z} \cdot \mathbf{x}}\omega^{\mathbf{l} \cdot \mathbf{x}}$),
the expectation value becomes:

$$\langle \mathcal{O}(\mathbf{l}, \mathbf{m}) \rangle = \mathbb{E}_{\mathbf{z} \sim U}\left[\exp\left(\frac{i\pi}{d}\mathbf{m}\cdot(2\mathbf{z} - \mathbf{l})\right) e^{i\Delta_{\boldsymbol{\theta}}^{\mathbf{l}}(\mathbf{z})} \left(\sum_{\mathbf{x}} \psi_{\mathbf{x}}^* \omega^{(\mathbf{l} - \mathbf{z})\cdot\mathbf{x}}\right)\left(\sum_{\mathbf{y}} \psi_{\mathbf{y}}\omega^{\mathbf{z}\cdot\mathbf{y}}\right)\right]$$

The two bracketed sums form the **initial-state correction factor** $H$,
which multiplies the default-input-state integrand element-wise.

### Matrix formulation of $H$

Let $X \in \mathbb{Z}_d^{N \times n}$ be the support matrix and
$\Psi \in \mathbb{C}^N$ the amplitude vector.  Define:

$$\widetilde{\Psi}^{(2)} = \omega^{Z \cdot X^T} \cdot \Psi \qquad (s \times 1)$$

$$F = \Psi^* \cdot \mathbf{1}_{1 \times s} \odot \omega^{-X \cdot Z^T} \qquad (N \times s)$$

$$\widetilde{\Psi}^{(1)} = \omega^{L \cdot X^T} \cdot F \qquad (|\mathcal{O}| \times s)$$

The correction matrix is then:

$$H = \widetilde{\Psi}^{(1)} \odot \left(\mathbf{1}_{|\mathcal{O}| \times 1} \cdot (\widetilde{\Psi}^{(2)})^T\right)$$

and the full integrand becomes $\text{mean}_1[J \odot e^{iE} \odot H]$.

The complexity of the input-state correction is $\mathcal{O}(|\mathcal{O}| \cdot N \cdot s)$.  When the number of observables exceeds the support size, this cost is subleading.  This construction is implemented in `_compute_initial_state_correction`.

---

## 4. Monte Carlo Statistics

The integrand $Y$ is a complex $|\mathcal{O}| \times s$ matrix.  For each
observable, the estimator (via `_compute_mc_statistics`) returns:

- **expvals**: The sample mean $\hat{\mu} = \frac{1}{s}\sum_r Y_r$.
- **cov**: The $2 \times 2$ covariance matrix of the mean estimator (real and imaginary parts), computed as the sample covariance divided by $s$.
- **mean_y_sq** (optional): $\frac{1}{s}\sum_r |Y_r|^2$.  This quantity is needed for the unbiased QQ correction in the MMD loss (§5.3).

---

## 5. Graph-Kernel MMD Loss

### 5.1 Graph-Spectral Kernels on $\mathbb{Z}_d^n$

To train the circuit to match a target data distribution, we use the squared
Maximum Mean Discrepancy (MMD) with a kernel derived from a graph Laplacian.

Let $G_i$ be a graph on $d$ vertices encoding closeness of category labels
at site $i$, with Laplacian eigenvalues $\lambda_k^{(i)}$.  The Cartesian
product graph $G = G_1 \square \cdots \square G_n$ has joint eigenvalues
$\Lambda_{\mathbf{l}} = \sum_i \lambda_{l_i}^{(i)}$.  A spectral kernel
$K = f(L)$ with $f(\lambda) = e^{-t\lambda}$ (heat kernel) gives

$$K(\mathbf{x}, \mathbf{y}) = \sum_{\mathbf{l}} \hat{\kappa}(\mathbf{l})u_{\mathbf{l}}(\mathbf{x})u_{\mathbf{l}}(\mathbf{y}), \qquad \hat{\kappa}(\mathbf{l}) = e^{-t\Lambda_{\mathbf{l}}} \geq 0.$$

When each $G_i$ is **vertex-transitive** (e.g. cycle $C_d$ or complete
graph $K_d$), its Laplacian is diagonalised by the characters of
$\mathbb{Z}_d$, the kernel becomes shift-invariant, and the graph-Fourier
moments coincide with the Heisenberg–Weyl expectations:

$$\tilde{p}(\mathbf{l}) \propto \mathbb{E}_{\mathbf{x} \sim p}[\omega^{\mathbf{l} \cdot \mathbf{x}}] = \langle \mathcal{O}(\mathbf{l}, 0) \rangle_p.$$

The squared MMD then diagonalises into a mixture of squared moment
differences:

$$\text{MMD}^2(p, q) = \mathbb{E}_{\mathbf{l} \sim \mathcal{P}}\left[|\langle \mathcal{O}(\mathbf{l}, 0)\rangle_p - \langle \mathcal{O}(\mathbf{l}, 0)\rangle_q|^2\right]$$

where $\mathcal{P}(\mathbf{l}) \propto \hat{\kappa}(\mathbf{l})$ is a
probability distribution over Fourier indices.  This is the key equation:
the MMD loss reduces to estimating the *same* displacement-operator
expectations that the circuit naturally computes, with $\mathbf{m} = 0$.

### 5.2 Spectral Sampling Distributions

The distribution $\mathcal{P}$ factorises across sites:
$\mathcal{P}(\mathbf{l}) = \prod_i \mathcal{P}_1(l_i)$, where the per-site
marginal depends on the graph choice.

**Cycle graph $C_d$** (`_cycle_marginal_probs`)**:**  The Laplacian
eigenvalue at index $k$ is $\lambda_k = 4\sin^2(\pi k/d)$, giving

$$\mathcal{P}_1(k) = \frac{e^{-4t\sin^2(\pi k/d)}}{Z_1}, \qquad Z_1 = \sum_{k=0}^{d-1} e^{-4t\sin^2(\pi k/d)}.$$

**Complete graph $K_d$** (`_complete_marginal_probs`)**:**  The Laplacian has
eigenvalue 0 for $k=0$ and $d$ for all $k \neq 0$:

$$\mathcal{P}_1(0) = \pi_0, \qquad \mathcal{P}_1(k \neq 0) = \frac{1 - \pi_0}{d - 1}, \qquad \pi_0 = \frac{1}{1 + (d-1)e^{-td}}.$$

To estimate the loss, we sample a batch of $|\mathcal{L}|$ Fourier indices
$\mathbf{l} \sim \mathcal{P}$ by drawing each component independently from
the $d$-way categorical marginal (`_sample_fourier_indices`).

### 5.3 Unbiased MMD Estimator

Expanding $|\mu_p - \mu_q|^2 = |\mu_p|^2 - 2\text{Re}(\mu_p^* \mu_q) + |\mu_q|^2$, each term is estimated separately.

#### PP term (data–data)

From the dataset $\mathcal{X} = \{\mathbf{x}_1, \dots, \mathbf{x}_m\}$,
the empirical moment is
$\hat{\mu}_p(\mathbf{l}) = \frac{1}{m}\sum_i \omega^{\mathbf{l} \cdot \mathbf{x}_i}$
(computed by `_empirical_fourier_moments`).
The naive $|\hat{\mu}_p|^2$ includes diagonal ($i = i'$) self-pairs.  Since
$|\omega^{\mathbf{l}\cdot\mathbf{x}}| = 1$, the U-statistic correction
(`_pp_term`) is:

$$\text{PP}(\mathbf{l}) = \frac{m|\hat{\mu}_p(\mathbf{l})|^2 - 1}{m - 1}$$

#### PQ term (data–model cross)

Since data and circuit samples are independent, no diagonal correction is
needed (`_pq_cross_term`):

$$\text{PQ}(\mathbf{l}) = 2\text{Re}\left(\hat{\mu}_p(\mathbf{l})^*\hat{\mu}_q(\mathbf{l})\right)$$

#### QQ term (model–model)

The circuit Monte Carlo produces per-sample values $y_r(\mathbf{l})$ with
$\hat{\mu}_q = \frac{1}{s}\sum_r y_r$.  Unlike the data side, the
per-sample values may not have unit modulus (especially with non-trivial
input states), so the diagonal correction uses the observed second moment
(`_qq_term`):

$$\text{QQ}(\mathbf{l}) = \frac{s|\hat{\mu}_q(\mathbf{l})|^2 - \bar{|y|^2}(\mathbf{l})}{s - 1}$$

where $\bar{|y|^2} = \frac{1}{s}\sum_r |y_r|^2$.  Gradients are stopped
through $\bar{|y|^2}$ so that optimisation differentiates through the mean,
not the variance correction.

#### Combining the terms

The unbiased MMD estimator averages over the sampled Fourier indices:

$$\widehat{\text{MMD}^2_u} = \frac{1}{|\mathcal{L}|}\sum_j \left[\text{PP}(\mathbf{l}_j) - \text{PQ}(\mathbf{l}_j) + \text{QQ}(\mathbf{l}_j)\right]$$

This combination is assembled by `_unbiased_mmd_squared`; the public entry
point `build_qudit_mmd_loss` returns a loss function that orchestrates the
full pipeline — sampling Fourier indices, estimating moments, and combining
the PP/PQ/QQ terms — for one or more heat-kernel bandwidths.
