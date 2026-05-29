This document explains the algorithm behind the decomposition of `qp.PCPhase` gates.

# Algorithm description

For this decomposition, we will decompose the generator of the `PCPhase` gate,
which we denote as $G$, into multiple projectors, which generate (multi-controlled)
phase shift gates. The generator of `PCPhase` is given as 


$$
G = 2 \left(\sum_{j=0}^{d-1} |j\rangle\langle j|\right) - \mathbb{I}_N
= \text{diag}(
\underset{d\text{ times}}{\underbrace{1, \dots, 1}},
\underset{(N-d)\text{ times}}{\underbrace{-1, \dots, -1}}
),
$$

where we denoted the overall dimension as $N=2^n$ for $n$ qubits and the
subspace dimension as $d$, which is called `dim` in code.
Our first step is to move to a slightly different convention in order to arrive at a
*projector* that has ones and zeros, instead of ones and
minus ones (note how $G$ is not a projector because $G^2=\mathbb{I}_N\neq G$).
To do this, we define

$$
\Pi = \begin{cases}
    \sum_{j=0}^{d-1} |j\rangle\langle j| & \text{for }\ d\leq \tfrac{N}{2}\\
    \sum_{j=d}^{N-1} |j\rangle\langle j| & \text{for }\ d> \tfrac{N}{2}
\end{cases},
$$

so that $G=2\Pi -\mathbb{I}_N$ for $d\leq \tfrac{N}{2}$ and
$G=\mathbb{I}_N - 2 \Pi$ for $d>\tfrac{N}{2}$. We may summarize
this as $G=\sigma (2\Pi -\mathbb{I}_N)$ with the sign $\sigma=\pm 1$
defined accordingly for the two cases.

Now we want to decompose $\Pi$, a sum of the computational basis state projectors
$|j\rangle\langle j|$ onto the $d$ consecutive smallest (largest, for
$\sigma=-1$) basis states,
into projectors that contain a power of two basis state projectors. We will see below that
these are exactly the generators of (multi-controlled) phase shift gates, allowing us
to implement the decomposition. As we may flip the sign of these individual phase shift
gates, we may also use subtraction when decomposing $\Pi$.

This decomposition of $\Pi$ is based on decomposing the integer $d$ into sums
and differences of powers of two, which is detailed in 
`pennylane/math/decomp_int_to_powers_of_two.md`. Such a decomposition reads

$$
d = \sum_{i=0}^{n-1} c_i 2^{n-1-i}, \quad c_i\in\{-1, 0, 1\}.
$$

Now we need to use this decomposition to combine projectors of $2^{n-1-i}$
computational basis state projectors into the desired $\Pi$.
We start with the scenario $\sigma=+1$, i.e., $d\leq \tfrac{N}{2}$, in which
case this decomposition has the form

$$
\Pi = \sum_{i=0}^{n-1} c_i \sum_{j=s_i}^{e_i} |j\rangle\langle j|,
$$

where $s_i$ and $e_i$ are the start and end positions
for the $i$\ th contribution to $\Pi$.
Besides computing the integer decomposition itself, this decomposition function is mostly
about figuring out the right control structure to realize the correct positioning
according to $s_i$ and $e_i$.

To begin, initialize empty lists of control wires and control values, corresponding
to $s_0=0$. Then, starting at $i=0$, which also marks the current wire,
go through the integer decomposition $\{c_i\}$ from above and perform the
following steps:

1. If $c_i=0$, go to step 2 immediately. Else, we want to
   add/subtract computational basis state projectors to the decomposition. The number of
   projectors $2^{n-1-i}$ is stored in the number of control wires that we already
   aggregated. In addition, $s_i$ (and thus $e_i$) is encoded in the control
   values, up to whether the projectors should be located in the $|0\rangle$
   or $|1\rangle$ subspace of the current target wire $i$. If we want to
   add projectors ($c_i=+1$), we append at the beginning of the current subspace,
   so in the $|0\rangle$ subspace. If we want to remove projectors ($c_i=-1$),
   we need to remove them from the end of the subspace, so from $|1\rangle$.
   Having figured out the subspace, we add the corresponding projector to the
   decomposition. In code, we immediately apply the corresponding phase gate, see below.

2. Add the current wire $i$ to the control wires. The control value to be added
   depends on the current $c_i$ and on the next non-zero coefficient, $d_i$.
   For $d_i=+1$, we are going to add more projectors, so we should control into
   the $|1\rangle$ subspace (add onto just added projectors for $c_i=+1$,
   or re-add just removed projectors for $c_i=-1$) unless we have $c_i=0$,
   in which case we
   did not just modify the decomposition and need to control into the $|0\rangle$
   subspace instead. In either case, this sets $s_{i+1}$ to $e_i+1$.
   For $d_i=-1$, we will subtract projectors. If we just subtracted projectors (from
   the $|1\rangle$ subspace, $c_i=-1$), we need to remove further from the
   $|0\rangle$ subspace and control into it. If we just added projectors (to the
   $|0\rangle$ subspace), we remove some of those again, and need to control into
   the $|0\rangle$ subspace as well. For $c_i=0$, we need to remove from
   the upper end, i.e., the $|1\rangle$ subspace.
   All of this is easily summarized: If the next non-zero coefficient, $d_i$,
   is positive (negative), set the next control value to $1$ ($0$). If
   the current coefficient is $c_i=0$, flip the control value.

3. Increment $i$ by one. If $i=n$, terminate, else go to step 1.

If $\sigma=-1$ instead, i.e., $d>\tfrac{N}{2}$, the start and end points
$s_i$ and $e_i$ need to be mapped via $x\mapsto N-1-x$. This
implies a change of the subspace acted on in step 1 above, and a change of the control
value in step 2 above, both via Boolean negation, i.e., $x\mapsto 1-x$.

To conclude, note that the control structure above, together with the information of
the subspace we want to add projectors to/subtract projectors from, allows us to
collect (multi-controlled) phase shift gates as the gate decomposition.
The sign of all phases is multiplied by $2\sigma$ to realize the first term of
$G=\sigma(2\Pi -\mathbb{I}_N)$, and a global phase with angle multiplied by
$\sigma$ is added to realize the second term.
