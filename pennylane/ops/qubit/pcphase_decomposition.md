This document explains the algorithm behind the decomposition of `qml.PCPhase` gates.

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
and differences of powers of two, which is detailed in a separate section below. This
decomposition reads

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
the subspace we want ot add projectors to/subtract projectors from, allows us to
collect (multi-controlled) phase shift gates as the gate decomposition.
The sign of all phases is multiplied by $2\sigma$ to realize the first term of
$G=\sigma(2\Pi -\mathbb{I}_N)$, and a global phase with angle multiplied by
$\sigma$ is added to realize the second term.

## Decomposing an integer into powers of two

Here we decompose an integer $d<=2^{n-1}$ into additions and subtractions of the
smallest-possible number of powers of two.
This is equivalent to computing the minimal bit operations required to produce
a bit string by subtraction and addition of weight-one bit strings:


$$
d = \sum_{i=0}^{n-1} c_i 2^{n-1-i}, \quad c_i\in\{-1, 0, 1\}.
$$

The number of powers of two is given by the OEIS sequence
[A007302](https://oeis.org/A007302) and can be computed as
`(d ^ (3 * d)).bit_count()`. Note that we use zero-based indexing.

### Algorithm

The algorithm works in the following steps, exploiting the insight from
[this post](https://stackoverflow.com/questions/57797157/minimum-number-of-powers-of-2-to-get-an-integer/57805889#57805889>) on stack overflow that two bits of $d$ can be set
correctly at a time. That is, given two bits of the target $d$ and the
corresponding two bits of some intermediate working bit string $s$ that we try
to set to $d$, we have 16 scenarios. We will only need to think about those
scenarios where the less significant bit already differs between $d$ and
$s$. In each such scenario, a single addition or subtraction to $s$
suffices to set the bits of $s$ to those of $d$:

| Bits | d=00 | d=01 | d=10 | d=11 |
|------|------|------|------|------|
| s=00 | 0    | +1   | 0    | -1   |
| s=01 | -1   | 0    | +1   | 0    |
| s=10 | 0    | -1   | 0    | +1   |
| s=11 | +1   | 0    | -1   | 0    |

Now let's lay out the algorithm itself. We will use this operation in step 3 below.

1. Set $s=0=0\dots 0_2$ to be the zero bit string of length $n$.

2. Initialize $i=n-1$ and an empty list $R=[]$ that will hold 2-tuples, each
   containing an exponent and a factor.

3. If $d_i=s_i$, insert $(i, 0)$ at the beginning of $R$.
   Else, i.e., if $d_i=1-s_i$, set $\sigma=1$ if $i<2$ and otherwise
   look at $d_{i-1}$ and select the sign $\sigma$ from the table above that
   will make $s_{i-1}s_i$ match $d_{i-1}d_i$.
   Insert $(i, \sigma)$ at the beginning of $R$ and add $\sigma 2^{n-i-1}$
   to $s$.

4. If $i=0$, terminate. Otherwise, update $i=i-1$ and go to step 3.

### Calculation example

Take $n=5$ and $d=7=00111_2$. If we only allowed for addition of powers of two,
we would simply aggregate the single non-zero bits $00001_2$, $00010_2$ and
$00100_2$, requiring three powers of two to be added. However, including substraction
allows us to decompose $d$ as $d=8-1=01000_2 - 00001_2$ instead, only
requiring two powers of two.
To compute this result, we go through the steps from above:

1. Set $s=00000_2$.

2. Initialize $i=n-1=4$ and $R=[]$.

3. $d_4=1=1-s_4$, so look at $d_3=1$ as well. The sign at $(00, 11)$
   in the table above is $-1$, so we insert $(4, -1)$
   and compute $s=00000_2 - 00001_2=11111_2$.

4. Update $i=4-1=3$.

5. $d_3=1=s_3$, so we insert $(3, 0)$ and move on.

6. Update $i=3-1=2$.

7. $d_2=1=s_2$, so we insert $(2, 0)$ and move on.

8. Update $i=2-1=1$.

9. $d_1=0=1-s_1$, and $i=1<2$, so we insert $(1, +1)$
   and compute $s=11111_2 + 01000_2=00111_2=d$.

10. Update $i=1-1=0$.

11. $d_0=0=s_0$, so we insert $(0, 0)$ and move on.

12. we have $i=0$, so we terminate.

Overall, this creates the list $R=[(0,0), (1,+1), (2, 0), (3, 0), (4, -1)]$,
corresponding to the decomposition we were after,
$d=2^{n-1-1} - 2^{n-4-1}=2^3-2^0=8-1$.

In the code below, we will actually *skip* the first entry of the 2-tuples in
$R$, because they always equal the position of the tuple within $R$
(after completing the algorithm).

As a second example, consider the number
$k=121_{10}=01111001_2$, which can be (trivially) decomposed into a sum of
five powers of two by reading off the bits:
$k = 2^6 + 2^5 + 2^4 + 2^3 + 2^0 = 64 + 32 + 16 + 8 + 1$.
The decomposition here, however, allows for minus signs and achieves the decomposition
$k = 2^7 - 2^3 + 2^0 = 128 - 8 + 1$, which only requires three powers of two.
