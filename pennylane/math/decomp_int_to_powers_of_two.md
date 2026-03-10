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
