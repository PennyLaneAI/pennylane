# Pauli word sparse representation

Pauli words, a.k.a. Pauli strings, are operators resulting from the tensor products of Pauli matrices or the identity. Namely:

$$
I = \begin{pmatrix}
     1 & 0 \\
     0 & 1
    \end{pmatrix}, \quad
X = \begin{pmatrix}
     0 & 1 \\
     1 & 0
    \end{pmatrix}, \quad
Y = \begin{pmatrix}
     0 & -i \\
     i & 0
    \end{pmatrix}, \quad
Z = \begin{pmatrix}
     1 & 0 \\
     0 & -1
    \end{pmatrix}.
$$

We can exploit their structure in combination with the tensor product to efficiently compute the sparse matrices of the resulting operators.

Before we dive into the details, let's do a brief recap on sparse matrices. Here, we focus on compressed sparse row (CSR) matrices, which are described by three one-dimensional arrays:
- $\texttt{val}$: contains the value of non-zero (NNZ) entries with shape `(NNZ,)`.
- $\texttt{col}$: contains the column indices of $\texttt{val}$ with shape `(NNZ,)`.
- $\texttt{row}$: contains the cumulative number of row-wise NNZ entries starting from zero with shape `(rows+1,)`.

> In scipy, $\texttt{val}$ is $\texttt{data}$, $\texttt{col}$ is $\texttt{indices}$, and $\texttt{row}$ is $\texttt{indptr}$.

If we look at the CSR representation of Pauli matrices, we already see some clear patterns in their sparse strucutre:

$$\begin{align*}
I &= \begin{cases}
      \texttt{val}=\left[1, 1\right]\\
      \texttt{col}=\left[0, 1\right]
     \end{cases} \\
X &= \begin{cases}
      \texttt{val}=\left[1, 1\right]\\
      \texttt{col}=\left[1, 0\right]
     \end{cases} \\
Y &= \begin{cases}
      \texttt{val}=\left[-i, i\right]\\
      \texttt{col}=\left[1, 0\right]
     \end{cases} \\
Z &= \begin{cases}
      \texttt{val}=\left[1, -1\right]\\
      \texttt{col}=\left[0, 1\right]
     \end{cases}
\end{align*}$$

$Z$ and $I$ have the same structure, and the same happens with $X$ and $Y$. All of them have the same $\texttt{row}=[0, 1, 2]$, as all the matrices have a unique NNZ element in every row.

Let's look at a simple example to illustrate the properties of the tensor product of Pauli matrices. Consider the Pauli word $XZI$, which can be created in PennyLane with `qp.pauli.PauliWord({0: "X", 1: "Z", 2: "I"})`. The corresponding operator is obtained by taking the tensor product of its elements:

$$
XZI = \begin{pmatrix}
       0 & 1\\
       1 & 0
      \end{pmatrix}
      \otimes \begin{pmatrix}
       1 & 0\\
       0 & -1
      \end{pmatrix}
      \otimes \begin{pmatrix}
       1 & 0\\
       0 & 1
      \end{pmatrix}
$$

Starting from the right, our initial matrix is $I$ with $\texttt{val}=\left[1, 1\right]$ and $\texttt{col}=\left[0, 1\right]$ (let us forget about $\texttt{row}$ for now). If we apply $Z$, we obtain:

$$
ZI = \begin{pmatrix}
      1 & 0\\
      0 & -1
     \end{pmatrix}
     \otimes
     \begin{pmatrix}
      1 & 0\\
      0 & 1
     \end{pmatrix}
     = 
     \left(\begin{array}{cc|cc}
      1 & 0 & & \\ 
      0 & 1 & & \\ 
     \hline 
      & & -1 & 0 \\
      & & 0 & -1
     \end{array}\right).
$$

The resulting CSR representation is $\texttt{val}=\left[1, 1, -1, -1\right]$ and $\texttt{col}=\left[0, 1, 2, 3\right]$.

```python
import numpy as np
import scipy

I = np.array([[1, 0], [0, 1]])
Z = np.array([[1, 0], [0, -1]])
X = np.array([[0, 1], [1, 0]])

ZI = np.kron(Z, I)
ZI_CSR = scipy.sparse.csr_matrix(ZI)
print(ZI)
print(f"val: {ZI_CSR[ZI_CSR.nonzero()].tolist()[0]}\ncol: {ZI_CSR.indices}")

[[ 1  0  0  0]
 [ 0  1  0  0]
 [ 0  0 -1  0]
 [ 0  0  0 -1]]
val: [1, 1, -1, -1]
col: [0 1 2 3]
```

Every time we perform a tensor product with a new Pauli matrix, we double the matrix dimensions ($\mathbb{C}^{m\times m}\to \mathbb{C}^{2m\times2m}$). Given their nature, we always replicate the structure of our existing matrix in blocks either along the diagonal (like in this case) or the anti-diagonal (like in the next operation). This means we always have as many NNZ elements as columns, and there is always one per column and row (hence why we ignore $\texttt{row}$). Thus, our $\texttt{val}$ and $\texttt{col}$ arrays double in size with every operation and the effect of each matrix in our sparse representation is well defined and consistent.

In this case, $Z$ extends $\texttt{val}$ by replicating it with a negative sign. $\texttt{col}$ is extended by replicating it and adding the dimension $m$ to values (see below for a summary of matrix effects).

Now if we apply $X$, we obtain:

$$ XZI = \begin{pmatrix}
          0 & 1\\
          1 & 0
         \end{pmatrix}
         \otimes 
         \begin{pmatrix}
          1 & 0 & 0 & 0 \\ 
          0 & 1 & 0 & 0 \\ 
          0 & 0 & -1 & 0 \\
          0 & 0 & 0 & -1
         \end{pmatrix} 
         = 
         \left(\begin{array}{cccc|cccc}
          & & & & 1 & 0 & 0 & 0 \\ 
          & & & & 0 & 1 & 0 & 0 \\
          & & & & 0 & 0 & -1 & 0 \\
          & & & & 0 & 0 & 0 & -1 \\
         \hline 
          1 & 0 & 0 & 0 & & & & \\
          0 & 1 & 0 & 0 & & & & \\
          0 & 0 & -1 & 0 & & & & \\
          0 & 0 & 0 & -1 & & & & \\
         \end{array}\right).$$

The new CSR representation is $\texttt{val}=\left[1, 1, -1, -1, 1, 1, -1, -1\right]$ and $\texttt{col}=\left[4, 5, 6, 7, 0, 1, 2, 3\right]$.
In this case, $X$ has simply duplicated the $\texttt{val}$, and then we replicate $\texttt{col}$ and add the dimension $m$ to the first indices.

```python
XZI = np.kron(X, ZI)
XZI_CSR = scipy.sparse.csr_matrix(XZI)
print(XZI)
print(f"val: {XZI_CSR[XZI_CSR.nonzero()].tolist()[0]}\ncol: {XZI_CSR.indices}")

[[ 0  0  0  0  1  0  0  0]
 [ 0  0  0  0  0  1  0  0]
 [ 0  0  0  0  0  0 -1  0]
 [ 0  0  0  0  0  0  0 -1]
 [ 1  0  0  0  0  0  0  0]
 [ 0  1  0  0  0  0  0  0]
 [ 0  0 -1  0  0  0  0  0]
 [ 0  0  0 -1  0  0  0  0]]
val: [1, 1, -1, -1, 1, 1, -1, -1]
col: [4 5 6 7 0 1 2 3]
```

In general, we take our current $\texttt{val}$ and replicate it multiplying by the Pauli operator's $\texttt{val}_P$, such that $\texttt{val}\leftarrow[\texttt{val}, \texttt{val}]\cdot\texttt{val}_P$ (assuming $\texttt{val}_P$ broadcasts accordingly).
For the $\texttt{col}$, we replicate it and add the previous matrix dimension $m$ times the operator's $\texttt{col}_P$, such that $\texttt{col}\leftarrow[\texttt{col}, \texttt{col}] + m\times\texttt{col}_P$.
Knowing the number of terms $n$, we can directly compute the resulting $\texttt{row}$ as $\texttt{row} = [0, 1, \dots, 2^n]$.
For long Pauli words ($n\gg1$), $\texttt{row}$ can be relatively demanding to generate, so we can cache it for repeated words with the same length.
Below, we list the explicit effect on the CSR representation of performing the tensor product with each Pauli matrix:

$$\begin{align*}
I &= \begin{cases}
      \texttt{val} \leftarrow [\texttt{val}, \texttt{val}]\\
      \texttt{col}\leftarrow [\texttt{col}, \texttt{col}+m]
     \end{cases} \\
X &= \begin{cases}
      \texttt{val} \leftarrow [\texttt{val}, \texttt{val}]\\
      \texttt{col}\leftarrow [\texttt{col}+m, \texttt{col}]
     \end{cases} \\
Y &= \begin{cases}
      \texttt{val} \leftarrow [-i\texttt{val}, i\texttt{val}]\\
      \texttt{col}\leftarrow [\texttt{col}+m, \texttt{col}]
     \end{cases} \\
Z &= \begin{cases}
      \texttt{val} \leftarrow [\texttt{val}, -\texttt{val}]\\
      \texttt{col}\leftarrow [\texttt{col}, \texttt{col}+m]
     \end{cases}
\end{align*}$$ 

# Pauli sentence sparse representation

Pauli sentences are linear combinations of Pauli words.
Again, we can exploit the properties of Pauli words to add their sparse representations in the most efficient way, which is the key aspect to compute the sparse matrix of Pauli sentences.

In general, adding two arbitrary sparse matrices requires us to check all their NNZ entries.
If they happen to be in the same position, we add their values together, otherwise, we add them as a new entry to the resulting matrix.
Therefore, we can exploit the information about the matching NNZ entries between two Pauli words to directly manipulate their data.

## Add matrices with the same sparse structure first

As we have seen before, $I$ and $Z$ have the same sparse structure (same $\texttt{col}$ and $\texttt{row}$), and so do $X$ and $Y$.
Thus, words with the same combination of $I$ or $Z$ with $X$ or $Y$ have the NNZ entries in the same positions.
This means that all their NNZ will be in the same positions, which allows us to directly add their $\texttt{val}$ without any additional consideration!

For example, consider the case of the Pauli word $XZI$ from above, with $\texttt{val}=\left[1, 1, -1, -1, 1, 1, -1, -1\right]$ and $\texttt{col}=\left[4, 5, 6, 7, 0, 1, 2, 3\right]$.
The Pauli word $YZZ$ has the same sparse structure:

$$ YZZ = \left(\begin{array}{cccc|cccc}
          & & & & -i & 0 & 0 & 0 \\ 
          & & & & 0 & i & 0 & 0 \\
          & & & & 0 & 0 & i & 0 \\
          & & & & 0 & 0 & 0 & -i \\
         \hline 
          i & 0 & 0 & 0 & & & & \\
          0 & -i & 0 & 0 & & & & \\
          0 & 0 & -i & 0 & & & & \\
          0 & 0 & 0 & i & & & & \\
         \end{array}\right),$$

with $\texttt{val}=\left[-i, i, i, -i, i, -i, -i, i\right]$ and $\texttt{col}=\left[4, 5, 6, 7, 0, 1, 2, 3\right]$.
The resulting matrix from $XZI + YZZ$ will have the same number of NNZ entries, preserving $\texttt{col}$ and $\texttt{row}$, and its $\texttt{val}$ will be the elementwise addition of the constituent $\texttt{val}$ arrays.

To identify Pauli words with the same sparse structure, we represnt the Pauli sentence as a matrix in which every row represents a Pauli word, and every column corresponds to a wire.
The entries denote the sparse structure of the Pauli operator acting on each wire: `0` for $I$ and $Z$, `1` for $X$ and $Y$.

For example:

$$XZI + YZZ + IIX + YYX \rightarrow
\begin{pmatrix}
1 & 0 & 0 \\
1 & 0 & 0 \\
0 & 0 & 1 \\
1 & 1 & 1
\end{pmatrix},
$$

which allows us to identify common patterns to add the matrices at nearly zero cost.

## Add matrices with different structure last

Given that Pauli words have a single NNZ per row and column, any pair of words that differ in (at least) one operator will not have any matching NNZ entry.

Hence, once we have added all the words with the same sparse structure, we can be certain that the intermediate results do not have any matching NNZ entries between themselves.
Again, we can exploit this information to add them by directly manipulating their representation data.

In this case, all the NNZ are new entries in the result matrix so we need to combine the $\texttt{col}$ and $\texttt{val}$ arrays of the summands.
In order to see this clearly, let us break down the addition of two small Pauli words with different structures, $XI$ and $ZZ$:

$$
\begin{align*}
XI &= \left(\begin{array}{cc|cc}
       & & 1 & 0 \\ 
       & & 0 & 1 \\ 
      \hline 
       1 & 0 & & \\
       0 & 1 & &
      \end{array}\right)
   &&= \begin{cases}
       \texttt{val}=\left[1, 1, 1, 1\right]\\
       \texttt{col}=\left[2, 3, 0, 1\right]
      \end{cases} \\
     \\
ZZ &= \left(\begin{array}{cc|cc}
       1 & 0 & & \\ 
       0 & -1 & & \\ 
      \hline 
       & & -1 & 0 \\
       & & 0 & 1
      \end{array}\right)
   &&= \begin{cases}
       \texttt{val}=\left[1, -1, -1, 1\right]\\
       \texttt{col}=\left[0, 1, 2, 3\right]
      \end{cases}
\end{align*}
$$

The resulting matrix is

$$
XI + ZZ = \begin{pmatrix}
           1 & 0 & 1 & 0 \\
           0 & -1 & 0 & 1 \\
           1 & 0 & -1 & 0 \\
           0 & 1 & 0 & 1
          \end{pmatrix}
          =
          \begin{cases}
           \texttt{val}=\left[1, 1, -1, 1, 1, -1, 1, 1\right]\\
           \texttt{col}=\left[0, 2, 1, 3, 0, 2, 1, 3\right]
          \end{cases}
$$

The result is the entry-wise concatenation of $\texttt{col}$ and $\texttt{val}$ sorted by $\texttt{col}$.
To do so, we start by arranging the $\texttt{col}$ arrays in a matrix and sorting them row-wise.

$$
\texttt{col}_{XI, ZZ} =
 \begin{bmatrix}
  2 & 0 \\
  3 & 1 \\
  0 & 2 \\
  1 & 3
 \end{bmatrix}
 \rightarrow 
 \begin{bmatrix}
  0 & 2 \\
  1 & 3 \\
  0 & 2 \\
  1 & 3
 \end{bmatrix}            
$$

Then, we concatenate the resulting matrix rows to obtain the final $\texttt{col}$.
We do the same with the $\texttt{val}$ arrays sorting them according to the $\texttt{col}$.
In `numpy` terms, we "take along axis" with the sorted $\texttt{col}$ indices.