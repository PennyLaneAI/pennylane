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

Let's look at a simple example to illustrate the properties of the tensor product of Pauli matrices. Consider the Pauli word $XZI$, which can be created in PennyLane with `qml.pauli.PauliWord({0: "X", 1: "Z", 2: "I"})`. The corresponding operator is obtained by taking the tensor product of its elements:

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
