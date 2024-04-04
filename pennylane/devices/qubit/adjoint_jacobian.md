# The adjoint jacobian algorithm

## Full jacobian with expectation values

For a statevector,

$$
| \Psi \rangle = U_n(x_n) \dots U_i (x_i) \dots U_0(x_0) |0\rangle
$$

we have the derivative of an expectation value of an observable $M$,

$$
\frac{\partial}{\partial x_i} \langle \Psi | M | \Psi \rangle
$$

$$
= \langle 0| \dots \frac{\partial U_i^{\dagger}}{\partial x_i} \dots M \dots
U_i \dots |0\rangle +
\langle 0 | \dots U_i^{\dagger} \dots M \dots \frac{\partial U_i}{\partial x_i} \dots |0 \rangle
$$

$$
= 2 \text{Re} \big( \langle \Psi | M U_n \dots \frac{\partial U_i}{\partial x_i}  \dots U_0 |0\rangle \big)
$$

We can then make the definitions:

$$
\langle b_i|  = 2 \langle \Psi | M U_n \dots U_{i+1}
$$

$$
|k_i \rangle  = U_{i-1} \dots U_0 |0\rangle
$$

$$
| \tilde{k}_i \rangle = \frac{\partial U_i}{\partial x_i} | k_i \rangle
$$

which allows us to expressed the expectation value derivative as:

$$
\text{Re}\big( \langle b_i | \tilde{k}_i \rangle \big)
$$

We can then also define $|k_i\rangle$ and $|b_i\rangle$ iteratively as:

$$
|b_{i-1} \rangle = U^{\dagger}_i |b_i\rangle \qquad |k\_{i-1}\rangle = U^{\dagger}\_{i-1} |k_i \rangle
$$

With the initial conditions:

$$
| b_n \rangle = 2  M |\Psi \rangle \qquad |k_n\rangle = U_{n-1} \dots |0\rangle
$$

where we iterate in reverse from $n$ to $0$.

For the adjoint differentiation algorithm with expectation values, we iterate in reverse over $i$, taking an inner product at each step for each partial derivative.

## Full statevector differentiation

If we want to differentiate the entire statevector, we have:

$$
\frac{\partial}{\partial x_i} | \Psi \rangle = 
U_n \dots \frac{\partial U_i}{\partial x_i} \dots U_0 |0\rangle
$$

At each step $i$, we have have the list of "Jacobian" statevectors.

$$
|J_{i,j}\rangle = U_i \dots \frac{\partial U_j}{\partial x_j} \dots U_0 |0\rangle
$$

Upon iteration, we update:

$$
| J_{i+1, j} \rangle = U_{i+1} |J_{i, j} \rangle
$$

And add a new element to the list

$$
| J_{i+1, i+1} \rangle = \frac{\partial U_{i+1}}{\partial x_{i+1}} U_{i} \dots U_0 |0\rangle = \frac{\partial U_{i+1}}{\partial x_{i+1}} | k_i \rangle
$$

We are able to make some improvements over performing one simulation per trainable parameter by iteratively calculating $|k_i\rangle$ and reusing it between each jacobian calculation.

Note that this iterates from $0$ to $n$, instead if in reverse like expectation value differentiation.

## Vector jacobian product

When performing backpropagation across a workflow, we can reduce the computational complexity by calculating the <b>Vector Jacobian Product</b> (VJP) instead of the full jacobian product.

For a full jacobian:

$$
J_{i, j} = \frac{\partial f_j(x)}{\partial x_i}
$$

And a workflow that depends on the output of $f$:

$$
y = h(f(x))
$$

we have the VJP:

$$
\frac{\partial y}{\partial x_i} = \frac{\partial y}{\partial f_j}  J_{i,j} 
$$

where we sum up over all $j$ indices.

Calculating this quantity instead has several benefits over calculating the full jacobian.  

1. We only need to calculate components for the full jacobian that actually influence final output.  If a component $j$  has $dy/df_j = 0$, we do not need to calculate derivatives for that output.

2. Reduction in memory, which leads to a reduction in overall computational cost. The full jacobian scales with the size of the output *product* the size of the input.  But the VJP accepts something that is the size of the output and returns something that is the size of the input.

For the VJP of expectation values, this looks like:

$$
\frac{\partial y}{\partial f_j}  \frac{\partial \langle M_j \rangle}{\partial x_i} = \frac{\partial \langle \Psi | \frac{\partial y}{\partial f_j} M_j |\Psi\rangle}{\partial x_i}
$$

So this is equivalent to performing a single jacobian calculation with a single "effective" observable of $\Sigma_j \frac{\partial y}{\partial f_j} M_j$.

For the VJP calculation where the output is the state , we have:

$$
\frac{\partial y}{\partial f_j} \frac{\partial |\Psi\rangle}{\partial x_i}
$$

but we can make the identification that $\partial y/ \partial f_j$ is the same size and shape as a statevector, so we can simply rewrite this:

$$
\langle \left(\frac{\partial y}{\partial f} \right)^{\dagger} | \frac{\partial \Psi }{\partial x_i} \rangle
$$

$$
\langle dy^{\dagger} | U_n \dots \frac{\partial U_i}{\partial x_i} \dots |0 \rangle
$$ 

We can then use the same algorithm as for adjoint differentiation but with the initial conditions 

$$
| b_n \rangle = |dy^{\dagger} \rangle  \qquad |k_n\rangle = U_{n-1} \dots |0\rangle
$$

instead of:

$$
| b_n \rangle = 2  M |\Psi \rangle \qquad |k_n\rangle = U_{n-1} \dots |0\rangle
$$