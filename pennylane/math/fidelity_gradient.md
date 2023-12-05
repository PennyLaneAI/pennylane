
# Gradient of fidelity

## Issue with autodiff
The fidelity between two density operators $\rho$ and $\sigma$ is given by

$$\begin{align}
F(\rho,\sigma)=\text{Tr}\left(\sqrt{\sqrt{\rho}\\,\sigma\\,\sqrt{\rho}}\right)^2\qquad\qquad\qquad\text{(1)}
\end{align}$$

Looking at the above, we multiply the square root of the operator $\rho$ with a generally non-commuting operator $\sigma$, so it is clear that computing $F$ requires the full eigendecomposition of $\rho$, including its eigenvectors. For comparison, the trace distance between $\rho$ and $\sigma$ is given by

$$\begin{align*}
T(\rho,\sigma)=\frac{1}{2}\text{Tr}\left(|\rho-\sigma|\right)
\end{align*}$$

which does not require the eigenvectors of $\rho$, $\sigma$, or any function of the two, since the trace can be directly computed from the eigenvalues of $\rho-\sigma$.

The backward gradient of the eigendecomposition $\rho=U\Lambda U^\dagger$ is given by

$$\frac{\mathcal{\partial L}}{\partial\rho}=U^*\left(\frac{\partial\mathcal{L}}{\partial \Lambda}+D\circ U^\dagger\frac{\partial\mathcal{L}}{\partial U}\right)U^T$$

where $D$ is the matrix defined as

$$D_{ij}=\begin{cases}\frac{1}{\lambda_j-\lambda_i}\quad&\text{if }i\neq j\\
0&\text{if }i=j\end{cases}$$

See section 4.17 [here](https://arxiv.org/pdf/1701.00392.pdf) for more details.

It is clear that if $\rho$ has two or more eigenvalues which are the same, then the gradient of the eigendecomposition will be undefined. Indeed, auto-differentiation frameworks like JAX produce NaN results when backpropagating through $F$ if the inputs $\rho$ or $\sigma$ are sparse (which is often the case for states close to pure). On the other hand, the fidelity is a measure of overlap and thus has a well-defined gradient in every scenario.

Our solution to the above is to skip the gradient computation of the eigendecomposition, and instead treat the fidelity function as a black-box while defining a custom gradient.

## Gradient derivation

Let $\sqrt{\rho}\\,\sigma\sqrt{\rho}=U\Lambda U^\dagger$ be the eigendecomposition of $\sqrt{\rho}\\,\sigma\sqrt{\rho}$. In einsum notation, we have

$$\Lambda_{k\ell} = U^\dagger_ {kp}\sqrt{\rho}_ {pm}\sigma_{mn}\sqrt{\rho}_ {nq}U_ {q\ell}$$

whence

$$\frac{\partial\Lambda_{k\ell}}{\partial \sigma_ {mn}}=\sqrt{\rho}_ {nq}U_{q\ell}U^\dagger_{kp}\sqrt{\rho}_{pm}$$

Then equation (1) gives

$$F(\rho,\sigma)=\text{Tr}\left(\sqrt{\sqrt{\rho}\\,\sigma\\,\sqrt{\rho}}\right)^2=\left(\sum_{k=1}^d\sqrt{\Lambda_{kk}}\right)^2$$

and the gradient is easily seen to be

$$\begin{align*}
\frac{\partial F}{\partial\sigma_{mn}}&=2\left(\sum_{k=1}^n\sqrt{\Lambda_{kk}}\right)\sum_{k=1}^d\sqrt{\rho}_ {nq}U_{qk}\frac{1}{2\sqrt{\Lambda_{kk}}}U^\dagger_{kp}\sqrt{\rho}_{pm}\\
\end{align*}$$

Converting back to matrix notation, this is

$$\begin{align}
\frac{\partial F}{\partial\sigma}=\sqrt{F(\rho,\sigma)}\left(\sqrt{\rho}\\,\left(\sqrt{\rho}\\,\sigma\sqrt{\rho}\right)^{-\frac{1}{2}}\sqrt{\rho}\right)^T\qquad\qquad\qquad\text{(2)}
\end{align}$$

Since $F(\rho,\sigma)=F(\sigma,\rho)$, it follows immediately that

$$\begin{align}
\frac{\partial F}{\partial\rho}=\sqrt{F(\rho,\sigma)}\left(\sqrt{\sigma}\\,\left(\sqrt{\sigma}\\,\rho\sqrt{\sigma}\right)^{-\frac{1}{2}}\sqrt{\sigma}\right)^T\qquad\qquad\qquad\text{(3)}
\end{align}$$

Equations (2) and (3) are what we use for the custom gradient of the fidelity function.

Note that these equations are what Autograd and JAX use for the gradient of a real-valued function defined on complex inputs. On the other hand, PyTorch and TensorFlow use the Wirtinger derivatives given by

$$\begin{align}
\frac{\partial F}{\partial\rho^\*}&=\sqrt{F(\rho,\sigma)}\sqrt{\sigma}\\,\left(\sqrt{\sigma}\\,\rho\sqrt{\sigma}\right)^{-\frac{1}{2}}\sqrt{\sigma}\\
\frac{\partial F}{\partial\sigma^\*}&=\sqrt{F(\rho,\sigma)}\sqrt{\rho}\\,\left(\sqrt{\rho}\\,\sigma\sqrt{\rho}\right)^{-\frac{1}{2}}\sqrt{\rho}
\end{align}$$
