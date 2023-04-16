---
title: Efficient Cholesky decomposition of low-rank updates
subtitle: A short and practical guide to efficiently computing the 
  Cholesky decomposition of matrices perturbed by low-rank updates
summary: A short and practical guide to efficiently computing the 
  Cholesky decomposition of matrices perturbed by low-rank updates
authors:
- admin
date: 2023-04-16T11:16:03.167Z
draft: false
featured: true
categories:
  - technical
tags:
- TensorFlow Probability
- Machine Learning
math: true
image:
  filename: featured
  focal_point: Smart
  preview_only: false
---

Suppose we're given a positive semidefinite (PSD) 
matrix {{< math >}}$\mathbf{A} \in \mathbb{R}^{N \times N}${{< /math >}} to
which we wish to update by some low-rank
matrix {{< math >}}$\mathbf{U} \mathbf{U}^\top \in \mathbb{R}^{N \times N}${{< /math >}},
{{< math >}}
$$\mathbf{B} \triangleq \mathbf{A} + \mathbf{U} \mathbf{U}^\top,$$
{{< /math >}} 
where the update factor matrix {{< math >}}$\mathbf{U} \in \mathbb{R}^{N \times M}${{< /math >}}.
To be more precise, the low-rank update is rank-$M$ for some $M \ll N$.

*What is the best way to calculate the Cholesky decomposition of {{< math >}}$\mathbf{B}${{< /math >}}?*

Given no additional information the obvious way is to calculate it directly, 
which incurs a cost of {{< math >}}$\mathcal{O}(N^3)${{< /math >}}.
But suppose we've already calculated the lower-triangular Cholesky factor 
{{< math >}}$\mathbf{L} \in \mathbb{R}^{N \times N}${{< /math >}} 
of {{< math >}}$\mathbf{A}${{< /math >}} (i.e., {{< math >}}$\mathbf{LL}^\top = \mathbf{A}${{< /math >}}).
Then, we can use it to calculate the Cholesky decomposition 
of {{< math >}}$\mathbf{B}${{< /math >}} at a reduced cost 
of {{< math >}}$\mathcal{O}(N^2M)${{< /math >}}.
Here's how.

## Rank-1 Updates

First, let's consider the simpler case involving just *rank-1 updates*
{{< math >}}
$$\mathbf{B} \triangleq \mathbf{A} + \mathbf{u} \mathbf{u}^\top,$$
{{< /math >}} 
where update factor vector {{< math >}}$\mathbf{u} \in \mathbb{R}^{N}${{< /math >}}.
With some clever manipulations[^seeger2004low], the details of which we won't
get into in this post, we can leverage {{< math >}}$\mathbf{L}${{< /math >}} to 
calculate the Cholesky decomposition of {{< math >}}$\mathbf{B}${{< /math >}} 
at a reduced cost of {{< math >}}$\mathcal{O}(N^2)${{< /math >}}.
Such a procedure for rank-1 updates is implemented in the old-school Fortran 
linear algebra software library [LINPACK](https://netlib.org/linpack/) 
(but unfortunately not in its successor [LAPACK](https://netlib.org/lapack/)),
and also in modern libraries like [TensorFlow Probability](https://www.tensorflow.org/probability) (TFP).

In TFP, this is implemented in the function named `tfp.math.cholesky_update`. 
For example,

```python
import tensorflow as tf
import tensorflow_probability as tfp

a_factor = tf.linalg.cholesky(a)  # O(N^3); suppose this is pre-computed and stored

b_factor_1 = tf.linalg.cholesky(a + update @ update.T)  # O(N^3), ignores `a_factor`
b_factor_2 = tfp.math.cholesky_update(a_factor, update)  # O(N^2), uses `a_factor`

assert_array_equal(b_factor_1, b_factor_2)
```

## Low-Rank Updates

Now let's return to rank-$M$ updates.
First let's write the update factor matrix $\mathbf{U}$ in terms of column 
vectors $\mathbf{u}_m \in \mathbb{R}^{N}$,
{{< math >}}
$$
\mathbf{U} \triangleq
\begin{bmatrix}
\mathbf{u}_1 & \cdots & \mathbf{u}_M
\end{bmatrix}.
$$
{{< /math >}} 
Now we can write the rank-$M$ update matrix as a sum of $M$ rank-1 matrices,
{{< math >}}
$$
\mathbf{U} \mathbf{U}^\top = 
\begin{bmatrix} \mathbf{u}_1 & \cdots & \mathbf{u}_M \end{bmatrix} 
\begin{bmatrix} \mathbf{u}_1^\top \\ \vdots \\ \mathbf{u}_M^\top \end{bmatrix} = 
\sum_{m=1}^{M} \mathbf{u}_m \mathbf{u}_m^\top.
$$
{{< /math >}} 
Thus seen, a low-rank update is nothing but a repeated application of rank-1 updates,
{{< math >}}
$$
\begin{align}
\mathbf{B} & = \mathbf{A} + \mathbf{U} \mathbf{U}^\top \\ & =
\mathbf{A} + \sum_{m=1}^{M} \mathbf{u}_m \mathbf{u}_m^\top \\ & = 
((\mathbf{A} + \mathbf{u}_1 \mathbf{u}_1^\top) + \cdots ) + \mathbf{u}_M \mathbf{u}_M^{\top}.
\end{align}
$$
{{< /math >}} 

Therefore, we can simply leverage the $O(N^2)$ procedure for Cholesky 
decompositions of rank-1 updates and apply it recursively $M$ times to obtain 
a $O(N^2M)$ procedure for rank-$M$ updates.

```python
import tensorflow as tf
import tensorflow_probability as tfp

a  # Tensor; [..., N, N]
update_factor  # Tensor; [..., N, M]

a_factor = tf.linalg.cholesky(a)  # O(N^3); suppose this is pre-computed and stored

b_factor_1 = low_rank(a_factor, update_factor)  # O(N^2M), uses `a_factor`
b_factor_2 = tf.linalg.cholesky(a + update_factor @ update_factor.T)  # O(N^3), ignores `a_factor`
```

where function `low_rank` is implemented as follows:

```python
def low_rank(chol, update_factor):
    value = chol
    for update in tf.unstack(update_factor, axis=-1):
        value = tfp.math.cholesky_update(value, update)
    return value
```

The astute reader will recognize that this is simply an special case of 
the [itertools.accumulate](https://docs.python.org/3/library/itertools.html#itertools.accumulate) 
or [functools.reduce](https://docs.python.org/3/library/functools.html#functools.reduce)
patterns, where 
the *binary operator* is `tfp.math.cholesky_update`, 
the *iterable* is `tf.unstack(update_factor, axis=-1)` and 
the *initial value* is `chol`.

Therefore, we can alternatively implement with the one-liner:

```python
from functools import reduce


def low_rank(chol, update_factor):
    return reduce(tfp.math.cholesky_update, tf.unstack(update_factor, axis=-1), chol)
```

To receive updates on more posts like this, follow me on [Twitter] and [GitHub]!

[Twitter]: https://twitter.com/louistiao
[GitHub]: https://github.com/ltiao


[^seeger2004low]: Seeger, M. (2004). Low rank updates for the Cholesky decomposition.
[^dongarra1979linpack]: Dongarra, J. J., Moler, C. B., Bunch, J. R., & Stewart, G. W. (1979). LINPACK users' guide. Society for Industrial and Applied Mathematics.
