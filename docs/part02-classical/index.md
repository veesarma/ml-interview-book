# Part II — Classical machine learning

Classical ML is where interviewers check whether you *derive* or *recite*. Every
model in this part fits in a few hundred lines of NumPy, every training rule can be
written on a whiteboard in ten lines, and every one of them is still in production
at the companies you are interviewing with: logistic regression scores ads at Google
and Meta, gradient-boosted trees rank listings at Airbnb and score fraud at Stripe,
k-means builds the codebooks inside every billion-scale vector index, PCA compresses
the embeddings that feed them, and kernel smoothing is the ancestor of attention.

The through-line of this part is that **each algorithm is an optimisation problem plus
a solver**, and the solver choices (closed form, gradient descent, Newton, coordinate
descent, EM, alternating minimisation, SMO) reappear unchanged in Parts III–VII.

## Chapters

| Chapter | The one thing to be able to derive cold | Code you should be able to write in 20 minutes |
|---|---|---|
| [Linear regression](01-linear-regression.md) | Normal equations; ridge shrinkage $s_i^2/(s_i^2+\lambda)$ via SVD; lasso soft-thresholding | OLS, ridge, lasso coordinate descent |
| [Logistic & softmax regression](02-logistic-softmax-regression.md) | $\nabla_w L = X^T(p - y)$; softmax Jacobian $\to$ $\partial L/\partial z = p - y$ | Stable sigmoid/softmax, GD and Newton fits |
| [Trees & ensembles](03-trees-and-ensembles.md) | Newton boosting leaf weight $w^* = -G/(H+\lambda)$; RF variance $\rho\sigma^2 + (1-\rho)\sigma^2/M$ | CART tree, random forest, gradient boosting |
| [KNN & K-means](04-knn-kmeans.md) | Lloyd as coordinate descent; k-means++ $O(\log k)$ guarantee | Brute-force KNN, kd-tree, k-means++ |
| [Probabilistic models & EM](05-probabilistic-models-em.md) | ELBO / Jensen derivation of EM; GDA $\Rightarrow$ logistic posterior | Naive Bayes, GDA, GMM-EM |
| [Dimensionality reduction](06-dimensionality-reduction.md) | PCA from variance maximisation *and* reconstruction error; randomized SVD | PCA via eig and SVD, whitening |
| [Kernel methods & SVMs](07-kernel-methods-svm.md) | Primal $\to$ Lagrangian $\to$ KKT $\to$ dual; kernel ridge $\alpha = (K+\lambda I)^{-1}y$ | Kernel ridge, SMO-lite SVM, Nadaraya–Watson |

## Prerequisites

From [Part I](../part01-math/index.md): the SVD and eigendecomposition
([linear algebra](../part01-math/01-linear-algebra.md)), the gradient of a quadratic
form and the chain rule for vector functions
([matrix calculus](../part01-math/02-calculus-matrix-calculus.md)), Bayes' rule and
the multivariate Gaussian ([probability](../part01-math/03-probability.md)), entropy
and KL divergence ([information theory](../part01-math/05-information-theory.md)), and
convexity, Lipschitz gradients and Newton's method
([optimization](../part01-math/06-optimization.md)). Every chapter states which of
these it uses at the top of its math section.

## If you have one day

Read in this order; each step builds on the previous one's solver.

1. **Morning.** Linear regression §2 (normal equations, SVD view of ridge, soft
   thresholding) then logistic/softmax §2 (derive $p - y$ twice: once for the
   sigmoid, once through the softmax Jacobian). These two derivations are asked in
   more interviews than everything else in this part combined.
2. **Early afternoon.** Trees §2.3–2.4: functional gradient descent and the
   second-order XGBoost objective. Be able to write $w^* = -G/(H+\lambda)$ and the
   split gain without notes. Skim the LightGBM/CatBoost ideas.
3. **Late afternoon.** K-means (Lloyd as alternating minimisation) then EM for GMMs
   (Jensen $\to$ ELBO $\to$ E/M steps) — they are the same argument at two levels of
   softness. PCA §2 as a 30-minute detour: both derivations lead to the top
   eigenvectors.
4. **Evening.** SVM dual derivation once, slowly, then the "attention is kernel
   smoothing" bridge in §2.6 of the kernel chapter. Finish by re-reading every
   chapter's *TL;DR — the interview card*.

## Code and tests

All implementations live in `src/mlbook/classical/` and are pure NumPy. Run

```bash
pytest tests/test_classical_*.py -q
```

Every chapter has a **Retype by hand** section naming the exact functions to
reproduce from memory, the test command that checks them, and a target time.
