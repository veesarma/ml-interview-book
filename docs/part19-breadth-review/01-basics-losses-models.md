# Basics, losses and models

> **Why this matters at staff level.** These are the warm-up questions, and they are
> scored differently from how candidates expect. Nobody is checking whether you can
> define overfitting. They are checking whether you reach for the data-generating
> process, whether you can say what probabilistic assumption a loss encodes, and
> whether you know which of your habits are load-bearing and which are cargo cult. A
> staff candidate who answers "why standardise features?" with the leakage trap has
> already signalled more than one who lists three model families.

## The questions

| # | Question | Taught properly in |
|---|---|---|
| 1 | [What breaks when data is not IID](#q1) | [Uncertainty and reliability](../part13-retrieval-eval-reliability/03-uncertainty-reliability.md) |
| 2 | [Overfitting vs underfitting, and how you diagnose which you have](#q2) | [Regularization](../part03-neural-nets/06-regularization.md) |
| 3 | [Two models, identical training loss, one generalises](#q3) | [Regularization](../part03-neural-nets/06-regularization.md) |
| 4 | [Cross-entropy vs MSE](#q4) | [Logistic and softmax regression](../part02-classical/02-logistic-softmax-regression.md) |
| 5 | [L1 vs L2 from a Bayesian view](#q5) | [Linear regression](../part02-classical/01-linear-regression.md) |
| 6 | [Ridge vs lasso under correlated predictors](#q6) | [Linear regression](../part02-classical/01-linear-regression.md) |
| 7 | [Hinge loss vs log loss](#q7) | [Kernel methods and SVMs](../part02-classical/07-kernel-methods-svm.md) |
| 8 | [Softmax vs one-vs-rest vs one-vs-one](#q8) | [Logistic and softmax regression](../part02-classical/02-logistic-softmax-regression.md) |
| 9 | [What makes a linear model linear, and how it relates to an MLP](#q9) | [MLPs and activations](../part03-neural-nets/01-mlp-and-activations.md) |
| 10 | [Why feature scaling matters, and when it does not](#q10) | [Optimization](../part01-math/06-optimization.md) |
| 11 | [How a CART tree chooses its splits](#q11) | [Trees and ensembles](../part02-classical/03-trees-and-ensembles.md) |
| 12 | [How bagging and feature subsampling reduce variance](#q12) | [Trees and ensembles](../part02-classical/03-trees-and-ensembles.md) |
| 13 | [Boosting vs gradient descent](#q13) | [Trees and ensembles](../part02-classical/03-trees-and-ensembles.md) |

## The frame everything hangs on

Supervised learning is an attempt to recover $f(x)$ from finite, noisy samples of a
joint distribution $P(X, Y)$. Two statements cover most of the questions in this
chapter.

For squared loss, expected test error at a point decomposes:

$$
\E\big[(y - \hat f(x))^2\big] = \underbrace{\big(\E[\hat f(x)] - f(x)\big)^2}_{\text{bias}^2} + \underbrace{\mathrm{Var}[\hat f(x)]}_{\text{variance}} + \underbrace{\sigma^2}_{\text{irreducible}}
$$

Bias is error from a hypothesis class too rigid to represent the truth. Variance is
error from a fit too sensitive to the particular training sample. The last term is the
entropy of $Y \mid X$, which no model removes.

Generalisation is a claim about a distribution, not about a dataset. The moment the
deployment distribution differs from the training distribution, every guarantee built
on the IID assumption weakens, and the arithmetic above stops describing your system.
Half the "basics" questions are one of these two statements in costume.

---

### 1. What breaks when data is not IID {#q1}

??? question "Q: What is the impact of non-IID data on machine learning?"
    **The answer.** IID bundles two assumptions, and breaking each has a different
    consequence, so name which one broke.

    *Identically distributed* fails under **distribution shift**: training and
    deployment draw from different $P$. Empirical risk minimisation optimised the
    wrong objective. Name the subtype, because the remedy depends on it. **Covariate
    shift** moves $P(X)$ with $P(Y \mid X)$ stable (a new camera sensor, a new
    traffic source). **Label or prior shift** moves $P(Y)$ (the fraud base rate jumps
    during a holiday weekend). **Concept drift** moves $P(Y \mid X)$ itself (the
    meaning of a feature changes, or the adversary adapts).

    *Independence* fails under **correlated samples**, and the consequence is that
    your effective sample size is far below your row count. Standard errors computed
    under independence are too narrow, so you declare wins that are noise.
    Random-split cross-validation leaks information between correlated train and test
    rows. Gradient estimates from a batch that is internally correlated are biased
    relative to the population gradient.

    !!! interview "Staff move"
        Tie it to one concrete failure and one specific remedy. "Frames from the same
        video clip are near-duplicates. A random frame-level split puts near-identical
        frames on both sides, so held-out accuracy is fiction and the model collapses
        on a new scene. The fix is grouped splitting at the clip or scene level. For
        shift, the questions I ask are: which kind, is it abrupt or gradual, and do I
        have any labelled target data? That triple decides between importance
        reweighting, domain adaptation, and a retraining cadence with drift monitors."
        Naming the kind of non-IID-ness and matching it to a specific remedy is the
        whole signal.

    **Goes deeper:** [uncertainty and reliability](../part13-retrieval-eval-reliability/03-uncertainty-reliability.md)
    for the shift taxonomy and detectors, and [statistics](../part01-math/04-statistics.md)
    for what correlated samples do to your error bars.

### 2. Overfitting vs underfitting, and how you diagnose which you have {#q2}

??? question "Q: Overfitting or underfitting, and how do you tell which one you have?"
    **The answer.** Underfitting is high bias: the model cannot capture the signal, so
    training and validation error are both high and close together. Overfitting is
    high variance: training error is low, validation error is much higher, and the gap
    widens with capacity or training time.

    The diagnostic is two curves. The **learning curve** plots error against training
    set size; the **training curve** plots error against epochs or capacity.

    * Both curves plateau high and close together: underfit. Add capacity, better
      features, longer training, less regularisation.
    * Large persistent train-validation gap: overfit. More data, stronger
      regularisation, augmentation, early stopping, less capacity.
    * Validation error still falling as you add data: data is the lever, and
      architecture work is the wrong project this quarter.

    !!! interview "Staff move"
        Say that the textbook U-curve is incomplete in the deep-learning regime.
        **Double descent** means test error can fall, rise near the interpolation
        threshold, then fall again as you go further overparameterised, so "bigger
        model means more overfitting" is unsafe to assert ([Nakkiran et al.,
        2019](https://arxiv.org/abs/1912.02292)). "With a 100M-parameter network on
        10k images I do not reason about overfitting from parameter count. I reason
        about effective capacity after augmentation, regularisation, and the implicit
        regularisation of SGD, and I treat the measured train-validation gap as the
        ground truth."

    **Goes deeper:** [regularization](../part03-neural-nets/06-regularization.md).

### 3. Two models, identical training loss, one generalises {#q3}

??? question "Q: Two models reach identical training loss and one generalises better. Give fundamentally different explanations and say how you would test each."
    **The answer.** This is a reasoning-quality question, so give a branching
    differential diagnosis and then pick the first test. Five distinct mechanisms:

    1. **Flatter minimum.** Same loss, but one sits in a flatter region, which
       correlates with better generalisation. *Test:* perturb the weights with noise
       and re-evaluate training loss, or estimate the top Hessian eigenvalues. The
       flatter model degrades less.
    2. **Lower effective capacity used.** One fits the same points with a simpler
       function: smaller weight norms, more low-frequency structure. *Test:* compare
       weight norms and the spectrum of the learned features, or probe with
       frequency-decomposed inputs.
    3. **One memorised, one learned structure.** Equal aggregate loss can hide that
       one model nailed the easy examples and memorised the noisy ones. *Test:*
       corrupt a fraction of the labels and retrain. Networks with capacity to spare
       fit random labels readily ([Zhang et al.,
       2016](https://arxiv.org/abs/1611.03530)); the generaliser resists. Per-example
       loss histograms show the same thing more cheaply.
    4. **Training stochasticity.** Data order, augmentation, and seed produced
       different solutions at equal final training loss. *Test:* re-run with
       controlled seeds and ablate augmentation. If the gap vanishes, it was never a
       property of the models.
    5. **A calibration or threshold artifact.** They may generalise equally in loss
       while you measure "better" with a thresholded metric, and one is better
       calibrated. *Test:* compare on a proper scoring rule and on calibration curves.

    !!! interview "Staff move"
        Open by refusing the premise until it is operationalised: "generalises better
        on which metric, under which distribution?" Then give the menu, then say which
        test you would run first and why. "I would start with label corruption and a
        sharpness probe. Both are cheap and they discriminate between the two
        explanations that are most often the real one." Interviewers are listening for
        cost-ordered hypothesis testing, and a candidate who runs the expensive test
        first has told them how they debug in production.

    **Goes deeper:** [regularization](../part03-neural-nets/06-regularization.md) and
    [optimization](../part01-math/06-optimization.md).

### 4. Cross-entropy vs MSE {#q4}

??? question "Q: Cross-entropy or MSE? What happens if you use MSE for classification?"
    **The answer.** A loss is a probabilistic modelling assumption in disguise.
    Minimising it is maximum likelihood under an assumed noise model, possibly with a
    prior. MSE is the MLE under Gaussian output noise, which is what you want for
    real-valued targets. Cross-entropy is the MLE under a categorical or Bernoulli
    output model, which is what you want for class probabilities.

    Using MSE for classification breaks in three ways, and the second is the one to
    derive at the whiteboard.

    1. **Wrong noise model.** Labels are not Gaussian around a mean, so the penalty
       assigned to a confident error does not match the log-likelihood of a
       categorical.

    2. **The gradient vanishes through the squashing non-linearity.** With a sigmoid
       output $\sigma(z)$ and loss $\tfrac12(\sigma(z) - y)^2$,

        $$
        \frac{\partial L}{\partial z} = (\sigma(z) - y)\,\sigma'(z)
        $$

        When the model is confidently wrong ($z \ll 0$ while $y = 1$), $\sigma'(z)
        \approx 0$ and the gradient vanishes, so learning stalls exactly where the
        error is largest. With cross-entropy the $\sigma'(z)$ cancels against the
        derivative of the log and

        $$
        \boxed{\;\frac{\partial L}{\partial z} = \sigma(z) - y\;}
        $$

        The gradient is proportional to the error. That cancellation is why
        cross-entropy trains classifiers.

    3. **Convexity.** MSE composed with a sigmoid is non-convex in the weights, while
       cross-entropy with a sigmoid is convex for logistic regression.

    !!! interview "Staff move"
        Hold the nuance rather than overclaiming. Squared error on probabilities is
        the **Brier score**, a legitimate proper scoring rule, and it is bounded,
        which makes it more robust to a single mislabelled outlier than log loss,
        which is unbounded. "For training deep classifiers I use cross-entropy because
        of the gradient cancellation. For evaluation under heavy label noise I also
        look at Brier, because one confidently wrong label can dominate a log-loss
        average. The vanishing-gradient argument is about trainability, not about
        squared error being an invalid way to score a probability."

    **Goes deeper:** [logistic and softmax regression](../part02-classical/02-logistic-softmax-regression.md)
    and [information theory](../part01-math/05-information-theory.md).

### 5. L1 vs L2 from a Bayesian view {#q5}

??? question "Q: Explain L1 and L2 regularization from a Bayesian perspective."
    **The answer.** A regularised loss is a negative log posterior. The data term is
    the negative log-likelihood and the penalty is the negative log-prior, so the
    minimiser is the MAP estimate.

    * **L2 (ridge)** is a **Gaussian prior**, $w \sim \mathcal{N}(0, \tau^2 I)$. Its
      log density contributes $-\norm{w}_2^2 / (2\tau^2)$, the squared-norm penalty.
    * **L1 (lasso)** is a **Laplace prior**, $p(w) \propto \exp(-\lvert w \rvert / b)$,
      whose log contributes the absolute-value penalty.

    The geometry follows the prior's shape. The Laplace density has a sharp peak at
    zero and heavy tails, which is what produces exact zeros in the MAP estimate. In
    constraint form, the L1 region is a cross-polytope whose corners sit on the axes,
    and the loss contours usually first touch it at a corner where some coordinates
    are exactly zero. The L2 ball has no corners, so it shrinks coefficients toward
    zero without setting them there.

    !!! interview "Staff move"
        Convert the prior into a decision. "I pick the prior that matches my belief
        about the weights. If I believe most features are irrelevant and I want
        feature selection baked into fitting, that is a Laplace prior and I use L1. If
        I believe every feature contributes a little and I want variance control under
        multicollinearity, that is a Gaussian prior and I use L2. Elastic net mixes
        the two when I want sparsity and L2's grouping behaviour at the same time."
        The follow-up is almost always question 6.

    **Goes deeper:** [linear regression](../part02-classical/01-linear-regression.md)
    and [probability](../part01-math/03-probability.md).

### 6. Ridge vs lasso under correlated predictors {#q6}

??? question "Q: What happens to ridge and lasso when the predictors are correlated?"
    **The answer.** This is the follow-up that separates people who know the geometry
    from people who know the slogan.

    * **Lasso is unstable and arbitrary** across a correlated group. It picks one
      predictor and zeroes the rest, and which one it picks is sensitive to noise, so
      small data perturbations flip the selection. It also cannot select more than $n$
      features when $p > n$.
    * **Ridge spreads the weight** across the group. Correlated predictors receive
      similar coefficients and are shrunk together, which is stable across retrains,
      and $p > n$ causes it no trouble.
    * **Elastic net** exists for this case. The L2 term gives a grouping effect, so
      correlated features enter and leave together, while the L1 term still delivers
      sparsity.

    !!! interview "Staff move"
        Make the instability an operational problem, because that is what it is. "With
        correlated features I rarely want raw lasso. The selected feature set is not
        reproducible across retrains, which means the feature set in the model card
        changes every month and nobody downstream can trust it. I reach for elastic
        net, or I decorrelate first by grouping or PCA, or I use a tree ensemble,
        which handles correlated features gracefully through feature subsampling."

    **Goes deeper:** [linear regression](../part02-classical/01-linear-regression.md)
    and [dimensionality reduction](../part02-classical/06-dimensionality-reduction.md).

### 7. Hinge loss vs log loss {#q7}

??? question "Q: Compare hinge loss and log loss."
    **The answer.** Both are convex surrogates for 0-1 loss, and they encode different
    priorities.

    **Hinge**, $\max(0, 1 - y f(x))$, is margin-focused and sparse in its support. It
    is exactly zero once a point sits on the correct side of the margin, so those
    points contribute no gradient and only the support vectors near the boundary
    matter. Its output is a score, so it gives you no calibrated probability, and it
    is non-differentiable at the kink, so you need a subgradient.

    **Log loss** is smooth and probabilistic. Every point contributes a shrinking
    gradient even when correctly classified, the output is a genuine probability, and
    it is more sensitive to outliers than hinge precisely because a badly
    misclassified point keeps contributing gradient without bound.

    !!! interview "Staff move"
        Route the choice through what happens downstream. "If anything downstream
        thresholds, ranks, or computes an expected value, I need calibrated
        probabilities, so log loss. If I only need a decision boundary and I want
        margin maximisation and support-vector sparsity, hinge. In deep learning I use
        cross-entropy almost always, because I want probabilities and smooth
        gradients, but margin losses come back in metric learning and contrastive
        setups where the quantity I care about is a relative distance."

    **Goes deeper:** [kernel methods and SVMs](../part02-classical/07-kernel-methods-svm.md).

### 8. Softmax vs one-vs-rest vs one-vs-one {#q8}

??? question "Q: Compare softmax, one-vs-rest and one-vs-one for multi-class classification, on consistency and calibration."
    **The answer.**

    **Softmax (multinomial)** models the categorical directly: one model, $K$ logits,
    normalised together. It is Fisher-consistent for the multi-class problem, the
    probabilities sum to one by construction, and it is the natural choice when the
    classes are mutually exclusive. Calibration is usually decent and still needs
    checking, with temperature scaling as the one-parameter fix.

    **One-vs-rest** trains $K$ binary classifiers, each "class $k$ against everything
    else". Three problems: the $K$ scores come from separate models on **incomparable
    scales**, so you must calibrate before comparing them; each binary problem is
    **imbalanced** by construction (one class against $K-1$); and the decomposition
    carries no consistency guarantee. In exchange it is simple, parallel, and works
    with any binary learner.

    **One-vs-one** trains $K(K-1)/2$ classifiers, one per pair, and votes. Each
    sub-problem is balanced and uses only the relevant data, which can be cheap per
    model, but you train $O(K^2)$ of them, the voting produces ambiguous regions, and
    the probability estimates are poor.

    !!! interview "Staff move"
        Name calibration as the differentiator, because that is the axis with
        production consequences. "For a neural network, softmax. There is no reason to
        decompose. OvR and OvO are artifacts of wanting to reuse binary learners like
        SVMs. Softmax gives me one coherent distribution that I calibrate with a
        single temperature; OvR gives me $K$ uncalibrated scores I have to reconcile,
        and that reconciliation breaks quietly under shift. I would consider OvO only
        when the per-pair problems are genuinely easier or when per-class data is so
        large that pairwise subsetting saves real compute."

    **Goes deeper:** [logistic and softmax regression](../part02-classical/02-logistic-softmax-regression.md).

### 9. What makes a linear model linear, and how it relates to an MLP {#q9}

??? question "Q: What does the 'linear' in linear and logistic regression refer to, and how do these relate to MLPs?"
    **The answer.** Linear means linear **in the parameters**, not in the inputs.
    $w^\top \phi(x)$ is a linear model however wild $\phi$ is. Logistic regression is a
    generalised linear model: a linear predictor $w^\top x + b$ passed through a link
    function to land in $[0,1]$, with a hyperplane decision boundary in feature space.

    An MLP is what you get when you **learn $\phi$ instead of designing it**. A
    one-hidden-layer network computes $g(x W_1 + b_1) W_2 + b_2$ with
    $X \in \R^{N \times d}$ row-major, so the hidden layer is a learned feature map and
    the output layer is linear or logistic regression on top of it. Logistic regression
    is the special case with no hidden layer. The expressivity comes entirely from the
    non-linearity $g$ between the linear maps: stack linear maps without one and they
    collapse to a single linear map with no gain in expressivity.

    !!! interview "Staff move"
        Refuse the power-ranking framing. "Linear models buy convexity, coefficients
        you can interpret and audit, cheap training, and stability across retrains.
        MLPs buy representation learning and pay in non-convexity, data hunger, and
        opacity. The question I actually ask is whether I need to learn features or
        whether I have good features and value auditability. For tabular, regulated,
        or low-data problems, logistic regression on well-engineered features is
        frequently the right call, and being willing to say that out loud in an
        interview at a deep-learning company is itself a signal."

    **Goes deeper:** [MLPs and activations](../part03-neural-nets/01-mlp-and-activations.md).

### 10. Why feature scaling matters, and when it does not {#q10}

??? question "Q: Why does feature scaling matter?"
    **The answer.** Scaling changes the fit for some model families and leaves others
    untouched, and knowing the boundary is the signal.

    Scaling changes the fit for:

    * **Gradient-based optimization.** Features on wildly different scales make the
      loss surface a stretched ellipse, so gradient descent zig-zags down a narrow
      valley and one global learning rate cannot serve every direction. Standardising
      makes the curvature more isotropic.
    * **Distance-based methods** (KNN, K-means, RBF-kernel SVM). A feature measured in
      millions dominates a Euclidean distance against one measured in fractions.
    * **Regularization.** L1 and L2 penalise coefficient magnitude, and if features
      live on different scales the penalty lands unevenly: a feature whose natural
      coefficient is large only because its units are small gets over-penalised.
      Standardise before you regularise.

    Scaling leaves **tree-based models** unchanged. Splits depend on ordering within a
    single feature, and any monotonic rescaling leaves the set of achievable
    partitions unchanged.

    !!! interview "Staff move"
        Volunteer the leakage trap. "The scaler is fit on the training fold only and
        applied to validation and test. Fitting it on the full dataset leaks test
        statistics into training, so it belongs inside the cross-validation loop or
        inside a pipeline object, never as a preprocessing step before the split. It
        is the most common subtle leakage bug I see, and it is invisible offline
        because the leak inflates exactly the number you are using to decide."

    **Goes deeper:** [optimization](../part01-math/06-optimization.md) for the
    conditioning argument and [KNN and K-means](../part02-classical/04-knn-kmeans.md)
    for the distance argument.

### 11. How a CART tree chooses its splits {#q11}

??? question "Q: How does a CART tree choose its splits?"
    **The answer.** Greedy recursive binary splitting to maximise impurity reduction.
    At each node, for every feature and every candidate threshold, compute the
    weighted impurity of the two children and take the split that reduces impurity
    most.

    For classification the impurity is **Gini** $1 - \sum_k p_k^2$ or **entropy**
    $-\sum_k p_k \log p_k$. For regression it is variance or MSE reduction: the split
    that most reduces within-node variance of the target.

    The procedure is greedy, with no global optimisation, because finding the optimal
    tree is NP-hard. Grown fully it overfits, so you control it with depth limits,
    minimum samples per leaf, and cost-complexity pruning that penalises tree size by
    $\alpha \lvert \text{leaves} \rvert$.

    !!! note "Correction to a common claim"
        You will hear that "Gini favours larger partitions and entropy favours
        balanced ones". Treat that as folklore. The two criteria agree on the chosen
        split in the large majority of cases and the accuracy difference is
        negligible; what you can say with confidence is that Gini is cheaper because
        it avoids the logarithm. Do not spend hyperparameter budget on this choice.

    !!! interview "Staff move"
        Move to what the greediness implies operationally. "Greedy splitting makes a
        single tree high-variance and sensitive to small data changes, which is
        exactly why we ship ensembles and not single trees. And if I need calibrated
        probabilities out of a tree I have to be careful, because raw leaf frequencies
        are not calibrated probabilities without smoothing or an explicit calibration
        step on held-out data."

    **Goes deeper:** [trees and ensembles](../part02-classical/03-trees-and-ensembles.md).

### 12. How bagging and feature subsampling reduce variance {#q12}

??? question "Q: How do bagging and feature subsampling reduce variance?"
    **The answer.** Start from the variance of an average. For $B$ identically
    distributed estimators each with variance $\sigma^2$ and pairwise correlation
    $\rho$,

    $$
    \boxed{\;\mathrm{Var}\Big(\tfrac{1}{B}\sum_{b=1}^{B} \hat f_b\Big) = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2\;}
    $$

    As $B \to \infty$ the second term vanishes and you are left with $\rho\sigma^2$.
    Adding trees hits a floor set by the correlation between them, so the lever is
    driving $\rho$ down.

    * **Bagging** trains each tree on a bootstrap resample, which decorrelates the
      trees somewhat without raising bias, since the trees stay deep.
    * **Feature subsampling**, the "random" in random forest, considers only a random
      subset of features at each split. It attacks $\rho$ directly by stopping every
      tree from keying on the same dominant feature, which lowers the floor. That is
      why a random forest beats plain bagged trees.

    !!! interview "Staff move"
        Turn the algebra into a tuning policy. "The formula says the lever is
        decorrelation and not count, which reorders my hyperparameter search:
        `max_features` is the variance knob and `n_estimators` only has to be large
        enough to be past diminishing returns. Bagging also hands me a free held-out
        estimate through out-of-bag samples, since each tree missed about 37% of the
        data." Be ready to derive the 37%: the probability a given row is excluded
        from a bootstrap of size $n$ is $(1 - 1/n)^n \to 1/e \approx 0.368$.

    **Goes deeper:** [trees and ensembles](../part02-classical/03-trees-and-ensembles.md).

### 13. Boosting vs gradient descent {#q13}

??? question "Q: How does boosting relate to gradient descent, and does boosting overfit?"
    **The answer.** Gradient boosting is gradient descent in **function space**.
    Ordinary gradient descent steps in parameter space, $\theta \leftarrow \theta -
    \eta \nabla_\theta L$. Gradient boosting builds an additive model $F_m = F_{m-1} +
    \eta h_m$ where each weak learner $h_m$ is fit to the negative gradient of the
    loss with respect to the current predictions, the pseudo-residuals. The shrinkage
    $\eta$ is the learning rate, each tree is one gradient step, and the ensemble is
    the trajectory.

    Set against bagging: bagging reduces **variance** by averaging independent
    low-bias learners, while boosting reduces **bias** by sequentially fitting the
    residual errors of a high-bias weak learner. They attack opposite terms of the
    decomposition, which is why their hyperparameters behave in opposite directions
    (deeper trees help bagging, shallower trees help boosting).

    Boosting does overfit if you boost too long. It is a bias-reduction machine and it
    will eventually fit the noise. What controls it: **shrinkage** (small $\eta$ with
    more rounds, the strongest regulariser you have), **shallow trees**, **row and
    column subsampling**, and **early stopping** on a validation fold. Modern
    implementations add explicit L1 and L2 penalties on the leaf weights.

    !!! interview "Staff move"
        Handle the folklore explicitly and then commit to a default. "The claim that
        boosting resists overfitting comes from the AdaBoost margin story and from
        shrinkage behaving like a small learning rate. I treat early stopping on a
        held-out fold as mandatory anyway. In production I default to gradient-boosted
        trees for tabular because they are the strongest off-the-shelf learner there,
        and I use a random forest when I want a robust, parallel, low-tuning baseline
        I can ship in a day."

    **Goes deeper:** [trees and ensembles](../part02-classical/03-trees-and-ensembles.md)
    and [optimization](../part01-math/06-optimization.md).

## References

* Nakkiran, Kaplun, Bansal, Yang, Barak, Sutskever. "Deep Double Descent: Where Bigger
  Models and More Data Hurt", 2019. [arXiv:1912.02292](https://arxiv.org/abs/1912.02292)
* Zhang, Bengio, Hardt, Recht, Vinyals. "Understanding deep learning requires
  rethinking generalization", ICLR 2017.
  [arXiv:1611.03530](https://arxiv.org/abs/1611.03530)
* Zou and Hastie. "Regularization and variable selection via the elastic net", Journal
  of the Royal Statistical Society Series B, 2005. (The grouping-effect result for
  correlated predictors.)
* Friedman. "Greedy Function Approximation: A Gradient Boosting Machine", Annals of
  Statistics, 2001. (Boosting as functional gradient descent.)
