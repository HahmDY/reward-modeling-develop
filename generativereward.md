# Generative Reward Modeling

## Bradley-Terry Model

The Bradley-Terry model is used to derive a reward function based on comparison preferences:

$$p((x, y_c) \succ (x, y_r)|\theta) = \frac{e^{r_\theta(x,y_c)}}{e^{r_\theta(x,y_c)} + e^{r_\theta(x,y_r)}}$$

Where:
- $x$ is the prompt
- $y_c$ is the chosen response
- $y_r$ is the rejected response
- $r_\theta(x, y)$ is the reward function

## Reward as Log Odds Ratio

The reward can be expressed as the log odds ratio:

$$r_\theta(x, y_c) - r_\theta(x, y_r) = \log \frac{p((x,y_c) \succ (x,y_r)|\theta)}{p((x,y_r) \succ (x,y_c)|\theta)}$$

For a class $c \in \{+, -\}$, the odds ratio can be expressed in terms of probabilities:

$$\log \frac{p(c = + | f_\theta(x, y))}{p(c = - | f_\theta(x, y))} = \log \frac{p(f_\theta(x, y)|c = +)p(c = +)}{p(f_\theta(x, y)|c = -)p(c = -)} = \log \frac{p(f_\theta(x, y)|c = +)}{p(f_\theta(x, y)|c = -)} + \log \frac{p(c = +)}{p(c = -)}$$

Therefore, the reward function can be defined as:

$$R(x,y) := \log \frac{p(f_\theta(x,y)|c = +)}{p(f_\theta(x,y)|c = -)}$$

## Gaussian Discriminative Analysis

Using Gaussian discriminative analysis, we model the class-conditional distributions as multivariate Gaussian distributions:

$$p(z|c = +) = \mathcal{N}(z|\mu_+, \Sigma)$$
$$p(z|c = -) = \mathcal{N}(z|\mu_-, \Sigma)$$

Where $z$ represents the feature embeddings and both classes share the same covariance matrix $\Sigma$.

The reward function can then be expressed as:

$$R(x,y) = (\mu_+ - \mu_-)^T \Sigma^{-1} f_\theta(x,y) + b$$
$$b = \frac{1}{2}(\mu_+^T \Sigma^{-1} \mu_+ - \mu_-^T \Sigma^{-1} \mu_-)$$

## Proof

### Multivariate Gaussian Distribution

The multivariate Gaussian distribution with a shared covariance matrix is:

$$\mathcal{N}(z; \mu, \Sigma) = \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(z-\mu)^T \Sigma^{-1}(z-\mu)\right)$$

### Log Probability

Taking the log:

$$\log p(z | c) = -\frac{d}{2}\log(2\pi) - \frac{1}{2}\log|\Sigma| - \frac{1}{2}(z-\mu_c)^T \Sigma^{-1}(z-\mu_c)$$

### Reward Derivation

The reward is the log probability ratio:

$$R(x,y) = \log p(f_\theta(x,y)|c = +) - \log p(f_\theta(x,y)|c = -)$$

Let $z = f_\theta(x,y)$:

$$R(z) = -\frac{1}{2}(z-\mu_+)^T \Sigma^{-1}(z-\mu_+) + \frac{1}{2}(z-\mu_-)^T \Sigma^{-1}(z-\mu_-)$$

### Algebraic Expansion

Expanding the quadratic forms:

$$(z-\mu_+)^T \Sigma^{-1}(z-\mu_+) = z^T \Sigma^{-1}z - 2\mu_+^T \Sigma^{-1}z + \mu_+^T \Sigma^{-1}\mu_+$$
$$(z-\mu_-)^T \Sigma^{-1}(z-\mu_-) = z^T \Sigma^{-1}z - 2\mu_-^T \Sigma^{-1}z + \mu_-^T \Sigma^{-1}\mu_-$$

### Final Reward Expression

After simplification, the reward function becomes:

$$R(x,y) = (\mu_+ - \mu_-)^T \Sigma^{-1} f_\theta(x,y) + b$$

Where the bias term is:

$$b = \frac{1}{2}(\mu_+^T \Sigma^{-1}\mu_+ - \mu_-^T \Sigma^{-1}\mu_-)$$

This formulation shows that the reward function is a linear function of the feature embeddings $f_\theta(x,y)$, with weights determined by the difference in means of the two classes and the inverse covariance matrix.
