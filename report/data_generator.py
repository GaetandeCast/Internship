import numpy as np
from sklearn.utils import check_random_state


def generate_gaussian_causal_network(
    n_sample=1000,
    b=np.zeros(10),
    v=np.ones(10),
    w=np.vstack([np.zeros((9, 10)), np.ones((1, 10))]),
    random_state=None,
):
    """
    Generate samples of D gaussian rvs whose interactions are modeled by a Directed Acyclic Graph.
    We assume the variables are numbered so that if `j \\in pa_i` then `j<i`.
    The conditional of `x_i` conditioned on its parents `pa_i` is `p(x_i|pa_i) = \\mathcal{N}(x_i | \\sum_{j \\in pa_i w_{i,j}x_j + b_i , v_i})`

    Parameters
    ----------
    n_samples : int
        The number of samples to draw.
    b : ndarray of shape (D,)
        The bias term of each random variable.
    v : ndarray of shape (D,)
        The variance term of each random variable.
    w : ndarray of shape (D, D)
        The weights representing the conditional relation between i and its parent j.
        Zero if i and j are marginally independant.
    random_state : int
        Seed of the random number generator.

    Returns
    ----------
    X : ndarray of shape (n_samples, D)
        The drawn samples.
    mu : ndarray of shape (D,)
        The theoretical mean vector of the multivariate gaussian.
    Sigma : ndarray of shape (D, D)
        The theoretical variance-covariance matrix of the multivariate gaussian.
    Note
    ----------
    - The random variables follows a multivariate gaussian distribution whose mean and covariance matrix can be computed recursively
        in the following manner (Bishop, section 8.1.4):
        - `\\mu_i = \\sum_{j \\in pa_i} w_{i,j} \\mu_j + b_i`
        - `\\Sigma_{i,j} = \\sum_{k \\in pa_j} w_{j,k} \\Sigma_{i,k} + I_{i=j}v_i`
    - The default is a model where the first 9 variables are sampled from independant standard distributions, and the last from a normal distribution
        centered on the sum of covariates and with variance 1. Essentially a simple linear model.
    """
    rng = check_random_state(random_state)
    D = b.shape[0]

    # mu = np.copy(b)
    # Sigma = np.diag(v)

    # # Bishop's computations
    # for j in range(D):
    #     for k in range(D):
    #         if w[j, k] > 0:
    #             mu[j] += w[j, k] * mu[k]

    #     for k in range(D):
    #         for l in range(D):
    #             if w[k, l] > 0:
    #                 Sigma[k, j] += w[k, l] * Sigma[j, l]

    # for j in range(D):
    #     for k in range(j, D):
    #         Sigma[j, k] = Sigma[k, j]
    mu = np.zeros(D)
    Sigma = np.zeros((D, D))

    for i in range(D):
        mu[i] = b[i]
        for j in range(D):
            mu[i] += w[i, j] * mu[j]
        
        for j in range(i, D):
            Sigma[i, j] = (i == j) * v[i]
            for k in range(D):
                Sigma[i, j] += w[j, k] * Sigma[i, k]
            Sigma[j, i] = Sigma[i, j]
     
    X = rng.multivariate_normal(mu, Sigma, n_sample)
    return X, mu, Sigma
