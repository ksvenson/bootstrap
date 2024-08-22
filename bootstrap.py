import numpy as np
import scipy as sp
import multiprocessing as mp
import matplotlib.pyplot as plt

"""
bootstrap.py
Author: Kai Svenson
Date: August 22, 2024

For the multiple histogram method, we draw samples from one distribution (`sample_dist`),
in order to estimate observables of another distribution (`real_dist`).

We are interested in the variance of `real_dist`. We can compute this using a sample from
`sample_dist`. What is tricky is estimating the error in this estimate.

Here we test two methods, both of which are a form of bootstrapping.
"""


rng = np.random.default_rng()
PROCESSES = None
ESTIMATORS = (0, 1, 2)
FIG_SAVE_OPTIONS = {'bbox_inches': 'tight'}


def get_num_bins(x):
    """
    Calculate ideal number of bins for histogram using Freedman–Diaconis rule.
    """
    r = np.max(x, axis=-1) - np.min(x, axis=-1)
    p75, p25 = np.percentile(x, [75, 25], axis=-1)
    iqr = p75 - p25
    width = 2 * iqr / (x.shape[-1]**(1/3))
    return round(np.mean(r / width))


def get_sample_weights(sample_dist, real_dist, size):
    """
    Get samples from sample_dist with shape `size`. Calculate un-normalized weights so
    the samples can use used to estimate parameters of `real_dist`.
    """
    raw_sample = sample_dist.rvs(size=size, random_state=rng)
    weights = real_dist.pdf(raw_sample) / sample_dist.pdf(raw_sample)
    return raw_sample, weights


def w_var(raw_sample, weights, axis=None, estimator=ESTIMATORS[0]):
    """
    Estimate the variance of `real_dist` using samples and weights from
    `get_sample_weights`. So far, we have three estimators:

    Estimator 0
    This estimator is proven to be un-biased. I'm not sure there is an intuitive way at arriving at it.

    Estimator 1
    Estimates \expval{X^2} - \expval{X}^2 using weights normalized to 1.

    Estimator 2
    Estimates \expval{X^2} - \expval{X}^2 using un-normalized weights, and then normalizes with `1/n`.
    """
    if axis:
        n = raw_sample.shape[axis]
    else:
        n = raw_sample.size

    mu_2_bar = np.sum(weights * raw_sample**2, axis=axis)  # Un-normalized estimator for \expval{X^2} if X ~ `real_dist`
    mu_bar = np.sum(weights * raw_sample, axis=axis)  # Un-normalized estimator for \expval{X} if X ~ `real_dist`

    if estimator == 0:
        weight_sum = np.sum(weights, axis=axis)
        return (mu_2_bar * weight_sum - mu_bar**2)/(n * (n-1))
    elif estimator == 1:
        norm_weights = weights / np.sum(weights, axis=axis, keepdims=True)
        return np.sum(norm_weights * raw_sample**2, axis=axis) - np.sum(norm_weights * raw_sample, axis=axis)**2
    elif estimator == 2:
        return mu_2_bar / n - (mu_bar / n)**2
    else:
        raise ValueError(f'Unknown estimator. Known estimators are {", ".join([str(i) for i in ESTIMATORS])}.')


def bootstrap(raw_sample, weights, bs_sample_weights=None):
    pool = mp.Pool(processes=PROCESSES)
    return np.var(pool.starmap(bootstrap_helper, [(raw_sample, weights, bs_sample_weights)]*len(raw_sample)))

def bootstrap_helper(raw_sample, weights, bs_sample_weights):
    bs_samples = rng.choice(raw_sample, size=len(raw_sample), p=bs_sample_weights)
    return w_var(bs_samples, weights)


def get_var_var(sample_dist, real_dist, n, n_trials):
    pool = mp.Pool(processes=PROCESSES)
    return np.var(pool.starmap(get_var_samples_helper, [(sample_dist, real_dist, n)]*n_trials))

def get_var_samples_helper(sample_dist, real_dist, n):
    raw_sample, weights = get_sample_weights(sample_dist, real_dist, n)
    return w_var(raw_sample, weights)


if __name__ == '__main__':
    sample_dist = sp.stats.norm(loc=10, scale=5)
    real_dist = sp.stats.norm(loc=7, scale=3)
    # sample_dist = sp.stats.uniform(loc=-2, scale=4)
    # real_dist = sp.stats.uniform(loc=-1, scale=2)

    n = int(1e4)
    n_trials = int(1e4)

    raw_sample, weights = get_sample_weights(sample_dist, real_dist, (n_trials, n))

    var_estimates = np.full((len(ESTIMATORS), n_trials), np.nan)
    for i in range(var_estimates.shape[0]):
        var_estimates[i] = w_var(raw_sample, weights, axis=-1, estimator=i)

    # var_bar = w_var(raw_sample, weights, axis=-1)
    # var_bar_BIASED = w_var_BIASED(raw_sample, weights, axis=-1)
    # var_bar_BIASED_2 = w_var_BIASED_2(raw_sample, weights, axis=-1)

    np.save('var_estimates', var_estimates)

    # var_var_uniform = bootstrap(raw_sample, weights)
    # var_var_cdf = bootstrap(raw_sample, weights, bs_sample_weights=weights)

    print(f'Real var: {real_dist.var()}')
    for i, arr in enumerate(var_estimates):
        print(f'Estimator {i}: {np.mean(arr)}')

    fig, ax = plt.subplots()
    num_bins = get_num_bins(var_estimates)
    ax.hist(var_estimates.T, histtype='stepfilled', bins=num_bins, alpha=0.5, label=[f'Estimator {i}' for i in ESTIMATORS])
    ax.axvline(x=real_dist.var(), label=f'True Variance: {real_dist.var()}', color='Black', linestyle='dashed')
    # ax.set_xticks(list(ax.get_xticks()) + [real_dist.var()])
    ax.set(xlabel='Variance Estimate', ylabel='Counts', title=
           'Comparison of Weighted Variance Estimators\n'
           + r'$\mathcal{N}$'
           + rf'$(\mu={sample_dist.mean()}, \sigma^2={sample_dist.var()}) \rightarrow$'
           + r'$\mathcal{N}$'
           + rf'$(\mu={real_dist.mean()}, \sigma^2={real_dist.var()})$')
    fig.legend(loc='right')    
    fig.savefig('fig.svg', **FIG_SAVE_OPTIONS)
    # print(f'Uniform var var estimate: {var_var_uniform}')
    # print(f'CDF var var estimate: {var_var_cdf}')
    # print('-'*100)

    # print(f'Sample var: {sample_dist.var()}')
    # print(f'Sample var estiamte: {np.var(raw_sample)}')

    # Many raw samples to estiamte real var var
    # n_trials = n
    # real_var_var = get_var_var(sample_dist, real_dist, n, n_trials)
    
    # print(f'Real var var: {real_var_var}')
