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

import numpy as np
import scipy as sp
import multiprocessing as mp
import matplotlib.pyplot as plt

rng = np.random.default_rng(seed=628)  # The better circle constant!
PROCESSES = None
AVG_ESTIMATORS = (0, 1)
VAR_ESTIMATORS = (0, 1, 2, 3)
FIG_SAVE_OPTIONS = {'bbox_inches': 'tight'}
LEGEND_OPTIONS = {'bbox_to_anchor': (1.04, 1), 'loc': 'upper left'}
DISP_DECI = 5


def get_num_bins(x):
    """
    Calculate ideal number of bins for a histogram using Freedman–Diaconis rule.
    """
    r = np.max(x, axis=-1) - np.min(x, axis=-1)
    p75, p25 = np.percentile(x, [75, 25], axis=-1)
    iqr = p75 - p25
    width = 2 * iqr / (x.shape[-1]**(1/3))
    return round(np.mean(r / width))


def title_helper(sample_dist, real_dist):
    return (r'$\mathcal{N}$'
            + rf'$(\mu={sample_dist.mean()}, \sigma^2={sample_dist.var()}) \rightarrow$'
            + r'$\mathcal{N}$'
            + rf'$(\mu={real_dist.mean()}, \sigma^2={real_dist.var()})$')


def get_sample_weights(sample_dist, real_dist, size):
    """
    Get samples from sample_dist with shape `size`. Calculate un-normalized weights so
    the samples can use used to estimate parameters of `real_dist`.
    """
    raw_sample = sample_dist.rvs(size=size, random_state=rng)
    weights = real_dist.pdf(raw_sample) / sample_dist.pdf(raw_sample)
    return raw_sample, weights


def w_avg(raw_sample, weights, axis=None, estimator=AVG_ESTIMATORS[0]):
    """
    Estimate the average of `real_dist` using samples and weights from `get_sample_weights`.
    """
    if axis:
        n = raw_sample.shape[axis]
    else:
        n = raw_sample.size

    if estimator == 0:
        norm_weights = weights / np.sum(weights, axis=axis, keepdims=True)
        return np.sum(norm_weights * raw_sample, axis=axis)
    elif estimator == 1:
        return np.sum(weights * raw_sample, axis=axis) / n


def w_var(raw_sample, weights, axis=None, estimator=VAR_ESTIMATORS[0]):
    """
    Estimate the variance of `real_dist` using samples and weights from
    `get_sample_weights`. So far, we have three estimators:

    Estimator 0
    This estimator is proven to be unbiased. I'm not sure there is an intuitive way at arriving at it.

    Estimator 1
    Same as Estimator 0, but instead normalizes using the sum of weights instead of n.

    Estimator 2
    Estimates \expval{X^2} - \expval{X}^2 using weights normalized to 1.

    Estimator 3
    Estimates \expval{X^2} - \expval{X}^2 using un-normalized weights, and then normalizes with `1/n`.
    """
    if axis:
        n = raw_sample.shape[axis]
    else:
        n = raw_sample.size

    mu_2_bar = np.sum(weights * raw_sample**2, axis=axis)  # Un-normalized estimator for \expval{X^2} if X ~ `real_dist`
    mu_bar = np.sum(weights * raw_sample, axis=axis)  # Un-normalized estimator for \expval{X} if X ~ `real_dist`
    weight_sum = np.sum(weights, axis=axis)

    if estimator == 0:
        return (mu_2_bar * weight_sum - mu_bar**2)/(n * (n-1))
    elif estimator == 1:
        return (mu_2_bar * weight_sum - mu_bar**2) / (weight_sum * (weight_sum - 1))
    elif estimator == 2:
        norm_weights = weights / np.sum(weights, axis=axis, keepdims=True)
        return np.sum(norm_weights * raw_sample**2, axis=axis) - np.sum(norm_weights * raw_sample, axis=axis)**2
    elif estimator == 3:
        return mu_2_bar / n - (mu_bar / n) ** 2
    else:
        raise ValueError(f'Unknown estimator. Known estimators are {", ".join([str(i) for i in VAR_ESTIMATORS])}.')


def bootstrap_sample_weights(raw_sample, weights, n_trials, bs_prob=None, estimator=VAR_ESTIMATORS[0]):
    """
    Draw `len(raw_sample)` elements randomly from `raw_sample` according to probability
    distribution `bs_weights`. Then, estimate the variance using `weights` and `estimator`.    
    Do this `n_trials` times, and return an array with shape `(n_trials,)`.
    """
    bs_samples_idxes = rng.choice(raw_sample.size, size=(n_trials, raw_sample.size), p=bs_prob)
    bs_samples = raw_sample[bs_samples_idxes]

    if bs_prob is None:
        bs_weights = weights[bs_samples_idxes]
        return w_var(bs_samples, bs_weights, axis=-1, estimator=estimator)
    else:
        return np.var(bs_samples, axis=-1)


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
    sample_dist = sp.stats.norm(loc=12, scale=3)
    real_dist = sp.stats.norm(loc=0, scale=1)
    # sample_dist = sp.stats.uniform(loc=-2, scale=4)
    # real_dist = sp.stats.uniform(loc=-1, scale=2)
    diff = np.abs(sample_dist.mean() - real_dist.mean()) / np.sqrt(sample_dist.var())

    n = int(1e4)
    n_trials = int(1e4)

    raw_sample, weights = get_sample_weights(sample_dist, real_dist, (n_trials, n))

    avg_estimates = np.full((len(AVG_ESTIMATORS), n_trials), np.nan)
    var_estimates = np.full((len(VAR_ESTIMATORS), n_trials), np.nan)
    for i in AVG_ESTIMATORS:
        avg_estimates[i] = w_avg(raw_sample, weights, axis=-1, estimator=i)
    for i in VAR_ESTIMATORS:
        var_estimates[i] = w_var(raw_sample, weights, axis=-1, estimator=i)
    np.save('avg_estimates', avg_estimates)
    np.save('var_estimates', var_estimates)
    avg_avg = np.mean(avg_estimates, axis=-1)
    avg_var = np.mean(var_estimates, axis=-1)

    print(f'Real var: {real_dist.var()}')
    for i in VAR_ESTIMATORS:
        print(f'Estimator {i}: {avg_var[i]}')

    fig, ax = plt.subplots()
    num_bins = get_num_bins(avg_estimates)
    ax.hist(avg_estimates.T, histtype='stepfilled', bins=num_bins, alpha=0.5, label=[f'Estimator {i} (Err: {round(real_dist.mean() - avg_avg[i], DISP_DECI)})' for i in AVG_ESTIMATORS])
    ax.axvline(x=real_dist.mean(), label=f'Real Average: {real_dist.mean()}', color='Black', linestyle='dashed')
    ax.set(xlabel='Average Estimate', ylabel='Counts', title='Comparison of Weighted Average Estimators\n' + title_helper(sample_dist, real_dist))
    fig.legend(**LEGEND_OPTIONS)
    fig.savefig(f'avg_est_comp_d{diff}.svg', **FIG_SAVE_OPTIONS)

    fig, ax = plt.subplots()
    num_bins = get_num_bins(var_estimates)
    ax.hist(var_estimates.T, histtype='stepfilled', bins=num_bins, alpha=0.5, label=[f'Estimator {i} (Err: {round(real_dist.var() - avg_var[i], DISP_DECI)})' for i in VAR_ESTIMATORS])
    ax.axvline(x=real_dist.var(), label=f'Real Variance: {real_dist.var()}', color='Black', linestyle='dashed')
    ax.set(xlabel='Variance Estimate', ylabel='Counts', title='Comparison of Weighted Variance Estimators\n' + title_helper(sample_dist, real_dist))
    fig.legend(**LEGEND_OPTIONS)    
    fig.savefig(f'var_est_comp_d{diff}.svg', **FIG_SAVE_OPTIONS)

    # Now we bootstrap
    raw_sample, weights = get_sample_weights(sample_dist, real_dist, n)
    print('-'*100)
    print(w_var(raw_sample, weights, estimator=0))
    for i in VAR_ESTIMATORS:
        uniform = bootstrap_sample_weights(raw_sample, weights, n_trials, bs_prob=None, estimator=i)
        weighted = bootstrap_sample_weights(raw_sample, weights, n_trials, bs_prob=weights/np.sum(weights), estimator=i)

        data = np.stack((var_estimates[i], uniform, weighted))

        fig, ax = plt.subplots()
        num_bins = get_num_bins(data)
        ax.hist(data.T, histtype='stepfilled', bins=num_bins, alpha=0.5, label=
                (f'Manual Distribution (Err: {round(real_dist.var() - avg_var[i], DISP_DECI)})',
                 f'Uniform Bootstrap (Err: {round(real_dist.var() - np.mean(uniform), DISP_DECI)})',
                 f'Weighted Bootstrap (Err: {round(real_dist.var() - np.mean(weighted), DISP_DECI)})'))
        ax.axvline(x=real_dist.var(), label=f'Real Variance: {real_dist.var()}', color='Black', linestyle='dashed')
        ax.set(xlabel='Variance Estimate', ylabel='Counts', title=f'Comparison of Bootstrap Methods for Estimator {i}\n' + title_helper(sample_dist, real_dist))
        fig.legend(**LEGEND_OPTIONS)    
        fig.savefig(f'bs_var_comp_e{i}_d{diff}.svg', **FIG_SAVE_OPTIONS)
