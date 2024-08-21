import numpy as np
import scipy as sp
import multiprocessing as mp


rng = np.random.default_rng(seed=628)

def w_var(raw_sample, weights, axis=None):
    return np.sum(weights * raw_sample**2, axis=axis) - np.sum(weights * raw_sample, axis=axis)**2


def bootstrap(raw_sample, weights, bs_sample_weights=None):
    n = len(raw_sample)
    pool = mp.Pool()
    return np.var(pool.starmap(bootstrap_helper, [(raw_sample, weights, bs_sample_weights)]*n))

def bootstrap_helper(raw_sample, weights, bs_sample_weights):
    n = len(raw_sample)
    bs_samples = rng.choice(raw_sample, size=n, p=bs_sample_weights)
    return w_var(bs_samples, weights)


if __name__ == '__main__':
    sample_dist = sp.stats.norm(loc=10, scale=5)
    real_dist = sp.stats.norm(loc=9, scale=5)
    # sample_dist = sp.stats.uniform(loc=-2, scale=4)
    # real_dist = sp.stats.uniform(loc=-1, scale=2)

    n = int(1e2)

    # One raw sample to test bootstrapping
    raw_sample = sample_dist.rvs(size=n, random_state=rng)

    weights = real_dist.pdf(raw_sample) / sample_dist.pdf(raw_sample)
    weights /= np.sum(weights)

    var_bar = w_var(raw_sample, weights)

    var_var_uniform = bootstrap(raw_sample, weights)
    var_var_cdf = bootstrap(raw_sample, weights, bs_sample_weights=weights)

    print(f'Real var: {real_dist.var()}')
    print(f'Real var estiamte: {var_bar}')
    print(f'Uniform var var estimate: {var_var_uniform}')
    print(f'CDF var var estimate: {var_var_cdf}')
    print('-'*100)

    # print(f'Sample var: {sample_dist.var()}')
    # print(f'Sample var estiamte: {np.var(raw_sample)}')

    # Many raw samples to estiamte real var var
    n_trials = n
    raw_sample = sample_dist.rvs(size=(n_trials, n), random_state=rng)
    
    weights = real_dist.pdf(raw_sample) / sample_dist.pdf(raw_sample)
    weights /= np.sum(weights, axis=-1)[:, np.newaxis]

    var_bar = w_var(raw_sample, weights, axis=-1)
    real_var_var = np.var(var_bar)
    
    print(f'Real var var: {real_var_var}')
