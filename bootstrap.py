import numpy as np
import scipy as sp
import multiprocessing as mp


rng = np.random.default_rng(seed=628)
PROCESSES = None


def get_sample_weights(sample_dist, real_dist, n):
    raw_sample = sample_dist.rvs(size=n, random_state=rng)
    
    weights = real_dist.pdf(raw_sample) / sample_dist.pdf(raw_sample)
    weights /= np.sum(weights)

    return raw_sample, weights


def w_var(raw_sample, weights, axis=None):
    return np.sum(weights * raw_sample**2, axis=axis) - np.sum(weights * raw_sample, axis=axis)**2


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
    real_dist = sp.stats.norm(loc=9, scale=5)
    # sample_dist = sp.stats.uniform(loc=-2, scale=4)
    # real_dist = sp.stats.uniform(loc=-1, scale=2)

    n = int(1e2)

    # One raw sample to test bootstrapping
    raw_sample, weights = get_sample_weights(sample_dist, real_dist, n)

    # raw_sample = sample_dist.rvs(size=n, random_state=rng)

    # weights = real_dist.pdf(raw_sample) / sample_dist.pdf(raw_sample)
    # weights /= np.sum(weights)

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
    real_var_var = get_var_var(sample_dist, real_dist, n, n_trials)
    
    print(f'Real var var: {real_var_var}')
