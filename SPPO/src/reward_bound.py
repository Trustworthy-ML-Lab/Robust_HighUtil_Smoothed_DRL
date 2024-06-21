import math

import pickle
import numpy as np
from scipy.stats import norm
eps = [0.002, 0.004, 0.006, 0.008, 0.01]
print('-----------------------------------------------------------------')
print('SPPO (SGLD)')
print('walker')
with open('sppo_sgld_walker/agents/attack-none-eps-same/rewards.pkl', 'rb') as f:
    r = pickle.load(f)
    for e in eps:
        p = norm.cdf(norm.ppf(0.5 - 0.0387) - e * math.sqrt(1000) / 0.2)
        print(np.quantile(r, p))
print('hopper')
with open('sppo_sgld_hopper/agents/attack-none-eps-same/rewards.pkl', 'rb') as f:
    r = pickle.load(f)
    for e in eps:
        p = norm.cdf(norm.ppf(0.5 - 0.0387) - e * math.sqrt(1000) / 0.2)
        print(np.quantile(r, p))
print('-----------------------------------------------------------------')
print('SPPO (radial)')
print('walker')
with open('sppo_radial_walker/agents/attack-none-eps-same/rewards.pkl', 'rb') as f:
    r = pickle.load(f)
    for e in eps:
        p = norm.cdf(norm.ppf(0.5 - 0.0387) - e * math.sqrt(1000) / 0.2)
        print(np.quantile(r, p))
print('hopper')
with open('sppo_radial_hopper/agents/attack-none-eps-same/rewards.pkl', 'rb') as f:
    r = pickle.load(f)
    for e in eps:
        p = norm.cdf(norm.ppf(0.5 - 0.0387) - e * math.sqrt(1000) / 0.2)
        print(np.quantile(r, p))
print('-----------------------------------------------------------------')
print('SPPO (s-atla)')
print('walker')
with open('sppo_atla_walker/agents/attack-none-eps-same/rewards.pkl', 'rb') as f:
    r = pickle.load(f)
    for e in eps:
        p = norm.cdf(norm.ppf(0.5 - 0.0387) - e * math.sqrt(1000) / 0.2)
        print(np.quantile(r, p))
print('hopper')
with open('sppo_atla_hopper/agents/attack-none-eps-same/rewards.pkl', 'rb') as f:
    r = pickle.load(f)
    for e in eps:
        p = norm.cdf(norm.ppf(0.5 - 0.0387) - e * math.sqrt(1000) / 0.2)
        print(np.quantile(r, p))
print('-----------------------------------------------------------------')
print('SPPO (Vanilla)')
print('walker')
with open('sppo_walker/agents/attack-none-eps-same/rewards.pkl', 'rb') as f:
    r = pickle.load(f)
    for e in eps:
        p = norm.cdf(norm.ppf(0.5 - 0.0387) - e * math.sqrt(1000) / 0.2)
        print(np.quantile(r, p))
print('hopper')
with open('sppo_hopper/agents/attack-none-eps-same/rewards.pkl', 'rb') as f:
    r = pickle.load(f)
    for e in eps:
        p = norm.cdf(norm.ppf(0.5 - 0.0387) - e * math.sqrt(1000) / 0.2)
        print(np.quantile(r, p))
print('-----------------------------------------------------------------')
print('SGLD+RS')
print('walker')
with open('robust_ppo_sgld_walker/agents/attack-none-eps-same/rewards.pkl', 'rb') as f:
    r = pickle.load(f)
    for e in eps:
        p = norm.cdf(norm.ppf(0.5 - 0.0387) - e * math.sqrt(1000) / 0.2)
        print(np.quantile(r, p))
print('hopper')
with open('robust_ppo_sgld_hopper/agents/attack-none-eps-same/rewards.pkl', 'rb') as f:
    r = pickle.load(f)
    for e in eps:
        p = norm.cdf(norm.ppf(0.5 - 0.0387) - e * math.sqrt(1000) / 0.2)
        print(np.quantile(r, p))
print('-----------------------------------------------------------------')
print('radial+RS')
print('walker')
with open('radial_ppo_walker/agents/attack-none-eps-same/rewards.pkl', 'rb') as f:
    r = pickle.load(f)
    for e in eps:
        p = norm.cdf(norm.ppf(0.5 - 0.0387) - e * math.sqrt(1000) / 0.2)
        print(np.quantile(r, p))
print('hopper')
with open('radial_ppo_hopper/agents/attack-none-eps-same/rewards.pkl', 'rb') as f:
    r = pickle.load(f)
    for e in eps:
        p = norm.cdf(norm.ppf(0.5 - 0.0387) - e * math.sqrt(1000) / 0.2)
        print(np.quantile(r, p))
print('-----------------------------------------------------------------')
print('atla+RS')
print('walker')
with open('atla_ppo_walker/agents/attack-none-eps-same/rewards.pkl', 'rb') as f:
    r = pickle.load(f)
    for e in eps:
        p = norm.cdf(norm.ppf(0.5 - 0.0387) - e * math.sqrt(1000) / 0.2)
        print(np.quantile(r, p))
print('hopper')
with open('atla_ppo_hopper/agents/attack-none-eps-same/rewards.pkl', 'rb') as f:
    r = pickle.load(f)
    for e in eps:
        p = norm.cdf(norm.ppf(0.5 - 0.0387) - e * math.sqrt(1000) / 0.2)
        print(np.quantile(r, p))
print('-----------------------------------------------------------------')
print('Vanilla+RS')
print('walker')
with open('vanilla_ppo_walker/agents/attack-none-eps-same/rewards.pkl', 'rb') as f:
    r = pickle.load(f)
    for e in eps:
        p = norm.cdf(norm.ppf(0.5 - 0.0387) - e * math.sqrt(1000) / 0.2)
        print(np.quantile(r, p))
print('hopper')
with open('vanilla_ppo_hopper/agents/attack-none-eps-same/rewards.pkl', 'rb') as f:
    r = pickle.load(f)
    for e in eps:
        p = norm.cdf(norm.ppf(0.5 - 0.0387) - e * math.sqrt(1000) / 0.2)
        print(np.quantile(r, p))