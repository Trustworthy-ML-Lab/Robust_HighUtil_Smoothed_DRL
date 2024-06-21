import math

import pickle
import numpy as np
from scipy.stats import norm
eps = [0.002, 0.004, 0.006, 0.008, 0.01]
print('-----------------------------------------------------------------')
print('SPPO (WocaR)')
print('walker')
with open('sppo_wocar_walker/agents/attack-none-eps-same/rewards.pkl', 'rb') as f:
    r = pickle.load(f)
    for e in eps:
        p = norm.cdf(norm.ppf(0.5 - 0.0387) - e * math.sqrt(1000) / 0.2)
        print(np.quantile(r, p))
print('hopper')
with open('sppo_wocar_hopper/agents/attack-none-eps-same/rewards.pkl', 'rb') as f:
    r = pickle.load(f)
    for e in eps:
        p = norm.cdf(norm.ppf(0.5 - 0.0387) - e * math.sqrt(1000) / 0.2)
        print(np.quantile(r, p))
print('-----------------------------------------------------------------')
print('WocaR+RS')
print('walker')
with open('wocar_walker/agents/attack-none-eps-same/rewards.pkl', 'rb') as f:
    r = pickle.load(f)
    for e in eps:
        p = norm.cdf(norm.ppf(0.5 - 0.0387) - e * math.sqrt(1000) / 0.2)
        print(np.quantile(r, p))
print('hopper')
with open('wocar_hopper/agents/attack-none-eps-same/rewards.pkl', 'rb') as f:
    r = pickle.load(f)
    for e in eps:
        p = norm.cdf(norm.ppf(0.5 - 0.0387) - e * math.sqrt(1000) / 0.2)
        print(np.quantile(r, p))