import torch
import numpy as np
from scipy.stats import norm
eps = [0.001, 0.002, 0.003, 0.004, 0.005]
print('-----------------------------------------------------------------')
print('SDQN')
print('pong')
r = torch.load('PongNoFrameskip-v4_attack_denoiser_0.1/test.log_attack-eps-0.0_nr-1000_clean_m-1_sigma-0.1_clip_nat_all_rewards.pt')
for e in eps:
    p = norm.cdf(norm.ppf(0.5-0.0387)-e*50/0.1)
    print(np.quantile(r, p))
print('Freeway')
r = torch.load('FreewayNoFrameskip-v4_attack_denoiser_0.1/test.log_attack-eps-0.0_nr-1000_clean_m-1_sigma-0.1_clip_nat_all_rewards.pt')
for e in eps:
    p = norm.cdf(norm.ppf(0.5-0.0387)-e*50/0.1)
    print(np.quantile(r, p))
print('RoadRunner')
r = torch.load('RoadRunnerNoFrameskip-v4_attack_denoiser_0.05/test.log_attack-eps-0.0_nr-1000_clean_m-1_sigma-0.05_clip_nat_all_rewards.pt')
for e in eps:
    p = norm.cdf(norm.ppf(0.5-0.0387)-e*50/0.05)
    print(np.quantile(r, p))
print('-----------------------------------------------------------------')
print('SDQN (Radial)')
print('pong')
r = torch.load('PongNoFrameskip-v4_attack_rad_denoiser_0.1/test.log_attack-eps-0.0_nr-1000_clean_m-1_sigma-0.1_clip_rad_all_rewards.pt')
for e in eps:
    p = norm.cdf(norm.ppf(0.5-0.0387)-e*50/0.1)
    print(np.quantile(r, p))
print('Freeway')
r = torch.load('FreewayNoFrameskip-v4_attack_rad_denoiser_0.1/test.log_attack-eps-0.0_nr-1000_clean_m-1_sigma-0.1_clip_rad_all_rewards.pt')
for e in eps:
    p = norm.cdf(norm.ppf(0.5-0.0387)-e*50/0.1)
    print(np.quantile(r, p))
print('RoadRunner')
r = torch.load('RoadRunnerNoFrameskip-v4_attack_rad_denoiser_0.05/test.log_attack-eps-0.0_nr-1000_clean_m-1_sigma-0.05_clip_rad_all_rewards.pt')
for e in eps:
    p = norm.cdf(norm.ppf(0.5-0.0387)-e*50/0.05)
    print(np.quantile(r, p))
print('-----------------------------------------------------------------')
print('SDQN (SPGD)')
print('pong')
r = torch.load('PongNoFrameskip-v4_attack_denoiser_adv_0.1/test.log_attack-eps-0.0_nr-1000_clean_m-1_sigma-0.1_clip_nat_all_rewards.pt')
for e in eps:
    p = norm.cdf(norm.ppf(0.5-0.0387)-e*50/0.1)
    print(np.quantile(r, p))
print('Freeway')
r = torch.load('FreewayNoFrameskip-v4_attack_denoiser_adv_0.1/test.log_attack-eps-0.0_nr-1000_clean_m-1_sigma-0.1_clip_nat_all_rewards.pt')
for e in eps:
    p = norm.cdf(norm.ppf(0.5-0.0387)-e*50/0.1)
    print(np.quantile(r, p))
print('RoadRunner')
r = torch.load('RoadRunnerNoFrameskip-v4_attack_denoiser_adv_0.05/test.log_attack-eps-0.0_nr-1000_clean_m-1_sigma-0.05_clip_nat_all_rewards.pt')
for e in eps:
    p = norm.cdf(norm.ppf(0.5-0.0387)-e*50/0.05)
    print(np.quantile(r, p))
print('-----------------------------------------------------------------')
print('Radial+RS')
print('pong')
r = torch.load('PongNoFrameskip-v4_attack_rad/test.log_attack-eps-0.0_nr-1000_clean_m-1_sigma-0.1_clip_rad_all_rewards.pt')
for e in eps:
    p = norm.cdf(norm.ppf(0.5-0.0387)-e*50/0.1)
    print(np.quantile(r, p))
print('Freeway')
r = torch.load('FreewayNoFrameskip-v4_attack_rad/test.log_attack-eps-0.0_nr-1000_clean_m-1_sigma-0.1_clip_rad_all_rewards.pt')
for e in eps:
    p = norm.cdf(norm.ppf(0.5-0.0387)-e*50/0.1)
    print(np.quantile(r, p))
print('RoadRunner')
r = torch.load('RoadRunnerNoFrameskip-v4_attack_rad/test.log_attack-eps-0.0_nr-1000_clean_m-1_sigma-0.05_clip_rad_all_rewards.pt')
for e in eps:
    p = norm.cdf(norm.ppf(0.5-0.0387)-e*50/0.05)
    print(np.quantile(r, p))
print('-----------------------------------------------------------------')
print('Convex+RS')
print('pong')
r = torch.load('PongNoFrameskip-v4_attack_cov/test.log_attack-eps-0.0_nr-1000_clean_m-1_sigma-0.1_clip_cov_all_rewards.pt')
for e in eps:
    p = norm.cdf(norm.ppf(0.5-0.0387)-e*50/0.1)
    print(np.quantile(r, p))
print('Freeway')
r = torch.load('FreewayNoFrameskip-v4_attack_cov/test.log_attack-eps-0.0_nr-1000_clean_m-1_sigma-0.1_clip_cov_all_rewards.pt')
for e in eps:
    p = norm.cdf(norm.ppf(0.5-0.0387)-e*50/0.1)
    print(np.quantile(r, p))
print('RoadRunner')
r = torch.load('RoadRunnerNoFrameskip-v4_attack_cov/test.log_attack-eps-0.0_nr-1000_clean_m-1_sigma-0.05_clip_cov_all_rewards.pt')
for e in eps:
    p = norm.cdf(norm.ppf(0.5-0.0387)-e*50/0.05)
    print(np.quantile(r, p))
print('-----------------------------------------------------------------')
print('WocaR+RS')
print('pong')
r = torch.load('PongNoFrameskip-v4_attack_woc/test.log_attack-eps-0.0_nr-1000_clean_m-1_sigma-0.1_clip_woc_all_rewards.pt')
for e in eps:
    p = norm.cdf(norm.ppf(0.5-0.0387)-e*50/0.1)
    print(np.quantile(r, p))
print('Freeway')
r = torch.load('FreewayNoFrameskip-v4_attack_woc/test.log_attack-eps-0.0_nr-1000_clean_m-1_sigma-0.1_clip_woc_all_rewards.pt')
for e in eps:
    p = norm.cdf(norm.ppf(0.5-0.0387)-e*50/0.1)
    print(np.quantile(r, p))
print('RoadRunner')
r = torch.load('RoadRunnerNoFrameskip-v4_attack_woc/test.log_attack-eps-0.0_nr-1000_clean_m-1_sigma-0.05_clip_woc_all_rewards.pt')
for e in eps:
    p = norm.cdf(norm.ppf(0.5-0.0387)-e*50/0.05)
    print(np.quantile(r, p))
print('-----------------------------------------------------------------')
print('Vanilla+RS')
print('pong')
r = torch.load('PongNoFrameskip-v4_attack/test.log_attack-eps-0.0_nr-1000_clean_m-1_sigma-0.1_clip_nat_all_rewards.pt')
for e in eps:
    p = norm.cdf(norm.ppf(0.5-0.0387)-e*50/0.1)
    print(np.quantile(r, p))
print('Freeway')
r = torch.load('FreewayNoFrameskip-v4_attack/test.log_attack-eps-0.0_nr-1000_clean_m-1_sigma-0.1_clip_nat_all_rewards.pt')
for e in eps:
    p = norm.cdf(norm.ppf(0.5-0.0387)-e*50/0.1)
    print(np.quantile(r, p))
print('RoadRunner')
r = torch.load('RoadRunnerNoFrameskip-v4_attack/test.log_attack-eps-0.0_nr-1000_clean_m-1_sigma-0.05_clip_nat_all_rewards.pt')
for e in eps:
    p = norm.cdf(norm.ppf(0.5-0.0387)-e*50/0.05)
    print(np.quantile(r, p))