import sys
sys.path.append("./common")
sys.path.append("./auto_LiRPA")
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from argparser import argparser
from eps_scheduler import EpsilonScheduler
from read_config import load_config
import numpy as np
import cpprb
import re
from attacks import attack
import random
from common.wrappers import make_atari, wrap_deepmind, wrap_pytorch, make_atari_cart
from models import QNetwork, model_setup
from models_denoiser import Denoiser, denoiser_setup
import torch.optim as optim
import torch
from torch.nn import CrossEntropyLoss
import torch.autograd as autograd
import math
import time
import os
import argparse
from datetime import datetime
from utils import CudaTensorManager, ActEpsilonScheduler, BufferBetaScheduler, Logger, update_target, get_acrobot_eps, plot 
from my_replay_buffer import ReplayBuffer, NaivePrioritizedBuffer
from common.replay_buffer import PrioritizedReplayBuffer
from async_env import AsyncEnv
from async_rb import AsyncReplayBuffer

import torch.nn.functional as F
from torch.distributions.normal import Normal


USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)
UINTS=[np.uint8, np.uint16, np.uint32, np.uint64]


class TimeLogger(object):
    def __init__(self):
        self.time_logs = {}

    def log_time(self, time_id, time):
        if time_id not in self.time_logs:
            self.time_logs[time_id] = 0.0
        self.time_logs[time_id] += time

    def __call__(self, time_id, time):
        self.log_time(time_id, time)

    def clear(self):
        self.time_logs = {}

    def print(self):
        print_str = ""
        for t in self.time_logs:
            print_str += "{}={:.4f} ".format(t, self.time_logs[t])
        print(print_str + "\n")

log_time = TimeLogger()
def action_scheduler(f_index, num_f, action_epsilon):
    return action_epsilon * ((num_f-f_index) / num_f)

def sigma_scheduler(f_index, num_f, sigma, start_sigma=1500000, end_increase_sigma=5500000):
    if f_index <= start_sigma:
        return 0.0
    elif f_index > end_increase_sigma:
        return sigma
    else:
        return sigma * ((float(f_index)-start_sigma) / (end_increase_sigma-start_sigma))

def compute_td_loss(current_denoiser, target_denoiser, current_model, target_model, batch_size, replay_buffer, per, use_cpp_buffer, use_async_rb, optimizer, gamma, memory_mgr, robust, a_dim, **kwargs):
    t = time.time()
    dtype = kwargs['dtype']
    if per:
        buffer_beta = kwargs['buffer_beta']
        if use_async_rb:
            if not replay_buffer.sample_available():
                replay_buffer.async_sample(batch_size, buffer_beta)
            res = replay_buffer.wait_sample()
            replay_buffer.async_sample(batch_size, buffer_beta)
        else:
            res = replay_buffer.sample(batch_size, buffer_beta)
        if use_cpp_buffer:
            state, action, reward, next_state, done, indices, weights = res['obs'], res['act'], res['rew'], res['next_obs'], res['done'], res['indexes'], res['weights']
        else:
            state, action, reward, next_state, done, weights, indices = res[0], res[1], res[2], res[3], res[4], res[5], res[6]
    else:
        if use_async_rb:
            if replay_buffer.sample_available():
                replay_buffer.async_sample(batch_size)
            res = replay_buffer.wait_sample()
            replay_buffer.async_sample(batch_size)
        else:
            res = replay_buffer.sample(batch_size)
        if use_cpp_buffer:
            state, action, reward, next_state, done = res['obs'], res['act'], res['rew'], res['next_obs'], res['done']
        else:
            state, action, reward, next_state, done = res[0], res[1], res[2], res[3], res[4]
    if use_cpp_buffer and not use_async_rb:
        action = action.transpose()[0].astype(int)
        reward = reward.transpose()[0].astype(int)
        done = done.transpose()[0].astype(int)
    log_time('sample_time', time.time() - t)

    t = time.time()
    numpy_weights = weights
    if per:
        state, next_state, action, reward, done, weights = memory_mgr.get_cuda_tensors(state, next_state, action, reward, done, weights)
    else:
        state, next_state, action, reward, done = memory_mgr.get_cuda_tensors(state, next_state, action, reward, done)

    bound_solver = kwargs.get('bound_solver', 'cov')
    optimizer[0].zero_grad()
    optimizer[1].zero_grad()

    state = state.to(torch.float)
    next_state = next_state.to(torch.float)
    # Normalize input pixel to 0-1
    if dtype in UINTS:
        state /= 255
        next_state /= 255
        state_max = 1.0
        state_min = 0.0
    else:
        state_max = float('inf')
        state_min = float('-inf')
    beta = kwargs.get('beta', 0)

    #certified radius training param
    sigma_schedule = kwargs['sigma_schedule']
    update_rl_model = kwargs['update_rl_model']

    im = current_denoiser(state, sigma_schedule=sigma_schedule)
    cur_q_logits = current_model(im)
    tgt_next_q_logits = target_model(next_state)

    cur_q_value = cur_q_logits.gather(1, action.unsqueeze(1)).squeeze(1)

    tgt_next_q_value = tgt_next_q_logits.max(1)[0]
    expected_q_value = reward + gamma * tgt_next_q_value * (1 - done)

    loss_fn = torch.nn.SmoothL1Loss(reduction='none')
    loss = loss_fn(cur_q_value, expected_q_value.detach())
    td = loss_fn(cur_q_value, expected_q_value.detach())

    mse_loss = torch.nn.MSELoss(size_average=None, reduce=None, reduction='sum')
    mse = mse_loss(im, state)


    if per:
        loss = loss * weights
        td = td * weights
        prios = loss + 1e-5
        weights_norm = np.linalg.norm(numpy_weights)

    batch_cur_q_value = torch.mean(cur_q_value)
    batch_exp_q_value = torch.mean(expected_q_value)
    loss = loss.mean() + mse
    td = td.mean()
    td_loss = td.clone()

    loss.backward()

    # Gradient clipping.
    grad_norm = 0.0
    max_norm = kwargs['grad_clip']
    if max_norm > 0:
        parameters = current_model.parameters()
        for p in parameters:
            grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = np.sqrt(grad_norm)
        clip_coef = max_norm / (grad_norm + 1e-6)
        if clip_coef < 1:
            for p in parameters:
                p.grad.data.mul_(clip_coef)

    # update weights
    if update_rl_model:
        optimizer[0].step()
        optimizer[1].step()
    else:
        optimizer[0].step()

    nn_time = time.time() - t
    log_time('nn_time', time.time() - t)
    t = time.time()
    if per:
        replay_buffer.update_priorities(indices, prios.data.cpu().numpy())
    log_time('reweight_time', time.time() - t)

    res = (loss, grad_norm, weights_norm, td_loss, batch_cur_q_value, batch_exp_q_value)

    return res


def mini_test(denoiser, model, config, logger, dtype, num_episodes=10, max_frames_per_episode=30000, sigma_schedule=0.0):
    logger.log('start mini test')
    training_config = config['training_config']
    env_params = training_config['env_params']
    env_params['clip_rewards'] = False
    env_params['episode_life'] = False
    env_id = config['env_id']

    if 'NoFrameskip' not in env_id:
        env = make_atari_cart(env_id)
    else:
        env = make_atari(env_id)
        env = wrap_deepmind(env, **env_params)
        env = wrap_pytorch(env)
    state = env.reset()
    all_rewards = []
    episode_reward = 0

    seed = random.randint(0, sys.maxsize)
    logger.log('reseting env with seed', seed)
    env.seed(seed)
    state = env.reset()

    episode_idx = 1
    this_episode_frame = 1
    for frame_idx in range(1, num_episodes * max_frames_per_episode + 1):
        state_tensor = torch.from_numpy(np.ascontiguousarray(state)).unsqueeze(0).cuda().to(torch.float32)
        if dtype in UINTS:
            state_tensor /= 255
        im = denoiser(state_tensor, sigma_schedule=sigma_schedule)
        action = model.act(im)[0]
        next_state, reward, done, _ = env.step(action)

        # logger.log(action)
        state = next_state
        episode_reward += reward
        if this_episode_frame == max_frames_per_episode:
            logger.log('maximum number of frames reached in this episode, reset environment!')
            done = True

        if done:
            logger.log('reseting env with seed', seed)
            state = env.reset()
            all_rewards.append(episode_reward)
            logger.log('episode {}/{} reward: {:6g}'.format(episode_idx, num_episodes, all_rewards[-1]))
            episode_reward = 0
            this_episode_frame = 1
            episode_idx += 1
            if episode_idx > num_episodes:
                break
        else:
            this_episode_frame += 1
    return np.mean(all_rewards)


def main(args):
    config = load_config(args)
    prefix = config['env_id']
    training_config = config['training_config']
    if config['name_suffix']:
        prefix += config['name_suffix']
    if config['path_prefix']:
       prefix = os.path.join(config['path_prefix'], prefix)
    if not os.path.exists(prefix):
        os.makedirs(prefix)

    train_log = os.path.join(prefix, 'train.log')
    logger = Logger(open(train_log, "w"))
    logger.log('Command line:', " ".join(sys.argv[:]))
    logger.log(args)
    logger.log(config)

    # certified radius training config
    sigma = training_config['sigma']
    sigma_start = training_config['sigma_start']
    end_increase_sigma = training_config['end_increase_sigma']
    lr_denoiser = training_config['lr_denoiser']
    action_epsilon = training_config['action_epsilon']
    update_rl_model = training_config['update_rl_model']
    logger.log("sigma: {}, sigma_start: {}, end_increase_sigma: {}, lr_denoiser: {}, action_epsilon: {}, update_rl_model: {}".format(sigma, sigma_start, end_increase_sigma, lr_denoiser, action_epsilon, update_rl_model))


    env_params = training_config['env_params']
    env_id = config['env_id']
    if "NoFrameskip" not in env_id:
        env = make_atari_cart(env_id)
    else:
        env = make_atari(env_id)
        env = wrap_deepmind(env, **env_params)
        env = wrap_pytorch(env)

    seed = training_config['seed']
    env.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    state = env.reset()
    dtype = state.dtype
    logger.log("env_shape: {}, num of actions: {}".format(env.observation_space.shape, env.action_space.n))
    if "NoFrameskip" in env_id:
        logger.log('action meaning:', env.unwrapped.get_action_meanings()[:env.action_space.n])

    robust = training_config.get('robust', False)
    adv_train = training_config.get('adv_train', False)
    bound_solver = training_config.get('bound_solver', 'cov')
    attack_config = {}

    model_width = training_config['model_width']
    robust_model = robust and bound_solver != 'pgd'
    dueling = training_config.get('dueling', True)
    current_denoiser = denoiser_setup(env, USE_CUDA)
    target_denoiser = denoiser_setup(env, USE_CUDA)
    current_model = model_setup(env_id, env, robust_model, logger, USE_CUDA, dueling, model_width)
    target_model = model_setup(env_id, env, robust_model, logger, USE_CUDA, dueling, model_width)

    load_path = training_config["load_model_path"]
    denoiser_path = training_config["load_denoiser_path"]
    load_frame = 1
    if load_path != "" and os.path.exists(load_path):
        current_model.features.load_state_dict(torch.load(load_path))
        target_model.features.load_state_dict(torch.load(load_path))
    if denoiser_path != "" and os.path.exists(denoiser_path):
        current_denoiser.load_state_dict(torch.load(denoiser_path))
        target_denoiser.load_state_dict(torch.load(denoiser_path))

    lr = training_config['lr']
    grad_clip = training_config['grad_clip']
    natural_loss_fn = training_config['natural_loss_fn']
    optimizer = optim.Adam(current_model.parameters(), lr=lr, eps=training_config['adam_eps'])
    denoiser_optimizer = optim.Adam(current_denoiser.parameters(), lr=lr_denoiser, eps=training_config['adam_eps'])
    # Do not evaluate gradient for target model.
    for param in target_model.features.parameters():
        param.requires_grad = False

    buffer_config = training_config['buffer_params']
    replay_initial = buffer_config['replay_initial']
    buffer_capacity = buffer_config['buffer_capacity']
    use_cpp_buffer = training_config["cpprb"]
    use_async_rb = training_config['use_async_rb']
    num_frames = training_config['num_frames']
    batch_size = training_config['batch_size']
    gamma = training_config['gamma']

    if use_cpp_buffer:
        logger.log('using cpp replay buffer')
        if use_async_rb:
            replay_buffer_ctor = AsyncReplayBuffer(initial_state=state, batch_size=batch_size)
        else:
            replay_buffer_ctor = cpprb.PrioritizedReplayBuffer
    else:
        logger.log('using python replay buffer')
    per = training_config['per']

    if per:
        logger.log('using prioritized experience replay.')
        alpha = buffer_config['alpha']
        buffer_beta_start = buffer_config['buffer_beta_start']
        buffer_beta_frames = buffer_config.get('buffer_beta_frames', -1)
        if buffer_beta_frames < replay_initial:
            buffer_beta_frames = num_frames - replay_initial
            logger.log('beffer_beta_frames reset to ', buffer_beta_frames)
        buffer_beta_scheduler = BufferBetaScheduler(buffer_beta_start, buffer_beta_frames, start_frame=replay_initial)
        if use_cpp_buffer:
            replay_buffer = replay_buffer_ctor(size=buffer_capacity,
                    # env_dict={"obs": {"shape": state.shape, "dtype": np.uint8},
                    env_dict={"obs": {"shape": state.shape, "dtype": dtype},
                        "act": {"shape": 1, "dtype": np.uint8},
                        "rew": {},
                        # "next_obs": {"shape": state.shape, "dtype": np.uint8},
                        "next_obs": {"shape": state.shape, "dtype": dtype},
                        "done": {}}, alpha=alpha, eps = 0.0)  # We add eps manually in training loop
        else:
            replay_buffer = PrioritizedReplayBuffer(buffer_capacity, alpha=alpha)

    else:
        logger.log('using regular replay.')
        if use_cpp_buffer:
            replay_buffer =cpprb.ReplayBuffer(buffer_capacity,
                    # {"obs": {"shape": state.shape, "dtype": np.uint8},
                    {"obs": {"shape": state.shape, "dtype": dtype},
                        "act": {"shape": 1, "dtype": np.uint8},
                        "rew": {},
                        # "next_obs": {"shape": state.shape, "dtype": np.uint8},
                        "next_obs": {"shape": state.shape, "dtype": dtype},
                        "done": {}})
        else:
            replay_buffer = ReplayBuffer(buffer_capacity)

    update_target(current_model, target_model)
    update_target(current_denoiser, target_denoiser)

    act_epsilon_start = training_config['act_epsilon_start']
    act_epsilon_final = training_config['act_epsilon_final']
    act_epsilon_decay = training_config['act_epsilon_decay']
    act_epsilon_method = training_config['act_epsilon_method']
    if training_config.get('act_epsilon_decay_zero', True):
        decay_zero = num_frames
    else:
        decay_zero = None
    act_epsilon_scheduler = ActEpsilonScheduler(act_epsilon_start, act_epsilon_final, act_epsilon_decay, method=act_epsilon_method, start_frame=replay_initial, decay_zero=decay_zero)

    # Use optimized cuda memory management
    memory_mgr = CudaTensorManager(state.shape, batch_size, per, USE_CUDA, dtype=dtype)

    losses = []
    td_losses = []
    batch_cur_q = []
    batch_exp_q = []

    sa = None
    kappa = None
    hinge = False

    if training_config['use_async_env']:
        # Create an environment in a separate process, run asychronously
        async_env = AsyncEnv(env_id, result_path=prefix, draw=training_config['show_game'], record=training_config['record_game'], env_params=env_params, seed=seed)

    # initialize parameters in logging
    all_rewards = []
    episode_reward = 0
    grad_norm = np.nan
    weights_norm = np.nan
    best_test_reward = -float('inf')
    buffer_stored_size = 0

    start_time = time.time()
    period_start_time = time.time()

    # Main Loop
    for frame_idx in range(load_frame, num_frames + 1):
        #certified radius training schedule
        sigma_schedule = sigma_scheduler(frame_idx, num_frames, sigma, start_sigma=sigma_start, end_increase_sigma=end_increase_sigma)
        # action_schedule = action_scheduler(frame_idx, num_frames, action_epsilon)
        # Step 1: get current action
        frame_start = time.time()
        t = time.time()

        eps = 0

        act_epsilon = act_epsilon_scheduler.get(frame_idx)

        with torch.no_grad():
            state_tensor = torch.from_numpy(np.ascontiguousarray(state)).unsqueeze(0).cuda().to(torch.float32)
            if dtype in UINTS:
                state_tensor /= 255
            ori_state_tensor = torch.clone(state_tensor)
            im = current_denoiser(state_tensor, sigma_schedule=sigma_schedule)
            action = current_model.act(im, act_epsilon)[0]

        # torch.cuda.synchronize()
        log_time('act_time', time.time() - t)

        # Step 2: run environment
        t = time.time()
        if training_config['use_async_env']:
            async_env.async_step(action)
        else:
            next_state, reward, done, _ = env.step(action)
        log_time('env_time', time.time() - t)


        # Step 3: save to buffer
        # For asynchronous env, defer saving
        if not training_config['use_async_env']:
            t = time.time()
            if use_cpp_buffer:
                replay_buffer.add(obs=state, act=action, rew=reward, next_obs=next_state, done=done)
            else:
                replay_buffer.push(state, action, reward, next_state, done)
            log_time('save_time', time.time() - t)

        if use_cpp_buffer:
            buffer_stored_size = replay_buffer.get_stored_size()
        else:
            buffer_stored_size = len(replay_buffer)

        beta = np.nan
        buffer_beta = np.nan
        t = time.time()

        if buffer_stored_size > replay_initial:
            if training_config['per']:
                buffer_beta = buffer_beta_scheduler.get(frame_idx)

            res = compute_td_loss(current_denoiser, target_denoiser, current_model, target_model, batch_size, replay_buffer, per, use_cpp_buffer, use_async_rb, [denoiser_optimizer, optimizer], gamma, memory_mgr, robust, env.action_space.n, buffer_beta=buffer_beta, grad_clip=grad_clip, natural_loss_fn=natural_loss_fn, eps=eps, beta=beta, sa=sa, kappa=kappa, dtype=dtype, hinge=hinge, hinge_c=training_config.get('hinge_c', 1), env_id=env_id, bound_solver=bound_solver, attack_config=attack_config, sigma_schedule=sigma_schedule, update_rl_model=update_rl_model)
            loss, grad_norm, weights_norm, td_loss, batch_cur_q_value, batch_exp_q_value = res[0], res[1], res[2], res[3], res[4], res[5]

            losses.append(loss.data.item())
            td_losses.append(td_loss.data.item())
            batch_cur_q.append(batch_cur_q_value.data.item())
            batch_exp_q.append(batch_exp_q_value.data.item())

        log_time('loss_time', time.time() - t)

        # Step 2: run environment (async)
        t = time.time()
        if training_config['use_async_env']:
            next_state, reward, done, _ = async_env.wait_step()
        log_time('env_time', time.time() - t)

        # Step 3: save to buffer (async)
        if training_config['use_async_env']:
            t = time.time()
            if use_cpp_buffer:
                replay_buffer.add(obs=state, act=action, rew=reward, next_obs=next_state, done=done)
            else:
                replay_buffer.push(state, action, reward, next_state, done)
            log_time('save_time', time.time() - t)

        # Update states and reward
        t = time.time()
        state = next_state
        episode_reward += reward
        if done:
            if training_config['use_async_env']:
                state = async_env.reset()
            else:
                state = env.reset()
            all_rewards.append(episode_reward)
            episode_reward = 0
        log_time('env_time', time.time() - t)

        # All kinds of result logging
        if frame_idx % training_config['print_frame'] == 0 or frame_idx==num_frames or abs(buffer_stored_size-replay_initial) < 5:
            logger.log('\nframe {}/{}, learning rate: {:.6g}, buffer beta: {:.6g}, action epsilon: {:.6g}'.format(frame_idx, num_frames, lr, buffer_beta, act_epsilon))
            logger.log('total time: {:.2f}, epoch time: {:.4f}, speed: {:.2f} frames/sec, last total loss: {:.6g}, avg total loss: {:.6g}, grad norm: {:.6g}, weights_norm: {:.6g}, latest episode reward: {:.6g}, avg 10 episode reward: {:.6g}'.format(
                time.time() - start_time,
                time.time() - period_start_time,
                training_config['print_frame'] / (time.time() - period_start_time),
                losses[-1] if losses else np.nan,
                np.average(losses[:-training_config['print_frame']-1:-1]) if losses else np.nan,
                grad_norm, weights_norm,
                all_rewards[-1] if all_rewards else np.nan,
                np.average(all_rewards[:-11:-1]) if all_rewards else np.nan))
            logger.log('last td loss: {:.6g}, avg td loss: {:.6g}'.format(
                    td_losses[-1] if td_losses else np.nan,
                    np.average(td_losses[:-training_config['print_frame']-1:-1]) if td_losses else np.nan))
            logger.log('last batch cur q: {:.6g}, avg batch cur q: {:.6g}'.format(
                    batch_cur_q[-1] if batch_cur_q else np.nan,
                    np.average(batch_cur_q[:-training_config['print_frame']-1:-1]) if batch_cur_q else np.nan))
            logger.log('last batch exp q: {:.6g}, avg batch exp q: {:.6g}'.format(
                    batch_exp_q[-1] if batch_exp_q else np.nan,
                    np.average(batch_exp_q[:-training_config['print_frame']-1:-1]) if batch_exp_q else np.nan))
            logger.log('sigma:{:.6g}'.format(sigma_schedule))

            period_start_time = time.time()
            log_time.print()
            log_time.clear()

        if frame_idx % training_config['save_frame'] == 0 or frame_idx==num_frames:
            plot(frame_idx, all_rewards, losses, prefix)
            torch.save(current_model.features.state_dict(), '{}/dqn_frame_{}.pth'.format(prefix, frame_idx))
            torch.save(current_denoiser.state_dict(), '{}/denoiser_frame_{}.pth'.format(prefix, frame_idx))

        # if frame_idx % training_config['update_target_frame'] == 0:
        #     update_target(current_model, target_model)
        #     update_target(current_denoiser, target_denoiser)

        if frame_idx % training_config.get('mini_test', 100000) == 0:
            test_reward = mini_test(current_denoiser, current_model, config, logger, dtype, sigma_schedule=sigma_schedule)
            logger.log('this test avg reward: {:6g}'.format(test_reward))
            if test_reward >= best_test_reward:
                best_test_reward = test_reward
                logger.log('new best reward {:6g} achieved, update checkpoint'.format(test_reward))
                torch.save(current_model.features.state_dict(), '{}/best_dqn_frame_{}.pth'.format(prefix, frame_idx))
                torch.save(current_denoiser.state_dict(), '{}/best_denoiser_frame_{}.pth'.format(prefix, frame_idx))

        log_time.log_time('total', time.time() - frame_start)


if __name__ == "__main__":
    args = argparser()
    main(args)
