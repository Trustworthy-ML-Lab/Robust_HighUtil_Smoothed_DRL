#!/bin/bash

trap "kill 0" SIGINT  # exit cleanly when pressing control+C
# Set number of threads if necessary
# export nthreads=128
source paad_attack_functions.sh

# Generate a random semaphore ID
semaphorename=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 8 | head -n 1)

# Set this flag so that the sem in attack scan does not wait.
export ATTACK_FOLDER_NO_WAIT=1
export ATTACK_MODEL_NO_WAIT=1

export ATTACK_MODEL_NO_STOCHASTIC=1

# Scan folders. Check carefully about folder name and config file name.
scan_exp_folder config_walker_pa_atla_ppo.json configs/agents_attack_pa_atla_walker_scan/attack_pa_atla_walker/agents models/walker_pa_atla_median.model $semaphorename


# To stop all running process:
# killall perl; killall perl

# wait for all attacks done.
sem --wait --semaphorename $semaphorename

