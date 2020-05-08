#!/bin/sh

# Copyright (c) 2020 Uber Technologies, Inc.

# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project. 

# See the License for the specific language governing permissions and
# limitations under the License.



game=${1:-1000}
demo=${2:-`pwd`/demos}
results=${3:-`pwd`/results}
frames=${4:-750000000}

python atari_reset/train_atari.py --sil_vf_c=0.1 --ffmem=128 --ffsh=1x256 --learning_r=0.0001 --demo_sel=normalize_by_target --nrst=40 --ent_coef=1e-05 --sil_coef=0.1 --extra_frames_exp_factor=4 --allowed_lag=10  --sil_ent=1e-05 --sil_weight_success_rate --sil_vf_relu --num_timesteps $frames --sil_pg_weight_by_value  --fetch_target_location=$game --extra_sil_from_start_prob=0 --extra_sil_before_demo_max=10 --fetch_type=boxes_1 --fetch_nsubsteps=80 --fetch_total_timestep=0.08 --nenvs=120 --n_sil_envs=8 --sd_multiply_explore=2 --inc_entropy_threshold=10 --fetch_incl_extra_full_state --game=fetch --demo $demo --gamma=0.99 --vf_coef=0.5 --steps_per_demo=100 --move_threshold=0.1 --save_path=$results
