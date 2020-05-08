#!/bin/sh

# Copyright (c) 2020 Uber Technologies, Inc.

# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project. 

# See the License for the specific language governing permissions and
# limitations under the License.



game=${1:-MontezumaRevenge}
demo=${2:-`pwd`/demos}
results=${3:-`pwd`/results}
frames=${4:-2500000000}

python atari_reset/train_atari.py --game=$game --sil_vf_coef=0.01 --demo_selection=normalize_by_target --ent_coef=1e-05 --sil_ent_coef=1e-05 --n_sil_envs=2 --extra_sil_from_start_prob=0.3 --autoscale_fn=mean --sil_coef=0.1 --gamma=0.999 --move_threshold=0.1 --autoscale=10 --from_start_demo_reward_interval_factor=20000000 --nrstartsteps=160   --sil_pg_weight_by_value --sil_weight_success_rate --sil_vf_relu --sticky --noops --max_demo_len=400000 --autoscale_fn=mean --num_timesteps=$frames --autoscale_value --nenvs=32 --demo=$demo --no_game_over_on_life_loss --test_from_start --steps_per_demo=200 --no_videos --save_path=$results
