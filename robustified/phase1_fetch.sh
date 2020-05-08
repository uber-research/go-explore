#!/bin/sh

# Copyright (c) 2020 Uber Technologies, Inc.

# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project. 

# See the License for the specific language governing permissions and
# limitations under the License.



game=${1:-1000}
results=${2:-results}
frames=${3:-20000000}

python goexplore_py/main.py --keep_checkpoints --door_resolution=0.2 --door_offset=0.195 --target_location=$game  --base_path=$results --fetch_type=boxes_1 --nsubsteps=80 --total_timestep=0.08 --minmax_grip_score=00 --game=fetch --seen_weight=1.0 --max_hours=128 --checkpoint_compute=500000 --max_compute_steps=$frames --repeat_action=10 --explore_steps=30
