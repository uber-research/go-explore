#!/bin/sh

# Copyright (c) 2020 Uber Technologies, Inc.

# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project. 

# See the License for the specific language governing permissions and
# limitations under the License.



game=${1:-MontezumaRevenge}
results=${2:-results}
frames=${3:-500000000}

python goexplore_py/main.py --seen_weight=1.0 --reset_cell_on_update --base_path=$results --game=generic_$game --cell_split_factor=0.125 --first_compute_archive_size=1 --first_compute_dynamic_state=10000 --max_archive_size=50000 --max_recent_frames=10000 --recent_frame_add_prob=0.01 --recompute_dynamic_state_every=10000000 --split_iterations=3000 --state_is_pixels --high_score_weight=0.0 --max_hours=256 --max_compute_steps=$frames --n_cpus=88 --batch_size=100 --dynamic_state
