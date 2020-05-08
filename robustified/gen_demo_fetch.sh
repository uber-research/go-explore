#!/bin/sh

# Copyright (c) 2020 Uber Technologies, Inc.

# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project. 

# See the License for the specific language governing permissions and
# limitations under the License.



source=$1
dest=${2:-`echo $source | sed -E 's/\/*$//'`_demo}
game=${3:-1000}

python gen_demo/new_gen_demo.py --fetch_target_location=$game --source=$source --destination=$dest --fetch_type=boxes_1 --fetch_nsubsteps=80 --fetch_total_timestep=0.08 --game=fetch --render --n_demos=10 --select_reward --compress=bz2 --min_compute_steps=0

