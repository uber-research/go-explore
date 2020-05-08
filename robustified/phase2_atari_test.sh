#!/bin/sh

# Copyright (c) 2020 Uber Technologies, Inc.

# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project. 

# See the License for the specific language governing permissions and
# limitations under the License.



game=${1:-MontezumaRevenge}
source=${2:-`pwd`/results}
results=${3:-`pwd`/test_results}

python atari_reset/check_atari.py --num_timesteps=100000000000 --noops --sticky --num_per_noop=1000 --load_path=$source --game=$game --save_path=$results
