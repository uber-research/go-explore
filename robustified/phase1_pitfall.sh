#!/bin/sh

# Copyright (c) 2020 Uber Technologies, Inc.

# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project. 

# See the License for the specific language governing permissions and
# limitations under the License.



results=${2:-results}
frames=${3:-250000000}

python goexplore_py/main.py --seen_weight=1 --chosen_weight=0 --chosen_since_new_weight=0 --high_score_weight=0 --horiz_weight=0 --vert_weight=0 --pitfall_treasure_type=none --remember_rooms --reset_cell_on_update --game=pitfall --max_hours=256 --max_compute_steps=$frames --base_path=$results

