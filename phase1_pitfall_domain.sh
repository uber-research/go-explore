#!/bin/sh
# Copyright (c) 2018-2019 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

./phase1.sh \
    --game=pitfall \
    --chosen_weight=1.0 \
    --chosen_since_new_weight=1.0 \
    --horiz_weight=1.0 \
    --reset_cell_on_update \
    --batch_size=1000 \
    --max_hours=500 \
    --max_compute_steps=1000000000
