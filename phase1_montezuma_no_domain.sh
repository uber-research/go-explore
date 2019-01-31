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
    --state_is_pixels \
    --seen_weight=3.0 \
    --chosen_weight=0.1 \
    --high_score_weight=0.0 \
    --max_hours=120 \
    --max_compute_steps=300000000
