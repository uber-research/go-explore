#!/bin/sh
# Copyright (c) 2018-2019 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

PYTHONPATH="${PYTHONPATH}:." python3 atari_reset/test_atari.py --game $1 --save_path=$3 --load_path=$2 --sticky --noops --num_timesteps=100000000
