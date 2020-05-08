#!/bin/sh

# Copyright (c) 2020 Uber Technologies, Inc.

# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project. 

# See the License for the specific language governing permissions and
# limitations under the License.



source=$1
dest=${2:-`echo $source | sed -E 's/\/*$//'`_demo}
game=${3:-MontezumaRevenge}

python gen_demo/new_gen_demo.py --source $source --destination=$dest --game=$game --select_done --n_demos=1 --compress=bz2 --min_compute_steps=0 $4

