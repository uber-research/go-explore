#!/bin/sh
#
# Copyright (c) 2020 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
# The settings below are for testing the code locally
# For the full experiment settings, change each setting to each "full experiment" value.

# Full experiment: 16
NB_MPI_WORKERS=2

# Full experiment: 16
NB_ENVS_PER_WORKER=2

# Full experiment: different for each run
SEED=0

# Full experiment: 200000000
CHECKPOINT=10000


# The game is run with both sticky actions and noops..
GAME_OPTIONS="--game pitfall --sticky_actions --noops"

# Both trajectory reward (goal_reward_factor) are 1, except for reaching the final cell, for which the reward is 3.
# Extrinsic (game) rewards are clipped to [-2, 2]. Because most Atari games have large rewards, this usually means that extrinsic rewards are twice that of the trajectory rewards.
REWARD_OPTIONS="--game_reward_factor 1 --goal_reward_factor 1 --clip_game_reward 1 --rew_clip_range=-2,2 --final_goal_reward 3"

# Cell selection is relative to: 1 / (1 + 0.5*number_of_actions_taken_in_cell).
CELL_SELECTION_OPTIONS="--selector weighted --selector_weights=attr,nb_actions_taken_in_cell,1,1,0.5 --base_weight 0"

# When the agent takes too long to reach the next cell, its intropy increases according to (inc_ent_fac*steps)^ent_inc_power.
# When exploring, this entropy increase starts when it takes more than expl_inc_ent_thresh (50) actions to reach a new cell.
# When returning, entropy increase starts relative to the time it originally took to reach the target cell.
ENTROPY_INC_OPTIONS="--entropy_strategy dynamic_increase --inc_ent_fac 0.01 --ent_inc_power 2 --ret_inc_ent_fac 1 --expl_inc_ent_thresh 50 --expl_ent_reset=on_new_cell --legacy_entropy 0"

# The cell representation for Montezuma's Revenge is a domain knowledge representation including level, room, number of keys, and the x, y coordinate of the agent.
# The x, y coordinate is discretized into bins of 36 by 18 pixels (note that the pixel of the x axis are doubled, so this is 18 by 18 on the orignal frame)
CELL_REPRESENTATION_OPTIONS="--cell_representation room_x_y --resolution=36,18"

# When following a trajectory, the agent is allowed to reach the goal cell, or any of the subsequent soft_traj_win_size (10) - 1 cells.
# While returning, the episode is terminated if it takes more than max_actions_to_goal (1000) to reach the current goal
# While exploring, the episode is terminated if it takes more than max_actions_to_new_cell (1000) to discover a new cell
# When the the final cell is reached, there is a random_exp_prob (0.5) chance that we explore by taking random actions, rather than by sampling from the policy.
EPISODE_OPTIONS="--trajectory_tracker sparse_soft --soft_traj_win_size 10 --random_exp_prob 0.5 --max_actions_to_goal 1000 --max_actions_to_new_cell 1000 --delay 0"

CHECKPOINT_OPTIONS="--checkpoint_compute ${CHECKPOINT} --clear_checkpoints trajectory"
TRAINING_OPTIONS="--goal_rep onehot --gamma 0.99 --learning_rate=2.5e-4 --no_exploration_gradients --sil=sil --max_compute_steps 10000000000"
MISC_OPTIONS="--low_prob_traj_tresh 0.01 --start_method spawn --log_info INFO --log_files __main__"
mpirun -n ${NB_MPI_WORKERS} python3 goexplore_start.py --base_path ~/temp --seed ${SEED} --nb_envs ${NB_ENVS_PER_WORKER} ${REWARD_OPTIONS} ${CELL_SELECTION_OPTIONS} ${ENTROPY_INC_OPTIONS} ${CHECKPOINT_OPTIONS} ${CELL_REPRESENTATION_OPTIONS} ${EPISODE_OPTIONS} ${GAME_OPTIONS} ${TRAINING_OPTIONS} ${MISC_OPTIONS}
