# Go-Explore

## Requirements

Tested with Python 3.6. `requirements.txt` gives the exact libraries used on a test machine
able to run all phases on Atari.

Required libraries for the exploration phase:
- matplotlib
- loky==2.3.1
- dataclasses
- gym
- opencv-python

The ALE/atari-py is not part of Go-Explore. If you are interested in running Go-Explore on Atari environments (for example to reproduce our experiments), you may install gym\[atari\] instead of just gym. Doing so will install atari-py. atari-py is licensed under GPLv2.

Additional libraries for demo generation:
- imageio-ffmpeg (optional)
- fire
- tqdm

Additional libraries for robustification:
- tensorflow=1.5.2 (or equivalent tensorflow-gpu)
- pandas
- horovod
- filelock
- mpi4py
- baselines
    - To avoid having to install mujoco-py, install commit 6d1c6c78d38dd25799145026a590cc584ea22c88 (`pip install git+git://github.com/openai/baselines.git@6d1c6c78d38dd25799145026a590cc584ea22c88`)

To run robustification, you will need to clone [uber-research/atari-reset](https://github.com/uber-research/atari-reset) (note: this is an improved fork of the original project, which you can find at [openai/atari-reset](https://github.com/openai/atari-reset)) and
put it, copy it or link to it as `atari_reset` in the same folder as `goexplore_py`.
E.g. you could run:

`git clone https://github.com/uber-research/atari-reset atari_reset`


Running the robotics environments requires a local installation of MuJoCo 2.0 (1.5 may work too),
as well as a corresponding version of mujoco-py. mujoco-py is not included in requirements.txt as it is unnecessary
for running the Atari environments.

## Usage

The exploration phase experiments on Atari with a downscaled representation can be run with:

`./phase1_downscaled.sh <game> <path> <timesteps>`

Running the exploration phase with domain knowledge on Montezuma's Revenge and Pitfall is done using:

`./phase1_montezuma.sh <path> <timesteps>`

and 

`./phase1_pitfall.sh <path> <timesteps>`

If any argument is not supplied, a default value will be used. The default game is MontezumaRevenge for 
the downscaled experiments (for the domain knowledge experiment, there is no game argument), the default
path is results and the default number of timesteps is 500,000 for the downscaled version, corresponding to the 2 billion frames used in the
paper (due to frame skipping, one timestep corresponds to 4 frames), and 250,000 for the domain knowledge version, corresponding
to the 1 billion frames used in the paper.

The exploration phase produces a folder called `results`, and subfolders for each experiment, of the form
`0000_fb6be589a3dc44c1b561336e04c6b4cb`, where the first element is an automatically increasing
experiment id and the second element is a random string that helps prevent race condition issues if
two experiments are started at the same time and assigned the same id.

To generate demonstrations for Atari, run

`./gen_demo_atari.sh <source> <destination> <game>`

source is mandatory and is the folder produced by the exploration phase, destination will default to <source>_demo,
game defaults to MontezumaRevenge. You may also pass `--render` as a fourth argument to generate a video of your
exploration phase agent playing the game.

To robustify, put a set of `.demo` files from different runs of Phase 1 into a folder
(we used 10 in all cases, a single demonstration can also work, but is less
likely to succeed). Then run `./phase2.sh <game> <demo_folder> <results_folder> <timesteps>`. The default game is MontezumaRevenge, default demo folder is `demos`, default resulst folder is `results`
and default timesteps is `2,500,000` (corresponding to 10 billion frames as used in the paper for most games). The robustification
code doesn't handle relative paths well so it is recommended to give it absolute paths.

Important: all of the robustification results in the paper were performed with 8 GPUs through MPI. The `phase2*.sh` scripts do not start MPI themselves, you will need to do so yourself when calling them, e.g. by running `mpirun -np 8 ./phase2.sh <arguments>`. 

You may then test the performance of your trained neural network using 
`./phase2_atari_test.sh <game> <neural_net> <test_results_folder>`
where <neural_net> is one of the files produced by the robustification phase and printed in the log as `Saving to ...`.
This will produce `.json` files for each possible number of no-ops (from 0 to 30) with scores, levels
and exact action sequences produced by the test runs.

For the fetch environments, the steps are similar but with the `.sh` files containing `fetch`. In this context, `game`
is the target shelf identifier, with the valid identifiers being `0001`, `0010`, `0100` and `1000`. These scripts also need to be run on 8 GPUs with MPI as described above to reproduce our exact results.

Note that the `gen_demo`
script for fetch produces 10 demos from a single exploration phase run, so you do not need to run the exploration phase
multiple times and combine the demo files to run robustification.

Crucially, the fetch environment requires that mujoco-py be installed, which itself requires that MuJoCo 2.0 be installed.
It is also important that the folder which contains `goexplore_py` be in the PYTHONPATH during robustification.

Finally, the two controls (vanilla PPO and PPO + IM) for fetch can be run using `./control_ppo_fetch.sh` and `./control_im_fetch.sh`.
Both take as parameters the target shelf, result output folder and number of frames.