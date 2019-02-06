# Go-Explore

Paper located at: [arxiv.org/abs/1901.10995](https://arxiv.org/abs/1901.10995)

## Requirements

Tested with Python 3.6. `requirements.txt` gives the exact libraries and versions used on a test machine
able to run all phases. Unless otherwise specified, libraries can be installed using `pip install <library_name>`.

**Required libraries for Phase 1:**
- matplotlib
- loky==2.3.1
- dataclasses
- tqdm
- gym
- opencv-python

These libraries are sufficient to run Go-Explore Phase 1 with custom environments, which you may model after `goexplore_py/pitfall_env.py` and `goexplore_py/montezuma_env.py`.

The ALE/atari-py is not part of Go-Explore. If you are interested in running Go-Explore on Atari environments (for example to reproduce our experiments), you may install `gym[atari]` instead of just `gym`. Doing so will install atari-py. atari-py is licensed under GPLv2.

**Additional libraries for demo generation:**
- ffmpeg (non-Python library, install using package manager)
- imageio
- fire

Additionally, to run `gen_demo`, you will need to clone [openai/atari-demo](https://github.com/openai/atari-demo) and
put a copy or link of the subfolder `atari_demo` at `gen_demo/atari_demo` in this codebase.

E.g. you could run:

`git clone https://github.com/openai/atari-demo`

`cp -r atari-demo/atari_demo gen_demo`

**Additional libraries for Phase 2:**
- [openmpi](https://www.open-mpi.org/software/ompi/v4.0/) (non-Python library, install for source or using package manager)
- tensorflow-gpu
- pandas
- horovod (install using `HOROVOD_WITH_TENSORFLOW=1 pip install horovod`
- baselines (ignore mujoco-related errors)

Additionally, to run Phase 2, you will need to clone [uber-research/atari-reset](https://github.com/uber-research/atari-reset) (note: this is an improved fork of the original project, which you can find at [openai/atari-reset](https://github.com/openai/atari-reset)) and
put it, copy it or link to it as `atari_reset` in the root folder for this project.
E.g. you could run:

`git clone https://github.com/uber-research/atari-reset atari_reset`

## Usage

Running Phase 1 of Go-Explore can be done using the `phase1.sh` script. To see the arguments
for Phase 1, run:

`./phase1.sh --help` 

The default arguments for Phase 1 will run a domain knowledge version of Go-Explore Phase 1 on
Montezuma's Revenge. However, the default parameters do not correspond to any experiment actually
presented in the paper. To reproduce Phase 1 experiments from the paper, run one of
`./phase1_montezuma_domain.sh`, `./phase1_montezuma_no_domain.sh` or `./phase1_pitfall_domain.sh`.

Phase 1 produces a folder called `results`, and subfolders for each experiment, of the form
`0000_fb6be589a3dc44c1b561336e04c6b4cb`, where the first element is an automatically increasing
experiment id and the second element is a random string that helps prevent race condition issues if
two experiments are started at the same time and assigned the same id.

To generate demonstrations, call `./gen_demo.sh <phase1_result_folder> <destination> --game <game>`. Where `<game>` is one of "montezuma" (default) or "pitfall". The destination
will be a directory containing a `.demo` file and a `.mp4` file corresponding to the video of the
demonstration.

To robustify (run Phase 2), put a set of `.demo` files from different runs of Phase 1 into a folder
(we used 10 for Montezuma and 4 for Pitfall, a single demonstration can also work, but is less
likely to succeed). Then run `./phase2.sh <game> <demo_folder> <results_folder>` where the game is 
one of `MontezumaRevenge` or `Pitfall`. This should work with `mpirun` if you are using distributed 
training (we used 16 GPUs). The indicator of success for Phase 2 is when one of the 
`max_starting_point` displayed in the log has reached a value near 0 (values less than around 80 are
typically good). You may then test the performance of your trained neural network using 
`./phase2_test.sh <game> <neural_net> <test_results_folder>`
where <neural_net> is one of the files produced by Phase 2 and printed in the log as `Saving to ...`.
This will produce `.json` files for each possible number of no-ops (from 0 to 30) with scores, levels
and exact action sequences produced by the test runs.
