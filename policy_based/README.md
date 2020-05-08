# Policy-based Go-Explore

Code accompanying the paper "First return then explore", available at: [arxiv.org/abs/2004.12919](https://arxiv.org/abs/2004.12919)


## Requirements

Tested with Python 3.6. The required libraries are listed below and in  `requirements.txt`. Libraries need to be installed in the specified order. Unless otherwise specified, libraries can be installed using `pip install <library_name>`.

**Required libraries:**
- tensorflow==1.15.2
- mpi4py
- gym[Atari]
- horovod
- baselines@git+https://github.com/openai/baselines@ea25b9e8b234e6ee1bca43083f8f3cf974143998
- Pillow
- imageio
- matplotlib
- loky
- joblib
- dataclasses
- opencv-python
- cloudpickle

## Usage

To test that everything is installed correctly, a local run of policy-based Go-Explore on Montezuma's Revenge or Pitfall can be started by executing `run_policy_based_ge_montezuma.sh` or `run_policy_based_ge_pitfall.sh`, respectively. To reproduce the experiments presented in the afformentioned paper, open each file and change the following settings to:

```
NB_MPI_WORKERS=16
NB_ENVS_PER_WORKER=16
SEED=0
CHECKPOINT=200000000
```

The seed should be changed for each run. Note that, to run effeciently, this code needs to be executed in a compute environment where each worker has access to a GPU. By default, results will be written to `~/temp`, though this can be changed by editing the `sh` files.
