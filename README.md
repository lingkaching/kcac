# KCAC
Knowledge Compilation for Action Contraints in Reinforcement Learning
Constrained Amortized Q-Learning using a PSDD

# Installation:
1. Clone this library
1. create a conda env with python 3.6. and optionally run `conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge` for GPU support
1. install libraries from `requirements.txt` into your python 3.6 environment
1. install gymGame, gym-ERSLE and gym-BSS from my other github repos.
1. run scripts from `scripts/`
1. if you get the error `ModuleNotFoundError: No module named 'pysdd.sdd'` try reinstalling the PySDD package by running `pip install -vvv --upgrade --force-reinstall --no-binary :all: --no-deps pysdd`.
