## Plot Hole Detection in Fantasy Novels

This code accompanies the paper "Extrauniversal Claim Verification and Fantasy Novel Plot Hole Detection" (to be provided at some point in the future).


Steps to run:
1. install conda env using conda_env.yml: `conda env create --file=conda_env_gpu.yml`
2. install spacy lexicon: `python -m spacy download en_core_web_sm`
3. see model training options and train a model: `python train.py --help`
4. `python clean_data.py` to delete all cached data ONLY for debugging purposes (warning: this will destroy your cached data!)


**Reproducibility:** Despite our efforts to the contrary, reproducibility is not assured. Results that are slightly different than our own can occur with even small changes such as different CUDA versions or different GPUs (https://pytorch.org/docs/stable/notes/randomness.html for more info).
