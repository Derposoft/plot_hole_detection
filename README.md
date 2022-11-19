Steps to run:

1. install conda env using conda_env.yml: `conda env create --file=conda_env_gpu.yml`
2. install spacy lexicon: `python -m spacy download en_core_web_sm`
3. see model training options and train a model: `python train.py --help`
4. `python clean_data.py` to delete all cached data ONLY for debugging purposes (warning: this will destroy your cached data!)
