# teren
Linking activation space features to model behavior

## setup instructions
1. install python3.10
1. setup your private SSH key
   1. put it under in `.ssh/id_[protocol]`
   1. `chmod 600 [key]`
   1. you can debug with `ssh -T -v git@github.com`
1. clone the repo via ssh `git@github.com:jettjaniak/teren.git`
1. make virtual env `python3.10 -m venv .venv`
1. activate virtual env `source .venv/bin/activate`
1. install project in editable state `pip install -e .`
1. install pre-commit hooks `pre-commit install`
1. run `pytest`

