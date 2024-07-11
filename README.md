# teren
Linking activation space features to model behavior

## setup instructions
1. setup your private SSH key
   1. put it under in `.ssh/id_[protocol]`
   2. `chmod 600 [key]`
   3. you can debug with `ssh -T -v git@github.com`
1. install python3.10
2. make virtual env `python3.10 -m venv .venv`
3. activate virtual env `source .venv/bin/activate`
4. install project in editable state `pip install -e .`
5. install pre-commit hooks `pre-commit install`
6. run `pytest`

