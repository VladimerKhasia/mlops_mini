# mlops_mini



Workflow:
`artifact_config.yaml` [`params.yaml`, `settings.py`] -> `schemas.py` -> `utils.py` -> `config_manager.py` -> `components` -> `pipeline` -> `main.py` -> `dvc.yaml` -> `app`

This is how requirements.txt was populated originally `pip freeze > requirements.txt`. You do not need to do this except when you install some additional package you want to be the part of requirements. Now, check `requirements.txt` :
- pytorch, torchvision, torchaudio etc. may have `+cu118` as suffix which must be commented out to avoid complications on windows.
- Install requirements: `pip install -r requirements.txt`
- Also, change `-e some_path/to_local_repo` into `-e .` and then 
- Additionally install accelerate if you are going to use GPU `pip install accelerate`
- Instead of `wandb` you can use `mlflow`. In that case `pip install mlflow`

For additional documentation you can:
pip install [mkdocs](https://www.mkdocs.org/) or [sphinx](https://www.sphinx-doc.org/) use `sphinx-quickstart`.

`dvc init` -> `dvc repro` -> [`git add dvc.lock` >> ` dvc config core.autostage true` (for `dvc add something`) >> `dvc remote add -d origin https://github.com/VladimerKhasia/mlops_mini` >> `dvc push`] -> `dvc dag` 
- `dvc repro` runs and tracks your pipelines. It also means that instead of running `python main.py` you run your code with `dvc repro` which uses your `dvc.yaml` file. DVC may ask you to remove some  directories from being traked by Git. Sometimes after adding some files and directories to .gitignore it still traks them. You can check the status if it is true and if it is you can remove them. e.g. this one remove artifacts directory `git rm -r --cached artifacts`
- `git add dvc.lock dvc.yaml` (adds these files to git to track, so that when you do git commit and git push they are in repository for collaborators to see them) >> `dvc config core.autostage true` (basically it performs git add automatically after dvc commands that change cause changes in files: dvc repro, dvc add, dvc remove) >> `dvc push` commands ensure the proper integration with gith and github. 
- `dvc dag` displays dependency graph of the stages in one or more pipelines. 
- `dvc list .` to show what directories and files does DVC track in the current directory `.`. If you want to see all subfolders and directories use `dvc list . -R`. Adding the flag `--dvc-only` in the end like `dvc list . --dvc-only` will give you directories and filestracked only by dvc and not by git if there are such files. You can allways modify your `.dvcignore` file accordingly.
- `dvc gc -a --all-branches --all-tags --all-commits` removes all cache from dvc. `dvc remove <something>` removes tracked files metadata and stops tracking.


You use DVC for github if you want to share model and data versioning. You use Git for publishing on github if you want to share just code versions.

    - activate the environment (venv) 
    - `git init`
    - `git add .`
    - `git commit -m "first commit`
    - `git branch -M main`
    - create empty repository on github mlops_mini
    - `git remote add origin https://github.com/VladimerKhasia/mlops_mini.git`  
    - `git push -u origin main`


    If you want to change remote repo: `git remote set-url origin https://github.com/VladimerKhasia mlops_mini.git` verify change with `git remote -v` and `git push -u origin main`

    If you want just add new remote repo: `git remote add main_2 <new_repository_url>` -> `git push main_2`
    Push local repo to the remote repository on github: `git push -u origin main_2`


To run fastapi app: `fastapi dev src/app/main.py` but before allways update your application to the latest changes with `dvc repro`

