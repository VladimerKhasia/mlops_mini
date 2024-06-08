import os
from pathlib import Path
import logging

#logging string
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

name = 'llm_fc'

list_of_files = [
    ".github/workflows/.gitkeep",
    "src/__init__.py",
    "src/settings.py",
    "src/app/__init__.py",
    "src/app/main.py",
    "src/app/routers/generation_service.py",
    f"src/{name}/__init__.py",
    f"src/{name}/components/__init__.py",
    f"src/{name}/pipeline/__init__.py",
    f"src/{name}/schemas.py",
    f"src/{name}/utils.py",
    f"src/{name}/config_manager.py",
    "main.py",
    ".env",
    "tests/__init__.py",
    "tests/conftest.py",
    "tests/test_app.py",
    "artifact_config.yaml",
    "dvc.yaml",
    "params.yaml",
    "requirements.txt",
    "setup.py",
    "experiments/FineTune.ipynb",

]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)


    if filedir !="":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")


    else:
        logging.info(f"{filename} is already exists")