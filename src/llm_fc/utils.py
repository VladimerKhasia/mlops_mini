import os
import yaml
from pathlib import Path
from typing import List
from llm_fc import logger #, schemas
from pydantic import ValidationError


def read_yaml(path_to_yaml: Path): 
    """reads yaml file and returns
    Args:
        path_to_yaml (str): path like input
    Returns:
        Pydantic object GetData.
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            if content is None:
                raise ValueError(f"YAML file {path_to_yaml} is empty")
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return content
    except ValidationError as ve:
        raise ve
    except (FileNotFoundError, yaml.YAMLError) as e:
        raise e
    


def create_directories(path_to_directories: List[Path], verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        if path.is_dir() and any(path.iterdir()):
            logger.info(f"directory exists at: {path}")
            pass
        else:
            os.makedirs(path, exist_ok=True)
            if verbose:
                logger.info(f"created directory at: {path}")







