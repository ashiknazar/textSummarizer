import os
from box.exceptions import BoxValueError
import yaml
from textSummarizer.logging import logger
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any


@ensure_annotations
def read_yaml(path_to_yaml:Path)->ConfigBox:
    """
    Args: 
         path_to_yaml(str):path like input
    return :
         ConfigBox:ConfigBox type
    """
    try:
        with open(path_to_yaml)as yaml_file:
            content=yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    

@ensure_annotations
def create_directories(path_to_directories:list,verbose=True):
    """
    Args:
       path_to directories

    """
    for path in path_to_directories:
        os.makedirs(path,exist_ok=True)
        if verbose:
            logger.info(f"created directory at {path}")


@ensure_annotations
def get_size(path:Path)->str:
    """
    Args:
      path of file
    return :
       size in kb :str
    """
    size_in_kb=round(os.path.getsize(path)/1024)
    return f"~{size_in_kb} KB"