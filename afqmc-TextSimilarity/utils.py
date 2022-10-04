import logging 
import random 
import torch 
import numpy as np

from os import path
from typing import Optional

from transformers import AutoTokenizer


def setup_cuda_device(no_cuda):
    logging.info(f'torch.cuda.is_available {torch.cuda.is_available()}')
    if no_cuda or not torch.cuda.is_available():
        device = torch.device("cpu")
        n_gpu = 0
    else:
        device = torch.device("cuda")
        n_gpu = torch.cuda.device_count()
    return device, n_gpu


def set_random_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)
    logging.info(f"Set random seed to {seed}")


def setup_tokenizer(model_name: str, cache_dir: Optional[str] = None):
    logging.info("***** Setting up tokenizer *****\n")

    model_path = model_name
    if cache_dir:
        logging.info("Loading tokenizer from cache files")
        model_path = path.join(cache_dir, model_name)
        if not path.isdir(model_path):
            raise FileNotFoundError(f"No cache directory {model_path} exists.")
    else:
        logging.info(f"Loading {model_path} tokenizer")
    do_lower_case = 'uncased' in model_name

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, do_lower_case=do_lower_case)

    return tokenizer
