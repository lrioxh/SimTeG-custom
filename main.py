import json
import logging
import os
import random

import numpy as np
import torch
import torch.multiprocessing as mp

from src.options import parse_args
from src.run import train, test
from src.utils import set_logging, is_dist
import logging

logger = logging.getLogger(__name__)

"""
TODO:
x check the loss computation, should not be that high (~500)
- Implement SIGN and check the performance
x run SAGN and check the performance

- Implement the baselines
    * X_PLM
    * LM-Ft
    * LM-Ft (x features) + SOTA GNN
"""


def set_env(random_seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if is_dist():
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        if is_dist():
            gpus = ",".join([str(_) for _ in range(int(os.environ["WORLD_SIZE"]))])
            os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)


def main():
    args = parse_args()
    set_logging()
    logger.info(args)
    set_env(args.random_seed)
    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test(args)
    else:
        raise NotImplementedError("running mode should be either 'train' or 'test'")


if __name__ == "__main__":
    main()
