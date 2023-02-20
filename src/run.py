import gc
import logging
import os
import time

import numpy as np
import torch
import torch.distributed as dist
from ogb.nodeproppred import Evaluator
from torch_geometric.transforms import ToUndirected

from .datasets import load_dataset
from .models.gbert.modeling_gbert import GBert
from .models.gnns.modeling_gnn import SAGN
from .models.lms.modeling_lm import AdapterDeberta, AdapterRoberta, Deberta, Roberta
from .utils import dataset2foldername, is_dist

logger = logging.getLogger(__name__)


def set_single_env(rank, world_size):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    torch.cuda.empty_cache()
    dist.destroy_process_group()
    gc.collect()


def get_trainer_class(model_type):
    if model_type in ["Roberta", "Deberta"]:
        from .models.lms.trainer import LM_Trainer as Trainer
    elif model_type in ["SAGN", "SIGN"]:
        from .models.gnns.trainer import GNN_Trainer as Trainer
    else:
        raise NotImplementedError("not implemented")
    return Trainer


def load_model(args):
    if args.model_type == "GBert":
        model = GBert(args)
    elif args.model_type == "SAGN":
        model = SAGN(args)
    elif args.model_type == "Roberta":
        model = AdapterRoberta(args) if args.use_adapter else Roberta(args)
    elif args.model_type == "Deberta":
        model = AdapterDeberta(args) if args.use_adapter else Deberta(args)
    else:
        raise NotImplementedError("Model {args.model_type} is not implemented")
    return model


def load_data(args):
    dataset = load_dataset(
        args.dataset,
        root=args.data_folder,
        transform=ToUndirected(),
        tokenizer=args.pretrained_model,
    )
    split_idx = dataset.get_idx_split()
    data = dataset.data
    # if use bert_x, change it
    if args.use_bert_x:
        saved_dir = os.path.join(args.data_folder, dataset2foldername(args.dataset), "processed", "bert_x.pt")
        bert_x = torch.load(saved_dir)
        data.x = bert_x
        logger.warning("using bert_x instead of original features!!!")
    evaluator = Evaluator(name=args.dataset)
    for split in ["train", "valid", "test"]:
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[split_idx[split]] = True
        data[f"{split}_mask"] = mask

    return data, split_idx, evaluator, dataset.processed_dir


def train(args):
    # setup running envs
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    set_single_env(rank, world_size)
    if rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()
    # setup dataset: [ogbn-arxiv]
    data, split_idx, evaluator, processed_dir = load_data(args)
    model = load_model(args)
    if rank == 0:
        torch.distributed.barrier()
    # trainer
    Trainer = get_trainer_class(args.model_type)
    trainer = Trainer(args, model, data, split_idx, evaluator)
    trainer.train()
    del trainer, model, data, split_idx, evaluator
    cleanup()


def test(args):
    if is_dist():
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        set_single_env(rank, world_size)
        if rank not in [-1, 0]:
            # Make sure only the first process in distributed training will download model & vocab
            torch.distributed.barrier()
    # setup dataset: [ogbn-arxiv]
    data, split_idx, evaluator, processed_dir = load_data(args)
    model = load_model(args)
    if is_dist() and rank == 0:
        torch.distributed.barrier()
    # trainer
    Trainer = get_trainer_class(args.model_type)
    trainer = Trainer(args, model, data, split_idx, evaluator)
    test_acc = trainer.evaluate(mode="test")
    logger.info("test_acc: {:.4f}".format(test_acc))
    valid_acc = trainer.evaluate(mode="valid")
    logger.info("valid_acc: {:.4f}".format(valid_acc))
    train_acc = trainer.evaluate(mode="train")
    logger.info("train_acc: {:.4f}".format(train_acc))
    del trainer, model, data, split_idx, evaluator
    cleanup()
    return train_acc, valid_acc, test_acc


def save_bert_x(args):
    if is_dist():
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        set_single_env(rank, world_size)
        if rank not in [-1, 0]:
            # Make sure only the first process in distributed training will download model & vocab
            torch.distributed.barrier()
    # setup dataset: [ogbn-arxiv]
    data, split_idx, evaluator, processed_dir = load_data(args)
    model = load_model(args)
    if is_dist() and rank == 0:
        torch.distributed.barrier()
    # trainer
    Trainer = get_trainer_class(args.model_type)
    trainer = Trainer(args, model, data, split_idx, evaluator)
    trainer.save_bert_x(data)
    del trainer, model, data, split_idx, evaluator
    cleanup()
