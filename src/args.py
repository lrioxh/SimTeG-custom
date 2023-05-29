import argparse
import json
import logging
import os

import torch

logger = logging.getLogger(__name__)

LM_LIST = ["all-roberta-large-v1", "all-mpnet-base-v2", "all-MiniLM-L6-v2", "e5-large"]
GNN_LIST = ["GAMLP", "SAGN", "SIGN", "SGC", "GraphSAGE", "GCN", "MLP"]
SAMPLING_GNN_LIST = ["GraphSAGE", "GCN"]
DECOUPLING_GNN_LIST = ["GAMLP", "SAGN", "SIGN", "SGC"]

LINK_PRED_DATASETS = ["ogbl-citation2"]
NODE_CLS_DATASETS = ["ogbn-arxiv", "ogbn-products"]


def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test", "save_bert_x"])
    parser.add_argument("--single_gpu", type=int, default=0)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--start_seed", type=int, default=42)
    parser.add_argument("--cont", type=bool, default=False)
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--suffix", type=str, default="main")
    parser.add_argument("--n_exps", type=int, default=1)

    # parameters for data and model storage
    parser.add_argument("--data_folder", type=str, default="../data")
    parser.add_argument("--dataset", type=str, default="ogbn-arxiv")
    parser.add_argument("--model_type", type=str, default="GBert")
    parser.add_argument("--task_type", type=str, default="node_cls")
    parser.add_argument("--output_dir", type=str)  # output dir
    parser.add_argument("--ckpt_dir", type=str)  # ckpt path to save
    parser.add_argument("--ckpt_name", type=str, default="TGRoberta-best.pt")  # ckpt name to be loaded
    parser.add_argument("--pretrained_dir", type=str, default="./pretrained")
    parser.add_argument("--pretrained_repo", type=str, help="has to be consistent with repo_id in huggingface")
    parser.add_argument("--bert_x_dir", type=str, help="used when use_bert_x is True")
    parser.add_argument("--giant_x_dir", type=str, help="used when use_bert_x is True")

    # dataset and fixed model args
    parser.add_argument("--num_labels", type=int)
    parser.add_argument("--num_feats", type=int)
    parser.add_argument("--hidden_size", type=int, default=768, help="hidden size of bert-like model")

    # flag
    parser.add_argument("--disable_tqdm", action="store_true", default=False)
    parser.add_argument("--use_bert_x", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False, help="will use mini dataset")
    parser.add_argument("--use_adapter", action="store_true", default=False)
    parser.add_argument(
        "--use_SLE", action="store_true", default=False, help="whether to use self-label-enhancement (SLE)"
    )
    parser.add_argument("--optuna", type=bool, default=False, help="use optuna to tune hyperparameters")
    parser.add_argument("--use_cache", action="store_true", default=False)
    parser.add_argument("--save_ckpt_per_valid", action="store_true", default=False)
    parser.add_argument("--eval_train_set", action="store_true", default=False)
    parser.add_argument("--inherit", action="store_true", default=False)
    parser.add_argument("--gnn_inherit", action="store_true", default=False)
    parser.add_argument("--fix_gnn", action="store_true", default=False, help="fix gnn model when finetuning bert")
    parser.add_argument("--compute_kl_loss", action="store_true", default=False)
    parser.add_argument("--use_default_config", action="store_true", default=False)
    parser.add_argument("--use_peft", action="store_true", default=False)
    parser.add_argument("--use_giant_x", action="store_true", default=False)

    # training hyperparameters
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--eval_batch_size", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--accum_interval", type=int, default=1)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1, help="consistent with the one in bert")
    parser.add_argument("--header_dropout_prob", type=float, default=0.2)
    parser.add_argument("--attention_dropout_prob", type=float, default=0.1)
    parser.add_argument("--adapter_hidden_size", type=int, default=768)
    parser.add_argument("--label_smoothing", type=float, default=0.3)
    parser.add_argument("--warmup_ratio", type=float, default=0.6)
    parser.add_argument("--num_iterations", type=int, default=4)
    parser.add_argument("--avg_alpha", type=float, default=0.5)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", choices=["linear", "constant"])
    # parameters for kl loss
    parser.add_argument("--kl_loss_weight", type=float, default=1)
    parser.add_argument("--kl_loss_temp", type=int, default=0, help="kl_loss *= 2**kl_loss_temp")
    # training hyperparameters for SLE
    parser.add_argument("--mlp_dim_hidden", type=int, default=128)
    parser.add_argument("--SLE_threshold", type=float, default=0.9)
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--eval_patience", type=int, default=50000)
    parser.add_argument("--SLE_mode", type=str, default="both", choices=["gnn", "lm", "both"])

    # module hyperparameters
    parser.add_argument("--lm_type", type=str, default="Deberta")
    # gnn parameters, alternative options for gnn when training gbert
    # NOTE: only used when training gbert, should set the general parameters when training sole gnn
    parser.add_argument("--gnn_eval_interval", type=int, default=5)
    parser.add_argument("--gnn_num_layers", type=int, default=4)
    parser.add_argument("--gnn_type", type=str, default="GAMLP")
    parser.add_argument("--gnn_dropout", type=float, default=0.2)
    parser.add_argument("--gnn_dim_hidden", type=int, default=256)
    parser.add_argument("--gnn_lr", type=float, default=5e-4)
    parser.add_argument("--gnn_weight_decay", type=float, default=1e-5)
    parser.add_argument("--gnn_label_smoothing", type=float, default=0.1)
    parser.add_argument("--gnn_batch_size", type=int, default=10000)
    parser.add_argument("--gnn_eval_batch_size", type=int, default=10000)
    parser.add_argument("--gnn_epochs", type=int, default=500)
    parser.add_argument("--gnn_warmup_ratio", type=float, default=0.25)
    parser.add_argument("--gnn_lr_scheduler_type", type=str, default="constant", choices=["constant", "linear"])

    # optuna hyperparameters
    parser.add_argument("--expected_valid_acc", type=float, default=0.6)
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--load_study", action="store_true", default=False)

    # peft lora hyperparameters
    parser.add_argument("--peft_r", type=int, default=8)
    parser.add_argument("--peft_lora_alpha", type=float, default=32)
    parser.add_argument("--peft_lora_dropout", type=float, default=0.1)

    # other hyperparameters
    parser.add_argument(
        "--train_mode",
        type=str,
        default="both",
        help="both: train lm and gnn in each iteration; lm: only train lm in each iteration and train GNN in iter 0",
    )
    args = parser.parse_args()
    args = _post_init(args)
    return args


def save_args(args, dir):
    if int(os.getenv("RANK", -1)) <= 0:
        FILE_NAME = "args.json"
        with open(os.path.join(dir, FILE_NAME), "w") as f:
            json.dump(args.__dict__, f, indent=2)
        logger.info("args saved to {}".format(os.path.join(dir, FILE_NAME)))


def load_args(dir):
    with open(os.path.join(dir, "args.txt"), "r") as f:
        args = argparse.Namespace(**json.load(f))
    return args


def _post_init(args):
    args = _set_dataset_specific_args(args)
    args = _set_pretrained_repo(args)
    args = _set_lm_and_gnn_type(args)
    return args


def _set_lm_and_gnn_type(args):
    if args.model_type in LM_LIST:
        args.lm_type = args.model_type
    elif args.model_type in GNN_LIST:
        args.gnn_type = args.model_type
    return args


def _set_pretrained_repo(args):
    dict = {
        "all-roberta-large-v1": "sentence-transformers/all-roberta-large-v1",
        "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
        "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
        "e5-large": "intfloat/e5-large",
    }

    if args.model_type in dict.keys():
        args.pretrained_repo = dict[args.model_type]
    else:
        assert args.lm_type in dict.keys()
        # assert args.pretrained_repo in dict[args.lm_type]
    return args


def _set_dataset_specific_args(args):
    if args.dataset == "ogbn-arxiv":
        args.num_labels = 40
        args.num_feats = 128
        args.expected_valid_acc = 0.6
        args.task_type = "node_cls"

    elif args.dataset == "ogbn-products":
        args.num_labels = 47
        args.num_feats = 100
        args.expected_valid_acc = 0.8
        args.task_type = "node_cls"

    elif args.dataset == "ogbl-citation2":
        args.num_feats = 128
        args.task_type = "link_pred"

    hidden_size_dict = {
        "all-roberta-large-v1": 1024,
        "all-mpnet-base-v2": 768,
        "all-MiniLM-L6-v2": 384,
        "e5-large": 1024,
    }

    if args.model_type in hidden_size_dict.keys():
        args.hidden_size = hidden_size_dict[args.model_type]
    elif args.use_bert_x and args.lm_type in hidden_size_dict.keys():
        args.num_feats = args.hidden_size = hidden_size_dict[args.lm_type]
    elif args.use_giant_x:
        args.num_feats = args.hidden_size = 768

    return args


if __name__ == "__main__":
    args = parse_args()
