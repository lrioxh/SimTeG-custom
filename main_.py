#!/usr/bin/env python
import csv
import gc
import logging
import math
import os
import random
import time
from tqdm import tqdm

import dgl
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
import torch_geometric.transforms as T

from src.utils import set_logging
from src.misc.revgat.loss import loss_kd_only
from src.model.lm_gnn import RevGAT, E5_model
from src.dataset import load_data_bundle
from src.args_ import parse_args

# -*- coding: utf-8 -*-


logger = logging.getLogger(__name__)



def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.random.seed(seed)


def preprocess(graph):
    # global n_node_feats

    # make bidirected
    # feat = graph.ndata["feat"]
    graph = dgl.to_bidirected(graph)
    # graph.ndata["feat"] = feat

    # add self-loop
    logger.info(f"Total edges before adding self-loop {graph.number_of_edges()}")
    graph = graph.remove_self_loop().add_self_loop()
    logger.info(f"Total edges after adding self-loop {graph.number_of_edges()}")

    graph.create_formats_()

    return graph


class LM_GNN():
    def __init__(self, args) -> None:
        self.args = args
        self.epsilon = 1 - math.log(2)
        # dataset = "ogbn-arxiv"
        self.n_node = 0 
        self.n_classes = 0
        self.device = None
        self.text_loader = None
        self.graph = None
        self.labels = None
        self.train_idx = None
        self.val_idx = None
        self.test_idx = None
        self.evaluator = None
        self.optimizer = None
        
        self.model_lm = None
        self.model_gnn = None
        self.feat = None
        self.feat_eval = None
        
    def custom_loss_function(self, x, labels, label_smoothing_factor):
        y = F.cross_entropy(x, labels[:, 0], reduction="none", label_smoothing=label_smoothing_factor)
        y = torch.log(self.epsilon + y) - math.log(self.epsilon)
        return torch.mean(y)

    def cal_labels(self, length, labels, idx):
        onehot = torch.zeros([length, self.n_classes], device=self.device)
        onehot[idx, labels[idx, 0]] = 1
        return onehot
    
    def prepare(self):
        if self.args.cpu:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(f"cuda:{self.args.gpu}")
        self.graph, self.labels, self.train_idx, self.val_idx, self.test_idx = map(
        lambda x: x.to(self.device), (self.graph, self.labels, self.train_idx, self.val_idx, self.test_idx)
    )
        # 初始化GradScaler
        self.scaler = GradScaler() if self.args.fp16 else None

    def adjust_learning_rate(self, lr, epoch):
        if epoch <= 50:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr * epoch / 50

    def save_pred(self, pred, run_num, kd_dir):
        os.makedirs(kd_dir,exist_ok=True)
        fname = os.path.join(kd_dir, "best_pred_run{}.pt".format(run_num))
        torch.save(pred.cpu(), fname)    
        
    def count_parameters(self):
        return sum([p.numel() for p in \
                                    list(self.model_gnn.parameters())+list(self.model_lm.parameters()) \
                                    if p.requires_grad])
            
    def load_data(self):
        assert self.args.dataset in [
                "ogbn-arxiv", "ogbl-citation2", "ogbn-products", "ogbn-arxiv-tape"
            ]
        data_graph = DglNodePropPredDataset(name=self.args.dataset, root="../dgl_data")
        self.evaluator = Evaluator(name=self.args.dataset)
        
        # text attr
        text_token, split_idx, evaluator = load_data_bundle(
            self.args.dataset,
            root=self.args.data_folder,
            tokenizer=self.args.pretrained_repo,
            tokenize=True)
        # process data
        if self.args.dataset == "ogbn-arxiv":
            transform = T.ToUndirected()    #TODO: 加入PE处理有向图
            text_token = transform(text_token)
            
        dataset = TensorDataset(text_token.input_ids, text_token.attention_mask)
        self.text_loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False)
        
        splitted_idx = data_graph.get_idx_split()
        self.train_idx, self.val_idx, self.test_idx = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]
        self.graph, self.labels = data_graph[0]

        # if args.use_bert_x:
        # graph.ndata["feat"] = text_token
        # graph.ndata["token"] = text_token
        logger.warning(
            "Loaded node tokens of shape={}".format(text_token["input_ids"].shape)
        )
        # TODO
        if self.args.use_gpt_preds:
            preds = []
            with open(f"src/misc/gpt_preds/ogbn-arxiv.csv", "r") as file:
                reader = csv.reader(file)
                for row in reader:
                    inner_list = []
                    for value in row:
                        inner_list.append(int(value))
                    preds.append(inner_list)
            pl = torch.zeros(len(preds), 5, dtype=torch.long)
            for i, pred in enumerate(preds):
                pl[i][: len(pred)] = torch.tensor(pred[:5], dtype=torch.long) + 1
            self.graph.ndata["feat"] = pl
            logger.warning(
                "Loaded pre-trained node embeddings of shape={} from gpt_preds".format(self.graph.ndata["feat"].shape)
            )

        # if args.use_gpt_preds:
        #     n_node_feats = args.n_hidden * 5
        # else:
        self.n_node = text_token["input_ids"].shape[0]
        self.n_classes = (self.labels.max() + 1).item()

        return 1


    def gen_model(self):
        if self.args.use_labels:
            n_node_feats_ = self.args.hidden_size + self.n_classes
        else:
            n_node_feats_ = self.args.hidden_size

        if self.args.gnn_type == "RevGAT":
            self.model_gnn = RevGAT(
                n_node_feats_,
                self.n_classes,
                n_hidden=self.args.n_hidden,
                n_layers=self.args.n_layers,
                n_heads=self.args.n_heads,
                activation=F.relu,
                dropout=self.args.dropout,
                input_drop=self.args.input_drop,
                attn_drop=self.args.attn_drop,
                edge_drop=self.args.edge_drop,
                use_attn_dst=not self.args.no_attn_dst,
                use_symmetric_norm=self.args.use_norm,
                use_gpt_preds=self.args.use_gpt_preds,
            )
        else:
            raise Exception("Unknown gnn")
        
        if self.args.lm_type == "e5-large":
            self.model_lm = E5_model(self.args)
        else:
            raise Exception("Unknown lm")

        return 1



    def train(
        self, epoch, evaluator, mode="teacher", teacher_output=None
    ):
        self.model_gnn.train()
        self.model_lm.train()
        if mode == "student":
            assert teacher_output != None

        alpha = self.args.alpha
        temp = self.args.temp

        # feat = graph.ndata["feat"]  #requires_grand=False

        self.optimizer.zero_grad()
        
        
        num_batches = len(self.text_loader)
        # feat = torch.empty((self.n_node, self.args.hidden_size), device=self.device)
        with tqdm(total=num_batches, desc=f'Epoch {epoch}/{self.args.n_epochs}', unit='batch') as pbar:
            for i, (input_ids, attention_mask) in enumerate(self.text_loader):
                # lm
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                if self.args.fp16:
                    with autocast():
                        with torch.no_grad():
                            outcls, feat \
                                            = self.model_lm(input_ids,attention_mask,return_hidden=True)
                else:
                    outcls, self.feat[i*self.args.batch_size:i*self.args.batch_size+input_ids.size[0],:self.args.hidden_size] \
                                        = self.model_lm(input_ids,attention_mask,return_hidden=True)
                
                
                # gnn
                if self.args.use_labels:
                    mask = torch.rand(self.train_idx.shape) < self.args.mask_rate

                    train_labels_idx = self.train_idx[mask]
                    train_pred_idx = self.train_idx[~mask]

                    onehot_labels = self.cal_labels(self.n_node, self.labels, train_labels_idx)
                    self.feat = torch.cat([self.feat, onehot_labels], dim=-1)
                else:
                    mask = torch.rand(self.train_idx.shape) < self.args.mask_rate

                    train_pred_idx = self.train_idx[~mask]
                    
                if self.args.n_label_iters > 0:
                    with torch.no_grad():
                        pred = self.model_gnn(self.graph, self.feat)
                else:
                    pred = self.model_gnn(self.graph, self.feat)

                if self.args.n_label_iters > 0:
                    unlabel_idx = torch.cat([train_pred_idx, self.val_idx, self.test_idx])
                    for _ in range(self.args.n_label_iters):
                        pred = pred.detach()    #requires_grad为false, 梯度向前传播到此为止
                        torch.cuda.empty_cache()
                        onehot_labels[unlabel_idx] = F.softmax(pred[unlabel_idx], dim=-1)
                        pred = self.model_gnn(self.graph, self.feat)

                if mode == "teacher":
                    loss = self.custom_loss_function(pred[train_pred_idx], self.labels[train_pred_idx],self.args.label_smoothing_factor)
                elif mode == "student":
                    loss_gt = self.custom_loss_function(pred[train_pred_idx], self.labels[train_pred_idx],self.args.label_smoothing_factor)
                    loss_kd = loss_kd_only(pred, teacher_output, temp)
                    loss = loss_gt * (1 - alpha) + loss_kd * alpha
                else:
                    raise Exception("unkown mode")

                if self.args.fp16:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                
                gc.collect()
                torch.cuda.empty_cache()
                pbar.update(1)
                
        
        # gc.collect()
        # torch.cuda.empty_cache()
        

        return evaluator(pred[self.train_idx], self.labels[self.train_idx]), loss.item()


    @torch.no_grad()
    def evaluate(self, evaluator):
        self.model_gnn.eval()
        self.model_lm.eval()

        # feat = graph.ndata["feat"]

        # lm
        num_batches = len(self.text_loader)
        with tqdm(total=num_batches, desc=f'Eval: ', unit='batch') as pbar:
            for i, (input_ids, attention_mask) in enumerate(self.text_loader):
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                outcls, self.feat_eval[i*self.args.batch_size:i*self.args.batch_size+input_ids.size[0],:self.args.hidden_size] \
                                = self.model_lm(input_ids,attention_mask,return_hidden=True)
                # 拼接
                
                pbar.update(1)
                
        
        # gnn        
        if self.args.use_labels:
            onehot_labels = self.cal_labels(self.n_node, self.labels, self.train_idx)
            self.feat_eval = torch.cat([self.feat_eval, onehot_labels], dim=-1)
        pred = self.model_gnn(self.graph, self.feat_eval)

        if self.args.n_label_iters > 0:
            unlabel_idx = torch.cat([self.val_idx, self.test_idx])
            for _ in range(self.args.n_label_iters):
                onehot_labels[unlabel_idx] = F.softmax(pred[unlabel_idx], dim=-1)
                pred = self.model_gnn(self.graph, self.feat_eval)

        train_loss = self.custom_loss_function(pred[self.train_idx], self.labels[self.train_idx], 0)
        val_loss = self.custom_loss_function(pred[self.val_idx], self.labels[self.val_idx], 0)
        test_loss = self.custom_loss_function(pred[self.test_idx], self.labels[self.test_idx], 0)

        return (
            evaluator(pred[self.train_idx], self.labels[self.train_idx]),
            evaluator(pred[self.val_idx], self.labels[self.val_idx]),
            evaluator(pred[self.test_idx], self.labels[self.test_idx]),
            train_loss,
            val_loss,
            test_loss,
            pred,
        )


    def run(self, n_running):
        evaluator_wrapper = lambda pred, labels: self.evaluator.eval(
            {"y_pred": pred.argmax(dim=-1, keepdim=True), "y_true": labels}
        )["acc"]

        # kd mode
        mode = self.args.mode

        # define model and optimizer
        #e5_revgat
        self.gen_model()
        logger.info(f"Number of params: {self.count_parameters()}")
        self.model_gnn.to(self.device)
        self.model_lm.to(self.device)
        self.optimizer = optim.RMSprop(list(self.model_gnn.parameters())+list(self.model_lm.parameters()), 
                                       lr=self.args.lr, weight_decay=self.args.wd)
        
        self.feat = torch.empty((self.n_node, self.args.hidden_size), 
                                dtype=torch.float16 if self.args.fp16 else torch.float32, device=self.device)
        self.feat_eval = torch.empty((self.n_node, self.args.hidden_size), 
                                dtype=torch.float16 if self.args.fp16 else torch.float32, device=self.device)

        # training loop
        total_time = 0
        best_val_acc, final_test_acc, best_val_loss = 0, 0, float("inf")
        final_pred = None

        accs, train_accs, val_accs, test_accs = [], [], [], []
        losses, train_losses, val_losses, test_losses = [], [], [], []

        for epoch in range(1, self.args.n_epochs + 1):
            tic = time.time()
            if mode == "student":
                teacher_output = torch.load("./{}/best_pred_run{}.pt".format(self.args.kd_dir, n_running)).cpu().cuda()
            else:
                teacher_output = None

            self.adjust_learning_rate(self.args.lr, epoch)

            acc, loss = self.train(
                epoch,
                evaluator_wrapper,
                mode=mode,
                teacher_output=teacher_output,
            )

            train_acc, val_acc, test_acc, train_loss, val_loss, test_loss, pred = \
                                                                self.evaluate(evaluator_wrapper)

            toc = time.time()
            total_time += toc - tic

            if epoch == 1:
                peak_memuse = torch.cuda.max_memory_allocated(self.device) / float(1024**3)
                logger.info("Peak memuse {:.2f} G".format(peak_memuse))

            if val_acc > best_val_acc:
                best_val_loss = val_loss
                best_val_acc = val_acc
                final_test_acc = test_acc
                final_pred = pred
                if mode == "teacher":
                    self.save_pred(final_pred, n_running, self.args.kd_dir)

            if epoch == self.args.n_epochs or epoch % self.args.log_every == 0:
                logger.info(
                    f"Run: {n_running}/{self.args.n_runs}, Epoch: {epoch}/{self.args.n_epochs}, Average epoch time: {total_time / epoch:.2f}\n"
                    f"Loss: {loss:.4f}, Acc: {acc:.4f}\n"
                    f"Train/Val/Test loss: {train_loss:.4f}/{val_loss:.4f}/{test_loss:.4f}\n"
                    f"Train/Val/Test/Best val/Final test acc: {train_acc:.4f}/{val_acc:.4f}/{test_acc:.4f}/{best_val_acc:.4f}/{final_test_acc:.4f}"
                )

            for l, e in zip(
                [accs, train_accs, val_accs, test_accs, losses, train_losses, val_losses, test_losses],
                [acc, train_acc, val_acc, test_acc, loss, train_loss, val_loss, test_loss],
            ):
                l.append(e)

        logger.info("*" * 50)
        logger.info(f"Best val acc: {best_val_acc}, Final test acc: {final_test_acc}")
        logger.info("*" * 50)

        if self.args.save_pred:
            os.makedirs(f"{self.args.output_dir}/cached_embs", exist_ok=True)
            torch.save(final_pred, f"{self.args.output_dir}/cached_embs/logits_seed{n_running}.pt")
            logger.warning(f"Saved logits to {self.args.output_dir}/cached_embs/logits_seed{n_running}.pt")

        return best_val_acc, final_test_acc



def main():
    # global device, n_node_feats, n_classes, epsilon
    set_logging()
    gbc = LM_GNN(parse_args())
    # args = parse_args()

    gbc.args.save = f"{gbc.args.output_dir}/{gbc.args.dataset}/{gbc.args.model_type}"
    os.makedirs(gbc.args.save,exist_ok=True)
    
    if not gbc.args.use_labels and gbc.args.n_label_iters > 0:
        raise ValueError("'--use-labels' must be enabled when n_label_iters > 0")

    # load data & preprocess
    gbc.load_data()
    gbc.graph = preprocess(gbc.graph)#

    # to device
    gbc.prepare()
    # gbc.gen_model()
    logger.info(gbc.args)
    # logger.info(f"Number of params: {gbc.count_parameters()}")

    # run
    val_accs, test_accs = [], []

    for i in range(gbc.args.n_runs):
        seed(gbc.args.seed + i)
        val_acc, test_acc = gbc.run(i + 1)
        val_accs.append(val_acc)
        test_accs.append(test_acc)

    logger.info(gbc.args)
    logger.info(f"Runned {gbc.args.n_runs} times")
    logger.info("Val Accs:")
    logger.info(val_accs)
    logger.info("Test Accs:")
    logger.info(test_accs)
    logger.info(f"Average val accuracy: {np.mean(val_accs)} ± {np.std(val_accs)}")
    logger.info(f"Average test accuracy: {np.mean(test_accs)} ± {np.std(test_accs)}")
    logger.info(f"Number of params: {gbc.count_parameters()}")


if __name__ == "__main__":
    main()
