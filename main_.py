#!/usr/bin/env python
import csv
import gc
import logging
import math
import os
import random
import time
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from functools import lru_cache

import dgl
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
import torch_geometric.transforms as T

from src.utils import set_logging
from src.misc.revgat.loss import loss_kd_only
from src.model.lm_gnn import RevGAT, E5_model
from src.dataset import load_data_bundle
from src.args_ import parse_args, save_args
import src.lora as lora

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

class ReplaceRowsFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, rows, replacement):
        ctx.save_for_backward(input, rows, replacement)
        output = input.clone()
        output[rows] = replacement
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, rows, replacement = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_replacement = grad_output[rows].clone()
        grad_input[rows] = 0
        return grad_input, None, grad_replacement
    
replace_rows = ReplaceRowsFunction.apply

        
class LM_GNN():
    def __init__(self, args) -> None:
        self.args = args
        self.epsilon = 1 - math.log(2)
        # dataset = "ogbn-arxiv"
        self.n_node = 0 
        self.n_classes = 0
        self.device = None
        self.text_data = None
        self.feat_static = None
        self.graph = None
        self.graph_loader = None
        self.labels = None
        self.train_idx = None
        self.val_idx = None
        self.test_idx = None
        self.evaluator = None
        self.optimizer = None
        
        self.model_lm = None
        self.model_gnn = None

        
    def custom_loss_function(self, x, labels, label_smoothing_factor):
        y = F.cross_entropy(x, labels[:, 0], reduction="none", label_smoothing=label_smoothing_factor)
        y = torch.log(self.epsilon + y) - math.log(self.epsilon)
        return torch.mean(y)

    def cal_labels(self, length, labels, idx):
        onehot = torch.zeros([length, self.n_classes], device=self.device, 
                            #  dtype=torch.float16 if self.args.fp16 else torch.float32
                            )
        if len(idx)>0:
            onehot[idx, labels[idx, 0]] = 1
            
        return onehot
    
    def prepare(self):
        if self.args.cpu:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(f"cuda:{self.args.gpu}")
        self.labels, self.val_idx, self.test_idx = map(
        lambda x: x.to(self.device), (self.labels, self.val_idx, self.test_idx)
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
        
    def save_model(self, run_num):
        out_dir = f"{self.args.save}/ckpt"
        os.makedirs(out_dir,exist_ok=True)
        fname_gnn = os.path.join(out_dir, f"best_run{run_num}_gnn.pt")
        torch.save(self.model_gnn.state_dict(), fname_gnn)  
        fname_lm = os.path.join(out_dir, f"best_run{run_num}_lm.pt")
        torch.save(self.model_lm.state_dict(), fname_lm)  
        
    def count_parameters(self):
        return sum([p.numel() for p in \
                                    list(self.model_gnn.parameters())+list(self.model_lm.parameters()) \
                                    if p.requires_grad])
    @lru_cache(8)
    def id_in_parent(self, parent,sub):
        if self.args.frozen_padding >= 0:
            # 第一步：对 tensor1 进行排序
            sorted_parent, sorted_indices = torch.sort(parent)

            # 第二步：使用 torch.searchsorted 查找 tensor2 中每个元素在排序后的 tensor1 中的位置
            sorted_pos = torch.searchsorted(sorted_parent, sub)

            # 第三步：将排序后的索引映射回原始的索引
            return sorted_indices[sorted_pos]
        else:
            return sub
        
    @torch.no_grad()
    def get_static_feat(self):
        text_loader = DataLoader(self.text_data, batch_size=self.args.batch_size, shuffle=False)
        num_batches = len(text_loader)
        interval = num_batches//10
        feat = torch.empty((self.n_node, self.args.hidden_size), 
                                dtype=torch.float16 if self.args.fp16 else torch.float32, device=self.device, requires_grad=False)
        with tqdm(total=num_batches, desc=f'LM ', unit='batch', file=open(os.devnull, 'w')) as pbar:
        #     with logging_redirect_tqdm():
        # pbar = tqdm(range(num_batches), file=open(os.devnull, 'w'))
            for i, (input_ids, attention_mask) in enumerate(text_loader):
                # lm
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                if self.args.fp16:
                    with autocast():
                        
                            outcls, feat[i*self.args.batch_size:i*self.args.batch_size+input_ids.shape[0],:self.args.hidden_size] \
                                            = self.model_lm(input_ids,attention_mask,return_hidden=True)
                else:
                    outcls, feat[i*self.args.batch_size:i*self.args.batch_size+input_ids.size[0],:self.args.hidden_size] \
                                        = self.model_lm(input_ids,attention_mask,return_hidden=True)
                
                if interval == 0: logger.info(str(pbar))
                elif (i-1) % interval == 0: logger.info(str(pbar))
                pbar.update(1)
            torch.cuda.empty_cache()
            gc.collect()
        return feat
         
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
        
        self.text_data = TensorDataset(text_token.input_ids, text_token.attention_mask) 
        
        splitted_idx = data_graph.get_idx_split()
        self.train_idx, self.val_idx, self.test_idx = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]
        self.graph, self.labels = data_graph[0]

        # if args.use_bert_x:
        # self.graph.ndata["input_ids"] = text_token.input_ids
        # self.graph.ndata["attention_mask"] = text_token.attention_mask
        # logger.warning(
        #     "Loaded node tokens of shape={}".format(text_token["input_ids"].shape)
        # )
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
        if self.args.debug > 0:
            # debug_size = 60000
            debug_idx = [i for i in range(self.args.debug)]
            self.train_idx = self.train_idx[self.train_idx < self.args.debug]
            self.val_idx = self.val_idx[self.val_idx < self.args.debug]
            self.test_idx = self.test_idx[self.test_idx < self.args.debug]
            self.labels = self.labels[:self.args.debug]
            self.graph = dgl.node_subgraph(self.graph, debug_idx)
            self.text_data = Subset(self.text_data, debug_idx)
        # if args.use_gpt_preds:
        #     n_node_feats = args.n_hidden * 5
        # else: 
        
        self.n_node = len(self.text_data)
        self.n_classes = (self.labels.max() + 1).item()
        logger.warning(
            f"Loaded node tokens of shape=({self.n_node},{text_token.input_ids.shape[1]})")      

        return 1

    def init_loader(self):
        if self.args.frozen_padding > 0: 
            sampler = dgl.dataloading.NeighborSampler(
                [1]+[-1 for _ in range(self.args.frozen_padding)]+[1 for _ in range(self.args.grad_padding)])
            self.graph_loader = dgl.dataloading.DataLoader(
                self.graph, self.train_idx, sampler,
                batch_size=self.args.kernel_size,
                shuffle=False,
                drop_last=False,
                num_workers=4)
        else:   # 包括frozen_padding=0
            # if self.args.grad_padding > 0:
            #     sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.args.grad_padding)
            #     batch_size = self.args.kernel_size
            # elif self.args.grad_padding == 0:
            sampler = dgl.dataloading.ShaDowKHopSampler([1 for _ in range(self.args.grad_padding)])
            batch_size = self.args.kernel_size
            self.graph_loader = dgl.dataloading.DataLoader(
                    self.graph, self.train_idx, sampler,
                    batch_size=batch_size,
                    shuffle=False,
                    drop_last=False,
                    num_workers=4)
            
        # self.graph, self.train_idx = map(
        #     lambda x: x.to(self.device), (self.graph, self.train_idx)
        # )

    def gen_model(self):
        if self.args.use_labels:
            self.args.n_node_feats = self.args.hidden_size + self.n_classes
        else:
            self.args.n_node_feats = self.args.hidden_size

        if self.args.gnn_type == "RevGAT":
            self.model_gnn = RevGAT(
                self.args,
                self.n_classes,
                activation=F.relu,
                dropout=self.args.dropout,
                input_drop=self.args.input_drop,
                attn_drop=self.args.attn_drop,
                edge_drop=self.args.edge_drop,
                use_attn_dst=not self.args.no_attn_dst,
                use_symmetric_norm=self.args.use_norm,
                use_gpt_preds=self.args.use_gpt_preds,
            )

            if self.args.ckpt_dir != '' and os.path.exists(self.args.ckpt_dir):
                self.model_gnn.load_state_dict(torch.load(self.args.ckpt_dir),strict=False)
                logger.info(f"Loaded PGM from {self.args.ckpt_dir}")
                self.model_gnn.convs[-1].reset_parameters()
        else:
            raise Exception("Unknown gnn")
        
        if self.args.lm_type == "e5-large":
            self.model_lm = E5_model(self.args)
        else:
            raise Exception("Unknown lm")

        return 1

    def lora_gnn(self):
        src_model = self.model_gnn.to('cpu')
        self.model_gnn = RevGAT(
                self.args,
                self.n_classes,
                activation=F.relu,
                dropout=self.args.dropout,
                input_drop=self.args.input_drop,
                attn_drop=self.args.attn_drop,
                edge_drop=self.args.edge_drop,
                use_attn_dst=not self.args.no_attn_dst,
                use_symmetric_norm=self.args.use_norm,
                use_gpt_preds=self.args.use_gpt_preds,
                lora_params={
                    'use_lora': self.args.use_peft,
                    'r': self.args.peft_r,
                    'lora_alpha': self.args.peft_lora_alpha,
                    'lora_dropout': self.args.peft_lora_dropout
                    }
            )
        self.model_gnn.load_state_dict(src_model.state_dict(), strict=False)
         
        gc.collect()
        torch.cuda.empty_cache()   
        self.model_gnn.to(self.device)
        lora.mark_only_lora_as_trainable(self.model_gnn)
        self.optimizer = optim.RMSprop(list(self.model_gnn.parameters())+list(self.model_lm.parameters()), 
                                       lr=self.args.lr, weight_decay=self.args.wd)
        
        logger.info("GM switched to LoRA")
        logger.info(f"Number of params: {self.count_parameters()}")

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
        if self.feat_static == None:
            self.feat_static = self.get_static_feat()
            
        if self.args.use_labels:
            self.feat_static = torch.cat([self.feat_static, 
                              torch.zeros((self.n_node, self.n_classes), 
                                          dtype=torch.float16 if self.args.fp16 else torch.float32, 
                                          device=self.device)],
                              dim=-1)
        # res = torch.zeros(self.labels.shape)
        
        # for 采样相邻节点id kernel_size为train_pred_idx， 扩充grad_padding为grad_idx
        num_batches = len(self.graph_loader)
        interval = num_batches//10
        with tqdm(total=num_batches, desc=f'{epoch}/{self.args.n_epochs} ', unit='batch', file=open(os.devnull, 'w')) as pbar:
            with self.graph_loader.enable_cpu_affinity():
                for i, (sub_idx, train_pred_idx, blocks) in enumerate(self.graph_loader):
                    if self.args.frozen_padding > 0:
                        # sub_idx = sub_idx.to(self.device)
                        graph = dgl.node_subgraph(self.graph, sub_idx, output_device=self.device)
                        grad_idx = blocks[-1].srcdata['_ID']
                        feat = self.feat_static[sub_idx]
                        train_idx = sub_idx[torch.isin(sub_idx, self.train_idx)]
                        # n_nodes = len(sub_idx)
                    elif self.args.frozen_padding == 0:
                        graph = blocks.to(device=self.device)
                        grad_idx = sub_idx
                        feat = self.feat_static[sub_idx]
                        train_idx = sub_idx[torch.isin(sub_idx, self.train_idx)]
                    else:
                        graph = self.graph.to(device=self.device)
                        feat = self.feat_static
                        grad_idx = sub_idx
                        train_idx = self.train_idx
                        # n_nodes = self.n_node
                    if len(grad_idx)>self.args.grad_size:
                        logger.info(f"grad_idx({len(grad_idx)}) sliced")
                        grad_idx = grad_idx[:self.args.grad_size]
                    self.optimizer.zero_grad()
                    feat = feat.detach()
                    subset = Subset(self.text_data, grad_idx) 
                    dataloader = DataLoader(subset, batch_size=len(grad_idx))
                    for _, (input_ids, attention_mask) in enumerate(dataloader):
                        input_ids = input_ids.to(self.device)
                        attention_mask = attention_mask.to(self.device)
                        if self.args.fp16:
                            with autocast():
                                out, embs = self.model_lm(input_ids, attention_mask, return_hidden=True)
                            # embs = embs.to(torch.float16)
                            feat = feat.to(dtype=torch.float32)
                        else:
                            out, embs = self.model_lm(input_ids, attention_mask, return_hidden=True)
                    
                    torch.cuda.empty_cache()    
                    gc.collect()
                    # gnn
                    if self.args.use_labels:
                       
                        train_labels_idx = set(train_idx.tolist()) - set(train_pred_idx.tolist())
                        train_labels_idx = torch.tensor(list(train_labels_idx))
                        onehot_labels = self.cal_labels(self.n_node, self.labels, train_labels_idx)
                        if len(train_labels_idx):
                            feat[self.id_in_parent(sub_idx, train_labels_idx),
                                -self.n_classes:] = onehot_labels[train_labels_idx]
                        embs = torch.cat([embs, onehot_labels[grad_idx]], dim=-1)
                    # else:
                    #     # mask = torch.rand(self.train_idx.shape) < self.args.mask_rate

                    #     # train_pred_idx = kernel_idx
                    #     ...
                    
                    feat = replace_rows(feat, self.id_in_parent(sub_idx, grad_idx), embs)
                    # static_feat[grad_idx] = embs
                    if self.args.n_label_iters > 0:
                        with torch.no_grad():
                            pred = self.model_gnn(graph, feat)
                    else:
                        # if self.args.fp16:
                        #     with autocast():
                        pred = self.model_gnn(graph, feat)

                    if self.args.n_label_iters > 0:
                        # unlabel_idx = torch.cat([train_pred_idx, self.val_idx, self.test_idx])
                        unlabel_idx = set(sub_idx.tolist()) - set(train_labels_idx.tolist())
                        unlabel_idx = torch.tensor(list(unlabel_idx))
                        for _ in range(self.args.n_label_iters):
                            pred = pred.detach()    #requires_grad为false, 梯度向前传播到此为止
                            torch.cuda.empty_cache()
                            onehot_labels[unlabel_idx] = F.softmax(
                                pred[self.id_in_parent(sub_idx, unlabel_idx)], dim=-1)
                            feat[self.id_in_parent(sub_idx, unlabel_idx), -self.n_classes:] \
                                = onehot_labels[unlabel_idx]
                            # embs = torch.cat([embs, onehot_labels[grad_idx]], dim=-1)
                            # feat = replace_rows(feat, grad_idx, embs)
                            pred = self.model_gnn(graph, feat)

                    if mode == "teacher":
                        loss = self.custom_loss_function(pred[self.id_in_parent(sub_idx, train_pred_idx)], self.labels[train_pred_idx],self.args.label_smoothing_factor)
                    elif mode == "student":
                        loss_gt = self.custom_loss_function(pred[train_pred_idx], self.labels[train_pred_idx],self.args.label_smoothing_factor)
                        loss_kd = loss_kd_only(pred, teacher_output, temp)
                        loss = loss_gt * (1 - alpha) + loss_kd * alpha
                    else:
                        raise Exception("unkown mode")
                    
                    torch.cuda.empty_cache()    
                    gc.collect()
                    if self.args.fp16:
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        self.optimizer.step()
                    if (i-1) % interval == 0: logger.info(str(pbar))
                    pbar.update(1)

        return evaluator(pred[self.id_in_parent(sub_idx, train_idx)], self.labels[train_idx]), loss.item()


    @torch.no_grad()
    def evaluate(self, evaluator):
        torch.cuda.empty_cache()    
        gc.collect()
        self.model_gnn.eval()
        self.model_lm.eval()

        # feat = graph.ndata["feat"]
        self.feat_static = self.get_static_feat()
        graph = self.graph.to(device=self.device)
        # gnn        
        if self.args.use_labels:
            onehot_labels = self.cal_labels(self.n_node, self.labels, self.train_idx)
            feat_eval = torch.cat([self.feat_static, onehot_labels], dim=-1)
        
        pred = self.model_gnn(graph, feat_eval)

        if self.args.n_label_iters > 0:
            unlabel_idx = torch.cat([self.val_idx, self.test_idx])
            for _ in range(self.args.n_label_iters):
                onehot_labels[unlabel_idx] = F.softmax(pred[unlabel_idx], dim=-1)
                pred = self.model_gnn(graph, feat_eval)

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
            {"y_pred": pred.argmax(dim=-1, keepdim=True), "y_true": labels} #onehot to cls
        )["acc"]

        # kd mode
        mode = self.args.kd_mode

        # define model and optimizer
        #e5_revgat
        self.gen_model()
        logger.info(f"Number of params: {self.count_parameters()}")
        self.model_gnn.to(self.device)
        self.model_lm.to(self.device)
        self.optimizer = optim.RMSprop(list(self.model_gnn.parameters())+list(self.model_lm.parameters()), 
                                       lr=self.args.lr, weight_decay=self.args.wd)
        
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
            if self.args.fullft + 1 == epoch:
                self.lora_gnn()
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

            # if epoch == 1:
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

    # gbc.args.save = f"{gbc.args.output_dir}/{gbc.args.dataset}/{gbc.args.model_type}/{gbc.args.suffix}"
    # os.makedirs(gbc.args.save,exist_ok=True)
    save_args(gbc.args, gbc.args.save)
    
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
        gbc.init_loader()
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
