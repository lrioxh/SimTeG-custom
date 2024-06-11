import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchviz import make_dot
from ogb.nodeproppred import DglNodePropPredDataset

class MultiLayerNeighborSampler(dgl.dataloading.BlockSampler):
    def __init__(self, fanouts):
        super().__init__(len(fanouts))

        self.fanouts = fanouts

    def sample_frontier(self, block_id, g, seed_nodes):
        fanout = self.fanouts[block_id]
        if fanout is None:
            frontier = dgl.in_subgraph(g, seed_nodes)
        else:
            frontier = dgl.sampling.sample_neighbors(g, seed_nodes, fanout)
        return frontier

def seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.random.seed(seed)
seed(0)

# g = dgl.graph(([0, 0, 1, 2, 3, 4], [1, 2, 3, 4, 4, 5]))
dataset = DglNodePropPredDataset(name="ogbn-arxiv", root="../dgl_data")
g, _ = dataset[0]
g = dgl.to_bidirected(g)
g = g.remove_self_loop().add_self_loop()
print(g)
train_nids = [i for i in range(128)]
dgl.dataloading.MultiLayerFullNeighborSampler(1) 
#  NeighborSampler([5, 10, 15]) LaborSampler([5, 10, 15]) dgl.dataloading.ShaDowKHopSampler([5, 10, 15]),MultiLayerNeighborSampler([-1,1])
# ClusterGCNSampler(g, num_parts=nodes//batch+1) 
# 
sampler = dgl.dataloading.NeighborSampler([1])
dataloader = dgl.dataloading.DataLoader(
    g, train_nids, sampler,
    batch_size=8,
    shuffle=False,
    drop_last=False,
    num_workers=4)

for input_nodes, output_nodes, blocks in dataloader:
    # blocks = [b.to(torch.device('cuda')) for b in blocks]
    grad_idx = blocks[-1].srcdata['_ID']
    
    sub = dgl.node_subgraph(g, input_nodes)
    ...
# for subg in dataloader:
#     ...
    
sg = dgl.sampling.sample_neighbors(g, train_nids, -1, )
print(sg)


# # 创建一个二维tensor，设置require_grad=False
# feat = torch.zeros((5, 3), requires_grad=False)

# # 创建几个require_grad=True的行向量
# row1 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
# row2 = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)

# # 覆盖feat的指定行
# feat[1] = row1
# feat[3] = row2

# # 定义一个简单的损失函数
# loss = (feat ** 2).sum()

# # 计算梯度
# loss.backward()

# # 查看行向量的梯度
# print(row1.grad)  # 输出: tensor([2., 4., 6.])
# print(row2.grad)  # 输出: tensor([ 8., 10., 12.])
# print(feat.grad)  # 输出: None

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

# 示例数据
feat = torch.randn(10, 5, requires_grad=False)
targets = torch.randn(10, 1)
rows_to_replace = torch.tensor([1, 3])
# replacements = torch.randn(3, 5, requires_grad=True)
# 使用自定义函数替换行
# rows = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0],
#                     [4.0, 5.0, 6.0, 0.0, 1.0]], requires_grad=True).retail_grad()

# 创建模型和优化器
model = nn.Linear(5, 1)
optimizer = optim.SGD(model.parameters(), lr=0.01)

def simple_replace(feat, idx, replacements):
    feat[idx] = replacements
    return feat

# 训练循环
for epoch in range(10):
    model.train()
        # 使用自定义函数替换行
    rows = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0],
                        [4.0, 5.0, 6.0, 0.0, 1.0]], requires_grad=True)
    feat_replaced = replace_rows(feat, rows_to_replace, rows)

    # feat_replaced = feat.clone()
    # feat_replaced = simple_replace(feat, rows_to_replace, rows)
    # feat_replaced[rows_to_replace] = rows
    
    # print(feat_replaced)

    # 前向传播
    outputs = model(feat_replaced)
    loss = ((outputs - targets) ** 2).mean()
    make_dot(loss, params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()

    # 查看哪些向量真正参与了反向传播
    for i, row in enumerate(rows):
        print(f"row[{i}] grad_fn: {row.grad_fn}")
        print(f"row[{i}] grad: {row.grad}")
    for i, row in enumerate(feat_replaced):
        print(f"feat[{i}] grad_fn: {row.grad_fn}")
        print(f"feat[{i}] grad: {row.grad}")
        
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    
    rows.grad.zero_()

print("Training finished")
