from ogb.linkproppred import Evaluator as LinkEvaluator
from ogb.nodeproppred import Evaluator as NodeEvaluator

from .ogbl_citation2 import OgblCitation2WithText
from .ogbn_arxiv import OgbnArxivWithText
from .ogbn_products import OgbnProductsWithText


def load_dataset(name, root="data", tokenizer=None, tokenize=True):
    datasets = {
        "ogbn-arxiv": OgbnArxivWithText,
        "ogbn-products": OgbnProductsWithText,
        "ogbl-citation2": OgblCitation2WithText,
    }
    assert name in datasets.keys()
    return datasets[name](root=root, tokenizer=tokenizer, tokenize=tokenize)


def load_data_bundle(name, root="data", tokenizer=None, tokenize=True):
    dataset = load_dataset(name, root=root, tokenizer=tokenizer, tokenize=tokenize)
    if name in ["ogbl-citation2"]:
        return dataset.data, dataset.get_edge_split(), LinkEvaluator(name)
    else:
        return dataset.data, dataset.get_idx_split(), NodeEvaluator(name)
