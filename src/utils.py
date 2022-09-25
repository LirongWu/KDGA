import torch
import numpy as np


def evaluation(logits, labels):
    preds = torch.argmax(logits, dim=1)
    correct = torch.sum(preds == labels)
    accuracy = correct.item() / len(labels)

    return accuracy

def SetSeed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)