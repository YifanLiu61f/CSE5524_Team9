import torch
from torch.nn.functional import one_hot

class LabelSmoothingCrossEntropy(torch.nn.Module):
    def __init__(self, eps: float = 0.1):
        super().__init__()
        self.eps = eps
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, logits, target):
        n = logits.size(1)
        log_probs = self.log_softmax(logits)
        loss = -(1 - self.eps) * log_probs[range(target.size(0)), target]
        loss -= self.eps * log_probs.mean(dim=1)
        return loss.mean()

def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """Computes the top‑k accuracy for the specified values of k"""
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / target.size(0)))
    return res