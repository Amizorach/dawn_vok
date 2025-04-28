import torch.nn as nn

class VCriterion:
    def __init__(self, model):
        self.criterion = nn.MSELoss()

    def __call__(self, batch_ret):
        return self.criterion(batch_ret[0], batch_ret[1])