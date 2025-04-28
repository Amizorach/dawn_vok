import torch.optim as optim

class VOptimizer:
    def __init__(self, model, lr=1e-3):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)


class SmartLROptimizer(VOptimizer):
    def __init__(self, model, lr=1e-3):
        super().__init__(model, lr)
        self.lr = lr
        self.best_loss = float('inf')
        self.no_improvement_epochs = 0
    def step(self):
        self.optimizer.step()
    
    def update_loss(self, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            self.no_improvement_epochs = 0
        else:
            self.no_improvement_epochs += 1
        if self.no_improvement_epochs > 100:
            self.lr *= 0.1
            self.no_improvement_epochs = 0
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        