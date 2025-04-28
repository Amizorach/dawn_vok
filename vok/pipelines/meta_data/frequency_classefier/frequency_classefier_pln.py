from dawn_vok.vok.pipelines.meta_data.frequency_classefier.frequency_classefier_dataset import FrequencyClassifierDataset
from dawn_vok.vok.pipelines.meta_data.frequency_classefier.frequency_classefier_model import FrequencyClassifierModel
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dawn_vok.vok.vmodel.pipeline.v_pipeline_node import VOKPipelineNode
from dawn_vok.vok.vmodel.trainer.v_trainer import VOKTrainer

class FrequencyClassifierTrainer(VOKTrainer):
    def __init__(self, node, device, load_model=True, **kwargs):
        super().__init__(node, device, load_model)
        # Use cross-entropy for classification
        self.input_dim = kwargs.get("input_dim", 16)

    def run_batch(self, data_batch):
        x, y = data_batch
        x = x.to(self.device)
        # y contains integer class labels (bin indices)
        y = y.to(self.device).long().view(-1)
        logits, latents = self.pipeline_node.model(x)
        return {'logits': logits, 'y': y, 'latents': latents}

    def time_collate_fn(self, batch):
        xs, ys = zip(*batch)
        xs = torch.stack(xs, dim=0)
        ys = torch.stack(ys, dim=0)
        return xs, ys

    def compute_loss(self, batch_ret):
        """
        Compute classification loss using cross-entropy between logits and integer labels.
        batch_ret: dict containing 'logits' and 'y'
        """
        logits = batch_ret['logits']  # [N, num_bins]
        targets = batch_ret['y']      # [N]
        # CrossEntropyLoss expects raw logits and class indices
        loss = F.cross_entropy(logits, targets)
        return loss

    def train(self, train_dataset, val_dataset=None, epochs=10, log_interval=10, lr=None, batch_size=128):
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=self.time_collate_fn,
            num_workers=0
        )
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True) if val_dataset else None
        best_loss = float('inf')

        for epoch in range(epochs):
            self.pipeline_node.train()
            epoch_loss = 0.0
            num_samples = 0

            for batch_idx, data_batch in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                batch_ret = self.run_batch(data_batch)
                loss = self.compute_loss(batch_ret)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * data_batch[0].size(0)
                num_samples += data_batch[0].size(0)
                if batch_idx == len(self.train_loader) - 1:
                    last_sample = data_batch[1][-1]
                    last_pred = torch.argmax(batch_ret['logits'][-1], dim=0)
            
                    print(f"Last sample: {last_sample}, Last pred: {last_pred}")
            
            avg_epoch_loss = epoch_loss / num_samples if num_samples > 0 else 0.0
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                self.pipeline_node.save_model()
            print(f"Epoch {epoch+1}/{epochs}  Avg CE Loss: {avg_epoch_loss:.6f}")

class FrequencyClassifierPLN(VOKPipelineNode):
    def __init__(self, node_id, name=None, model_info=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = FrequencyClassifierModel()
        super().__init__(node_id, name, model_info, model=model)
        self.load_trainer()

    def load_trainer(self):
        if self.trainer is None:
            self.trainer = FrequencyClassifierTrainer(self, device=self.device, load_model=True)
        return self.trainer

    def train_model(self, dataset=None):
        self.load_trainer()
        dataset = dataset or FrequencyClassifierDataset(length=10000)
        print('train_model', len(dataset))
        self.trainer.train(dataset, epochs=1000)

if __name__ == "__main__":
    pln = FrequencyClassifierPLN(node_id='test_frequency_classifier', name='FrequencyClassifierPLN')
    pln.train_model()