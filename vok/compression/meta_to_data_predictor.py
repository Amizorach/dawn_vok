from fileinput import filename
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from dawn_vok.vok.data_request.models.raw_data_retriever import FakeRawDataRetriever

class MetaToDataPredictor(nn.Module):
    def __init__(self, input_dim=136, hidden_dims=[ 256, 256], output_dim=1024):
        super(MetaToDataPredictor, self).__init__()
        mlp_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            mlp_layers += [
                nn.Linear(prev_dim, h_dim),
                nn.ReLU()
            ]
            prev_dim = h_dim
        self.mlp = nn.Sequential(*mlp_layers)
        # self.lstm = nn.LSTM(prev_dim, prev_dim, batch_first=True)
        self.final_fc = nn.Linear(prev_dim, output_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.mlp(x)  # -> (batch, seq_len, hidden_dims[-1])
        # x, _ = self.lstm(x)  # -> (batch, seq_len, hidden_dims[-1])
        out = self.final_fc(x)  # -> (batch, seq_len, output_dim)
        return out

class MultiHeadMetaToDataPredictor(nn.Module):
    def __init__(self, input_dim=136, hidden_dims=[512, 256, 128], output_dim=1024, num_heads=24):
        super(MultiHeadMetaToDataPredictor, self).__init__()
        self.num_heads = num_heads
        self.heads = nn.ModuleList([
            MetaToDataPredictor(input_dim, hidden_dims, output_dim)
            for _ in range(num_heads)
        ])

    def forward(self, x, head_idx=None):
        # x: (batch, seq_len, input_dim)
        batch_size = x.size(0)
        if head_idx is None:
            # randomly assign a head per example
            head_idx = torch.randint(0, self.num_heads, (batch_size,), device=x.device)
        outputs = []
        # process each sample with its assigned head
        for i in range(batch_size):
            head = int(head_idx[i].item())
            xi = x[i:i+1]  # keep batch dimension
            out_i = self.heads[head](xi)
            outputs.append(out_i)
        # concatenate back into batch
        return torch.cat(outputs, dim=0)

class MetaToDataPredictorDataset(torch.utils.data.Dataset):
    def __init__(self, full_embeddings, samples, head_ids=None, device='cpu'):
        self.full_embeddings = full_embeddings
        self.samples = samples
        self.head_ids = head_ids if head_ids is not None else None
        self.device = device
        # convert to tensors
        for i in range(len(self.full_embeddings)):
            self.full_embeddings[i] = torch.tensor(self.full_embeddings[i], dtype=torch.float32, device=self.device)
            self.samples[i] = torch.tensor(self.samples[i], dtype=torch.float32, device=self.device)

    def __len__(self):
        return len(self.full_embeddings)
    
    def __getitem__(self, idx):
        x = self.full_embeddings[idx]
        y = self.samples[idx]
        if self.head_ids is not None:
            head = self.head_ids[idx]
            head = torch.tensor(head, dtype=torch.long, device=self.device) if head is not None else None
            return x, y, head
        else:
            return x, y, 0

class Trainer:
    def __init__(self, model, train_loader, val_loader=None, lr=1e-3, device='cpu', multi_head=False, filename='best_model1.pth', load_model=True):
        self.model = model.to(device)
        self.filename = filename or 'best_model1.pth'
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.multi_head = multi_head
        self.criterion = nn.MSELoss()
        if load_model:
            self.load_model(self.filename)

        plt.ion()
        self.loss_history = []
        self.fig, (self.ax_loss, self.ax_embed) = plt.subplots(1, 2, figsize=(12, 4))

    def train(self, epochs=1000, log_interval=10, lr=None):
        best_loss = float('inf')
        if lr is not None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        for epoch in range(1, epochs+1):
            self.model.train()
            total_loss = 0.0
            for X, y, head_idx in self.train_loader:
                X, y = X.to(self.device), y.to(self.device)
                # head_idx may be None or a Tensor
               
                self.optimizer.zero_grad()
                if self.multi_head:
                    head_idx = head_idx.to(self.device)
                    preds = self.model(X, head_idx)
                else:
                    preds = self.model(X)
                loss = self.criterion(preds, y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * X.size(0)
            avg_loss = total_loss / len(self.train_loader.dataset)
            self.loss_history.append(avg_loss)
            print(f"Epoch {epoch}/{epochs} - Train MSE: {avg_loss:.6f} - best: {best_loss:.6f} - lr: {self.optimizer.param_groups[0]['lr']}")
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_model(self.filename)
            if epoch % log_interval == 0:
                # plot last batch
                self.plot_samples(y, preds)
            if self.val_loader is not None:
                self.validate(epoch)

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for X, y, head_idx in self.val_loader:
                X, y = X.to(self.device), y.to(self.device)
                head_idx = head_idx.to(self.device) if head_idx is not None else None
                preds = self.model(X, head_idx)
                loss = self.criterion(preds, y)
                total_loss += loss.item() * X.size(0)
        avg_loss = total_loss / len(self.val_loader.dataset)
        print(f"Epoch {epoch} - Val   MSE: {avg_loss:.6f}")

    def plot_samples(self, gt, z):
        self.ax_loss.clear()
        lh = min(len(self.loss_history), 200)
        self.ax_loss.plot(range(1, lh + 1), self.loss_history[-lh:])
        self.ax_loss.set_xlabel("Epoch")
        self.ax_loss.set_ylabel("Avg Loss")
        self.ax_loss.set_title("Training Loss")
        self.ax_loss.grid(True)

        self.ax_embed.clear()
        self.ax_embed.plot(gt.cpu().detach().numpy()[0], linestyle='--', label='Ground-Truth')
        self.ax_embed.plot(z.cpu().detach().numpy()[0], linestyle='-', label='Predicted')
        self.ax_embed.set_xlabel("Dimension Index")
        # self.ax_embed.set_ylim(0, 1)           # <— lock y‑axis between 0 and 1

        self.ax_embed.legend()
        self.ax_embed.grid(True)
        self.fig.canvas.draw_idle()
        plt.pause(0.01)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        try:
            state_dict = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error loading model: {e}")

def print_memory_usage(model, train_dataset, retriever):
    num_params = sum(p.numel() for p in model.parameters())
    model_size_bytes = num_params * 4
    model_size_mb = model_size_bytes / (1024 ** 2)
    emb0, target0, _ = train_dataset[0]
    per_elem_bytes = emb0.element_size()
    sample_bytes = (emb0.numel() + target0.numel()) * per_elem_bytes
    total_bytes = sample_bytes * len(train_dataset)
    total_mb = total_bytes / (1024 ** 2)
    print(f"Model size: {model_size_mb:.2f} MB ({num_params:,} parameters)")
    print(f"Dataset size: {total_mb:.2f} MB ({len(train_dataset)} samples)")
    print(f"Original size: {retriever.get_original_size()/1024**2:.2f} MB")

if __name__ == "__main__":
    # instantiate multi-head model
    # model = MultiHeadMetaToDataPredictor(output_dim=144, num_heads=4)
    model = MetaToDataPredictor(output_dim=144)
    file_names=['ims_ariel_21.pkl', 'ims_paran_207.pkl', 'ims_haifa_university_42.pkl', 'ims_bet_dagan_54.pkl']
    file_names = 'ims_bet_dagan_54.pkl'
    file_names = [ 'ims_paran_207.pkl']
    # file_names = ['ims_afeq.pkl']
    # file_names = ['synoptic_c0933_2025_04_16_altimeter_temperature.pkl']
    allowed_sensor_types = ['temperature', 'humidity']
    retriever = FakeRawDataRetriever(file_names, allowed_sensor_types=allowed_sensor_types)
                                                                                                                                                                        
    full_embeddings, embeddings, samples = retriever.create_samples(sample_size=144, sample_resolution=6, max_samples=2000)

    # create head assignments (example: round-robin)
    if isinstance(model, MultiHeadMetaToDataPredictor):
        head_ids = [i % model.num_heads for i in range(len(samples))]
    else:
        head_ids = None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = MetaToDataPredictorDataset(full_embeddings, samples, head_ids=head_ids, device=device)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True)

    # print memory usage
  
    print_memory_usage(model, train_dataset, retriever)
    fn = 'best_model_11.pth'
    log_interval = 10
    trainer = Trainer(model, train_loader, lr=0.01, filename=fn, load_model=True)
   # trainer.train(epochs=50, log_interval=log_interval, lr=0.1)
    trainer.train(epochs=3000, log_interval=log_interval, lr=0.01)
    trainer.train(epochs=4000, log_interval=log_interval, lr=0.001)
    trainer.train(epochs=1000, log_interval=log_interval, lr=0.0001)
    print_memory_usage(model, train_dataset, retriever)
