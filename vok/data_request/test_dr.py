import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
class POCEncoderCosineModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder: 456 → 128 → 32
        self.encoder = nn.Sequential(
            nn.Linear(456, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 32),
        )

    def forward(self, x: torch.Tensor, gt: torch.Tensor):
        """
        x  : Tensor of shape (batch_size, 456)
        gt : Tensor of shape (batch_size, 32)  (ground‑truth embeddings)
        Returns:
            cos_sim: Tensor of shape (batch_size,)
            encoded: Tensor of shape (batch_size, 32)
        """
        # 1) Encode down to 32
        encoded = self.encoder(x)

        # 2) Normalize both vectors
        encoded_norm = F.normalize(encoded[4:32], p=2, dim=-1)
        gt_norm      = F.normalize(gt[4:32],      p=2, dim=-1)

        # 3) Cosine similarity per sample
        cos_sim = F.cosine_similarity(encoded_norm, gt_norm, dim=-1)

        return cos_sim, encoded


def train(model: nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epochs: int = 10):
    """
    A simple training loop that maximizes cosine similarity:
    loss = 1 - cos_sim.mean()

    Args:
        model      : your POCEncoderCosineModel
        dataloader : yields (x, gt) batches
        optimizer  : torch optimizer (e.g. Adam)
        device     : cpu or cuda
        epochs     : number of training epochs
    """
    model.to(device)
    model.train()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for x, gt in dataloader:
            x, gt = x.to(device), gt.to(device)
            optimizer.zero_grad()

            cos_sim, _ = model(x, gt)
            # we want to maximize cos_sim → minimize (1 - cos_sim)
            loss = 1.0 - cos_sim.mean()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(dataloader.dataset)
        print(f"Epoch {epoch:02d} — avg loss: {avg_loss:.4f}")

if __name__ == "__main__":
    # --- Example with dummy data ---
    batch_size = 32
    num_samples = 1024
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dummy tensors
    X  = torch.randn(num_samples, 456)
    GT = torch.randn(num_samples, 32)

    dataset = TensorDataset(X, GT)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = POCEncoderCosineModel()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    train(model, loader, optim, device, epochs=20)
