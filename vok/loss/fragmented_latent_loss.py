import torch
import torch.nn as nn
import torch.nn.functional as F

class FragmentedLatentLoss(nn.Module):
    def __init__(self, fragments, latent_dim: int):
        """
        fragments: list of dicts, each with keys:
            - 'start': int, start index (inclusive)
            - 'end':   int, end index (inclusive if equal to start for single element,
                       otherwise exclusive)
            - 'loss':  'mse' or 'cosine'
            - 'weight': float
        latent_dim: the full size of your latent vectors
        """
        super().__init__()
        self.latent_dim = latent_dim

        # validate fragments
        for frag in fragments:
            s, e = frag['start'], frag['end']
            if not (0 <= s <= e < latent_dim):
                raise ValueError(
                    f"Fragment indices out of bounds or inverted: "
                    f"start={s}, end={e}, latent_dim={latent_dim}"
                )
            if frag['loss'] not in ('mse', 'cosine'):
                raise ValueError(f"Unsupported loss type: {frag['loss']}")
            if not isinstance(frag['weight'], (int, float)):
                raise ValueError(f"Weight must be a number, got {frag['weight']!r}")
        self.fragments = fragments

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred, target: (batch_size, latent_dim)
        """
        if pred.shape != target.shape:
            raise ValueError(f"pred and target must have same shape, got {pred.shape} vs {target.shape}")
        if pred.size(1) != self.latent_dim:
            raise ValueError(f"Expected latent_dim={self.latent_dim}, but got {pred.size(1)}")

        total_loss = 0.0
        for frag in self.fragments:
            s, e = frag['start'], frag['end']
            # if start==end, take exactly that one index
            slice_end = s + 1 if s == e else e

            p_slice = pred[:, s:slice_end]
            t_slice = target[:, s:slice_end]

            if frag['loss'] == 'mse':
                l = F.mse_loss(p_slice, t_slice, reduction='mean')
            else:  # 'cosine'
                cos_sim = F.cosine_similarity(p_slice, t_slice, dim=1)
                l = (1 - cos_sim).mean()

            total_loss = total_loss + frag['weight'] * l

        return total_loss
