import torch
import torch.nn.functional as F

class VOKSearchableDB:
    def __init__(self, latent_dim=144, meta_dim=10, device='cpu'):
        """
        latent_dim: dimensionality of the embedding portion
        meta_dim:  number of metadata fields prefixed to each vector
        full_latent_dim = meta_dim + latent_dim
        """
        self.latent_dim = latent_dim
        self.meta_dim = meta_dim
        self.full_latent_dim = latent_dim + meta_dim
        self.device = device

        self._buffer = []
        self.db_tensor = None
        self.sorted_by = None

        # unique (1:1) index
        self.unique_index_keys = None
        self.unique_index_vals = None
        self.unique_index_position = None

        # multi (1:many) CSR index
        self.multi_index_keys = None
        self.multi_index_ptrs = None
        self.multi_index_vals = None
        self.multi_index_position = None

    def add_latents(self, latents: torch.Tensor):
        """Buffer new rows of shape [N, full_latent_dim] before finalization."""
        assert latents.ndim == 2 and latents.shape[1] == self.full_latent_dim, \
            f"Expected [N, {self.full_latent_dim}], got {latents.shape}"
        self._buffer.append(latents)

    def finalize(self, sort_by_meta_pos: int = None):
        """
        Concatenate buffer, optionally sort by one metadata column,
        and move to self.device.
        """
        assert self._buffer, "No latents buffered"
        full = torch.cat(self._buffer, dim=0)  # [N, full_latent_dim]

        if sort_by_meta_pos is not None:
            assert 0 <= sort_by_meta_pos < self.meta_dim, "Invalid sort position"
            order = torch.argsort(full[:, sort_by_meta_pos])
            full = full[order]
            self.sorted_by = sort_by_meta_pos

        self.db_tensor = full.to(self.device)
        self._buffer.clear()

    def move_to(self, device: str):
        """Move all stored tensors (db + indexes) to `device`."""
        self.device = device
        if self.db_tensor is not None:
            self.db_tensor = self.db_tensor.to(device)
        for attr in (
            'unique_index_keys', 'unique_index_vals',
            'multi_index_keys', 'multi_index_ptrs', 'multi_index_vals'
        ):
            tensor = getattr(self, attr)
            if tensor is not None:
                setattr(self, attr, tensor.to(device))

    def prepare_for_use(
        self,
        sort_by_meta_pos: int = None,
        unique_index_positions: list[int] = None,
        multi_index_positions: list[int] = None,
        device: str = None
    ):
        """
        Convenience method to finalize the DB, build requested indexes,
        and move everything to `device` for immediate querying.
        - sort_by_meta_pos: metadata column to sort+enable binary_search
        - unique_index_positions: list of positions for 1:1 index
        - multi_index_positions: list of positions for CSR 1:many index
        - device: torch device string (e.g. 'cuda' or 'cpu')
        """
        # finalize & sort
        self.finalize(sort_by_meta_pos)

        # build unique indexes
        if unique_index_positions:
            for pos in unique_index_positions:
                self.create_unique_index(pos)

        # build multi indexes
        if multi_index_positions:
            for pos in multi_index_positions:
                self.create_multi_index(pos)

        # move to target device
        if device:
            self.move_to(device)

    # ------------------------------------------------------------------
    # COSINE SIMILARITY SEARCH
    # ------------------------------------------------------------------
    def search(
        self,
        query: torch.Tensor,
        top_k: int = 1,
        latent_range: tuple[int,int] = None,
        latent_mask: torch.Tensor = None,
        return_indices: bool = False
    ):
        """
        Cosine search over a slice or mask of the latent portion.
        - query: [Q, D], where D matches the selected dims
        - latent_range: (start, end) over full vector dims
        - latent_mask: bool mask [full_latent_dim], overrides range
        - return_indices: if True, also return the indices
        Returns:
            results: [Q, top_k, full_latent_dim]
            scores:  [Q, top_k]
            (optional) indices: [Q, top_k]
        """
        assert self.db_tensor is not None, "DB not finalized"

        # select dims
        if latent_mask is not None:
            assert latent_mask.shape == (self.full_latent_dim,) and latent_mask.dtype == torch.bool
            db_slice = self.db_tensor[:, latent_mask]
        else:
            start, end = latent_range or (self.meta_dim, self.full_latent_dim)
            assert 0 <= start < end <= self.full_latent_dim
            db_slice = self.db_tensor[:, start:end]

        # normalize and compute similarity
        db_slice = F.normalize(db_slice, dim=-1)
        q = F.normalize(query.to(self.device), dim=-1)
        if q.ndim == 1:
            q = q.unsqueeze(0)
        assert q.shape[1] == db_slice.shape[1], "Query dim mismatch"

        sim = torch.matmul(q, db_slice.T)           # [Q, N]
        vals, idx = sim.topk(top_k, dim=-1)         # [Q, top_k]
        res = self.db_tensor[idx]                   # [Q, top_k, full_latent_dim]

        return (res, vals, idx) if return_indices else (res, vals)

    # ------------------------------------------------------------------
    # META SEARCH (LINEAR & BINARY)
    # ------------------------------------------------------------------
    def find_by_meta(self, value: float, position: int, return_indices: bool = False):
        """Linear scan for exact metadata match."""
        assert self.db_tensor is not None
        assert 0 <= position < self.meta_dim

        col = self.db_tensor[:, position]
        mask = col == value
        return mask.nonzero(as_tuple=False).squeeze(1) if return_indices else self.db_tensor[mask]

    def binary_search_by_meta(self, value: float, position: int):
        """Binary search after finalize(sort_by_meta_pos=position)."""
        assert self.db_tensor is not None
        assert self.sorted_by == position, "DB not sorted by this position"
        cpu = self.db_tensor.to('cpu')
        col = cpu[:, position]
        N = col.size(0)

        def lb():
            l, r = 0, N
            while l < r:
                m = (l + r)//2
                if col[m].item() < value: l = m+1
                else: r = m
            return l
        def ub():
            l, r = 0, N
            while l < r:
                m, cmp = (l+r)//2, col[(l+r)//2].item() <= value
                if cmp: l = m+1
                else: r = m
            return l

        s, e = lb(), ub()
        return cpu[s:e].to(self.device)

    # ------------------------------------------------------------------
    # UNIQUE INDEX (1:1 mapping)
    # ------------------------------------------------------------------
    def create_unique_index(self, position: int):
        """Build O(log K) lookup tables for unique keys at `position`."""
        assert self.db_tensor is not None
        assert 0 <= position < self.meta_dim

        col = self.db_tensor[:, position].to('cpu')
        sorted_col, sorted_idx = torch.sort(col)
        keys, counts = torch.unique_consecutive(sorted_col, return_counts=True)
        ptrs = torch.cat([torch.tensor([0]), counts.cumsum(0)])
        first_ptrs = ptrs[:-1]
        first_inds = sorted_idx[first_ptrs]

        self.unique_index_keys = keys.to(self.device)
        self.unique_index_vals = first_inds.to(self.device)
        self.unique_index_position = position

    def get_from_unique_index(self, value: float):
        """Retrieve single full vector for exact key."""
        assert self.unique_index_keys is not None
        key = torch.tensor([value], device=self.device)
        pos = torch.searchsorted(self.unique_index_keys, key).item()
        if pos >= len(self.unique_index_keys) or self.unique_index_keys[pos] != value:
            return torch.empty((0, self.full_latent_dim), device=self.device)
        return self.db_tensor[self.unique_index_vals[pos]]

    # ------------------------------------------------------------------
    # MULTI INDEX (CSR-style mapping)
    # ------------------------------------------------------------------
    def create_multi_index(self, position: int):
        """Build CSR mapping for many-to-many metadata lookup."""
        assert self.db_tensor is not None
        assert 0 <= position < self.meta_dim

        col = self.db_tensor[:, position].to('cpu')
        sorted_col, sorted_idx = torch.sort(col)
        db_inds = torch.arange(len(col), dtype=torch.long)[sorted_idx]
        keys, counts = torch.unique_consecutive(sorted_col, return_counts=True)
        ptrs = torch.cat([torch.tensor([0]), counts.cumsum(0)])

        self.multi_index_keys      = keys.to(self.device)
        self.multi_index_ptrs      = ptrs.to(self.device)
        self.multi_index_vals      = db_inds.to(self.device)
        self.multi_index_position = position

    def get_from_multi_index(self, value: float):
        """Retrieve all full vectors matching `value` via CSR index."""
        assert self.multi_index_keys is not None
        key = torch.tensor([value], device=self.device)
        pos = torch.searchsorted(self.multi_index_keys, key).item()
        if pos >= len(self.multi_index_keys) or self.multi_index_keys[pos] != value:
            return torch.empty((0, self.full_latent_dim), device=self.device)

        start = self.multi_index_ptrs[pos].item()
        end   = self.multi_index_ptrs[pos+1].item()
        rows  = self.multi_index_vals[start:end]
        return self.db_tensor[rows]


def test_vok_searchable_db():
    import torch

    # 1) Create a small toy dataset
    #    - meta_dim = 2, latent_dim = 4 → full_latent_dim = 6
    torch.manual_seed(0)
    meta = torch.tensor([
        [1.0, 10.0],
        [2.0, 20.0],
        [2.0, 20.0],  # duplicate on purpose for multi-index
        [3.0, 30.0],
        [5.0, 50.0],
    ])
    latents = torch.randn(5, 4)
    full = torch.cat([meta, latents], dim=1)  # [5,6]

    # 2) Initialize DB and buffer rows
    db = VOKSearchableDB(latent_dim=4, meta_dim=2, device='cpu')
    db.add_latents(full)

    # 3) Finalize (no sort), then move to CPU
    db.finalize(sort_by_meta_pos=None)
    print("DB tensor:")
    print(db.db_tensor)

    # 4) Cosine search: pick the first embedded vector as query
    q = full[0, 2:].unsqueeze(0)  # [1,4]
    res, scores = db.search(q, top_k=3)
    print("\nCosine search (top 3 for row 0):")
    print("Scores:", scores)
    print("Results:\n", res)

    # 5) Linear meta search
    matches = db.find_by_meta(2.0, position=0)
    print("\nfind_by_meta(value=2.0, position=0) found rows:")
    print(matches)

    # 6) Build and test unique index on meta position 0
    db.create_unique_index(position=0)
    print("\nUnique index keys:", db.unique_index_keys)
    print("Unique index vals:", db.unique_index_vals)
    row = db.get_from_unique_index(3.0)
    print("get_from_unique_index(3.0) →", row)

    # 7) Build and test multi index on meta position 0
    db.create_multi_index(position=0)
    print("\nMulti-index keys:", db.multi_index_keys)
    print("Multi-index ptrs:", db.multi_index_ptrs)
    print("Multi-index vals:", db.multi_index_vals)
    rows = db.get_from_multi_index(2.0)
    print("get_from_multi_index(2.0) →")
    print(rows)

    # 8) Sort by meta position 1 and do binary search
    db = VOKSearchableDB(latent_dim=4, meta_dim=2, device='cpu')
    db.add_latents(full)
    db.finalize(sort_by_meta_pos=1)
    print("\nDB sorted by meta position 1:")
    print(db.db_tensor[:, :2])  # print only meta columns
    bs = db.binary_search_by_meta(20.0, position=1)
    print("binary_search_by_meta(20.0, position=1) →")
    print(bs)

if __name__ == "__main__":
    test_vok_searchable_db()
