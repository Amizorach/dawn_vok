import time
import torch
import pandas as pd

from dawn_vok.vok.embedding.db.vector_db.vok_searchable_db import VOKSearchableDB

def test_performance_vok_db():
    latent_dim = 144
    meta_dim   = 10
    full_dim   = latent_dim + meta_dim
    num_records = 1_000_000
    num_queries = 1_000

    # 1) Generate synthetic data
    torch.manual_seed(0)
    data = torch.randn(num_records, full_dim)
    query = torch.randn(1, latent_dim)

    results = []

    # CPU benchmark
    db_cpu = VOKSearchableDB(latent_dim=latent_dim, meta_dim=meta_dim, device='cpu')
    db_cpu.add_latents(data)
    t0 = time.time()
    db_cpu.finalize()
    cpu_finalize = time.time() - t0

    _ = db_cpu.search(query, top_k=5)  # warm‑up
    t0 = time.time()
    _ = db_cpu.search(query, top_k=5)
    cpu_single = time.time() - t0

    t0 = time.time()
    for _ in range(num_queries):
        _ = db_cpu.search(query, top_k=5)
    cpu_1000 = time.time() - t0

    results.extend([
        {'device': 'cpu', 'operation': 'finalize',      'time_s': cpu_finalize},
        {'device': 'cpu', 'operation': 'single_search', 'time_s': cpu_single},
        {'device': 'cpu', 'operation': '1000_searches', 'time_s': cpu_1000},
    ])

    # GPU benchmark
    if torch.cuda.is_available():
        db_gpu = VOKSearchableDB(latent_dim=latent_dim, meta_dim=meta_dim, device='cuda')
        db_gpu.add_latents(data)
        torch.cuda.synchronize()
        t0 = time.time()
        db_gpu.finalize()
        torch.cuda.synchronize()
        gpu_finalize = time.time() - t0

        _ = db_gpu.search(query, top_k=5)  # warm‑up
        torch.cuda.synchronize()
        t0 = time.time()
        _ = db_gpu.search(query, top_k=5)
        torch.cuda.synchronize()
        gpu_single = time.time() - t0

        t0 = time.time()
        for _ in range(num_queries):
            _ = db_gpu.search(query, top_k=5)
        torch.cuda.synchronize()
        gpu_1000 = time.time() - t0

        results.extend([
            {'device': 'gpu', 'operation': 'finalize',      'time_s': gpu_finalize},
            {'device': 'gpu', 'operation': 'single_search', 'time_s': gpu_single},
            {'device': 'gpu', 'operation': '1000_searches', 'time_s': gpu_1000},
        ])
    else:
        print("CUDA not available; skipping GPU benchmarks.")

    # Use pandas to tabulate
    df = pd.DataFrame(results)
    print(df.to_string(index=False))

if __name__ == "__main__":
    test_performance_vok_db()
