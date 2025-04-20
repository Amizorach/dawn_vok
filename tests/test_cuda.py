import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Using device:", device)

# model = nn.Sequential(
#     nn.Linear(10, 5),
#     nn.ReLU(),
#     nn.Linear(5, 2)
# ).to(device)

# x = torch.randn(1, 10).to(device)

# with torch.no_grad():
#     output = model(x)

# print("Output:", output)

# from sentence_transformers import SentenceTransformer

# model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
# embedding1 = model.encode("temperature")
# embedding2 = model.encode("humidity")
# embedding3 = model.encode("wind speed")

# print("Embedding shape:", embedding1.shape)
# print("Embedding shape:", embedding2.shape)
# print("Embedding shape:", embedding3.shape)

# # print("Cosine similarity between embedding1 and embedding2:", torch.nn.functional.cosine_similarity(embedding1, embedding2))
# # print("Cosine similarity between embedding1 and embedding3:", torch.nn.functional.cosine_similarity(embedding1, embedding3))


# embedding11 = model.encode("temperature")
# embedding22 = model.encode("humidity")
# embedding33 = model.encode("wind speed")
# print("Embedding11:", embedding11)
# print("Embedding22:", embedding22)
# print("Embedding33:", embedding33)
# print("Embedding shape:", embedding11 - embedding1)
# print("Embedding shape:", embedding3 - embedding33)

import os
import numpy as np
import joblib
import matplotlib
matplotlib.use("Qt5Agg")  # or fallback to "TkAgg" if needed
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from typing import List, Optional

# Load the model once globally
model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

# Full list of sensor types for training PCA
sensor_types = [
    "temperature", "temperature max 10 minutes", "temperature min 10 minutes", "temperature wet", "humidity", "humidity max 10 minutes", "humidity min 10 minutes",
    "humidity wet", "radiation", "tension", "wind", "light", "pressure",
    "CO2", "pH", "salinity", "soil moisture", "rainfall", "airflow",
    "solar radiation", "leaf wetness", "chlorophyll", "NDVI", "barometric pressure",
    "water level", "dew point", "vapor pressure", "conductivity", "PAR"
]


def get_semantic_embedding(
    text: str,
    target_dim: int = 16,
    pca_cache_dir: str = "./pca_cache",
    visualize: bool = False
) -> np.ndarray:
    """
    Returns a static reduced-dimensionality semantic embedding for a given word.

    Args:
        text (str): The word or phrase to encode.
        target_dim (int): Target output dimension.
        pca_cache_dir (str): Where to cache PCA model.
        visualize (bool): If True, show a 2D PCA scatter plot of sensor types.

    Returns:
        np.ndarray: Static reduced embedding of size target_dim.
    """
    os.makedirs(pca_cache_dir, exist_ok=True)

    # Unique PCA model cache filename
    model_id = "MiniLM_L3"
    pca_id = f"{model_id}_to_{target_dim}.joblib"
    pca_path = os.path.join(pca_cache_dir, pca_id)

    # Fit or load PCA
    if os.path.exists(pca_path):
        pca = joblib.load(pca_path)
    else:
        base_embeddings = model.encode(sensor_types)
        dim = min(target_dim, len(base_embeddings))  # Ensure we don't exceed available samples
        pca = PCA(n_components=dim)
        pca.fit(base_embeddings)
        joblib.dump(pca, pca_path)

    if visualize:
        _visualize_embeddings(model, sensor_types)

    # Generate and reduce the embedding
    full_vector = model.encode([text])[0]
    reduced_vector = pca.transform([full_vector])[0]
    return reduced_vector

def _visualize_embeddings(model, terms: List[str]):
    """
    Plots a 2D PCA of embeddings for the given terms.
    """
    embeddings = model.encode(terms)
    reduced = PCA(n_components=2).fit_transform(embeddings)

    plt.figure(figsize=(10, 6))
    for i, label in enumerate(terms):
        x, y = reduced[i]
        plt.scatter(x, y, alpha=0.7)
        plt.text(x + 0.01, y + 0.01, label, fontsize=9)
    plt.title("2D PCA Visualization of Sensor Type Embeddings")
    plt.grid(True)
    plt.tight_layout()
    print("Saving visualization to embedding_visualization.png")
    plt.savefig("embedding_visualization.png", dpi=150)

    plt.show()
vec = get_semantic_embedding("temperature", target_dim=16, visualize=True)
print(vec.shape)  # (16,)
print(vec)
get_semantic_embedding("humidity", target_dim=16, visualize=True)
