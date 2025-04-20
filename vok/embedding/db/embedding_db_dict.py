

import time
import numpy as np
import torch
from dawn_vok.db.mongo_utils import MongoUtils
from dawn_vok.vok.embedding.db.embedding_db import EmbeddingDB


class EmbeddingDBMongo(EmbeddingDB):
    def __init__(self, table_id, generator_id=None, db_name=None, latent_dim=32, device="cpu"):
        super().__init__(table_id, generator_id, latent_dim, "mongo", db_name)
        self.current_total_count = 0
        self.device = device or "cpu"
    def to_dict(self):
        ret = super().to_dict()
        return ret
    
    def populate_from_dict(self, d):
        super().populate_from_dict(d)
        self.current_total_count = d.get("current_total_count", self.current_total_count)
        # self.search_matrix = d.get("search_matrix", self.search_matrix)
        # self.search_device = d.get("search_device", self.search_device)
        self.seed = d.get("seed", self.seed)
        return self
    
    def get_or_assign_vectors(self, column_ids_input):
        """
        Gets or creates indices and generates corresponding latent vectors on CPU
        using per-index deterministic generation (seed + advance).

        Uses external atomic function ('MongoUtils.atm_increment_value')
        to manage the index counter in the database for new index requests.

        Args:
            column_ids_input:
                - A single non-negative int: Get vector for this index.
                - A list of non-negative ints: Get vectors for these indices.
                - A negative int -k: Request k new unique indices and their vectors.

        Returns:
            A tuple containing:
            - A numpy array where columns are the requested vectors (shape: [dimension, num_requested]).
            - A list of the integer indices corresponding to the vectors.
        """
        # print(f"Received request for column_ids: {column_ids_input}") # Optional Debug
        actual_column_ids = []

        if isinstance(column_ids_input, int) and column_ids_input < 0:
            # --- Request for multiple NEW indices ---
            num_new_indices = column_ids_input * -1
            if num_new_indices <= 0:
                raise ValueError("Number of requested new indices must be positive.")

            # print(f"Requesting {num_new_indices} new indices from DB...") # Optional Debug
            try:
                # ** Use External Atomic Increment Function **
                start_index = MongoUtils.atm_increment_value(
                    db_name=self.get_db_name(),
                    collection_name=self.get_collection_name(),
                    document_id=self.generator_id,
                    variable_path="next_available_index", # Field to increment
                    inc=num_new_indices
                )
                actual_column_ids = list(range(start_index, start_index + num_new_indices))
                # Keep local counter consistent with DB state *after* increment
                self.next_available_index = start_index + num_new_indices
                # print(f"Atomically assigned indices: {actual_column_ids}. Updated local counter to {self.next_available_index}") # Optional Debug

            except Exception as e:
                raise RuntimeError(f"Failed to get next ID block via atomic increment. Error: {e}") from e

        elif isinstance(column_ids_input, int) and column_ids_input >= 0:
             # --- Request for a SINGLE existing index ---
             actual_column_ids = [column_ids_input]
        elif isinstance(column_ids_input, (list, tuple)):
             # --- Request for a LIST of existing indices ---
             if not column_ids_input: return np.empty((self.dimension, 0), dtype=np.float32), []
             if not all(isinstance(i, int) and i >= 0 for i in column_ids_input):
                 raise ValueError("List must contain only non-negative integers.")
             actual_column_ids = list(column_ids_input)
        else:
            raise TypeError("column_ids must be an int or list/tuple of ints.")

        if not actual_column_ids:
            return np.empty((self.dimension, 0), dtype=np.float32), []

        # --- Generate requested vectors individually using seed and advance ---
        # print(f"Generating vectors for indices: {actual_column_ids} using seed {self.seed}") # Optional Debug
        result_vectors_list = []
        try:
            for index in actual_column_ids:
                 # --- Generate vector for this specific index ---
                 if not isinstance(index, int) or index < 0: raise ValueError(f"Invalid index encountered: {index}") # Validation
                 bit_generator = np.random.PCG64(self.seed) # Re-seed
                 offset = index * self.dimension
                 if offset > 0:
                     try: bit_generator.advance(offset) # Advance state
                     except Exception as e: raise RuntimeError(f"Failed to advance PRNG state for index {index}") from e
                 rng = np.random.Generator(bit_generator) # Use advanced state
                 vector = rng.standard_normal(size=self.dimension, dtype=np.float32) # Generate vector
                 result_vectors_list.append(vector)
        except Exception as e:
             raise RuntimeError(f"Failed during vector generation loop. Error: {e}") from e

        # Stack the individually generated vectors into the result matrix
        if not result_vectors_list:
             return np.empty((self.dimension, 0), dtype=np.float32), actual_column_ids

        result_vectors = np.stack(result_vectors_list, axis=-1) # Stack along new last axis (columns)
        # print("Vector generation complete.") # Optional Debug

        return result_vectors, actual_column_ids
    def prepare_for_search(self,):
        """
        Generates the full matrix of latent vectors up to current_total_count
        and loads it onto the specified device (CPU or GPU) for searching.
        Uses PER-INDEX generation to ensure consistency with get_or_assign_vectors.

        Args:
            current_total_count: The total number of vectors (indices 0 to N-1)
                                 to generate and prepare. Load this from DB state.
            device: The target device: 'cpu' or 'cuda' (or specific 'cuda:0' etc.).
        """
        if not isinstance(self.current_total_count, int) or self.current_total_count < 0:
             raise ValueError("current_total_count must be a non-negative integer.")
        if self.current_total_count == 0:
             print("Warning: current_total_count is 0. No search matrix to prepare.")
             self._search_matrix = None; self._search_device = None; self._prepared_count = 0; return

        print(f"Preparing search matrix for {self.current_total_count} vectors on device '{self.search_device}' (using per-index generation)...")
        start_time = time.time()
        num_vectors = self.current_total_count

        # --- Generate ALL vectors ON CPU using reliable per-index method ---
        vectors_list = []
        try:
            print(f"Generating {num_vectors} vectors individually (Dim: {self.dimension})...")
            for index in range(num_vectors):
                 # --- Generate vector for this specific index ---
                 bit_generator = np.random.PCG64(self.seed) # Re-seed
                 offset = index * self.dimension
                 if offset > 0:
                     try: bit_generator.advance(offset) # Advance state
                     except Exception as e: raise RuntimeError(f"Failed to advance PRNG state for index {index}") from e
                 rng = np.random.Generator(bit_generator) # Use advanced state
                 vector = rng.standard_normal(size=self.dimension, dtype=np.float32) # Generate vector
                 vectors_list.append(vector)

            # Stack them into the matrix (columns are vectors)
            if not vectors_list: # Should only happen if num_vectors was 0
                 cpu_matrix = np.empty((self.dimension, 0), dtype=np.float32)
            else:
                 cpu_matrix = np.stack(vectors_list, axis=-1)
            print("CPU matrix generated via per-index method.")

        except Exception as e:
             raise RuntimeError(f"Failed during per-index vector generation loop for prepare_for_search. Error: {e}") from e

        # --- Load onto target device (same logic as before) ---
        if self.device.lower() == 'cpu':
            self._search_matrix = cpu_matrix
            self._search_device = 'cpu'
            print("Search matrix prepared on CPU.")
        elif 'cuda' in self.device.lower():
            try:
                # (GPU transfer logic remains the same)
                print(f"Transferring {cpu_matrix.shape} matrix to GPU device '{self.device}'...")
                gpu_tensor = torch.from_numpy(cpu_matrix).to(self.device)
                self._search_matrix = gpu_tensor
                self._search_device = self.device
                print("Search matrix prepared on GPU.")
            except Exception as e: raise RuntimeError(f"Failed to prepare search matrix on GPU device '{self.device}'. Error: {e}") from e
        else: raise ValueError(f"Unsupported device specified: '{self.device}'. Use 'cpu' or 'cuda'.")

        self._prepared_count = num_vectors
        end_time = time.time()
        print(f"Preparation finished in {end_time - start_time:.2f} seconds.")
   
    
    
    def find_closest(
            self,
            query_vector: np.ndarray,
            k: int = 1,
            return_vectors: bool = False
        ) :
            """
            Finds the k indices of the vectors in the prepared search matrix that are
            closest to the query_vector (using Euclidean distance). Optionally returns
            the vectors themselves.

            Requires prepare_for_search() to have been called successfully first.

            Args:
                query_vector: A numpy array of shape (dimension,) representing the
                            vector to search for.
                k: The number of nearest neighbors to find. Defaults to 1.
                return_vectors: If True, also returns the actual latent vectors
                                of the k nearest neighbors. Defaults to False.

            Returns:
                A tuple containing:
                - A list of the integer indices of the k closest vectors found.
                - EITHER a numpy array (shape: [dimension, k]) containing the k vectors
                if return_vectors is True, OR None if return_vectors is False.
                Returns None if the matrix is not prepared or an error occurs.
            """
            if self._search_matrix is None or self._search_device is None or self._prepared_count == 0:
                print("Error: Search matrix not prepared or empty. Call prepare_for_search() first.")
                return None
            if query_vector.shape != (self.dimension,):
                raise ValueError(f"query_vector must have shape ({self.dimension},), but got {query_vector.shape}")
            if query_vector.dtype != np.float32:
                query_vector = query_vector.astype(np.float32)
            if not isinstance(k, int) or k <= 0:
                raise ValueError("k must be a positive integer.")

            k = min(k, self._prepared_count)
            if k == 0: return ([], None)

            # print(f"Searching for {k} closest vector(s) on {self._search_device}...") # Optional Debug
            start_time = time.time()
            closest_indices = []
            closest_vectors = None

            try:
                if self._search_device == 'cpu':
                    # --- CPU Search ---
                    if not isinstance(self._search_matrix, np.ndarray): raise TypeError("...")
                    diff = self._search_matrix - query_vector[:, np.newaxis]
                    sq_distances = np.sum(diff * diff, axis=0)
                    sorted_indices = np.argsort(sq_distances)
                    closest_indices = sorted_indices[:k].tolist()
                    if return_vectors: closest_vectors = self._search_matrix[:, closest_indices]

                elif 'cuda' in self._search_device:
                    # --- GPU Search ---
                    if not isinstance(self._search_matrix, torch.Tensor): raise TypeError("...")
                    query_tensor = torch.from_numpy(query_vector).to(self._search_device)
                    query_tensor_T = query_tensor.unsqueeze(0)
                    matrix_T = self._search_matrix.T
                    distances = torch.cdist(query_tensor_T, matrix_T, p=2).squeeze()
                    _, closest_indices_tensor = torch.topk(distances, k, largest=False, sorted=True)
                    closest_indices = closest_indices_tensor.cpu().tolist()
                    if return_vectors:
                        closest_vectors_tensor = self._search_matrix[:, closest_indices_tensor]
                        closest_vectors = closest_vectors_tensor.cpu().numpy()
                else: raise RuntimeError(f"...")

            except Exception as e:
                print(f"Error during search on {self._search_device}. Error: {e}")
                return None

            end_time = time.time()
            # print(f"Search completed in {end_time - start_time:.4f} seconds. Found {len(closest_indices)} closest indices: {closest_indices[:10]}{'...' if k>10 else ''}") # Optional Debug
            return (closest_indices, closest_vectors)

    def clear_search_data(self):
        """Clears the potentially large search matrix from memory."""
        print("Clearing search matrix...")
        self._search_matrix = None; self._search_device = None; self._prepared_count = 0
        if  'cuda' in str(self._search_device) and torch.cuda.is_available():
             try: torch.cuda.empty_cache()
             except Exception: pass
        print("Search data cleared.")

    @property
    def is_prepared_for_search(self):
        return self._search_matrix is not None and self._prepared_count > 0

    @property
    def prepared_info(self):
        if not self.is_prepared_for_search: return {"status": "Not prepared"}
        return {"status": "Prepared", "device": self._search_device, "vector_count": self._prepared_count, "dimension": self.latent_dim}
