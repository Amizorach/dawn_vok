import torch
import torch.nn as nn
import numpy as np
import time
import warnings
# Import Faiss
import torch
import os
import warnings
import time
import faiss # Make sure faiss-cpu or faiss-gpu is installed
import numpy as np

class CategoricalEmbeddingLayer():
    """
    Manages a categorical embedding matrix (lookup table) using PyTorch
    and provides efficient decoding using a Faiss ANN index.

    Handles initialization, saving/loading of both the embedding tensor
    and the Faiss index.
    """
    def __init__(self, cat_id, max_categories, embedding_dim):
        """
        Initializes the configuration, creates the initial matrix, and builds the Faiss index.

        Args:
            cat_id (str or int): An identifier for this categorical dimension.
            max_categories (int): The total number of unique categories.
            embedding_dim (int): The size of the embedding vector.
        """
        if not isinstance(max_categories, int) or max_categories <= 0:
            raise ValueError("max_categories must be a positive integer.")
        if not isinstance(embedding_dim, int) or embedding_dim <= 0:
            raise ValueError("embedding_dim must be a positive integer.")

        self.cat_id = cat_id
        self.max_categories = max_categories
        self.embedding_dim = embedding_dim
        self.embedding_matrix = None # PyTorch Tensor
        self.ann_index = None      # Faiss Index

        print(f"Initializing CategoricalEmbeddingLayer for '{self.cat_id}'...")
        self.initialize_embedding_matrix() # Creates matrix and builds index
        print(f"  Initialized matrix shape: {self.embedding_matrix.shape}")
        if self.ann_index:
            print(f"  Built initial Faiss index (ntotal={self.ann_index.ntotal}).")
        else:
             print("  Warning: Faiss index could not be built during initialization.")


    def _build_faiss_index(self, embedding_matrix):
        """ Helper function to build a Faiss index from an embedding matrix. """
        if embedding_matrix is None:
            print("  Error: Cannot build Faiss index from None matrix.")
            return None
        try:
            dimension = embedding_matrix.shape[1]
            # Using IndexFlatL2 for exact Euclidean distance search.
            index = faiss.IndexFlatL2(dimension)
            # Faiss requires float32 numpy arrays on CPU
            embeddings_np = embedding_matrix.cpu().numpy().astype('float32')
            index.add(embeddings_np)
            print(f"    Built Faiss index: is_trained={index.is_trained}, total={index.ntotal}")
            return index
        except Exception as e:
            print(f"    ERROR building Faiss index: {e}")
            print("    Ensure Faiss is installed correctly and matrix data is valid.")
            return None # Return None if index building fails


    def initialize_embedding_matrix(self, init_method='randn'):
        """
        Initializes or re-initializes the embedding matrix tensor and rebuilds the Faiss index.

        Args:
            init_method (str): Initialization method ('randn', 'xavier', 'kaiming').
        """
        print(f"  Initializing embedding matrix ({self.max_categories} x {self.embedding_dim}) using '{init_method}' method...")
        new_matrix = torch.empty(self.max_categories, self.embedding_dim)

        if init_method == 'randn':
            new_matrix.normal_(mean=0.0, std=0.02)
        elif init_method == 'xavier':
            torch.nn.init.xavier_uniform_(new_matrix)
        elif init_method == 'kaiming':
             torch.nn.init.kaiming_normal_(new_matrix)
        else:
            warnings.warn(f"Unknown init_method '{init_method}'. Using default 'randn'.")
            new_matrix.normal_(mean=0.0, std=0.02)

        self.embedding_matrix = new_matrix
        # --- Rebuild Faiss index whenever matrix changes ---
        print("  Rebuilding Faiss index after initialization...")
        self.ann_index = self._build_faiss_index(self.embedding_matrix)


    def save(self, base_file_path):
        """
        Saves the embedding matrix (.pt) and the Faiss index (.index)
        using a base file path.

        Args:
            base_file_path (str): The base path for saving (e.g., './data/source_id').
                                  '.pt' and '.index' will be appended.
        Returns:
            bool: True if both saves were successful, False otherwise.
        """
        matrix_path = base_file_path + ".pt"
        index_path = base_file_path + ".index"
        matrix_saved = False
        index_saved = False

        # 1. Save Embedding Matrix
        if self.embedding_matrix is None:
            print(f"Error: Embedding matrix for '{self.cat_id}' is not initialized. Cannot save.")
        else:
            print(f"Saving embedding matrix for '{self.cat_id}' to {matrix_path}...")
            try:
                os.makedirs(os.path.dirname(matrix_path), exist_ok=True)
                torch.save(self.embedding_matrix, matrix_path)
                print("  Matrix save successful.")
                matrix_saved = True
            except Exception as e:
                print(f"  Error saving embedding matrix: {e}")

        # 2. Save Faiss Index
        if self.ann_index is None:
             print(f"Error: Faiss index for '{self.cat_id}' is not initialized. Cannot save.")
        else:
            print(f"Saving Faiss index for '{self.cat_id}' to {index_path}...")
            try:
                os.makedirs(os.path.dirname(index_path), exist_ok=True)
                faiss.write_index(self.ann_index, index_path)
                print("  Index save successful.")
                index_saved = True
            except Exception as e:
                 print(f"  Error saving Faiss index: {e}")

        return matrix_saved and index_saved

    def load(self, base_file_path):
        """
        Loads the embedding matrix (.pt) and the Faiss index (.index)
        from a base file path. Performs shape validation for the matrix.

        Args:
            base_file_path (str): The base path for loading (e.g., './data/source_id').
                                  '.pt' and '.index' will be appended.
        Returns:
            bool: True if both loads were successful and consistent, False otherwise.
        """
        matrix_path = base_file_path + ".pt"
        index_path = base_file_path + ".index"
        loaded_matrix = None
        loaded_index = None
        matrix_loaded_ok = False
        index_loaded_ok = False

        # 1. Load Embedding Matrix
        print(f"Loading embedding matrix for '{self.cat_id}' from {matrix_path}...")
        if not os.path.exists(matrix_path):
            print(f"  Error: Matrix file not found at {matrix_path}")
        else:
            try:
                loaded_matrix = torch.load(matrix_path)
                # --- Validation ---
                if not isinstance(loaded_matrix, torch.Tensor):
                    print(f"  Error: Loaded object is not a PyTorch tensor (type: {type(loaded_matrix)}).")
                else:
                    expected_shape = (self.max_categories, self.embedding_dim)
                    if loaded_matrix.shape != expected_shape:
                        print(f"  Error: Shape mismatch! Loaded matrix shape is {loaded_matrix.shape}, "
                              f"but expected {expected_shape}.")
                    else:
                        print(f"  Matrix load successful. Shape: {loaded_matrix.shape}")
                        matrix_loaded_ok = True
            except Exception as e:
                print(f"  Error loading embedding matrix: {e}")

        # 2. Load Faiss Index
        print(f"Loading Faiss index for '{self.cat_id}' from {index_path}...")
        if not os.path.exists(index_path):
             print(f"  Error: Index file not found at {index_path}")
        else:
            try:
                loaded_index = faiss.read_index(index_path)
                # --- Validation ---
                if loaded_index.d != self.embedding_dim:
                     print(f"  Error: Dimension mismatch! Loaded index dimension is {loaded_index.d}, "
                           f"but expected {self.embedding_dim}.")
                elif matrix_loaded_ok and loaded_index.ntotal != loaded_matrix.shape[0]:
                     # Check consistency if matrix also loaded
                     print(f"  Error: Count mismatch! Loaded index ntotal is {loaded_index.ntotal}, "
                           f"but loaded matrix rows are {loaded_matrix.shape[0]}.")
                elif loaded_index.ntotal != self.max_categories:
                     # Check consistency even if matrix didn't load (against config)
                     print(f"  Error: Count mismatch! Loaded index ntotal is {loaded_index.ntotal}, "
                           f"but expected {self.max_categories}.")
                else:
                    print(f"  Index load successful. Dimension={loaded_index.d}, NTotal={loaded_index.ntotal}")
                    index_loaded_ok = True
            except Exception as e:
                 print(f"  Error loading Faiss index: {e}")

        # 3. Assign if both loaded successfully
        if matrix_loaded_ok and index_loaded_ok:
            self.embedding_matrix = loaded_matrix
            self.ann_index = loaded_index
            print("Successfully loaded and assigned both matrix and index.")
            return True
        else:
            print("Loading failed for one or both components. State not updated.")
            # Optionally revert any partial load, e.g., set self.embedding_matrix back if index failed
            return False


    def get_embedding(self, index):
        """
        Retrieves the embedding vector for a specific category index.

        Args:
            index (int): The category index to retrieve.

        Returns:
            torch.Tensor or None: The embedding vector if index is valid, otherwise None.
        """
        if self.embedding_matrix is None:
            print(f"Error: Embedding matrix for '{self.cat_id}' is not initialized.")
            return None
        if not isinstance(index, int) or not (0 <= index < self.max_categories):
            print(f"Error: Index {index} out of bounds for '{self.cat_id}' (max categories: {self.max_categories}).")
            return None
        # Return a detached copy
        return self.embedding_matrix[index].clone().detach()


    def decode(self, latent):
        """
        Decodes an embedding vector (latent) back to the closest category index
        using the pre-built Faiss ANN index.

        Args:
            latent (torch.Tensor): The embedding vector to decode (shape [embedding_dim]).

        Returns:
            int or None: The index of the closest matching category, or None if input is invalid or index not ready.
        """
        # print(f"Decoding latent vector for '{self.cat_id}' using Faiss...") # Less verbose
        if self.ann_index is None:
            print("  Error: Faiss index is not initialized. Cannot decode.")
            return None
        if not isinstance(latent, torch.Tensor):
             print(f"  Error: Input latent must be a PyTorch tensor (got {type(latent)}).")
             return None
        if latent.shape != (self.embedding_dim,):
            print(f"  Error: Input latent shape is {latent.shape}, expected ({self.embedding_dim},).")
            return None

        # --- Faiss ANN Search ---
        try:
            # Prepare query vector for Faiss (needs to be float32 numpy array on CPU, shape [1, dim])
            query_vector_np = latent.detach().cpu().numpy().astype('float32').reshape(1, -1)
            k = 1 # Find the 1 nearest neighbor
            # Perform the search
            distances, indices = self.ann_index.search(query_vector_np, k)

            # indices[0][0] contains the index of the closest embedding
            closest_idx = indices[0][0]
            # print(f"  Decoding successful. Closest index found: {closest_idx}")
            return int(closest_idx) # Return as standard Python int
        except Exception as e:
            print(f"  Error during Faiss search: {e}")
            return None


    def __repr__(self):
        """ String representation of the object. """
        status = "Initialized" if self.embedding_matrix is not None else "Not Initialized"
        shape = self.embedding_matrix.shape if self.embedding_matrix is not None else "N/A"
        index_status = f"Ready (ntotal={self.ann_index.ntotal})" if self.ann_index else "Not Ready"
        return (f"<CategoricalEmbeddingLayer id='{self.cat_id}' "
                f"max_categories={self.max_categories} "
                f"embedding_dim={self.embedding_dim} "
                f"status='{status}' shape={shape} "
                f"ANN_status='{index_status}'>")

# --- Example Usage ---

# 1. Create instance (initializes matrix and builds Faiss index)
num_cats_demo = 50000
emb_dim_demo = 128
print(f"\n--- Creating Layer for Demo (Cats={num_cats_demo}, Dim={emb_dim_demo}) ---")
layer = CategoricalEmbeddingLayer(
    cat_id='demo_id',
    max_categories=num_cats_demo,
    embedding_dim=emb_dim_demo
)
print(layer)

# 2. Get embedding for an index (using get_embedding directly)
target_index = 4321
print(f"\n--- Getting embedding for index {target_index} ---")
latent_vector = layer.get_embedding(target_index)

# 3. Decode the resulting vector using Faiss
if latent_vector is not None:
    print(f"\n--- Decoding the generated latent vector using Faiss ---")
    start_time = time.time()
    decoded_index = layer.decode(latent_vector)
    end_time = time.time()
    # This should be very fast now, regardless of num_cats_demo
    print(f"Decode execution time: {end_time - start_time:.6f} seconds")

    if decoded_index is not None:
        print(f"\nOriginal Index: {target_index}")
        print(f"Decoded Index:  {decoded_index}")
        print(f"Match?          {target_index == decoded_index}")
    else:
        print("Decoding failed.")
else:
    print("Getting embedding failed.")

# 4. Save matrix and index
save_base_path = "./embedding_data/demo_layer" # Will save demo_layer.pt and demo_layer.index
print(f"\n--- Saving data to base path: {save_base_path} ---")
save_success = layer.save(save_base_path)
if save_success:
    print("Save completed.")

    # 5. Create a new instance and load data
    print("\n--- Creating new layer instance ---")
    new_layer = CategoricalEmbeddingLayer(
        cat_id='demo_id_loaded',
        max_categories=num_cats_demo,
        embedding_dim=emb_dim_demo
    )
    print(f"--- Loading data from base path: {save_base_path} ---")
    load_success = new_layer.load(save_base_path)

    if load_success:
        print("\n--- Verifying loaded data by decoding again ---")
        # Use the same latent vector from before
        decoded_index_loaded = new_layer.decode(latent_vector)
        if decoded_index_loaded is not None:
            print(f"Original Index: {target_index}")
            print(f"Decoded Index (from loaded data): {decoded_index_loaded}")
            print(f"Match? {target_index == decoded_index_loaded}")

# Clean up dummy files (optional)
# import shutil
# if os.path.exists("./embedding_data"):
#      shutil.rmtree("./embedding_data")

exit()

class SpecificationCoder:
    """
    Encodes a specification dictionary into a concatenated vector (embeddings + continuous)
    and decodes the vector back into the specification dictionary using Faiss ANN index
    for efficient nearest-neighbor lookup on embeddings.

    Manages embedding layers, stored matrices, and Faiss indices.
    """
    def __init__(self, dimension_configs):
        """
        Initializes the SpecificationCoder and builds Faiss indices.

        Args:
            dimension_configs (list): Configuration for each dimension. (See previous example)
        """
        super().__init__()

        self.dimension_configs = dimension_configs
        self.embedding_layers = nn.ModuleDict()
        self.stored_embeddings = {} # Still store original tensors if needed elsewhere
        self.category_maps = {}
        self.continuous_var_names = []
        self.total_spec_vector_dim = 0
        self.dim_slices = {}
        self.ann_indices = {} # To store Faiss indices

        print("Initializing SpecificationCoder and building Faiss indices...")

        current_offset = 0
        for i, config in enumerate(self.dimension_configs):
            dim_name = config['name']
            dim_type = config['type']
            start_offset = current_offset

            if dim_type == 'categorical':
                num_categories = config.get('num_categories')
                categories = config.get('categories')
                embedding_dim = config['embedding_dim']

                if num_categories is None and categories is None:
                    raise ValueError(f"Categorical dimension '{dim_name}' must have 'num_categories' or 'categories' defined.")
                if num_categories is None:
                    num_categories = len(categories)

                # Create and store embedding layer
                self.embedding_layers[dim_name] = nn.Embedding(
                    num_embeddings=num_categories,
                    embedding_dim=embedding_dim
                )
                # Store weights for lookup (detached)
                embedding_matrix = self.embedding_layers[dim_name].weight.data.clone().detach()
                self.stored_embeddings[dim_name] = embedding_matrix
                print(f"  Dim {i} ('{dim_name}'): Categorical, Num={num_categories}, EmbDim={embedding_dim}, Shape={self.stored_embeddings[dim_name].shape}")

                # --- Build Faiss Index ---
                try:
                    faiss_index = self._build_faiss_index(embedding_matrix)
                    self.ann_indices[dim_name] = faiss_index
                    print(f"    Built Faiss index for '{dim_name}'.")
                except Exception as e:
                    print(f"    ERROR building Faiss index for '{dim_name}': {e}")
                    print("    Ensure Faiss is installed correctly (faiss-cpu or faiss-gpu).")
                    raise e # Re-raise the error

                # Store maps if categories list provided
                if categories:
                    map_to_idx = {name: idx for idx, name in enumerate(categories)}
                    map_to_name = {idx: name for idx, name in enumerate(categories)}
                    self.category_maps[dim_name] = {'to_idx': map_to_idx, 'to_name': map_to_name}
                else:
                     self.category_maps[dim_name] = None

                current_offset += embedding_dim
                self.dim_slices[dim_name] = {'start': start_offset, 'end': current_offset, 'type': 'categorical', 'num_categories': num_categories}

            elif dim_type == 'continuous':
                self.continuous_var_names.append(dim_name)
                current_offset += 1 # Continuous variables take 1 slot
                self.dim_slices[dim_name] = {'start': start_offset, 'end': current_offset, 'type': 'continuous'}
                print(f"  Dim {i} ('{dim_name}'): Continuous")
            else:
                raise ValueError(f"Unknown dimension type '{dim_type}' for dimension '{dim_name}'")

        self.total_spec_vector_dim = current_offset
        print(f"Total Specification Vector Dimension: {self.total_spec_vector_dim}")
        print("SpecificationCoder Initialized with Faiss indices.")

    def _build_faiss_index(self, embedding_matrix):
        """ Helper function to build a Faiss index. """
        dimension = embedding_matrix.shape[1]
        # Using IndexFlatL2 for exact Euclidean distance search.
        # For better speed/memory trade-offs on huge datasets, consider
        # approximate indices like IndexIVFFlat or IndexHNSWFlat.
        index = faiss.IndexFlatL2(dimension)
        # Faiss requires float32 numpy arrays on CPU
        embeddings_np = embedding_matrix.cpu().numpy().astype('float32')
        index.add(embeddings_np)
        print(f"    Faiss index details: is_trained={index.is_trained}, total={index.ntotal}")
        return index

    def encode(self, spec_dict):
        """
        Generates the concatenated specification vector from a dictionary.
        (Identical to previous version)

        Args:
            spec_dict (dict): Contains desired values for each dimension defined in config.
                              e.g., {'source_id': 12345, 'sensor_type': 'sensor_3', 'voltage': 1.1, ...}

        Returns:
            torch.Tensor: The flat specification vector, or None on error. Shape [total_spec_vector_dim].
        """
        vector_parts = []
        try:
            for config in self.dimension_configs:
                dim_name = config['name']
                dim_type = config['type']
                value = spec_dict[dim_name] # Get value from input dict

                if dim_type == 'categorical':
                    emb_layer = self.embedding_layers[dim_name]
                    num_categories = config.get('num_categories') or len(config['categories'])
                    cat_map = self.category_maps.get(dim_name)
                    idx = -1

                    if cat_map: # Mapping from name exists
                        idx = cat_map['to_idx'].get(value)
                        if idx is None:
                            raise ValueError(f"Invalid category name '{value}' for dimension '{dim_name}'")
                    else: # Assume value is already the index
                        if not isinstance(value, int) or not (0 <= value < num_categories):
                             raise ValueError(f"Invalid index {value} for dimension '{dim_name}'. Must be int in [0, {num_categories-1}]")
                        idx = value

                    idx_tensor = torch.tensor([idx], dtype=torch.long)
                    # Get embedding, remove batch dim, detach from graph
                    embedded_part = emb_layer(idx_tensor).squeeze(0).detach()
                    vector_parts.append(embedded_part)

                elif dim_type == 'continuous':
                    # Ensure float and wrap in tensor
                    vector_parts.append(torch.tensor([float(value)], dtype=torch.float))

        except KeyError as e:
            print(f"Error encoding vector: Missing key {e} in spec_dict or maps")
            return None
        except ValueError as e:
            print(f"Error encoding vector: {e}")
            return None

        # Concatenate all parts into a single flat vector
        return torch.cat(vector_parts, dim=0)


    def decode(self, spec_vector):
        """
        Interprets the specification vector back into a dictionary of values
        using Faiss ANN index for efficient nearest-neighbor lookup.

        Args:
            spec_vector (torch.Tensor): The flat specification vector (shape [total_spec_vector_dim]).

        Returns:
            dict: The interpreted specification dictionary, or None on error.
        """
        if spec_vector.shape[0] != self.total_spec_vector_dim:
            print(f"Error decoding vector: Input vector dimension {spec_vector.shape[0]} "
                  f"does not match expected dimension {self.total_spec_vector_dim}")
            return None

        interpreted_spec = {}
        # Faiss CPU index requires CPU data
        spec_vector_cpu = spec_vector.detach().cpu()

        try:
            for config in self.dimension_configs:
                dim_name = config['name']
                slice_info = self.dim_slices[dim_name]
                start, end = slice_info['start'], slice_info['end']
                vector_slice = spec_vector_cpu[start:end]

                if slice_info['type'] == 'categorical':
                    # --- Faiss ANN Search ---
                    faiss_index = self.ann_indices.get(dim_name)
                    if faiss_index is None:
                        raise RuntimeError(f"Faiss index not found for dimension '{dim_name}'")

                    # Prepare query vector for Faiss (needs to be float32 numpy array, shape [1, dim])
                    query_vector_np = vector_slice.numpy().astype('float32').reshape(1, -1)

                    # Search for the 1 nearest neighbor (k=1)
                    k = 1
                    distances, indices = faiss_index.search(query_vector_np, k)

                    # indices[0][0] contains the index of the closest embedding in the original matrix
                    closest_idx = indices[0][0]
                    # --- End Faiss Search ---

                    cat_map = self.category_maps.get(dim_name)
                    if cat_map: # Map index back to name if possible
                        interpreted_spec[dim_name] = cat_map['to_name'].get(closest_idx, f"Unknown Index {closest_idx}")
                    else: # Otherwise, the index is the value
                         interpreted_spec[dim_name] = int(closest_idx) # Ensure it's an int

                elif slice_info['type'] == 'continuous':
                    interpreted_spec[dim_name] = vector_slice.item() # Get scalar float value

        except KeyError as e:
             print(f"Error decoding vector: Internal configuration error for key {e}")
             return None
        except Exception as e:
            print(f"Error decoding vector: {e}")
            return None

        return interpreted_spec


# --- Example Usage ---

# 1. Define Configuration
dimension_config_list = [
    # Reduce num_categories for faster initialization in this example if needed
    {'name': 'source_id', 'type': 'categorical', 'num_categories': 1000, 'embedding_dim': 64},
    # {'name': 'source_id', 'type': 'categorical', 'num_categories': 10000, 'embedding_dim': 128}, # Smaller for demo
    {'name': 'sensor_type', 'type': 'categorical', 'categories': [f'sensor_{i}' for i in range(50)], 'embedding_dim': 16},
    {'name': 'voltage', 'type': 'continuous'},
    {'name': 'temperature', 'type': 'continuous'},
]

# 2. Instantiate the Coder (This will build Faiss indices)
print("--- Instantiating Coder ---")
coder = SpecificationCoder(dimension_config_list)
print("--- Coder Instantiated ---")


# 3. Define a desired specification
my_spec = {
    'source_id': 555, # Using index directly
    'sensor_type': 'sensor_15', # Using name
    'voltage': 0.95,
    'temperature': 18.0
}
print(f"\nOriginal Specification:\n{my_spec}")

# 4. Encode the specification into a vector
encoded_vector = coder.encode(my_spec)

if encoded_vector is not None:
    print(f"\nEncoded Vector (shape: {encoded_vector.shape}):")
    # print(f"{encoded_vector[:10]}...{encoded_vector[-10:]}") # Print snippet

    # 5. Decode the vector back into a specification using Faiss
    print("\nDecoding vector (using Faiss ANN)...")
    start_time = time.time()
    # Move vector to CPU if it's on GPU before decoding, Faiss index is on CPU here
    decoded_spec = coder.decode(encoded_vector.cpu())
    end_time = time.time()
    print(f"...Decoding finished in {end_time - start_time:.4f} seconds.") # Should be much faster now

    print(f"\nDecoded Specification:\n{decoded_spec}")

    # Verification
    is_match = (my_spec == decoded_spec) if decoded_spec else False
    print(f"\nDoes decoded spec match original? {is_match}")
    if not is_match and decoded_spec:
        # Compare elements individually for clarity if mismatch occurs
        for key in my_spec:
             if my_spec[key] != decoded_spec.get(key):
                  print(f"  Mismatch on '{key}': Original='{my_spec[key]}', Decoded='{decoded_spec.get(key)}'")
        print("Note: Exact float recovery might have minor precision differences.")
        print("Note: If using approximate ANN indices, the closest index might occasionally differ.")

else:
    print("Encoding failed.")

# Example of error handling
invalid_spec = {
    'source_id': 55555,
    'sensor_type': 'sensor_99', # Invalid category name
    'voltage': 0.95,
    'temperature': 18.0
}
print("\n--- Testing Invalid Spec for Encoding ---")
invalid_encoded = coder.encode(invalid_spec)
if invalid_encoded is None:
    print("Encoding correctly failed for invalid spec.")

invalid_vector = torch.randn(coder.total_spec_vector_dim - 5) # Wrong dimension
print("\n--- Testing Invalid Vector for Decoding ---")
invalid_decoded = coder.decode(invalid_vector)
if invalid_decoded is None:
    print("Decoding correctly failed for invalid vector.")

