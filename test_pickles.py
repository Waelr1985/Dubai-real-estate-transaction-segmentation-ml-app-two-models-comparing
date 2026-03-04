import pickle
import traceback
import sys

print(f"Python Version: {sys.version}")
try:
    import numpy as np
    print(f"NumPy Version: {np.__version__}")
except Exception as e:
    print(f"NumPy import failed: {e}")

print("\n--- Testing PCA Pipeline ---")
try:
    with open('models/pca_segmentation_pipeline.pkl', 'rb') as f:
        pipe_pca = pickle.load(f)
    print("PCA Pipeline Loaded Successfully.")
except Exception as e:
    print(f"PCA Pipeline Load Failed: {e}")
    traceback.print_exc()

print("\n--- Testing UMAP Pipeline ---")
try:
    with open('models/segmentation_pipeline.pkl', 'rb') as f:
        pipe_umap = pickle.load(f)
    print("UMAP Pipeline Loaded Successfully.")
except Exception as e:
    print(f"UMAP Pipeline Load Failed: {e}")
    traceback.print_exc()
