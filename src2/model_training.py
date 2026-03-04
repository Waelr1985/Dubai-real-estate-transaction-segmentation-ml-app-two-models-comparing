import logging
import pickle
import os
import json
import numpy as np
from sklearn.cluster import KMeans
import umap
from sklearn.pipeline import Pipeline

from src2.config import MODELS_DIR
from src2.data_ingestion import load_data
from src2.data_validation import validate_data
from src2.data_preprocessing import get_preprocessor, apply_target_encoding

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# UMAP Components
UMAP_N_COMPONENTS = 5

def train_model(n_clusters: int = 5, sample_frac: float = 0.1): # Max 10% recommended for UMAP memory
    logging.info("--- Starting Model Training Pipeline (Strategy E: UMAP) ---")
    
    df = load_data(sample_frac=sample_frac)
    
    # Optional: We drop the ID column as it should not be clustered
    if 'transaction_id' in df.columns:
        transaction_ids = df['transaction_id']
        df = df.drop(columns=['transaction_id'])
    
    df_clean = validate_data(df)
    
    # Calculate and save baseline stats for drift monitoring
    from src2.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES
    baseline_stats = {'numeric_medians': {}, 'categorical_modes': {}}
    
    for col in NUMERIC_FEATURES:
        if col in df_clean.columns:
            baseline_stats['numeric_medians'][col] = float(df_clean[col].median())

    for col in CATEGORICAL_FEATURES:
        if col in df_clean.columns:
            baseline_stats['categorical_modes'][col] = str(df_clean[col].mode()[0])

    stats_path = os.path.join(MODELS_DIR, 'baseline_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(baseline_stats, f, indent=4)
    logging.info(f"Baseline statistics for drift monitoring saved to {stats_path}")

    # --- Strategy D: Target-encode high-cardinality features ---
    logging.info("Applying target encoding for area_name_en...")
    df_clean = apply_target_encoding(df_clean)
    
    # Save the target encoding mapping for use during prediction
    if 'area_name_en' in df_clean.columns and 'actual_worth' in df_clean.columns:
        area_medians = df_clean.groupby('area_name_en')['actual_worth'].median().to_dict()
        te_path = os.path.join(MODELS_DIR, 'target_encoding_mappings.json')
        # Convert numpy types to native Python for JSON serialization
        area_medians_serializable = {k: float(v) for k, v in area_medians.items()}
        with open(te_path, 'w') as f:
            json.dump({'area_name_en': area_medians_serializable}, f, indent=4)
        logging.info(f"Target encoding mappings saved to {te_path}")

    preprocessor = get_preprocessor(df_clean)
    
    logging.info(f"Training KMeans with {n_clusters} clusters (Strategy E: TE + UMAP)...")
    
    # Preprocess the data
    X_preprocessed = preprocessor.fit_transform(df_clean)
    logging.info(f"Feature space after preprocessing: {X_preprocessed.shape[1]} columns")
    
    # --- Strategy E: Apply UMAP for non-linear dimensionality reduction ---
    logging.info("Fitting UMAP... This may take a few minutes on large datasets.")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=UMAP_N_COMPONENTS, random_state=42)
    X_reduced = reducer.fit_transform(X_preprocessed)
    logging.info(f"Feature space after UMAP: {X_reduced.shape[1]} components")
    
    # Train KMeans on the reduced feature space
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X_reduced)
    
    # Save all components separately for maximum flexibility
    # 1. Save the preprocessor
    preprocessor_path = os.path.join(MODELS_DIR, 'preprocessor.pkl')
    with open(preprocessor_path, 'wb') as f:
        pickle.dump(preprocessor, f)
    logging.info(f"Preprocessor saved to {preprocessor_path}")
    
    # 2. Save the UMAP model
    umap_path = os.path.join(MODELS_DIR, 'umap_model.pkl')
    with open(umap_path, 'wb') as f:
        pickle.dump(reducer, f)
    logging.info(f"UMAP model saved to {umap_path}")
    
    # 3. Save the KMeans model
    kmeans_path = os.path.join(MODELS_DIR, 'kmeans_model.pkl')
    with open(kmeans_path, 'wb') as f:
        pickle.dump(kmeans, f)
    logging.info(f"KMeans model saved to {kmeans_path}")
    
    # 4. Also save a combined pipeline for backward compatibility
    # Note: This pipeline expects pre-target-encoded data
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('umap', reducer),
        ('kmeans', kmeans)
    ])
    pipeline_path = os.path.join(MODELS_DIR, 'segmentation_pipeline.pkl')
    with open(pipeline_path, 'wb') as f:
        pickle.dump(full_pipeline, f)
    logging.info(f"Full pipeline saved to {pipeline_path}")
    
    logging.info(f"--- Training Complete ---")
    logging.info(f"  Clusters: {n_clusters}")
    logging.info(f"  UMAP components: {X_reduced.shape[1]}")
    logging.info(f"  Original features: {X_preprocessed.shape[1]} -> Reduced: {X_reduced.shape[1]}")
    
    return full_pipeline, df_clean

if __name__ == "__main__":
    train_model(n_clusters=5, sample_frac=0.1) # Constrain to 10% for UMAP memory limit
