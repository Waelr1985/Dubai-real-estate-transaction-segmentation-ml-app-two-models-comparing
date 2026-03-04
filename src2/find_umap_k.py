import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src2.config import DATA_DIR, MODELS_DIR
from src2.data_ingestion import load_data
from src2.data_validation import validate_data
from src2.data_preprocessing import apply_target_encoding

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_umap_k():
    logging.info("Loading latest dataset...")
    df = load_data('sample_transactions.csv')
    
    logging.info("Validating and cleaning missing data...")
    df = validate_data(df)
    
    logging.info("Applying Target Encoding...")
    df = apply_target_encoding(df)

    logging.info("Loading preprocessor and UMAP model...")
    preprocessor = joblib.load(os.path.join(MODELS_DIR, 'preprocessor.pkl'))
    umap_model = joblib.load(os.path.join(MODELS_DIR, 'umap_model.pkl'))

    # For K-evaluation, memory limits usually allow for ~30k-50k rows to compute Silhouette efficiently
    max_eval_samples = 40000
    if len(df) > max_eval_samples:
        df_eval = df.sample(n=max_eval_samples, random_state=42)
        logging.info(f"Sampled {max_eval_samples} rows for K-evaluation.")
    else:
        df_eval = df
    
    logging.info("Preprocessing data...")
    X_processed = preprocessor.transform(df_eval)
    
    logging.info("Applying UMAP projection...")
    X_reduced = umap_model.transform(X_processed)
    
    logging.info("Evaluating K from 2 to 10...")
    
    results = []
    
    for k in range(2, 11):
        logging.info(f"  -> Testing k={k}...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(X_reduced)
        
        sil = silhouette_score(X_reduced, labels)
        db = davies_bouldin_score(X_reduced, labels)
        ch = calinski_harabasz_score(X_reduced, labels)
        
        results.append({
            'k': k,
            'Silhouette Score': sil,
            'Davies-Bouldin Index': db,
            'Calinski-Harabasz Score': ch
        })
        print(f"k={k} | Silhouette: {sil:.4f} | DB: {db:.4f} | CH: {ch:.1f}")

    results_df = pd.DataFrame(results)
    
    print("\n--- Final K Evaluation Table (UMAP) ---")
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    evaluate_umap_k()
