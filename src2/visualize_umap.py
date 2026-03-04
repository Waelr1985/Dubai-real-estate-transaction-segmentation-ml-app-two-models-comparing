import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data
from src2.data_ingestion import load_data
from src2.data_validation import validate_data
from src2.data_preprocessing import apply_target_encoding

print("Loading data...")
df = load_data(sample_frac=0.1)
df_original = df.copy() # Keep for profiling later
if 'transaction_id' in df.columns:
    df = df.drop(columns=['transaction_id'])
df_clean = validate_data(df)

# Load pipeline
print("Loading model pipeline...")
pipeline_path = 'models/segmentation_pipeline.pkl'
with open(pipeline_path, 'rb') as f:
    pipeline = pickle.load(f)

preprocessor = pipeline.named_steps['preprocessor']
reducer = pipeline.named_steps['umap']
kmeans = pipeline.named_steps['kmeans']

# Preprocess and predict
print("Applying target encoding...")
df_encoded = apply_target_encoding(df_clean)

print("Transforming and predicting...")
X_prep = preprocessor.transform(df_encoded)
X_umap = reducer.transform(X_prep) # This is 5D
labels = kmeans.predict(X_umap)

# 1. VISUALIZATION
print("Generating 2D Visualization...")
plt.figure(figsize=(12, 8))
# We just take the first two UMAP components for a 2D visual, similar to taking PC1 and PC2
sns.scatterplot(
    x=X_umap[:, 0], 
    y=X_umap[:, 1], 
    hue=labels, 
    palette='viridis', 
    s=5, 
    alpha=0.6
)
plt.title('UMAP Non-Linear Clustering (2D Representation)')
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

vis_path = 'models/umap_clusters_2d.png'
plt.savefig(vis_path, dpi=300)
print(f"Visualization saved to {vis_path}")

# 2. PROFILING (What do these groups mean?)
print("Generating Cluster Profiles...")
# Add labels back to the original subset data to see the raw values
df_original['Cluster'] = labels

# Numeric profiles (medians)
numeric_cols = df_original.select_dtypes(include=[np.number]).columns.drop('Cluster', errors='ignore')
numeric_profile = df_original.groupby('Cluster')[numeric_cols].median()

# Categorical profiles (modes)
cat_cols = ['property_usage_en', 'property_type_en', 'area_name_en']
cat_profile = df_original.groupby('Cluster')[cat_cols].agg(lambda x: x.value_counts().index[0])

# Combine
profile = pd.concat([numeric_profile, cat_profile], axis=1)

profile_path = 'models/umap_cluster_profile.csv'
profile.to_csv(profile_path)
print(f"Cluster profiles saved to {profile_path}")

print("Done.")
