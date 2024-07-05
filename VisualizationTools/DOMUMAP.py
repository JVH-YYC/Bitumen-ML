"""
UMAP and PCA clustering for DOM data, comparing FIG, compound classes
and EEM PARAFAC components
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import umap
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid

def visualize_important_features(df_path):
    df = pd.read_csv(df_path)
    categorical_columns = df.iloc[:, 1:5]
    data_columns = df.iloc[:, 5:]

    # Encode the Categorical Labels
    label_encoders = {}
    encoded_labels = pd.DataFrame()

    for col in categorical_columns.columns:
        le = LabelEncoder()
        encoded_labels[col] = le.fit_transform(categorical_columns[col])
        label_encoders[col] = le

    # Calculate Feature Importance using Mutual Information
    feature_importances = {}

    for col in encoded_labels.columns:
        mi = mutual_info_classif(data_columns, encoded_labels[col])
        feature_importances[col] = mi

    # Convert to DataFrame
    feature_importances_df = pd.DataFrame(feature_importances, index=data_columns.columns)

    # Calculate Feature Importance using Random Forest
    feature_importances_rf = {}

    for col in encoded_labels.columns:
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(data_columns, encoded_labels[col])
        feature_importances_rf[col] = rf.feature_importances_

    # Convert to DataFrame
    feature_importances_rf_df = pd.DataFrame(feature_importances_rf, index=data_columns.columns)

    # Plotting Feature Importances
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))

    for ax, col in zip(axes.flatten(), feature_importances_df.columns):
        feature_importances_df[col].sort_values(ascending=False).plot(kind='bar', ax=ax)
        ax.set_title(f'Feature Importance for {col}')

    plt.tight_layout()
    plt.show()

    return

def select_important_features(df,
                              categorical_label,
                              threshold,
                              control_label=None):
    # Separate the categorical columns and the data columns
    categorical_columns = df.iloc[:, :5]
    data_columns = df.iloc[:, 5:]
    
    # Encode the Categorical Labels
    label_encoders = {}
    encoded_labels = pd.DataFrame()
    
    for col in categorical_columns.columns:
        le = LabelEncoder()
        encoded_labels[col] = le.fit_transform(categorical_columns[col])
        label_encoders[col] = le

    # Check if the given categorical_label exists
    if categorical_label not in encoded_labels.columns:
        raise ValueError(f"The categorical label '{categorical_label}' is not found in the first four columns.")
    
    # Calculate Feature Importance using Mutual Information
    mi = mutual_info_classif(data_columns, encoded_labels[categorical_label])
    # Convert to DataFrame
    feature_importances_df = pd.DataFrame(mi, index=data_columns.columns, columns=['Importance'])
    # Determine the threshold to use
    if control_label is not None:
        if control_label not in data_columns.columns:
            raise ValueError(f"The control label '{control_label}' is not found in the data columns.")
        
        control_importance = feature_importances_df.loc[control_label, 'Importance']
        threshold = control_importance

    # Filter columns based on the threshold
    important_features = feature_importances_df[feature_importances_df['Importance'] > threshold]
    
    # Create new DataFrame with important features
    important_data_columns = data_columns[important_features.index]
    
    return feature_importances_df, important_data_columns

def find_optimal_clustering(df_path, categorical_label, control_label=None):
    # Read the DataFrame
    df = pd.read_csv(df_path)
        
    # Hyperparameter grid for UMAP
    param_grid = {
        'n_neighbors': [2, 4, 6, 8],
        'min_dist': [0.1, 0.2, 0.3],
        'threshold': [0.15, 0.25, 0.35,]
    }
    
    best_params = None
    best_score = -np.inf
    best_embed = None
    
    # Perform Grid Search
    for params in ParameterGrid(param_grid):
        raw_output, important_features_df = select_important_features(df, categorical_label, threshold=params['threshold'], control_label=control_label)
        reducer = umap.UMAP(n_neighbors=params['n_neighbors'], min_dist=params['min_dist'])
        embed = reducer.fit_transform(important_features_df)
        
        # Evaluate clustering (use a suitable metric, e.g., silhouette score, Davies-Bouldin index, etc.)
        # For simplicity, we use a placeholder function to evaluate the clustering
        # Here we assume a higher score indicates better clustering
        score = evaluate_clustering(embed, df[categorical_label])
        print('Score:', score, 'Params:', params)
        if score > best_score:
            best_score = score
            best_params = params
            best_embed = embed
            best_labels = important_features_df.columns

    # Plot the best result
    plt.scatter(best_embed[:, 0], best_embed[:, 1], c=[sns.color_palette()[x] for x in df[categorical_label]])
    plt.title(f"Best Clustering with n_neighbors={best_params['n_neighbors']} and min_dist={best_params['min_dist']} /n using labels {best_labels}")
    plt.show()
    
    print("Best Hyperparameters:", best_params)
    print("Best Score:", best_score)
    return best_params, best_embed

def evaluate_clustering(embed, labels):
    # Placeholder function to evaluate clustering
    # Implement a suitable clustering evaluation metric, such as silhouette score, Davies-Bouldin index, etc.
    from sklearn.metrics import silhouette_score
    score = silhouette_score(embed, labels)
    return score



# Example usage:
df_path = '/Users/jeffreyvanhumbeck/Documents/GitHub/Bitumen-ML/Clustering/DOM FIG first test.csv'
visualize_important_features(df_path)
best_params, best_embed = find_optimal_clustering(df_path, 'location', control_label='(8, 11, 1, 2, 0)')
print("Best Parameters:", best_params)
print("Embedding Shape:", best_embed.shape)

# # Example usage:
# df_path = '/Users/jeffreyvanhumbeck/Documents/GitHub/Bitumen-ML/Clustering/DOM FIG first test.csv'
# best_params, best_embed = find_optimal_clustering(df_path, 'land_use', 0.25, control_label=None)
# print("Best Parameters:", best_params)
# print("Embedding Shape:", best_embed.shape)
# # Example usage:
# df = pd.read_csv('/Users/jeffreyvanhumbeck/Documents/GitHub/Bitumen-ML/Clustering/DOM FIG first test.csv')
# visualize_important_features(df)
# raw_output, important_features_df = select_important_features(df, 'land_use', 0.25)

# reducer = umap.UMAP(n_neighbors=7,
#                     min_dist=0.1)
# embed = reducer.fit_transform(important_features_df)

# # Plotting attempt - location is the categorical label
# plt.scatter(embed[:, 0],
#             embed[:, 1],
#             c=[sns.color_palette()[x] for x in df['land_use']])
# print(embed)            
# plt.show()
