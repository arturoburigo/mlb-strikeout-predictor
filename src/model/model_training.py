import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

def create_features_from_cleaned_data(pitchers_df):
    """
    Create features from the cleaned pitcher data format.
    
    Args:
        pitchers_df (DataFrame): DataFrame with columns: Pitcher, Opp, IP, SO, WPA, BF, Pit, Str, StS, StL, opp_so_avg
        
    Returns:
        DataFrame: DataFrame with engineered features
    """
    # Create a copy to avoid modifying the original
    df = pitchers_df.copy()
    
    # Basic rate statistics
    df['SO_per_IP'] = df['SO'] / df['IP']
    df['BB_per_IP'] = (df['BF'] - df['Str']) / df['IP']  # Estimate BB from BF and Str
    df['Str_rate'] = df['Str'] / df['Pit']
    df['StS_rate'] = df['StS'] / df['Pit']
    df['StL_rate'] = df['StL'] / df['Pit']
    
    # Efficiency metrics
    df['Pit_per_IP'] = df['Pit'] / df['IP']
    df['BF_per_IP'] = df['BF'] / df['IP']
    
    # Strikeout efficiency
    df['SO_per_BF'] = df['SO'] / df['BF']
    df['SO_per_Pit'] = df['SO'] / df['Pit']
    
    # Handle infinity and NaN values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        df[col] = df[col].replace([np.inf, -np.inf], 0)
        df[col] = df[col].fillna(0)
    
    # Clip extreme values
    df['SO_per_IP'] = df['SO_per_IP'].clip(0, 20)
    df['BB_per_IP'] = df['BB_per_IP'].clip(0, 10)
    df['Str_rate'] = df['Str_rate'].clip(0, 1)
    df['StS_rate'] = df['StS_rate'].clip(0, 1)
    df['StL_rate'] = df['StL_rate'].clip(0, 1)
    df['Pit_per_IP'] = df['Pit_per_IP'].clip(0, 50)
    df['BF_per_IP'] = df['BF_per_IP'].clip(0, 10)
    df['SO_per_BF'] = df['SO_per_BF'].clip(0, 1)
    df['SO_per_Pit'] = df['SO_per_Pit'].clip(0, 1)
    
    # Create rolling averages for each pitcher
    df_sorted = df.sort_values(['Pitcher', 'Opp']).reset_index(drop=True)
    
    # Calculate rolling averages for each pitcher
    df_sorted['SO_rolling_3'] = df_sorted.groupby('Pitcher')['SO'].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
    df_sorted['SO_rolling_5'] = df_sorted.groupby('Pitcher')['SO'].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
    df_sorted['IP_rolling_3'] = df_sorted.groupby('Pitcher')['IP'].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
    df_sorted['IP_rolling_5'] = df_sorted.groupby('Pitcher')['IP'].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
    
    # Calculate pitcher-specific averages
    pitcher_stats = df_sorted.groupby('Pitcher').agg({
        'SO': 'mean',
        'IP': 'mean',
        'Pit': 'mean',
        'Str': 'mean',
        'StS': 'mean',
        'StL': 'mean',
        'BF': 'mean',
        'WPA': 'mean'
    }).reset_index()
    
    pitcher_stats.columns = ['Pitcher'] + [f'avg_{col}' for col in pitcher_stats.columns if col != 'Pitcher']
    
    # Merge pitcher averages back to main dataframe
    df_with_stats = df_sorted.merge(pitcher_stats, on='Pitcher', how='left')
    
    # Create additional features
    df_with_stats['SO_vs_avg'] = df_with_stats['SO'] - df_with_stats['avg_SO']
    df_with_stats['IP_vs_avg'] = df_with_stats['IP'] - df_with_stats['avg_IP']
    df_with_stats['Pit_vs_avg'] = df_with_stats['Pit'] - df_with_stats['avg_Pit']
    
    # Normalize opponent strikeout average
    df_with_stats['opp_so_avg_norm'] = (df_with_stats['opp_so_avg'] - df_with_stats['opp_so_avg'].mean()) / df_with_stats['opp_so_avg'].std()
    
    return df_with_stats

def train_model(pitchers_df, k_percentage_df=None):
    """
    Train and evaluate machine learning models to predict pitcher strikeouts.
    
    Args:
        pitchers_df (DataFrame): DataFrame containing pitcher data in cleaned format
        k_percentage_df (DataFrame, optional): Team strikeout percentages (not used in this version)
        
    Returns:
        tuple: (best_model, results_dict) containing the best trained model and evaluation metrics
    """
    # Create features from the cleaned data
    print("Creating features from cleaned pitcher data...")
    engineered_data = create_features_from_cleaned_data(pitchers_df)
    
    # Check if we have data to work with
    if engineered_data is None or engineered_data.empty:
        raise ValueError("No data available after feature engineering. Check your input data.")
    
    # Define features to use for training
    feature_columns = [
        'IP', 'BF', 'Pit', 'Str', 'StS', 'StL', 'WPA',
        'SO_per_IP', 'BB_per_IP', 'Str_rate', 'StS_rate', 'StL_rate',
        'Pit_per_IP', 'BF_per_IP', 'SO_per_BF', 'SO_per_Pit',
        'SO_rolling_3', 'SO_rolling_5', 'IP_rolling_3', 'IP_rolling_5',
        'avg_SO', 'avg_IP', 'avg_Pit', 'avg_Str', 'avg_StS', 'avg_StL', 'avg_BF', 'avg_WPA',
        'SO_vs_avg', 'IP_vs_avg', 'Pit_vs_avg',
        'opp_so_avg', 'opp_so_avg_norm'
    ]
    
    # Ensure all required features exist
    for feature in feature_columns:
        if feature not in engineered_data.columns:
            engineered_data[feature] = 0
    
    # Add debug print to check data shape
    print(f"Training data shape: {engineered_data.shape}")
    print(f"Number of unique pitchers: {engineered_data['Pitcher'].nunique()}")
    
    # Prepare features and target
    X = engineered_data[feature_columns].fillna(0)
    y = engineered_data['SO']
    
    # Add verification that we have data
    if len(X) == 0:
        raise ValueError(f"No data available for training. Engineered data shape: {engineered_data.shape}")
    
    # Print data statistics for debugging
    print("\nFeature statistics:")
    for feature in feature_columns[:10]:  # Show first 10 features
        print(f"{feature}: min={X[feature].min():.2f}, max={X[feature].max():.2f}, mean={X[feature].mean():.2f}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'RandomForest': RandomForestRegressor(random_state=42, n_estimators=100),
        'XGBoost': XGBRegressor(random_state=42, n_estimators=100),
        'LightGBM': LGBMRegressor(random_state=42, num_leaves=31, min_data_in_leaf=1, max_depth=-1, verbose=-1, n_estimators=100),
        'GradientBoosting': GradientBoostingRegressor(random_state=42, n_estimators=100)
    }

    best_model = None
    best_score = -np.inf
    best_model_name = ""
    results = {}

    for name, model in models.items():
        try:
            pipeline = make_pipeline(RobustScaler(), model)
            
            scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')
            avg_score = np.mean(scores)
            
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            test_r2 = r2_score(y_test, y_pred)
            test_mae = mean_absolute_error(y_test, y_pred)
            
            results[name] = {
                'CV_R2': avg_score,
                'Test_R2': test_r2,
                'Test_MAE': test_mae
            }
            
            print(f"{name:15} | CV R2: {avg_score:.4f} | Test R2: {test_r2:.4f} | Test MAE: {test_mae:.4f}")
            
            if avg_score > best_score:
                best_score = avg_score
                best_model = pipeline
                best_model_name = name
        except Exception as e:
            print(f"Error training {name} model: {e}")
            continue

    if best_model is None:
        raise ValueError("All models failed to train. Check your data for issues.")
        
    print(f"\nBest model: {best_model_name} with R2 score: {best_score:.4f}")
    return best_model, results

def main():
    """
    Main function to train and evaluate machine learning models for predicting pitcher strikeouts.
    Loads the cleaned pitcher data and trains models.
    """
    try:
        print("Loading cleaned pitcher data...")
        
        # Load the cleaned pitcher data
        pitchers_df = pd.read_csv('pitchers_data_with_opp_so_cleaned.csv')
        print(f"Loaded data for {pitchers_df['Pitcher'].nunique()} pitchers")
        print(f"Total records: {len(pitchers_df)}")
        
        # Train the models
        print("\nTraining machine learning models to predict pitcher strikeouts...")
        best_model, results = train_model(pitchers_df)
        
        # Print detailed results
        print("\nDetailed model evaluation results:")
        print("-" * 65)
        print(f"{'Model':<15} | {'CV R²':<10} | {'Test R²':<10} | {'Test MAE':<10}")
        print("-" * 65)
        
        for model_name, metrics in results.items():
            print(f"{model_name:<15} | {metrics['CV_R2']:<10.4f} | {metrics['Test_R2']:<10.4f} | {metrics['Test_MAE']:<10.4f}")
        
        # Get feature importances if available (for tree-based models)
        print("\nFeature importances (if available):")
        try:
            model = best_model.steps[-1][1]  # Get the model from the pipeline
            
            if hasattr(model, 'feature_importances_'):
                # Get the feature names from the training data
                feature_names = [
                    'IP', 'BF', 'Pit', 'Str', 'StS', 'StL', 'WPA',
                    'SO_per_IP', 'BB_per_IP', 'Str_rate', 'StS_rate', 'StL_rate',
                    'Pit_per_IP', 'BF_per_IP', 'SO_per_BF', 'SO_per_Pit',
                    'SO_rolling_3', 'SO_rolling_5', 'IP_rolling_3', 'IP_rolling_5',
                    'avg_SO', 'avg_IP', 'avg_Pit', 'avg_Str', 'avg_StS', 'avg_StL', 'avg_BF', 'avg_WPA',
                    'SO_vs_avg', 'IP_vs_avg', 'Pit_vs_avg',
                    'opp_so_avg', 'opp_so_avg_norm'
                ]
                
                # Create a DataFrame of feature importances
                importances = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model.feature_importances_
                })
                
                # Sort by importance
                importances = importances.sort_values('Importance', ascending=False)
                
                # Display top 15 features
                print(importances.head(15))
            else:
                print("Feature importances not available for this model type")
        
        except Exception as e:
            print(f"Could not extract feature importances: {e}")
        
        print("\nModel training completed successfully!")
        return best_model, results
        
    except Exception as e:
        print(f"Error occurred during model training: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    main()