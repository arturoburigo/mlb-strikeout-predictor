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
import sys
import os
import pickle
import joblib
from scipy.stats import norm

# Add the parent directory to the path to import feature engineering
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feature_engineering import engineer_features

warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

def save_lightgbm_model(pipeline, feature_columns, model_path='models/lightgbm_model.pkl'):
    """
    Save the LightGBM model pipeline and feature information.
    
    Args:
        pipeline: Trained LightGBM pipeline
        feature_columns: List of feature column names
        model_path: Path where to save the model
    """
    try:
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save the pipeline
        joblib.dump(pipeline, model_path)
        
        # Save feature columns for later use
        feature_info_path = model_path.replace('.pkl', '_features.pkl')
        with open(feature_info_path, 'wb') as f:
            pickle.dump(feature_columns, f)
        
        print(f"\n=== LIGHTGBM MODEL SAVED ===")
        print(f"Model saved to: {model_path}")
        print(f"Feature info saved to: {feature_info_path}")
        print(f"Number of features: {len(feature_columns)}")
        
    except Exception as e:
        print(f"Error saving LightGBM model: {e}")

def load_lightgbm_model(model_path='models/lightgbm_model.pkl'):
    """
    Load the saved LightGBM model and feature information.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        tuple: (pipeline, feature_columns) or (None, None) if loading fails
    """
    try:
        # Load the pipeline
        pipeline = joblib.load(model_path)
        
        # Load feature columns
        feature_info_path = model_path.replace('.pkl', '_features.pkl')
        with open(feature_info_path, 'rb') as f:
            feature_columns = pickle.load(f)
        
        print(f"LightGBM model loaded successfully from: {model_path}")
        print(f"Number of features: {len(feature_columns)}")
        
        return pipeline, feature_columns
        
    except Exception as e:
        print(f"Error loading LightGBM model: {e}")
        return None, None

def load_engineered_features(csv_path='pitchers_data_engineered.csv', use_cached=True):
    """
    Load engineered features from the pre-processed CSV file.
    
    Args:
        csv_path (str): Path to the engineered features CSV file
        use_cached (bool): Whether to use cached engineered features if available
        
    Returns:
        DataFrame: Engineered features dataframe
    """
    try:
        # If relative path doesn't work, try from parent directories
        if not os.path.exists(csv_path):
            # Try different possible locations
            possible_paths = [
                csv_path,
                f'../{csv_path}',
                f'../../{csv_path}',
                f'../../../{csv_path}'
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    csv_path = path
                    break
            else:
                raise FileNotFoundError(f"Could not find {csv_path} in any expected location")
        
        print(f"Loading engineered features from: {csv_path}")
        data = pd.read_csv(csv_path)
        
        # Convert Season to int if it's not already
        if 'Season' in data.columns:
            data['Season'] = pd.to_numeric(data['Season'], errors='coerce')
        
        print(f"Loaded {len(data)} records with {len(data.columns)} features")
        print(f"Season range: {data['Season'].min()} - {data['Season'].max()}")
        
        return data
        
    except Exception as e:
        print(f"Error loading engineered features: {e}")
        return None

def prepare_features_for_prediction(betting_data_path, engineered_data):
    """
    Prepare features for 2025 season predictions using betting data.
    
    Args:
        betting_data_path (str): Path to betting data CSV
        engineered_data (DataFrame): Historical engineered data
        
    Returns:
        DataFrame: Features ready for 2025 predictions
    """
    try:
        # Load betting data
        betting_data = pd.read_csv(betting_data_path)
        print(f"Loaded betting data with {len(betting_data)} pitchers")
        
        # Get unique pitcher names from betting data
        betting_pitchers = betting_data['Player'].unique()
        print(f"Unique pitchers in betting data: {len(betting_pitchers)}")
        
        # Get latest data for each pitcher (most recent season available)
        prediction_features = []
        
        for pitcher in betting_pitchers:
            # Find this pitcher in engineered data
            pitcher_data = engineered_data[engineered_data['Pitcher_Name'] == pitcher]
            
            if len(pitcher_data) > 0:
                # Get the most recent data for this pitcher
                latest_data = pitcher_data.sort_values('Date', ascending=False).iloc[0]
                
                # Create feature row for prediction
                feature_row = latest_data.copy()
                feature_row['Season'] = 2025  # Set for 2025 predictions
                feature_row['Date'] = '2025-01-01'  # Placeholder date
                
                # Get opponent from betting data
                betting_row = betting_data[betting_data['Player'] == pitcher].iloc[0]
                feature_row['Opp'] = betting_row['Opponent']
                feature_row['Home'] = betting_row['Home']
                
                prediction_features.append(feature_row)
            else:
                print(f"Warning: No historical data found for {pitcher}")
        
        if prediction_features:
            prediction_df = pd.DataFrame(prediction_features)
            print(f"Prepared features for {len(prediction_df)} pitchers")
            return prediction_df
        else:
            print("No prediction features could be prepared")
            return None
            
    except Exception as e:
        print(f"Error preparing features for prediction: {e}")
        return None

def train_model(csv_path='../../pitchers_data_engineered.csv', use_cached_features=True):
    """
    Train and evaluate machine learning models to predict pitcher strikeouts using engineered features.
    
    Args:
        csv_path (str): Path to the engineered features CSV file
        use_cached_features (bool): Whether to use cached engineered features
        
    Returns:
        tuple: (best_model, results_dict) containing the best trained model and evaluation metrics
    """
    # Load engineered features
    print("Loading engineered features...")
    engineered_data = load_engineered_features(csv_path, use_cached_features)
    
    # Check if we have data to work with
    if engineered_data is None or engineered_data.empty:
        raise ValueError("No data available. Check your input data.")
    
    # Filter for recent seasons (2023-2024) for training
    recent_data = engineered_data[engineered_data['Season'].isin([2023, 2024])].copy()
    
    if len(recent_data) == 0:
        raise ValueError("No recent data (2023-2024) available for training.")
    
    print(f"Using {len(recent_data)} records from 2023-2024 for training")
    
    # Define features to use for training (exclude identifier columns and target)
    exclude_columns = ['Season', 'Pitcher_ID', 'Team_x', 'Pitcher_Name', 'Date', 'SO']
    feature_columns = [col for col in recent_data.columns if col not in exclude_columns]
    
    # Add debug print to check data shape
    print(f"Training data shape: {recent_data.shape}")
    print(f"Number of unique pitchers: {recent_data['Pitcher_Name'].nunique()}")
    print(f"Number of features: {len(feature_columns)}")
    
    # Prepare features and target
    X = recent_data[feature_columns].copy()
    
    # Convert categorical features to numeric (handle all object columns)
    for col in X.columns:
        if X[col].dtype == 'object':
            # Convert to numeric, replacing non-numeric with 0
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
            print(f"Converted categorical column '{col}' to numeric")
    
    X = X.fillna(0)
    y = recent_data['SO']
    
    # Add verification that we have data
    if len(X) == 0:
        raise ValueError(f"No data available for training. Engineered data shape: {recent_data.shape}")
    
    # Print feature statistics for debugging
    print("\n=== FEATURE STATISTICS ===")
    print(f"Features used: {len(feature_columns)}")
    print("Sample of features:")
    for i, feature in enumerate(feature_columns[:10]):  # Show first 10 features
        try:
            if X[feature].dtype in ['int64', 'float64']:
                print(f"  {i+1:2d}. {feature:<25} | min={X[feature].min():.3f}, max={X[feature].max():.3f}, mean={X[feature].mean():.3f}")
            else:
                print(f"  {i+1:2d}. {feature:<25} | dtype={X[feature].dtype}, unique_values={X[feature].nunique()}")
        except Exception as e:
            print(f"  {i+1:2d}. {feature:<25} | Error: {str(e)}")
    
    if len(feature_columns) > 10:
        print(f"  ... and {len(feature_columns) - 10} more features")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    # Define models
    models = {
        'RandomForest': RandomForestRegressor(random_state=42, n_estimators=100),
        'XGBoost': XGBRegressor(random_state=42, n_estimators=100),
        'LightGBM': LGBMRegressor(random_state=42, num_leaves=31, min_data_in_leaf=1, 
                                  max_depth=-1, verbose=-1, n_estimators=100),
        'GradientBoosting': GradientBoostingRegressor(random_state=42, n_estimators=100)
    }
    
    # Store LightGBM model separately for saving
    lightgbm_pipeline = None

    best_model = None
    best_score = -np.inf
    best_model_name = ""
    results = {}

    print("\n=== MODEL TRAINING RESULTS ===")
    print("-" * 70)
    print(f"{'Model':<15} | {'CV R²':<10} | {'Test R²':<10} | {'Test MAE':<10}")
    print("-" * 70)

    for name, model in models.items():
        try:
            # Create pipeline with scaling
            pipeline = make_pipeline(RobustScaler(), model)
            
            # Cross-validation
            scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')
            avg_score = np.mean(scores)
            
            # Train and evaluate on test set
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            test_r2 = r2_score(y_test, y_pred)
            test_mae = mean_absolute_error(y_test, y_pred)
            
            # Store results
            results[name] = {
                'CV_R2': avg_score,
                'Test_R2': test_r2,
                'Test_MAE': test_mae
            }
            
            print(f"{name:<15} | {avg_score:<10.4f} | {test_r2:<10.4f} | {test_mae:<10.4f}")
            
            # Store LightGBM model separately
            if name == 'LightGBM':
                lightgbm_pipeline = pipeline
            
            # Track best model
            if avg_score > best_score:
                best_score = avg_score
                best_model = pipeline
                best_model_name = name
                
        except Exception as e:
            print(f"{name:<15} | ERROR: {str(e)}")
            continue

    if best_model is None:
        raise ValueError("All models failed to train. Check your data for issues.")
    
    print("-" * 70)
    print(f"Best model: {best_model_name} with CV R² score: {best_score:.4f}")
    
    # Get feature importances if available
    print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
    try:
        model = best_model.steps[-1][1]  # Get the model from the pipeline
        
        if hasattr(model, 'feature_importances_'):
            # Create a DataFrame of feature importances
            importances = pd.DataFrame({
                'Feature': feature_columns,
                'Importance': model.feature_importances_
            })
            
            # Sort by importance
            importances = importances.sort_values('Importance', ascending=False)
            
            # Display top 15 features
            print("Top 15 most important features:")
            print("-" * 50)
            for i, (_, row) in enumerate(importances.head(15).iterrows()):
                print(f"{i+1:2d}. {row['Feature']:<25} | {row['Importance']:.4f}")
        else:
            print("Feature importances not available for this model type")
    
    except Exception as e:
        print(f"Could not extract feature importances: {e}")
    
    # Save LightGBM model if it was trained successfully
    if lightgbm_pipeline is not None:
        save_lightgbm_model(lightgbm_pipeline, feature_columns)
    
    return best_model, results

def make_2025_predictions(betting_data_path=None, 
                         engineered_data_path='../../pitchers_data_engineered.csv'):
    """
    Make predictions for 2025 season using betting data and trained model.
    Adds probability/confidence columns for over/under using model MAE as sigma.
    """
    try:
        print("=== MAKING 2025 SEASON PREDICTIONS ===")
        
        # Generate betting data path based on current date if not provided
        if betting_data_path is None:
            current_date = pd.Timestamp.now().strftime('%Y-%m-%d')
            betting_data_path = f'../../betting_data_{current_date}.csv'
            print(f"Using betting data path: {betting_data_path}")
        
        # Check if betting data file exists
        if not os.path.exists(betting_data_path):
            print(f"Warning: Betting data file not found at {betting_data_path}")
            print("Please ensure the betting data file exists for the current date")
            return None
        
        # Load engineered data
        engineered_data = load_engineered_features(engineered_data_path)
        if engineered_data is None:
            raise ValueError("Could not load engineered data")
        
        # Load trained model
        pipeline, feature_columns = load_lightgbm_model()
        if pipeline is None:
            raise ValueError("Could not load trained model")
        
        # Prepare features for 2025 predictions
        prediction_features = prepare_features_for_prediction(betting_data_path, engineered_data)
        if prediction_features is None:
            raise ValueError("Could not prepare prediction features")
        
        # Prepare features for model input
        X_pred = prediction_features[feature_columns].copy()
        for col in X_pred.columns:
            if X_pred[col].dtype == 'object':
                X_pred[col] = pd.to_numeric(X_pred[col], errors='coerce').fillna(0)
        X_pred = X_pred.fillna(0)
        
        # Make predictions
        predictions = pipeline.predict(X_pred)
        
        # Create results dataframe
        results = prediction_features[['Pitcher_Name', 'Opp', 'Home']].copy()
        results['Predicted_SO'] = predictions.round(2)
        
        # Add betting data info
        betting_data = pd.read_csv(betting_data_path)
        results = results.merge(betting_data[['Player', 'Over Line', 'Under Line']], 
                              left_on='Pitcher_Name', right_on='Player', how='left')
        
        # Add confidence indicators (margin)
        results['Over_Confidence'] = (results['Predicted_SO'] - results['Over Line']).round(2)
        results['Under_Confidence'] = (results['Under Line'] - results['Predicted_SO']).round(2)
        results['Max_Confidence'] = results[['Over_Confidence', 'Under_Confidence']].abs().max(axis=1)
        
        # --- NEW: Add probability/confidence columns using model MAE as sigma ---
        # Use the MAE from the last training (or a default if not available)
        # We'll use the LightGBM MAE from the last run, or fallback to 0.65
        try:
            # Load last MAE from file if available
            mae = None
            if os.path.exists('last_mae.txt'):
                with open('last_mae.txt', 'r') as f:
                    mae = float(f.read().strip())
            if mae is None or mae <= 0:
                mae = 0.65
        except Exception:
            mae = 0.65
        sigma = mae
        # Over probability: P(pred > line) = 1 - CDF(line, pred, sigma)
        results['Over_Prob'] = 1 - norm.cdf(results['Over Line'], loc=results['Predicted_SO'], scale=sigma)
        results['Under_Prob'] = norm.cdf(results['Under Line'], loc=results['Predicted_SO'], scale=sigma)
        results['Max_Prob'] = results[['Over_Prob', 'Under_Prob']].max(axis=1)
        
        # Sort by prediction confidence (probability)
        results = results.sort_values('Max_Prob', ascending=False)
        
        print(f"\n=== 2025 PREDICTIONS COMPLETE ===")
        print(f"Made predictions for {len(results)} pitchers")
        
        # Save predictions
        output_path = f"predicted_2025_{pd.Timestamp.now().strftime('%Y-%m-%d')}.csv"
        results.to_csv(output_path, index=False)
        print(f"Predictions saved to: {output_path}")
        
        # Print summary with probabilities
        print("\n=== PREDICTION SUMMARY (WITH PROBABILITIES) ===")
        print(f"Total predictions made: {len(results)}")
        print(f"Average predicted SO: {results['Predicted_SO'].mean():.2f}")
        print(f"Highest predicted SO: {results['Predicted_SO'].max():.2f}")
        print(f"Lowest predicted SO: {results['Predicted_SO'].min():.2f}")
        print("\n=== TOP 5 MOST CONFIDENT PREDICTIONS (BY PROBABILITY) ===")
        top_5 = results.head(5)
        for _, row in top_5.iterrows():
            direction = "OVER" if row['Over_Prob'] > row['Under_Prob'] else "UNDER"
            prob = row['Over_Prob'] if direction == "OVER" else row['Under_Prob']
            print(f"{row['Pitcher_Name']:<20} | {row['Predicted_SO']:>5.2f} | {direction:<5} | Prob: {prob:>6.2%}")
        
        return results
        
    except Exception as e:
        print(f"Error making 2025 predictions: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """
    Main function to train and evaluate machine learning models for predicting pitcher strikeouts.
    Uses engineered features and prepares for 2025 season predictions.
    """
    try:
        print("=== PITCHER STRIKEOUT PREDICTION MODEL ===")
        print("Training on engineered features for 2025 season predictions")
        print("=" * 60)
        
        # Train the models
        best_model, results = train_model()
        
        # Print final summary
        print("\n=== FINAL SUMMARY ===")
        print(f"Best performing model ready for 2025 predictions.")
        print(f"Features used: Engineered features from historical data")
        print(f"Training data: 2023-2024 seasons")
        
        print("\nModel training completed successfully!")
        return best_model, results
        
    except Exception as e:
        print(f"Error occurred during model training: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_saved_lightgbm_model(model_path='models/lightgbm_model.pkl'):
    """
    Test the saved LightGBM model with a simple prediction.
    
    Args:
        model_path: Path to the saved model
    """
    try:
        pipeline, feature_columns = load_lightgbm_model(model_path)
        
        if pipeline is None:
            print("Could not load LightGBM model for testing")
            return
        
        # Create dummy data with the same features
        dummy_data = pd.DataFrame(np.zeros((1, len(feature_columns))), columns=feature_columns)
        
        # Make prediction
        prediction = pipeline.predict(dummy_data)
        
        print(f"\n=== LIGHTGBM MODEL TEST ===")
        print(f"Model loaded successfully")
        print(f"Sample prediction (all zeros input): {prediction[0]:.4f}")
        print("Model is ready for use!")
        
    except Exception as e:
        print(f"Error testing LightGBM model: {e}")

if __name__ == "__main__":
    main()