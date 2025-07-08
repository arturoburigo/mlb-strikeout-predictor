import numpy as np
import pandas as pd
import os
import re
from datetime import datetime
import warnings
import sys
import joblib
import pickle

# Add the parent directory to the path to import feature engineering
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feature_engineering import calculate_simple_performance

warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

def check_model_files_exist(model_path='models/lightgbm_model.pkl'):
    """
    Check if the model files exist.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        bool: True if both model and feature files exist
    """
    model_exists = os.path.exists(model_path)
    feature_info_path = model_path.replace('.pkl', '_features.pkl')
    features_exist = os.path.exists(feature_info_path)
    
    if not model_exists:
        print(f"✗ Model file not found: {model_path}")
    if not features_exist:
        print(f"✗ Feature info file not found: {feature_info_path}")
    
    return model_exists and features_exist

def load_trained_model(model_path='models/lightgbm_model.pkl'):
    """
    Load the pre-trained LightGBM model and feature information.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        tuple: (pipeline, feature_columns) or (None, None) if loading fails
    """
    # First check if files exist
    if not check_model_files_exist(model_path):
        return None, None
    
    try:
        # Load the pipeline
        pipeline = joblib.load(model_path)
        
        # Load feature columns
        feature_info_path = model_path.replace('.pkl', '_features.pkl')
        with open(feature_info_path, 'rb') as f:
            feature_columns = pickle.load(f)
        
        print(f"✓ Pre-trained model loaded successfully from: {model_path}")
        print(f"✓ Number of features: {len(feature_columns)}")
        
        return pipeline, feature_columns
        
    except Exception as e:
        print(f"✗ Error loading pre-trained model: {e}")
        return None, None

def create_features_for_prediction(pitchers_df, pitcher_name, opponent_team=None):
    """
    Create features for a specific pitcher using the existing engineered features.
    This extracts features directly from the engineered data instead of recreating them.
    
    Args:
        pitchers_df (DataFrame): Engineered DataFrame with pitcher data
        pitcher_name (str): Pitcher name
        opponent_team (str, optional): Opponent team abbreviation (not used but kept for compatibility)
        
    Returns:
        DataFrame: Single row with features for prediction using exact model features
    """
    # Filter data for the specific pitcher
    pitcher_data = pitchers_df[pitchers_df['Pitcher_Name'] == pitcher_name].copy()
    
    if pitcher_data.empty:
        print(f"No data found for pitcher: {pitcher_name}")
        return None
    
    # Get the most recent data for this pitcher (typically 2025 season)
    latest_record = pitcher_data.iloc[-1]
    
    # Load model features to know exactly what we need
    try:
        with open('models/lightgbm_model_features.pkl', 'rb') as f:
            model_features = pickle.load(f)
    except Exception as e:
        print(f"Error loading model features: {e}")
        return None
    
    # Create feature vector using exact model features
    feature_vector = {}
    
    # Define columns to exclude (same as in training)
    exclude_columns = ['Season', 'Pitcher_ID', 'Team_x', 'Pitcher_Name', 'Date', 'SO']
    available_features = [col for col in pitchers_df.columns if col not in exclude_columns]
    
    # Extract features that exist in the data
    for feature in model_features:
        if feature in available_features:
            feature_vector[feature] = latest_record[feature]
        else:
            # If feature is missing, set to 0 (will be handled during retraining)
            feature_vector[feature] = 0
            print(f"⚠️  Feature {feature} not found in data, setting to 0")
    
    # Create DataFrame with exact model features
    features = pd.DataFrame([feature_vector])
    
    # Convert categorical features to numeric (same as in training)
    for col in features.columns:
        if features[col].dtype == 'object':
            features[col] = pd.to_numeric(features[col], errors='coerce').fillna(0)
    
    # Fill NaN values with 0
    features = features.fillna(0)
    
    # Handle infinite values
    for col in features.columns:
        if features[col].dtype in [np.float64, np.int64]:
            features[col] = features[col].replace([np.inf, -np.inf], 0)
    
    return features

def predict_strikeouts_with_confidence(model, pitchers_df, pitcher_name, opponent_team, strikeout_line):
    """
    Make prediction with confidence metrics for a pitcher vs opponent.
    
    Args:
        model: Trained ML pipeline
        pitchers_df: Pitchers historical data
        pitcher_name: Pitcher name
        opponent_team: Opponent team abbreviation
        strikeout_line: Betting line for strikeouts
        
    Returns:
        Dictionary with prediction, recommendation and confidence metrics
        or None if prediction couldn't be made
    """
    try:
        # Create features for prediction
        features = create_features_for_prediction(pitchers_df, pitcher_name, opponent_team)
        
        if features is None:
            return None
        
        # Make prediction
        predicted_strikeouts = float(model.predict(features)[0])
        
        # Calculate confidence based on model type
        if hasattr(model, 'named_steps'):
            model_step = None
            for step_name, step_model in model.named_steps.items():
                if hasattr(step_model, 'estimators_'):
                    model_step = step_model
                    break
            
            if model_step and hasattr(model_step, 'estimators_'):
                # For ensemble models - use tree variance
                try:
                    predictions = [float(tree.predict(features)[0]) for tree in model_step.estimators_]
                    std_dev = float(np.std(predictions))
                    confidence = max(0, min(1 - (std_dev / 3), 1))
                except:
                    # Fallback confidence calculation
                    std_dev = 0.0
                    confidence = 0.8 - (abs(predicted_strikeouts - strikeout_line) / 10)
                    confidence = max(0, min(confidence, 1))
            else:
                # For other models - use simple confidence based on line proximity
                std_dev = 0.0
                confidence = 0.8 - (abs(predicted_strikeouts - strikeout_line) / 10)
                confidence = max(0, min(confidence, 1))
        else:
            # Fallback confidence calculation
            std_dev = 0.0
            confidence = 0.8 - (abs(predicted_strikeouts - strikeout_line) / 10)
            confidence = max(0, min(confidence, 1))
        
        # Determine recommendation
        recommended_side = "Over" if predicted_strikeouts > strikeout_line else "Under"
        
        return {
            'predicted_value': predicted_strikeouts,
            'recommended_side': recommended_side,
            'confidence_percentage': float(confidence * 100),
            'std_dev': std_dev
        }
        
    except Exception as e:
        print(f"Prediction failed for {pitcher_name}: {str(e)}")
        return None

def process_betting_data(model, pitchers_df, betting_data_path, output_dir=None):
    """
    Process betting data and add predictions using the trained model.
    
    Args:
        model: Trained ML pipeline
        pitchers_df: Engineered pitchers historical data (should be from pitchers_data_engineered.csv)
        betting_data_path: Path to betting data CSV
        output_dir: Directory to save the output file (defaults to current directory)
        
    Returns:
        Updated betting DataFrame with predictions
    """
    
    # Load betting data
    betting_data = pd.read_csv(betting_data_path)
    
    # Initialize prediction columns
    betting_data['ML Predict Value'] = np.nan
    betting_data['ML Recommend Side'] = pd.Series(dtype='object')
    betting_data['ML Confidence Percentage'] = np.nan
    betting_data['Has Historical Data'] = False
    
    def has_historical_data(pitcher_name):
        """Check if pitcher has historical data available."""
        return not pitchers_df[pitchers_df['Pitcher_Name'] == pitcher_name].empty
    
    for index, row in betting_data.iterrows():
        pitcher_name = row['Player']  # Use actual pitcher name, not team abbreviation
        opponent_team = row['Opponent'] if row['Opponent'] != 'N/A' else None
        strikeout_line = row['Over Line']  # Use Over Line as the betting line
        
        has_data = has_historical_data(pitcher_name)
        betting_data.at[index, 'Has Historical Data'] = has_data
        
        if not has_data:
            print(f"⚠️ No historical data for {pitcher_name}")
            continue
        
        # Make prediction
        result = predict_strikeouts_with_confidence(
            model=model,
            pitchers_df=pitchers_df,
            pitcher_name=pitcher_name,
            opponent_team=opponent_team,
            strikeout_line=strikeout_line
        )
        
        if result:
            betting_data.at[index, 'ML Predict Value'] = result['predicted_value']
            betting_data.at[index, 'ML Recommend Side'] = result['recommended_side']
            betting_data.at[index, 'ML Confidence Percentage'] = result['confidence_percentage']
            
            # Print progress
            print(f"{pitcher_name} vs {opponent_team or 'Unknown'}: "
                  f"Line {strikeout_line:.1f} → Pred {result['predicted_value']:.1f} "
                  f"({result['recommended_side']}, {result['confidence_percentage']:.0f}%)")
        else:
            print(f"⚠️ No prediction for {pitcher_name} vs {opponent_team or 'Unknown'} - prediction failed")
    
    # Filter rows with valid predictions
    filtered_data = betting_data.dropna(subset=['ML Predict Value'])
    
    # Extract date from input filename or use current date
    date_match = re.search(r'betting_data_(\d{4}-\d{2}-\d{2})\.csv', betting_data_path)
    if date_match:
        file_date = date_match.group(1)
    else:
        file_date = datetime.now().strftime("%Y-%m-%d")
    
    # Create output filename
    output_filename = f"predicted_{file_date}.csv"
    
    # Set output path
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
    else:
        output_path = output_filename
    
    # Save the filtered data
    filtered_data.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")
    
    return filtered_data

def get_top_picks(predictions_df, n=10, verbose=True):
    """
    Get the top n highest confidence picks from the predictions DataFrame.
    
    Args:
        predictions_df (pd.DataFrame): DataFrame containing predictions
        n (int): Number of top picks to return (default: 10)
        verbose (bool): Whether to print the picks (default: True)
        
    Returns:
        pd.DataFrame: DataFrame containing the top n picks
        str: Formatted string of the top picks (if verbose=True)
    """
    if predictions_df is None or predictions_df.empty:
        if verbose:
            print("No predictions available.")
        return None, "No predictions available."
    
    # Get top n picks
    top_picks = predictions_df.sort_values('ML Confidence Percentage', ascending=False).head(n)
    
    # Create formatted string
    formatted_output = ""
    if verbose:
        formatted_output = f"\nTop {n} highest confidence picks:\n\n"
        for _, pick in top_picks.iterrows():
            formatted_output += (f"{pick['Player']} ({pick['Team']}): |"
                                f"Over Odds: {pick['Over Odds']:.2f} | "
                                f"Under Odds: {pick['Under Odds']:.2f} | "
                                f"ML Pred: {pick['ML Predict Value']:.1f} | "
                                f"API Proj: {pick['API Projected Value']:.1f} | "
                                f"{pick['ML Recommend Side']} {pick['Over Line']} | "
                                f"Confidence: {pick['ML Confidence Percentage']:.1f}%\n")
        print(formatted_output)
    
    return top_picks, formatted_output

def force_retrain_model():
    """
    Force retraining of the model by deleting existing model files and training new ones.
    """
    print("🔄 Force retraining model...")
    
    # Delete existing model files
    model_files = ['models/lightgbm_model.pkl', 'models/lightgbm_model_features.pkl']
    for file_path in model_files:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"✓ Deleted: {file_path}")
    
    # Train new model
    from src.model.model_training import train_model
    model, results = train_model()
    
    # Print model performance summary
    print("\nModel performance summary:")
    best_model_name = max(results, key=lambda k: results[k]['CV_R2'])
    print(f"Best model: {best_model_name}")
    print(f"CV R2 Score: {results[best_model_name]['CV_R2']:.4f}")
    print(f"Test R2 Score: {results[best_model_name]['Test_R2']:.4f}")
    print(f"Test MAE: {results[best_model_name]['Test_MAE']:.4f}")
    
    return model

def main(force_retrain=False):
    """
    Main function to run the strikeout prediction workflow.
    Loads data, loads or trains the model, and processes betting data.
    
    Args:
        force_retrain (bool): If True, force retraining of the model
    """
    try:
        print("=== PITCHER STRIKEOUT PREDICTION WORKFLOW ===")
        print("Loading engineered pitcher data...")
        
        # Load the ENGINEERED pitcher data (not the raw data)
        pitchers_df = pd.read_csv('pitchers_data_engineered.csv')
        print(f"✓ Loaded engineered data for {pitchers_df['Pitcher_Name'].nunique()} pitchers")
        print(f"✓ Total records: {len(pitchers_df)}")
        print(f"✓ Seasons available: {sorted(pitchers_df['Season'].unique())}")
        
        # Handle model loading/training
        if force_retrain:
            model = force_retrain_model()
        else:
            # Load the pre-trained model
            print("\nLoading pre-trained model...")
            model, feature_columns = load_trained_model()
            
            # If model loading fails, fall back to training
            if model is None:
                print("\nPre-trained model not found or corrupted. Training new model...")
                from model_training import train_model
                model, results = train_model()
                
                # Print model performance summary
                print("\nModel performance summary:")
                if results:
                    best_model_name = max(results, key=lambda k: results[k]['CV_R2'])
                    print(f"Best model: {best_model_name}")
                    print(f"CV R2 Score: {results[best_model_name]['CV_R2']:.4f}")
                    print(f"Test R2 Score: {results[best_model_name]['Test_R2']:.4f}")
                    print(f"Test MAE: {results[best_model_name]['Test_MAE']:.4f}")
            else:
                print("✓ Using pre-trained model for predictions")
                print(f"✓ Model expects {len(feature_columns)} features")
                print("💡 To retrain the model, delete the files in the 'models/' directory")
        
        # Find betting data file
        betting_files = [f for f in os.listdir('.') if f.startswith('betting_data_') and f.endswith('.csv')]
        if not betting_files:
            print("❌ No betting data files found!")
            print("💡 Make sure you have a betting_data_YYYY-MM-DD.csv file in the current directory")
            return
        
        # Use the most recent betting file
        betting_files.sort(reverse=True)
        betting_file = betting_files[0]
        print(f"\n✓ Using betting file: {betting_file}")
        
        # Process betting data with predictions
        print("\n=== MAKING PREDICTIONS ===")
        predictions = process_betting_data(
            model=model,
            pitchers_df=pitchers_df,
            betting_data_path=betting_file
        )
        
        # Display prediction summary
        if predictions is not None and not predictions.empty:
            total_predictions = len(predictions)
            over_count = (predictions['ML Recommend Side'] == 'Over').sum()
            under_count = (predictions['ML Recommend Side'] == 'Under').sum()
            high_confidence = (predictions['ML Confidence Percentage'] >= 70).sum()
            avg_confidence = predictions['ML Confidence Percentage'].mean()
            
            print(f"\n=== PREDICTION SUMMARY ===")
            print(f"Total predictions made: {total_predictions}")
            print(f"Average predicted SO: {predictions['ML Predict Value'].mean():.2f}")
            print(f"Highest predicted SO: {predictions['ML Predict Value'].max():.2f}")
            print(f"Lowest predicted SO: {predictions['ML Predict Value'].min():.2f}")
            print(f"Over recommendations: {over_count} ({over_count/total_predictions*100:.1f}%)")
            print(f"Under recommendations: {under_count} ({under_count/total_predictions*100:.1f}%)")
            print(f"Average confidence: {avg_confidence:.1f}%")
            print(f"High confidence picks (≥70%): {high_confidence} ({high_confidence/total_predictions*100:.1f}%)")
        else:
            print("❌ No predictions were generated!")
            print("💡 Check that your pitchers in the betting data have historical data in the engineered features file")
        
        print("\n✅ Workflow completed successfully!")
        return predictions
        
    except Exception as e:
        print(f"❌ Error occurred during prediction workflow: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Pitcher Strikeout Prediction')
    parser.add_argument('--retrain', action='store_true', 
                       help='Force retraining of the model')
    
    args = parser.parse_args()
    main(force_retrain=args.retrain)