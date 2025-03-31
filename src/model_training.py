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
from data_utils import load_data


from feature_engineering import calculate_weighted_performance

warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

def train_model(pitchers_df, k_percentage_df):
    """
    Train and evaluate machine learning models to predict pitcher strikeouts.
    
    Args:
        pitchers_df (DataFrame): Historical pitcher data
        k_percentage_df (DataFrame): Team strikeout percentages
        
    Returns:
        tuple: (best_model, results_dict) containing the best trained model and evaluation metrics
    """
    # Calculate weighted performance for each pitcher
    weighted_pitcher_data = []
    pitchers_df = pitchers_df[(pitchers_df['Season'] == 2023) | (pitchers_df['Season'] == 2024)]

    for pitcher in pitchers_df['Pitcher'].unique():
        pitcher_data = pitchers_df[pitchers_df['Pitcher'] == pitcher].copy()
        
        if 'SO' not in pitcher_data.columns:
            pitcher_data['SO'] = 0
            
        pitcher_data = pitcher_data.sort_values('Season')
        pitcher_data['SO_rolling_5'] = pitcher_data['SO'].rolling(5, min_periods=1).mean()
        pitcher_data['SO_rolling_10'] = pitcher_data['SO'].rolling(10, min_periods=1).mean()
        
        pitcher_data['Home_IP'] = pitcher_data[pitcher_data['Home'] == 1.0]['IP'].mean()
        pitcher_data['Away_IP'] = pitcher_data[pitcher_data['Home'] == 0.0]['IP'].mean()
        pitcher_data['Home_SO'] = pitcher_data[pitcher_data['Home'] == 1.0]['SO'].mean()
        pitcher_data['Away_SO'] = pitcher_data[pitcher_data['Home'] == 0.0]['SO'].mean()
        
        performance = calculate_weighted_performance(pitcher_data, current_season=2024, last_season=2023)
        performance['Pitcher'] = pitcher
        performance['Opp_K%'] = pitcher_data['Opp_K%'].iloc[0] if not pitcher_data.empty else k_percentage_df['%K'].mean()
        performance['Team_K%'] = pitcher_data['Team_K%'].iloc[0] if not pitcher_data.empty else k_percentage_df['%K'].mean()
        weighted_pitcher_data.append(performance)

    weighted_df = pd.DataFrame(weighted_pitcher_data)
    
    weighted_df['IP'] = weighted_df['IP'].replace(0, 1)
    weighted_df['SO_per_IP'] = weighted_df['SO'] / weighted_df['IP']
    weighted_df['BB_per_IP'] = weighted_df['BB'] / weighted_df['IP']
    weighted_df['K-BB%'] = weighted_df['SO_per_IP'] - weighted_df['BB_per_IP']
    
    required_features = [
        'IP', 'H', 'BB', 'ERA', 'FIP', 'SO_per_IP', 'BB_per_IP', 'K-BB%', 
        'Opp_K%', 'Team_K%', 'Home', 'SO_rolling_5', 'SO_rolling_10',
        'Home_IP', 'Away_IP', 'Home_SO', 'Away_SO'
    ]
    
    for feature in required_features:
        if feature not in weighted_df.columns:
            weighted_df[feature] = 0
            #print(f"Warning: Initialized missing feature {feature} with zeros")
    
    X = weighted_df[required_features].fillna(0)
    y = weighted_df['SO']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'RandomForest': RandomForestRegressor(random_state=42),
        'XGBoost': XGBRegressor(random_state=42),
        'LightGBM': LGBMRegressor(random_state=42, num_leaves=31, min_data_in_leaf=1, max_depth=-1, verbose=-1),
        'GradientBoosting': GradientBoostingRegressor(random_state=42)
    }

    best_model = None
    best_score = -np.inf
    best_model_name = ""
    results = {}

    for name, model in models.items():
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

    print(f"\nBest model: {best_model_name} with R2 score: {best_score:.4f}")
    return best_model, results

def main():
    """
    Main function to train and evaluate machine learning models for predicting pitcher strikeouts.
    Loads necessary data, trains models, and displays evaluation results.
    """
    try:
        print("Loading pitcher data and team strikeout percentage data...")
        
        # Load the necessary data files
        # First try to use the load_data function if available
        try:
            pitchers_df, k_percentage_df, _ = load_data()
            print("Data loaded using load_data() function")
        except (ImportError, ModuleNotFoundError):
            # Fallback to direct loading if the function isn't available
            print("Fallback to direct data loading...")
            pitchers_df = pd.read_csv('pitchers_data.csv')
            k_percentage_df = pd.read_csv('team_strikeout_percentage.csv')
            print("Data loaded directly from CSV files")
        
        print(f"Loaded data for {pitchers_df['Pitcher'].nunique()} pitchers")
        print(f"Loaded strikeout percentages for {len(k_percentage_df)} teams")
        
        # Train the models
        print("\nTraining machine learning models to predict pitcher strikeouts...")
        best_model, results = train_model(pitchers_df, k_percentage_df)
        
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
                    'IP', 'H', 'BB', 'ERA', 'FIP', 'SO_per_IP', 'BB_per_IP', 'K-BB%', 
                    'Opp_K%', 'Team_K%', 'Home', 'SO_rolling_5', 'SO_rolling_10',
                    'Home_IP', 'Away_IP', 'Home_SO', 'Away_SO'
                ]
                
                # Create a DataFrame of feature importances
                importances = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model.feature_importances_
                })
                
                # Sort by importance
                importances = importances.sort_values('Importance', ascending=False)
                
                # Display top 10 features
                print(importances.head(10))
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