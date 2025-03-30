import warnings
warnings.filterwarnings('ignore')

from data_utils import load_data
from model_training import train_model
from predictions import process_betting_data

def main():
    """
    Main execution function to run the entire pipeline:
    1. Load and preprocess data
    2. Train prediction model
    3. Process betting data and generate predictions
    """
    # Step 1: Load data
    print("=== Loading and preprocessing data ===")
    pitchers_df, k_percentage_df, betting_file_used = load_data()
    
    # Step 2: Train model
    print("\n=== Training prediction model ===")
    model, model_results = train_model(pitchers_df, k_percentage_df)
    
    # Step 3: Process betting data
    print("\n=== Generating predictions ===")
    results_df = process_betting_data(
        model=model,
        pitchers_df=pitchers_df,
        k_percentage_df=k_percentage_df,
        betting_data_path=betting_file_used,
        output_dir='predictions'
    )
    
    print("\n=== Pipeline completed successfully ===")
    return results_df

if __name__ == "__main__":
    main()