# main.py
import os
import sys

# Ensure the project root is in sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    import config
    from src import data_collector
    from src import preprocess
    from src import model_trainer
    from src import analysis
except ImportError as e:
    print(f"Error importing modules. Ensure 'config.py' is in the project root and 'src' directory is present. Details: {e}")
    # Attempt to add project root to PYTHONPATH if running from a different CWD
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"Added {project_root} to sys.path. Please try running again.")
    exit(1)


def main():
    print("Starting the financial investment pipeline...")

    # Step 1: Data Collection (Generates sample data)
    print("\n--- Running Data Collection ---")
    if not hasattr(config, 'DATA_PATH') or not config.DATA_PATH:
        print("Error: DATA_PATH not configured in config.py.")
        return
    # Ensure the base data directory and raw/processed subdirectories exist
    os.makedirs(os.path.join(config.DATA_PATH, "raw"), exist_ok=True)
    os.makedirs(os.path.join(config.DATA_PATH, "processed"), exist_ok=True)
    
    data_collector.fetch_and_save_all_data(config.START_YEAR, config.END_YEAR)
    print("--- Data Collection Finished ---")

    # Step 2: Preprocessing
    print("\n--- Running Preprocessing ---")
    preprocess.run_preprocessing()
    print("--- Preprocessing Finished ---")

    # Step 3: Model Training
    print("\n--- Running Model Training ---")
    best_model = model_trainer.run_training_pipeline() 
    if best_model:
        print(f"Best model trained: {type(best_model).__name__}")
        model_save_path = os.path.join("models", "best_model.joblib")
        print(f"Model saved to {model_save_path}")
    else:
        print("Model training failed or returned no model.")
        return 
    print("--- Model Training Finished ---")

    # Step 4: Backtesting and Analysis
    print("\n--- Running Backtesting and Analysis ---")
    analysis.run_backtest_and_analyze(best_model)
    print("--- Backtesting and Analysis Finished ---")

    print("\nPipeline execution complete.")

if __name__ == "__main__":
    main()
