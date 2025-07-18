import pandas as pd
from feature_engineering import preprocess_data
from feature_engineering import extract_features
from train_models import train_and_select_model

def run_pipeline(json_path, save_dir="models/"):
    print("Loading data...")
    df = pd.read_json(json_path)

    print("Preprocessing...")
    df_1 = preprocess_data(df)

    print("Feature Engineering...")
    features_df = extract_features(df_1)




    print("Training and evaluating models...")
    results, best_model_name = train_and_select_model(features_df, save_dir, json_path)

    return results, best_model_name

if __name__ == "__main__":
    run_pipeline("data/user-wallet-transactions.json")
