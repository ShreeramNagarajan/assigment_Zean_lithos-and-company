import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score
from feature_engineering import preprocess_data,extract_features
def evaluate_saved_model(json_path, model_path):
    """
    Evaluate a saved model on the dataset.
    """
    df = pd.read_json(json_path, lines=True)
    df = preprocess_data(df)
    features_df = extract_features(df)

    X = features_df.drop(columns=["credit_score"])
    y = features_df["credit_score"]

    model = joblib.load(model_path)
    preds = model.predict(X)

    mse = mean_squared_error(y, preds)
    r2 = r2_score(y, preds)

    return {"mse": mse, "r2": r2}
