import os
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from evaluate_models import evaluate_saved_model

def train_and_select_model(features_df, save_dir, json_path):
    X = features_df.drop(columns=['credit_score','userWallet'])
    y = features_df['credit_score']

    # Split
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Models
    models = {
        "LinearRegression": LinearRegression(),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "DecisionTree": DecisionTreeRegressor(random_state=42),
        "RandomForest": RandomForestRegressor(random_state=42)
    }

    results = {}
    best_model_name = None
    best_r2 = float('-inf')
    best_model = None

    print("Training models...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        results[name] = {'mse': mse, 'r2': r2}
        print(f"{name} - MSE: {mse:.4f}, R2: {r2:.4f}")

        if r2 > best_r2:
            best_model_name = name
            best_model = model
            best_r2 = r2

    # Save best model
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, f"{best_model_name}.pkl")
    joblib.dump(best_model, model_path)
    print(f"\nBest model saved: {best_model_name} at {model_path}")

    # Evaluate final
    

    return results, best_model_name
