import streamlit as st
import joblib
import pandas as pd
import numpy as np
# from scripts.feature_engineering import preprocess_data, extract_features  # assuming this file has the functions you pasted
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the raw transaction data.
    """
    if 'actionData' not in df.columns:
        raise ValueError("Missing 'actionData' in input data.")

    # Expand nested actionData
    action_data = pd.json_normalize(df['actionData'])
    df = pd.concat([df.drop(columns=['actionData']), action_data], axis=1)

    # Remove duplicate columns if any
    df = df.loc[:, ~df.columns.duplicated()]

    # Rename columns for consistency
    df.rename(columns={
        'type': 'action',
        'amount': 'action_amount',
        'assetPriceUSD': 'asset_price_usd'
    }, inplace=True)

    # Select required columns
    expected_columns = ['userWallet', 'action', 'action_amount', 'asset_price_usd', 'timestamp']
    df = df[[col for col in expected_columns if col in df.columns]]

    # Convert values to numeric and datetime
    df['action_amount'] = pd.to_numeric(df['action_amount'], errors='coerce') / 1e6  # Adjusting for token decimals
    df['asset_price_usd'] = pd.to_numeric(df['asset_price_usd'], errors='coerce')
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')

    # Drop rows with critical missing data
    df.dropna(subset=['userWallet', 'action', 'action_amount', 'asset_price_usd', 'timestamp'], inplace=True)

    return df


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract wallet-level features and compute credit score.
    """
    # Deduplicate columns if needed
    df = df.loc[:, ~df.columns.duplicated()]

    # Ensure required columns exist
    required_cols = {'userWallet', 'action', 'action_amount', 'asset_price_usd', 'timestamp'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required_cols - set(df.columns)}")

    # Calculate USD value per transaction
    df['usd_value'] = df['action_amount'] * df['asset_price_usd']

    # Safely lowercase the 'action' column
    df['action'] = df['action'].astype(str).map(str.lower)

    # Define feature extraction function
    def user_group(group):
        features = {
            'total_usd': group['usd_value'].sum(),
            'avg_usd': group['usd_value'].mean(),
            'tx_count': len(group),
            'active_days': (group['timestamp'].max() - group['timestamp'].min()).days + 1,
            'last_action_days_ago': (pd.Timestamp.now() - group['timestamp'].max()).days
        }

        for action in ['deposit', 'borrow', 'repay', 'redeemunderlying', 'liquidationcall']:
            features[f'{action}_count'] = len(group[group['action'] == action])

        # Ratios (avoid division by 0)
        features['repay_to_borrow_ratio'] = features['repay_count'] / (features['borrow_count'] + 1)
        features['redeem_to_deposit_ratio'] = features['redeemunderlying_count'] / (features['deposit_count'] + 1)
        features['liquidation_rate'] = features['liquidationcall_count'] / (features['tx_count'] + 1)
        features['tx_per_day'] = features['tx_count'] / max(features['active_days'], 1)

        return pd.Series(features)

    # Apply to each wallet group
    features_df = df.groupby('userWallet', group_keys=False).apply(user_group).reset_index()

    # Credit score formula
    features_df['credit_score'] = (
    400
    + 50 * np.log1p(features_df['total_usd'])  # log dampens large USD influence
    + 100 * features_df['repay_to_borrow_ratio']
    + 50 * features_df['redeem_to_deposit_ratio']
    + 10 * features_df['tx_per_day']
    - 120 * features_df['liquidation_rate']
    - 0.8 * features_df['last_action_days_ago']
).clip(0, 1000)

    return features_df

st.set_page_config(page_title="Aave Credit Score Predictor", layout="centered")
st.title("📊 Aave Wallet Credit Score Predictor")

uploaded_file = st.file_uploader("Upload your wallet JSON file", type=["json"])

if uploaded_file:
    try:
        df_raw = pd.read_json(uploaded_file)
        df_clean = preprocess_data(df_raw)
        df_features = extract_features(df_clean)


        # Drop the credit_score column if it exists (model should predict it)
       # Preserve userWallet before dropping for model input
        user_wallets = df_features['userWallet'].copy()

        if 'credit_score' in df_features.columns:
            df_features = df_features.drop(columns=['credit_score','userWallet'])

        


        model = joblib.load("C:\\Users\\nrshr\\OneDrive\\Desktop\\projects\\assessments\\models\\GradientBoosting.pkl")

        predictions = model.predict(df_features)
        df_features['predicted_score'] = predictions
        df_features['userWallet'] = user_wallets

        st.success("✅ Credit scores predicted for uploaded wallet data!")
        st.dataframe(df_features[['userWallet', 'predicted_score']])

        # Download
        csv = df_features.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results as CSV", data=csv, file_name="credit_scores.csv", mime="text/csv")

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
else:
    st.info("📥 Upload a JSON file to begin.")
