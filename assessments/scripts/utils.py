def get_feature_target(df):
    X = df.drop(columns=['userWallet', 'credit_score'])
    y = df['credit_score']
    return X, y
