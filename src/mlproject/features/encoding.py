from sklearn.preprocessing import LabelEncoder

def encode_categorical(df, columns):
    """Label encode categorical columns."""
    for col in columns:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    return df
