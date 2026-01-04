from sklearn.preprocessing import StandardScaler

def scale_features(df, columns):
    """Scale numerical columns using StandardScaler."""
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df
