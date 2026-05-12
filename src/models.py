from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_models(df, features):
    """Trains Logistic Regression and Random Forest models"""
    
    X = df[features]
    y = df['target']

    # Time-series split (keep chronological order)
    split_index = int(len(df) * 0.8)

    # Create train/test sets
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train models
    log_model = LogisticRegression(max_iter=1000)
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    log_model.fit(X_train_scaled, y_train)
    rf_model.fit(X_train, y_train)

    # Make predictions
    log_preds = log_model.predict(X_test_scaled)
    rf_preds = rf_model.predict(X_test)

    # Store results
    results = {
        "Logistic Regression": {
            "model": log_model,
            "predictions": log_preds,
            "accuracy": accuracy_score(y_test, log_preds)
        },
        "Random Forest": {
            "model": rf_model,
            "predictions": rf_preds,
            "accuracy": accuracy_score(y_test, rf_preds)
        }
    }

    # Prepare test data
    test_df = df.iloc[split_index:].copy()

    return results, test_df