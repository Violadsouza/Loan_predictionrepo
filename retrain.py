import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

def main():
    # Load data
    df = pd.read_csv('data/loan_data.csv')

    # Drop 'Loan_ID' column
    if 'Loan_ID' in df.columns:
        df = df.drop('Loan_ID', axis=1)

    # Separate numerical and categorical columns
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns

    # Fill missing values
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())  # median for numeric

    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])  # mode for categorical

    # Separate features and target variable
    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']

    # One-hot encode categorical features
    X = pd.get_dummies(X)

    # Train Random Forest classifier
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    # Save the trained model
    joblib.dump(model, 'best_loan_model.pkl')

    print("Retraining complete and model saved.")

if __name__ == '__main__':
    main()
