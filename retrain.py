import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib


def main():
    # Load data from CSV
    data = pd.read_csv('data/loan_data.csv')
    data = data.drop('Loan_ID', axis=1)


    # Separate features (X) and target (y)
    X = data.drop('Loan_Status', axis=1)
    y = data['Loan_Status']

    # Encode categorical features (one-hot encoding used here)
    X = pd.get_dummies(X)

    # Train model - replace or customize with your model and parameters
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    # Save the trained model to file
    joblib.dump(model, 'best_loan_model.pkl')

    print('Retraining completed and model saved.')


if __name__ == '__main__':
    main()
