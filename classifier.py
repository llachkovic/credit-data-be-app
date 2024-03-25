import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo


def get_model_and_scaler():
    column_mapping = {
        'Attribute1': 'checkingAccountStatus',
        'Attribute2': 'durationInMonths',
        'Attribute3': 'creditHistory',
        'Attribute4': 'purpose',
        'Attribute5': 'creditAmount',
        'Attribute6': 'savingsAccount',
        'Attribute7': 'employmentDuration',
        'Attribute8': 'installmentRate',
        'Attribute9': 'personalStatus',
        'Attribute10': 'otherDebtors',
        'Attribute11': 'presentResidenceSince',
        'Attribute12': 'property',
        'Attribute13': 'age',
        'Attribute14': 'otherInstallmentPlans',
        'Attribute15': 'housing',
        'Attribute16': 'numberOfExistingCredits',
        'Attribute17': 'job',
        'Attribute18': 'numberOfPeopleLiableForMaintenance',
        'Attribute19': 'telephone',
        'Attribute20': 'foreignWorker',
    }

    statlog_german_credit_data = fetch_ucirepo(id=144)

    features: pd.DataFrame = statlog_german_credit_data.data.features
    targets: pd.DataFrame = statlog_german_credit_data.data.targets

    features.rename(columns=column_mapping, inplace=True)

    features_numeric = pd.get_dummies(features)
    print(features_numeric.head())

    X_train, X_test, y_train, y_test = train_test_split(features_numeric, targets, test_size=0.2, random_state=42)

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression()

    rfe = RFE(model, n_features_to_select=None)
    X_train_rfe = rfe.fit_transform(X_train_scaled, y_train_resampled)

    model.fit(X_train_rfe, y_train_resampled)

    X_test_rfe = rfe.transform(X_test_scaled)
    y_pred_rfe = model.predict(X_test_rfe)

    accuracy_rfe = accuracy_score(y_test, y_pred_rfe)
    print(f"Accuracy with feature selection and SMOTE: {accuracy_rfe:.2%}")

    print(f"Optimal number of features selected by RFE: {rfe.n_features_}")

    scaler.fit_transform(X_train_rfe)

    return model, scaler
