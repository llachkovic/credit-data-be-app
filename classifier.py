import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
from sklearn.feature_selection import SelectFromModel


def get_model():
    random_state = 42
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

    # Separate numerical and categorical features
    numerical_features = features.select_dtypes(include=['int64', 'float64'])
    categorical_features = features.select_dtypes(include=['object'])

    categorical_features = pd.get_dummies(categorical_features)

    features_processed = pd.concat([numerical_features, categorical_features], axis=1)

    pd.set_option('display.max_columns', None)

    X_train, X_test, y_train, y_test = train_test_split(features_processed, targets, test_size=0.2, random_state=random_state)

    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    model = LogisticRegression(penalty='l1', solver='liblinear', random_state=random_state, class_weight={1: 1, 2: 1})

    # Feature selection using Lasso (L1 regularization)
    selector = SelectFromModel(estimator=model)
    selector.fit(X_train_resampled, y_train_resampled)

    # Transform training and testing sets
    X_train_selected = selector.transform(X_train_resampled)
    X_test_selected = selector.transform(X_test)

    # Train the model
    model.fit(X_train_selected, y_train_resampled)

    # Evaluate the model
    y_pred = model.predict(X_test_selected)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    print("Selected feature names:")
    selected_feature_names = features_processed.columns[selector.get_support()]
    print(selected_feature_names)

    # Additional evaluation metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model
