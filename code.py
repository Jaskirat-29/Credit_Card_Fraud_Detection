
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("train_transaction.csv")

df.head()

df.tail()

df.shape

print(df.columns.tolist())

df.describe()

df.info()

df.isnull().sum()

df.duplicated()

df.dtypes

(df.isnull().sum()/len(df))*100

round((df.isnull().sum()/len(df))*100,2)

df.duplicated().sum()

df[df.duplicated()]

df['isFraud'].value_counts()

sns.countplot(x='isFraud', data=df)
plt.show()

missing_percent = df.isnull().mean() * 100

high_missing_cols = missing_percent[missing_percent > 80].index

print("Columns with >80% missing:", len(high_missing_cols))
print(high_missing_cols.tolist())

df = df.drop(columns=high_missing_cols)

print("New shape:", df.shape)

df = df.drop(columns=['TransactionID'])

print("Shape:", df.shape)

print("\nData types:")
print(df.dtypes.value_counts())

print("\nCategorical columns:")
print(df.select_dtypes(include='object').columns.tolist())

print("\nRemaining missing values:")
print(df.isnull().sum().sort_values(ascending=False).head(20))

missing_percent = (df.isnull().mean() * 100).sort_values(ascending=False)

print(missing_percent.head(50))

print("More than 70% missing:",
      (missing_percent > 70).sum())

print("More than 60% missing:",
      (missing_percent > 60).sum())

print("More than 50% missing:",
      (missing_percent > 50).sum())

print("More than 40% missing:",
      (missing_percent > 40).sum())

for col in df.select_dtypes(include='object').columns:
    print("\n", col)
    print("Unique:", df[col].nunique())
    print("Missing %:", round(df[col].isna().mean() * 100, 2))
    print(df[col].value_counts(dropna=False).head(10))

missing_percent = df.isnull().mean() * 100

cols_to_drop = missing_percent[missing_percent > 70].index

print("Number of columns to drop:", len(cols_to_drop))
print(cols_to_drop.tolist())

df = df.drop(columns=cols_to_drop)

print("New shape:", df.shape)

print('R_emaildomain' in df.columns)
print('P_emaildomain' in df.columns)

categorical_cols = df.select_dtypes(include='object').columns

print("Categorical columns:")
print(categorical_cols.tolist())

print("\nNumber of categorical columns:", len(categorical_cols))

for col in categorical_cols:
    print(
        col,
        "| unique:", df[col].nunique(),
        "| missing:", round(df[col].isna().mean() * 100, 2), "%"
    )

categorical_cols = df.select_dtypes(include='object').columns

for col in categorical_cols:
    df[col] = df[col].fillna('Unknown')

print(df[categorical_cols].isnull().sum())

numerical_cols = df.select_dtypes(include=np.number).columns

numerical_cols = numerical_cols.drop('isFraud')

for col in numerical_cols:
    df[col] = df[col].fillna(df[col].median())

print("Total missing values:", df.isnull().sum().sum())

print(df.shape)
print(df.isnull().sum().sum())
print(df.dtypes.value_counts())

df['TransactionDT_hour'] = (
    df['TransactionDT'] // 3600
) % 24

df['TransactionDT_day'] = (
    df['TransactionDT'] // (24 * 3600)
)

df = df.drop(columns=['TransactionDT'])

X = df.drop(columns=['isFraud'])
y = df['isFraud']

print("X shape:", X.shape)
print("y shape:", y.shape)

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training:")
print(y_train.value_counts())

print("\nValidation:")
print(y_val.value_counts())

print('TransactionDT' in df.columns)
print('TransactionDT_hour' in df.columns)
print('TransactionDT_day' in df.columns)
print(df.shape)

X = df.drop(columns=['isFraud'])
y = df['isFraud']

print("X shape:", X.shape)
print("y shape:", y.shape)

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("X_train:", X_train.shape)
print("X_val:", X_val.shape)
print("y_train:", y_train.shape)
print("y_val:", y_val.shape)

categorical_cols = X_train.select_dtypes(
    include='object'
).columns.tolist()

numerical_cols = X_train.select_dtypes(
    include=np.number
).columns.tolist()

print("Categorical columns:")
print(categorical_cols)

print("\nNumber of categorical columns:", len(categorical_cols))

print("\nNumber of numerical columns:", len(numerical_cols))

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

preprocessor = ColumnTransformer(
    transformers=[
        (
            'num',
            StandardScaler(),
            numerical_cols
        ),
        (
            'cat',
            OneHotEncoder(
                handle_unknown='ignore',
                sparse_output=True
            ),
            categorical_cols
        )
    ]
)

print("Preprocessor created successfully.")

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

logistic_model = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    ))
])

print("Logistic Regression pipeline created.")

logistic_model.fit(X_train, y_train)

print("Logistic Regression training completed.")

y_pred = logistic_model.predict(X_val)

y_prob = logistic_model.predict_proba(X_val)[:, 1]

print("Predictions completed.")

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

print("Accuracy :", accuracy_score(y_val, y_pred))
print("Precision:", precision_score(y_val, y_pred))
print("Recall   :", recall_score(y_val, y_pred))
print("F1 Score :", f1_score(y_val, y_pred))
print("ROC-AUC  :", roc_auc_score(y_val, y_prob))
print("PR-AUC   :", average_precision_score(y_val, y_prob))

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_pred))

print("\nClassification Report:")
print(classification_report(y_val, y_pred))

from sklearn.ensemble import RandomForestClassifier

rf_model = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ))
])

print("Random Forest pipeline created.")

rf_model.fit(X_train, y_train)

print("Random Forest training completed.")

rf_pred = rf_model.predict(X_val)
rf_prob = rf_model.predict_proba(X_val)[:, 1]

print("Accuracy :", accuracy_score(y_val, rf_pred))
print("Precision:", precision_score(y_val, rf_pred))
print("Recall   :", recall_score(y_val, rf_pred))
print("F1 Score :", f1_score(y_val, rf_pred))
print("ROC-AUC  :", roc_auc_score(y_val, rf_prob))
print("PR-AUC   :", average_precision_score(y_val, rf_prob))

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, rf_pred))

print("\nClassification Report:")
print(classification_report(y_val, rf_pred))

if_preprocessor = rf_model.named_steps['preprocessor']

X_train_encoded = if_preprocessor.transform(X_train)
X_val_encoded = if_preprocessor.transform(X_val)

print("Training encoded shape:", X_train_encoded.shape)
print("Validation encoded shape:", X_val_encoded.shape)

from sklearn.ensemble import IsolationForest


sample_size = 100000

X_if_train = X_train_encoded[:sample_size]

print("Isolation Forest training data:", X_if_train.shape)

isolation_model = IsolationForest(
    n_estimators=200,
    max_samples=100000,
    contamination=0.035,
    random_state=42,
    n_jobs=-1
)

print("Isolation Forest model created.")

isolation_model.fit(X_if_train)

print("Isolation Forest training completed.")

if_pred = isolation_model.predict(X_val_encoded)

print("Prediction completed.")

if_pred_binary = (if_pred == -1).astype(int)

print("Predicted normal:", (if_pred_binary == 0).sum())
print("Predicted anomaly:", (if_pred_binary == 1).sum())

print("Accuracy :", accuracy_score(y_val, if_pred_binary))
print("Precision:", precision_score(y_val, if_pred_binary))
print("Recall   :", recall_score(y_val, if_pred_binary))
print("F1 Score :", f1_score(y_val, if_pred_binary))

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, if_pred_binary))

print("\nClassification Report:")
print(classification_report(y_val, if_pred_binary))

if_scores = -isolation_model.decision_function(X_val_encoded)

print("ROC-AUC:", roc_auc_score(y_val, if_scores))
print("PR-AUC :", average_precision_score(y_val, if_scores))

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

precision, recall, thresholds = precision_recall_curve(
    y_val,
    if_scores
)

plt.figure(figsize=(8, 5))
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Isolation Forest - Precision-Recall Curve")
plt.grid()
plt.show()

from sklearn.ensemble import RandomForestClassifier

rf_balanced = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf_balanced.fit(X_train_encoded, y_train)

rf_pred = rf_balanced.predict(X_val_encoded)
rf_prob = rf_balanced.predict_proba(X_val_encoded)[:, 1]

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

print("Accuracy :", accuracy_score(y_val, rf_pred))
print("Precision:", precision_score(y_val, rf_pred))
print("Recall   :", recall_score(y_val, rf_pred))
print("F1 Score :", f1_score(y_val, rf_pred))
print("ROC-AUC  :", roc_auc_score(y_val, rf_prob))
print("PR-AUC   :", average_precision_score(y_val, rf_prob))

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, rf_pred))

print("\nClassification Report:")
print(classification_report(y_val, rf_pred))

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

for threshold in thresholds:
    
    rf_pred_threshold = (rf_prob >= threshold).astype(int)
    
    print(f"\nThreshold: {threshold}")
    print("Accuracy :", round(accuracy_score(y_val, rf_pred_threshold), 4))
    print("Precision:", round(precision_score(y_val, rf_pred_threshold), 4))
    print("Recall   :", round(recall_score(y_val, rf_pred_threshold), 4))
    print("F1 Score :", round(f1_score(y_val, rf_pred_threshold), 4))

import matplotlib.pyplot as plt

thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

precisions = []
recalls = []
f1_scores = []

for threshold in thresholds:
    
    pred = (rf_prob >= threshold).astype(int)
    
    precisions.append(precision_score(y_val, pred))
    recalls.append(recall_score(y_val, pred))
    f1_scores.append(f1_score(y_val, pred))

plt.plot(thresholds, precisions, marker='o', label='Precision')
plt.plot(thresholds, recalls, marker='o', label='Recall')
plt.plot(thresholds, f1_scores, marker='o', label='F1 Score')

plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Random Forest - Threshold Tuning")
plt.legend()
plt.grid()
plt.show()

import numpy as np

best_index = np.argmax(f1_scores)

print("Best threshold:", thresholds[best_index])
print("Best F1 Score :", f1_scores[best_index])
print("Precision     :", precisions[best_index])
print("Recall        :", recalls[best_index])

thresholds_fine = np.arange(0.30, 0.51, 0.01)

results = []

for threshold in thresholds_fine:
    pred = (rf_prob >= threshold).astype(int)

    results.append({
        'threshold': threshold,
        'precision': precision_score(y_val, pred),
        'recall': recall_score(y_val, pred),
        'f1': f1_score(y_val, pred)
    })

results_df = pd.DataFrame(results)

print(results_df.loc[results_df['f1'].idxmax()])

best_threshold = results_df.loc[
    results_df['f1'].idxmax(), 'threshold'
]

rf_final_pred = (rf_prob >= best_threshold).astype(int)

print("Best Threshold:", best_threshold)
print("\nConfusion Matrix:")
print(confusion_matrix(y_val, rf_final_pred))

print("\nClassification Report:")
print(classification_report(y_val, rf_final_pred))