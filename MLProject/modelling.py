"""
Modelling - Heart Disease Classification (MLflow Project Entry Point)
Author: Josa Pratama
Description: Entry point untuk MLflow Project pada Workflow CI.
             Kompatibel dengan mlflow run -- tidak memanggil set_experiment()
             atau start_run() karena sudah dihandle MLflow Project.
"""

import argparse
import os
import json
import urllib.request
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)

RANDOM_SEED = 42
DATA_DIR = 'heart_disease_preprocessing'
DATASET_URL = 'https://raw.githubusercontent.com/dsrscientist/dataset1/master/heart_disease.csv'
RAW_PATH = 'heart_disease_raw.csv'


def _create_fallback_dataset(save_path):
    np.random.seed(RANDOM_SEED)
    n = 303
    data = {
        'age': np.random.randint(29, 77, n),
        'sex': np.random.randint(0, 2, n),
        'cp': np.random.randint(0, 4, n),
        'trestbps': np.random.randint(94, 200, n),
        'chol': np.random.randint(126, 564, n),
        'fbs': np.random.randint(0, 2, n),
        'restecg': np.random.randint(0, 3, n),
        'thalach': np.random.randint(71, 202, n),
        'exang': np.random.randint(0, 2, n),
        'oldpeak': np.round(np.random.uniform(0, 6.2, n), 1),
        'slope': np.random.randint(0, 3, n),
        'ca': np.random.randint(0, 4, n),
        'thal': np.random.randint(0, 4, n),
        'target': np.random.randint(0, 2, n),
    }
    pd.DataFrame(data).to_csv(save_path, index=False)
    print(f'[INFO] Dataset fallback dibuat: {save_path}')


def load_and_preprocess():
    if not os.path.exists(RAW_PATH):
        try:
            urllib.request.urlretrieve(DATASET_URL, RAW_PATH)
            print(f'[INFO] Dataset diunduh: {RAW_PATH}')
        except Exception as e:
            print(f'[WARNING] Download gagal: {e}')
            _create_fallback_dataset(RAW_PATH)

    df = pd.read_csv(RAW_PATH)
    expected = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    try:
        df.columns = expected
    except Exception:
        pass

    # Missing values
    for col in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']:
        df[col] = df[col].fillna(df[col].median())
    for col in ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Duplikat
    df = df.drop_duplicates().reset_index(drop=True)

    # Outlier IQR
    for col in ['trestbps', 'chol', 'thalach', 'oldpeak']:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

    df = df.reset_index(drop=True)
    df = pd.get_dummies(df, columns=['cp', 'restecg', 'slope', 'ca', 'thal'],
                        drop_first=False, dtype=int)

    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    scaler = StandardScaler()
    num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    os.makedirs(DATA_DIR, exist_ok=True)
    pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)\
        .to_csv(f'{DATA_DIR}/train.csv', index=False)
    pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)\
        .to_csv(f'{DATA_DIR}/test.csv', index=False)

    print(f'[INFO] Preprocessing selesai. Train: {X_train.shape}, Test: {X_test.shape}')
    return X_train, X_test, y_train, y_test


def main(args):
    print('=' * 55)
    print('  MLFLOW PROJECT - HEART DISEASE CLASSIFIER')
    print(f'  n_estimators={args.n_estimators}, max_depth={args.max_depth}')
    print('=' * 55)

    X_train, X_test, y_train, y_test = load_and_preprocess()
    feature_names = list(X_train.columns)

    params = {
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'min_samples_split': args.min_samples_split,
        'min_samples_leaf': args.min_samples_leaf,
        'random_state': args.random_state,
    }

    # Log params ke active run yang sudah dibuat oleh MLflow Project
    for k, v in params.items():
        mlflow.log_param(k, v)

    # Training
    model = RandomForestClassifier(**params, n_jobs=-1)
    model.fit(X_train, y_train)

    # Metrics
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

    metrics = {
        'test_accuracy': accuracy_score(y_test, y_pred),
        'test_precision': precision_score(y_test, y_pred, average='weighted'),
        'test_recall': recall_score(y_test, y_pred, average='weighted'),
        'test_f1': f1_score(y_test, y_pred, average='weighted'),
        'test_roc_auc': roc_auc_score(y_test, y_proba),
        'train_accuracy': accuracy_score(y_train, model.predict(X_train)),
        'cv_accuracy_mean': cv_scores.mean(),
        'cv_accuracy_std': cv_scores.std(),
    }

    for k, v in metrics.items():
        mlflow.log_metric(k, v)

    # Log model
    mlflow.sklearn.log_model(model, artifact_path='model')

    # Artefak: confusion matrix
    y_pred_cm = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_cm)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Sehat', 'Penyakit'],
                yticklabels=['Sehat', 'Penyakit'])
    plt.title('Confusion Matrix', fontweight='bold')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('training_confusion_matrix.png', dpi=100)
    plt.close()
    mlflow.log_artifact('training_confusion_matrix.png')
    os.remove('training_confusion_matrix.png')

    # Artefak: feature importance
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:15]
    feats = [feature_names[i] for i in idx]
    vals = importances[idx]
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feats)), vals[::-1], color='steelblue', alpha=0.8)
    plt.yticks(range(len(feats)), feats[::-1])
    plt.xlabel('Importance Score')
    plt.title('Top 15 Feature Importances', fontweight='bold')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=100)
    plt.close()
    mlflow.log_artifact('feature_importance.png')
    os.remove('feature_importance.png')

    # Artefak: metric info json
    metric_info = {'model': 'RandomForestClassifier', 'params': params,
                   'metrics': {k: round(float(v), 6) for k, v in metrics.items()}}
    with open('metric_info.json', 'w') as f:
        json.dump(metric_info, f, indent=2)
    mlflow.log_artifact('metric_info.json')
    os.remove('metric_info.json')

    # Simpan run_id
    run_id = mlflow.active_run().info.run_id
    with open('latest_run_id.txt', 'w') as f:
        f.write(run_id)

    print(f'\n[INFO] Test Accuracy : {metrics["test_accuracy"]:.4f}')
    print(f'[INFO] Test F1       : {metrics["test_f1"]:.4f}')
    print(f'[INFO] Test ROC AUC  : {metrics["test_roc_auc"]:.4f}')
    print(f'[INFO] Run ID        : {run_id}')
    print('[INFO] MLflow Project run selesai!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=10)
    parser.add_argument('--min_samples_split', type=int, default=5)
    parser.add_argument('--min_samples_leaf', type=int, default=2)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--random_state', type=int, default=42)
    args = parser.parse_args()
    main(args)
