"""
Modelling - Heart Disease Classification (MLflow Project Entry Point)
Author: Josa Pratama
Description: Entry point untuk MLflow Project pada Workflow CI.
             Menerima hyperparameter via argparse dan menyimpan artefak ke MLflow.
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

# ─── Konstanta ────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
DATA_DIR = 'heart_disease_preprocessing'
DATASET_URL = 'https://raw.githubusercontent.com/dsrscientist/dataset1/master/heart_disease.csv'
RAW_PATH = 'heart_disease_raw.csv'


# ─── Helper: Dataset ──────────────────────────────────────────────────────────

def _create_fallback_dataset(save_path: str):
    """Buat dataset sintetis jika download gagal."""
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


def load_and_preprocess() -> tuple:
    """Download, load, dan preprocess dataset Heart Disease."""
    # Download jika belum ada
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

    # Missing values → median/mode
    for col in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']:
        df[col].fillna(df[col].median(), inplace=True)
    for col in ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Hapus duplikat
    df.drop_duplicates(inplace=True)

    # Outlier IQR
    for col in ['trestbps', 'chol', 'thalach', 'oldpeak']:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

    df.reset_index(drop=True, inplace=True)

    # One-hot encoding
    df = pd.get_dummies(df, columns=['cp', 'restecg', 'slope', 'ca', 'thal'],
                        drop_first=False, dtype=int)

    # Split
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    # Standarisasi
    num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    # Simpan
    os.makedirs(DATA_DIR, exist_ok=True)
    pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1) \
        .to_csv(f'{DATA_DIR}/train.csv', index=False)
    pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1) \
        .to_csv(f'{DATA_DIR}/test.csv', index=False)

    print(f'[INFO] Preprocessing selesai. Train: {X_train.shape}, Test: {X_test.shape}')
    return X_train, X_test, y_train, y_test


# ─── Helper: Artefak ─────────────────────────────────────────────────────────

def save_confusion_matrix(model, X_test, y_test, path='training_confusion_matrix.png'):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Sehat', 'Penyakit'],
                yticklabels=['Sehat', 'Penyakit'],
                linewidths=0.5)
    plt.title('Confusion Matrix - Heart Disease Classifier', fontweight='bold')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def save_feature_importance(model, feature_names, path='feature_importance.png', top_n=15):
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]
    feats = [feature_names[i] for i in idx]
    vals = importances[idx]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feats)), vals[::-1], color='steelblue', edgecolor='black', alpha=0.8)
    plt.yticks(range(len(feats)), feats[::-1])
    plt.xlabel('Importance Score')
    plt.title(f'Top {top_n} Feature Importances', fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def save_metric_info(metrics: dict, params: dict, path='metric_info.json'):
    info = {
        'model': 'RandomForestClassifier',
        'params': params,
        'metrics': {k: round(float(v), 6) for k, v in metrics.items()}
    }
    with open(path, 'w') as f:
        json.dump(info, f, indent=2)
    return path


# ─── Main Training ────────────────────────────────────────────────────────────

def main(args):
    print('=' * 55)
    print('  MLFLOW PROJECT - HEART DISEASE CLASSIFIER')
    print(f'  n_estimators={args.n_estimators}, max_depth={args.max_depth}')
    print('=' * 55)

    # Preprocessing
    X_train, X_test, y_train, y_test = load_and_preprocess()
    feature_names = list(X_train.columns)

    # Setup MLflow
    tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000')
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment('Heart Disease Classification')

    params = {
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'min_samples_split': args.min_samples_split,
        'min_samples_leaf': args.min_samples_leaf,
        'random_state': args.random_state,
    }

    with mlflow.start_run(run_name='CI_RandomForest') as run:
        # Train
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

        # MLflow log params
        for k, v in params.items():
            mlflow.log_param(k, v)

        # MLflow log metrics
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # Log model
        mlflow.sklearn.log_model(model, artifact_path='model',
                                 registered_model_name='heart-disease-ci')

        # Log artefak tambahan
        cm_path = save_confusion_matrix(model, X_test, y_test)
        fi_path = save_feature_importance(model, feature_names)
        mi_path = save_metric_info(metrics, params)

        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(fi_path)
        mlflow.log_artifact(mi_path)

        report = classification_report(y_test, y_pred, target_names=['Sehat', 'Penyakit'])
        mlflow.log_text(report, 'classification_report.txt')

        # Simpan run_id untuk digunakan step berikutnya
        run_id = run.info.run_id
        with open('latest_run_id.txt', 'w') as f:
            f.write(run_id)

        print(f'\n[INFO] Test Accuracy : {metrics["test_accuracy"]:.4f}')
        print(f'[INFO] Test F1 Score : {metrics["test_f1"]:.4f}')
        print(f'[INFO] Test ROC AUC  : {metrics["test_roc_auc"]:.4f}')
        print(f'[INFO] Run ID        : {run_id}')

        # Cleanup lokal
        for p in [cm_path, fi_path, mi_path]:
            if os.path.exists(p):
                os.remove(p)

    print('\n[INFO] MLflow Project run selesai!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Heart Disease Classifier - MLflow Project')
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=10)
    parser.add_argument('--min_samples_split', type=int, default=5)
    parser.add_argument('--min_samples_leaf', type=int, default=2)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--random_state', type=int, default=42)
    args = parser.parse_args()
    main(args)
