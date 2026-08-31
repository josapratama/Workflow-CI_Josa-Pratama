# Workflow-CI - Heart Disease Classifier
**Author:** Josa Pratama  
**Kelas:** Membangun Sistem Machine Learning - Dicoding

## Deskripsi
Repository ini berisi CI/CD workflow menggunakan GitHub Actions dan MLflow Project untuk melatih model klasifikasi Heart Disease secara otomatis.

## Struktur
```
Workflow-CI/
+-- .github/workflows/ci.yml   # GitHub Actions workflow
+-- MLProject/
    +-- MLProject               # MLflow project config
    +-- conda.yaml              # Environment dependencies
    +-- modelling.py            # Training script
```

## Cara Menjalankan Lokal
```bash
pip install mlflow==2.19.0 scikit-learn==1.5.2 pandas==2.2.2 numpy==1.26.4 matplotlib==3.9.2 seaborn==0.13.2
mlflow run MLProject --env-manager=local
```

## CI Workflow Steps
1. Checkout repository
2. Setup Python 3.12.7
3. Check environment
4. Install dependencies
5. Run MLflow project (training)
6. Get latest MLflow run_id
7. Install dependencies for upload
8. Upload artifacts to GitHub
9. Commit artifacts to repository
