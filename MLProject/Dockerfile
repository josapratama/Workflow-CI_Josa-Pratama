FROM python:3.12.7-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    mlflow==2.19.0 \
    scikit-learn==1.5.2 \
    pandas==2.2.2 \
    numpy==1.26.4 \
    waitress

# Copy model artifacts (akan di-copy saat build dari CI)
COPY model/ /app/model/

# Expose port serving
EXPOSE 8080

# Jalankan MLflow model serve
CMD ["python", "-m", "mlflow", "models", "serve", \
     "-m", "/app/model", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--no-conda"]
