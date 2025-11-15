# ===============================
# 🚀 Dockerfile – Dr Data 2.0 Analysis Engine (Optimisé)
# ===============================

FROM python:3.11-slim

# ====== ENV (Performances + Compatibilité) ======
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# ====== Dépendances système ======
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    libatlas-base-dev \
    liblapack-dev \
    libpng-dev \
    libjpeg62-turbo-dev \
    libxml2-dev \
    libxslt1-dev \
    libz-dev \
    && rm -rf /var/lib/apt/lists/*


# ====== Répertoire de travail ======
WORKDIR /app

# ====== Dépendances Python ======
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ====== Copie du moteur d’analyse ======
COPY App.py ./app.py

# ====== Clé API (optionnelle) ======
ENV SERVICE_API_KEY="change-me"

# ====== Port exposé ======
EXPOSE 8080

# ====== Healthcheck ======
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl --fail http://localhost:8080/health || exit 1

# ====== Lancement Uvicorn optimisé ======
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]

