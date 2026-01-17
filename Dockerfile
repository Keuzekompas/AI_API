# 1. Gebruik een lichte Python basis
FROM python:3.11-slim

# 2. Zet de werkmap in de container
WORKDIR /app

# 3. Installeer systeem tools (Nu ook wget toegevoegd voor de download!)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 4. Kopieer requirements
COPY requirements.txt .

# 5. Installeer CPU Torch (tegen 2GB overhead)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 6. Installeer de rest
RUN grep -v "torch" requirements.txt > requirements_no_torch.txt
RUN pip install --no-cache-dir -r requirements_no_torch.txt

# Download NLTK data tijdens de build, zodat het niet tijdens startup hoeft
RUN python -m nltk.downloader punkt punkt_tab

# 7. Kopieer de rest van je code
COPY . .

# 8. DOWNLOAD HET MODEL
# We downloaden het model naar een map BUITEN de workdir (/app).
# Dit voorkomt dat een volume mount (die vaak /app overschrijft) het model verbergt.
RUN mkdir -p /model
RUN wget https://huggingface.co/Q0xuzBEFIs/keuzekompas-model/resolve/main/model.joblib -O /model/model.joblib

# Stel de variabele in zodat de app weet waar het model staat
ENV MODEL_PATH=/model/model.joblib

# 9. Open poort 80
EXPOSE 80

# 10. Start de app
ENV PYTHONPATH=/app
CMD ["gunicorn", "--bind", "0.0.0.0:80", "--timeout", "600", "-k", "uvicorn.workers.UvicornWorker", "app.main:app"]