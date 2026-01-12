# 1. Gebruik een lichte Python basis
FROM python:3.11-slim

# 2. Zet de werkmap in de container
WORKDIR /app

# 3. Installeer systeem tools die nodig kunnen zijn voor builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Kopieer requirements
COPY requirements.txt .

# 5. SLIMME TRUC: Installeer EERST de lichte CPU-versie van Torch
# Dit voorkomt dat pip de 2GB GPU-versie downloadt.
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 6. Filter torch uit requirements.txt (want die hebben we al) en installeer de rest
RUN grep -v "torch" requirements.txt > requirements_no_torch.txt
RUN pip install --no-cache-dir -r requirements_no_torch.txt

# 7. Kopieer de rest van je code (app map, model map, etc.)
COPY . .

# 8. Open poort 8000
EXPOSE 8000

# 9. Start de app
# We gebruiken pythonpath zodat hij de modules goed vindt
ENV PYTHONPATH=/app
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--timeout", "600", "-k", "uvicorn.workers.UvicornWorker", "app.main:app"]