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

# 8. DOWNLOAD HET MODEL (Nieuwe stap!)
# We maken eerst de map aan voor de zekerheid
RUN mkdir -p app/model

# Vervang onderstaande URL door jouw 'resolve' link van Hugging Face!
# Let op de output vlag -O: die zorgt dat hij in de juiste map komt.
RUN wget https://huggingface.co/Q0xuzBEFIs/keuzekompas-model/resolve/main/model.joblib -O app/model/model.joblib

# 9. Open poort 8000
EXPOSE 8000

# 10. Start de app
ENV PYTHONPATH=/app
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--timeout", "600", "-k", "uvicorn.workers.UvicornWorker", "app.main:app"]