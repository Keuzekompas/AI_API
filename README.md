# Keuzekompas AI API

De AI-service voor het Keuzekompas project, verantwoordelijk voor het genereren van aanbevelingen en voorspellingen met behulp van machine learning.

## Vereisten

*   **Python 3.11** of nieuwer
*   **pip** (pakketbeheerder voor Python, meestal meegeleverd)
*   **Git**
*   **MongoDB** (Lokaal of cloud connection string)

## Installatie

1.  **Clone de repository**
    ```bash
    git clone <REPOSITORY_URL>
    cd AI_API
    ```

2.  **Maak een virtual environment aan**
    *   Windows:
        ```bash
        python -m venv venv
        venv\Scripts\activate
        ```
    *   macOS / Linux:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

3.  **Installeer afhankelijkheden**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Model downloaden**
    Het modelbestand is te groot voor Git en moet apart gedownload worden.
    *   Maak de map aan: `mkdir -p app/model`
    *   Download `model.joblib` van [HuggingFace](https://huggingface.co/Q0xuzBEFIs/keuzekompas-model/resolve/main/model.joblib) en plaats deze in `app/model/`.
    
    *Structuur:* `app/model/model.joblib`

## Configuratie

Maak een bestand genaamd `.env` in de hoofdmap en voeg de volgende configuratie toe:

```ini
# Database
MONGO_URI=mongodb://localhost:27017
DB_NAME=keuzekompas
COLLECTION_NAME=modules

# Security
JWT_SECRET=geheim_wachtwoord
JWT_ALGORITHM=HS256

# CORS
CORS_POLICY=http://localhost:3000,http://127.0.0.1:3000
```

## Opstarten

Start de applicatie met uvicorn:

```bash
uvicorn app.main:app --reload
```

De API is nu beschikbaar op: `http://127.0.0.1:8000`

## Testen

Voer de unit tests uit om de installatie te verifiëren:

```bash
python -m pytest
```