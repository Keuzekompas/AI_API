# AI API Service (FastAPI)

Dit is de AI-service voor het project. Deze applicatie is gebouwd met Python en **FastAPI**.
De service draait los van de andere onderdelen en verwerkt AI-verzoeken.

## 🚀 Aan de slag

Volg deze stappen om het project lokaal op je machine te draaien.

### 1. Vereisten

Zorg dat je **Python** geïnstalleerd hebt (versie 3.8 of hoger).
Controleer dit in je terminal:

```bash
python --version
```

### 2. Virtual Environment aanmaken (Eénmalig)

Om conflicten met andere projecten te voorkomen, maken we een virtuele omgeving aan. Zorg dat je in de map `ai_api` staat in je terminal.

**Windows:**
```powershell
python -m venv venv
```

**Mac/Linux:**
```bash
python3 -m venv venv
```

### 3. De omgeving activeren (Elke keer als je gaat werken)

Voordat je commando's uitvoert, moet je zorgen dat je in de `(venv)` zit.

**Windows (PowerShell):**
```powershell
.\venv\Scripts\activate
```
*(Als je een foutmelding krijgt over scripts, voer dan eerst `Set-ExecutionPolicy RemoteSigned -Scope Process` uit).*

**Mac/Linux:**
```bash
source venv/bin/activate
```

✅ **Check:** Je zou nu `(venv)` voor je command line moeten zien staan.

### 4. Dependencies installeren

Installeer alle benodigde packages die in `requirements.txt` staan.

```bash
pip install -r requirements.txt
```

### 5. De Server Starten

Start de ontwikkelserver met hot-reload (zodat hij herstart als je code opslaat).

```bash
uvicorn app.main:app --reload --port 8000
```

De API draait nu op: [http://localhost:8000](http://localhost:8000)

## 📚 Documentatie (Swagger UI)

FastAPI genereert automatische documentatie. Ga in je browser naar:

👉 [http://localhost:8000/docs](http://localhost:8000/docs)

Hier kun je alle endpoints bekijken en direct testen.

## 🛠 Nieuwe package toevoegen?

Als je een nieuwe library installeert (bijv. `pandas` of `torch`), vergeet dan niet de requirements file bij te werken zodat je teamgenoten deze ook krijgen:

```bash
pip install pakket-naam
pip freeze > requirements.txt
```
