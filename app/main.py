from fastapi import FastAPI

app = FastAPI(title="Mijn AI API")

@app.get("/")
def read_root():
    return {"status": "AI Service is running"}