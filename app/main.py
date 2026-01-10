from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
import nltk
from .schemas import StudentInput
import os
from dotenv import load_dotenv

from .schemas import RecommendationResponse, StudentInput
from .services.state import state
from .services.loader import load_data_and_model
from .services.predictor import predict_recommendations
from .services.auth import verify_token
from .train_model import train_and_save_model
from .utils import sanitize_recursive

load_dotenv()

# NLTK Setup
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

app = FastAPI()

# --- CORS MIDDLEWARE ---
origins = [os.getenv("CORS_POLICY")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# Startup Event
@app.on_event("startup")
def startup_event():
    load_data_and_model()

# --- ENDPOINTS ---

@app.post("/api/predict", response_model=RecommendationResponse)
@limiter.limit("4/minute") 
def predict_study(
    request: Request, 
    student: StudentInput, 
    language: str = "NL", 
    token: dict = Depends(verify_token)
):
    if not state.is_ready():
        raise HTTPException(status_code=503, detail="AI Model or Database not ready.")
    
    student.sanitize()
    
    # 1. Haal de ruwe data op uit de service
    raw_data = predict_recommendations(student, language)
    
    # 2. Pak de ECHTE lijst met modules eruit.
    # Volgens jouw JSON structuur zit de lijst in raw_data['recommendations']
    modules_list = raw_data.get('recommendations', [])
    
    # 3. Sanitize alleen de lijst met modules
    clean_modules = sanitize_recursive(modules_list)
    
    # 4. Return exact wat je RecommendationResponse schema verwacht
    return {
        "recommendations": clean_modules, 
        "language": language
    }

@app.post("/api/refresh-data")
    try:
        load_data_and_model()  
        return {"status": "success", "message": "Database reloaded and embeddings updated."}
    except Exception as e:
        # Note: Exception in Python 3 does not catch SystemExit, KeyboardInterrupt, GeneratorExit
        raise HTTPException(status_code=500, detail=str(e))
    
def run_training_and_reload():
    print("Background task: Training model...")
    try:
        train_and_save_model() 
        print("Training done. Reloading data...")
        load_data_and_model()
        print("System updated!")
    except Exception as e:
        print(f"Error during background training: {e}")

@app.post("/api/train")
def trigger_training(background_tasks: BackgroundTasks, token: dict = Depends(verify_token)):
    background_tasks.add_task(run_training_and_reload)
    return {
        "status": "accepted", 
        "message": "Training started in background."
    }