from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import nltk
import os
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from fastapi import FastAPI

from .schemas import RecommendationResponse, StudentInput, LanguageEnum
from .services.state import state
from .services.loader import load_data_and_model
from .services.predictor import predict_recommendations
from .services.auth import verify_token
from .train_model import train_and_save_model
from .utils import sanitize_recursive

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP LOGIC ---
    print("Starting up: loading model and data...")
    try:
        load_data_and_model() 
        print("Model and Data loaded successfully.")
    except Exception as e:
        print(f"Error during startup: {e}")
    
    yield
    # --- SHUTDOWN LOGIC ---
    print("Shutting down...")

# NLTK Setup
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

# Rate-Limiting Setup
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler) # Rate limit handler

# --- CORS MIDDLEWARE ---
cors_env = os.getenv("CORS_POLICY", "")
origins = cors_env.split(",") if cors_env else []

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# --- ENDPOINTS ---

@app.post("/api/predict", response_model=RecommendationResponse)
@limiter.limit("4/minute") 
def predict_study(
    request: Request, 
    student: StudentInput, 
    language: LanguageEnum = LanguageEnum.NL, # Default language
    token: dict = Depends(verify_token)
):
    if not state.is_ready():
        raise HTTPException(status_code=503, detail="AI Model or Database not ready.")
    
    student.sanitize()

    raw_data = predict_recommendations(student, language.value)
    modules_list = raw_data.get('recommendations', [])
    
    # Sanitize all strings in the recommendations recursively
    clean_modules = sanitize_recursive(modules_list)
    return {
        "recommendations": clean_modules, 
        "language": language.value
    }

@app.post("/api/refresh-data")
@limiter.limit("5/minute")
def refresh_data(request: Request, token: dict = Depends(verify_token)):
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
@limiter.limit("2/minute")
def trigger_training(
    request: Request, 
    background_tasks: BackgroundTasks, 
    token: dict = Depends(verify_token)
):
    background_tasks.add_task(run_training_and_reload)
    return {
        "status": "accepted", 
        "message": "Training started in background."
    }