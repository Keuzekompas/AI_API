from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import joblib
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import os
import torch
import io
import pickle
from pymongo import MongoClient
from dotenv import load_dotenv
import nltk
from .train_model import train_and_save_model

# Check for NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

# Laad environment variables
load_dotenv()

app = FastAPI()

# Global variables
model = None
df = None
module_embeddings = None

# --- CONFIGURATIE ---
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "KeuzeKompas")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "modules")

# --- LOADER HELPERS ---
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        return super().find_class(module, name)

def load_data_and_model():
    global model, df, module_embeddings
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Load Model
    model_path = os.path.join(BASE_DIR, 'model', 'model.joblib')
    print(f"Loading model from: {model_path}")
    try:
        try:
            model = joblib.load(model_path)
        except Exception as e:
            if "CUDA" in str(e):
                print("CUDA mismatch detected. Retrying with custom CPU Unpickler...")
                with open(model_path, 'rb') as f:
                    model = CPU_Unpickler(f).load()
            else:
                raise e
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Critical Error: Could not load model. {e}")
        # We zetten model op None, API zal errors gooien (geen fallback meer)
        model = None

    # 2. Load Data (MongoDB)
    print(f"Connecting to MongoDB: {DB_NAME} -> {COLLECTION_NAME}...")
    try:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        
        data_from_mongo = list(collection.find())
        
        if len(data_from_mongo) > 0:
            df = pd.DataFrame(data_from_mongo)
            print(f"✅ Dataset loaded from MongoDB ({len(df)} records).")

            required_cols = [
                'name_en', 'description_en', 'shortdescription_en', 'learningoutcomes_en', 'module_tags_en', 
                'name_nl', 'description_nl', 'module_tags_nl', # <--- TOEGEVOEGD
                'studycredit', 'location'
            ]
            
            for col in required_cols:
                if col not in df.columns:
                    df[col] = None

            # 2. Vul lege waardes op (zowel EN als NL)
            text_cols = ['description_en', 'shortdescription_en', 'learningoutcomes_en', 'name_en', 'description_nl', 'name_nl']
            df[text_cols] = df[text_cols].fillna('')
            
            # 3. Maak de Slimme Context (Hier plakken we NL en EN aan elkaar)
            if 'ai_context' not in df.columns:
                def clean_tags(tags):
                    try: 
                        from ast import literal_eval
                        if isinstance(tags, list): return ", ".join(tags)
                        return ", ".join(literal_eval(tags))
                    except: return str(tags)
                
                # Tags opschonen (indien aanwezig)
                df['clean_tags_en'] = df['module_tags_en'].apply(clean_tags)
                df['clean_tags_nl'] = df['module_tags_nl'].apply(clean_tags) # <--- OOK NL TAGS

                # DE BELANGRIJKSTE REGEL:
                df['ai_context'] = (
                    "Title (EN): " + df['name_en'] + ". " +
                    "Titel (NL): " + df['name_nl'] + ". " +
                    "Tags: " + df['clean_tags_en'] + ", " + df['clean_tags_nl'] + ". " +
                    "Description (EN): " + df['shortdescription_en'] + " " + df['description_en'] + ". " +
                    "Beschrijving (NL): " + df['description_nl']
                )
            
            # 3. Generate Embeddings
            if model:
                print("Generating embeddings for database...")
                module_embeddings = model.encode(df['ai_context'].tolist(), convert_to_tensor=True)
                print("✅ Embeddings ready.")
        else:
            print("⚠️ Warning: MongoDB collection is empty.")
            df = None

    except Exception as e:
        print(f"❌ Error connecting to MongoDB: {e}")
        df = None

# Run startup load
load_data_and_model()


# --- API DEFINITIONS ---

class StudentInput(BaseModel):
    description: str         
    preferred_location: str | None = None
    current_ects: int

@app.post("/predict")
def predict_study(student: StudentInput):
    # 1. Error Handling: Check initialization
    if model is None:
        raise HTTPException(status_code=503, detail="AI Model failed to load. Check server logs.")
    if df is None or module_embeddings is None:
        raise HTTPException(status_code=503, detail="Database data not available. Check MongoDB connection.")

    # 2. Embedding (Multilingual handling implicitly done by model or strictly English)
    # Note: 'all-MiniLM-L6-v2' is English trained. Dutch input might yield lower quality matches 
    # unless translated. For now, we assume input is compatible or user accepts this limitation.
    student_embedding = model.encode(student.description, convert_to_tensor=True)

    # 3. Semantic Matching
    cosine_scores = util.cos_sim(student_embedding, module_embeddings)[0]
    scores = cosine_scores.cpu().numpy()

    # 4. Business Logic: Location Bonus
    if student.preferred_location:
        # Case insensitive match
        loc_matches = df['location'].str.contains(student.preferred_location, case=False, na=False)
        scores[loc_matches] += 0.15

    # 5. Business Logic: ECTS Penalty
    if student.current_ects:
        # Convert to numeric, coerce errors to NaN then 0
        study_credits = pd.to_numeric(df['studycredit'], errors='coerce').fillna(0)
        ects_diff = abs(study_credits - student.current_ects)
        scores -= (ects_diff.to_numpy() * 0.01)

    # 6. Retrieve Top Results
    top_k = 5
    # Get indices of top k scores
    top_indices = scores.argsort()[-top_k:][::-1]

    results = []
    
    for idx in top_indices:
        idx = int(idx) # ensure python int
        row = df.iloc[idx]
        score = float(scores[idx])
        
        # --- Explainable AI Logic (The 'Why') ---
        # We split the module text into sentences and find the best matching sentence
        full_text = f"{row['shortdescription_en']} {row['description_en']}"
        sentences = nltk.sent_tokenize(full_text)
        
        ai_reason = "Match based on general profile overlap."
        if sentences:
            # Encode sentences to find the specific trigger
            sent_embeddings = model.encode(sentences, convert_to_tensor=True)
            sent_scores = util.cos_sim(student_embedding, sent_embeddings)[0]
            best_sent_idx = sent_scores.argmax().item()
            best_sentence = sentences[best_sent_idx].strip()
            # Truncate if too long
            if len(best_sentence) > 200:
                best_sentence = best_sentence[:197] + "..."
            ai_reason = f'💡 AI-Inzicht: "...{best_sentence}..."'

        # Location Check formatted string
        loc_str = str(row['location'])
        if student.preferred_location and student.preferred_location.lower() in loc_str.lower():
            loc_check = "✅ Locatie match"
        else:
            loc_check = f"📍 in {loc_str}"

        results.append({
            "ID": str(row['_id']),
            "Module Naam": row['name_en'],
            "Score": round(score, 2),
            "AI_Reden": ai_reason,
            "Details": {
                "ects": int(row['studycredit']) if pd.notna(row['studycredit']) else 0,
                "location": loc_str
            }
        })

    return {
        "aanbevelingen": results,
        "aantal_resultaten": len(results),
        "status": "success"
    }

@app.post("/refresh-data")
def refresh_data():
    try:
        load_data_and_model() 
        return {"status": "success", "message": "Database reloaded and embeddings updated."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
def run_training_and_reload():
    """
    Deze functie draait op de achtergrond.
    1. Hij start de training (zwaar werk).
    2. Als dat klaar is, herlaadt hij de API data direct.
    """
    print("⏳ Achtergrondtaak gestart: Model trainen...")
    try:
        # Stap 1: Trainen (dit kan minuten duren)
        train_and_save_model() 
        
        # Stap 2: Data verversen (zodat de API direct het nieuwe model gebruikt)
        print("🔄 Training klaar. API data herladen...")
        load_data_and_model()
        print("✅ Alles geüpdatet!")
        
    except Exception as e:
        print(f"❌ Fout tijdens achtergrond training: {e}")

@app.post("/train")
def trigger_training(background_tasks: BackgroundTasks):
    """
    Endpoint om training te starten.
    Geeft direct antwoord, training draait op de achtergrond.
    """
    # Voeg de taak toe aan de wachtrij
    background_tasks.add_task(run_training_and_reload)
    
    return {
        "status": "accepted", 
        "message": "Training is gestart op de achtergrond. Dit kan enkele minuten duren. Check de server logs voor voortgang."
    }