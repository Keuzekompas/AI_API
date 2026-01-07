from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
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

# --- CORS MIDDLEWARE ---
# This allows the frontend (running on a different domain) to talk to the API.
origins = [
    "*", # Allow all origins for development
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allow all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"], # Allow all headers
)

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
    current_ects: int | None = None
    tags: list[str] = []

@app.post("/api/predict")
def predict_study(student: StudentInput, language: str = "NL"):
    # 1. Error Handling: Check initialization
    if model is None:
        raise HTTPException(status_code=503, detail="AI Model failed to load. Check server logs.")
    if df is None or module_embeddings is None:
        raise HTTPException(status_code=503, detail="Database data not available. Check MongoDB connection.")

    # 2. Hard Filter: ECTS
    # If student has a specific ECTS requirement (15 or 30), we strictly filter the database.
    filtered_df = df.copy()
    if student.current_ects:
        # Ensure studycredit is numeric for comparison, handling potential data issues
        filtered_df['studycredit_num'] = pd.to_numeric(filtered_df['studycredit'], errors='coerce').fillna(0)
        filtered_df = filtered_df[filtered_df['studycredit_num'] == student.current_ects]
        
        if filtered_df.empty:
             return {
                "recommendations": [],
                "number_of_results": 0,
                "status": "success",
                "message": "No modules found with this ECTS."
            }

    # 3. Embedding
    # We voegen de tags toe aan de input text voor een betere match
    tags_text = ", ".join(student.tags) if student.tags else ""
    input_text = f"Tags: {tags_text}. Description: {student.description}"
    
    student_embedding = model.encode(input_text, convert_to_tensor=True)

    # 4. Semantic Matching
    # We moeten nu de embeddings ophalen die horen bij de gefilterde rows.
    # Omdat module_embeddings overeenkomt met de originele df index, is het makkelijker 
    # om de gefilterde subset opnieuw te encoderen OF de indices te mappen.
    # Gezien de dataset klein is (<1000?), is filteren en slicen van embeddings prima, 
    # maar embeddings zijn tensor objects.
    
    # Slimme aanpak: We gebruiken de indices van de filtered_df om de juiste embeddings te pakken.
    # df.index is de originele index.
    filtered_indices = filtered_df.index.tolist()
    
    # Selecteer de juiste embeddings uit de grote tensor
    relevant_embeddings = module_embeddings[filtered_indices]
    
    cosine_scores = util.cos_sim(student_embedding, relevant_embeddings)[0]
    scores = cosine_scores.cpu().numpy()

    # 5. Business Logic: Location Bonus
    if student.preferred_location and student.preferred_location != "Geen":
        # Case insensitive match
        loc_matches = filtered_df['location'].str.contains(student.preferred_location, case=False, na=False)
        scores[loc_matches] += 0.15

    # 5b. Business Logic: Explicit Tag Boost
    # Check if user tags appear in the module's tags (NL or EN).
    if student.tags:
        # Normalize user tags for comparison (lowercase)
        user_tags = [t.lower() for t in student.tags]
        
        # Function to check overlap
        def calculate_tag_boost(row):
            boost = 0.0
            # Combine all tags from row
            row_tags = []
            
            # Helper to safely extract list or string
            def extract(val):
                if isinstance(val, list): return val
                if isinstance(val, str):
                    try:
                        import ast
                        res = ast.literal_eval(val)
                        if isinstance(res, list): return res
                        return [val]
                    except:
                        if "," in val: return [x.strip() for x in val.split(",")]
                        return [val]
                return []

            row_tags.extend(extract(row.get('module_tags_en', [])))
            row_tags.extend(extract(row.get('module_tags_nl', [])))
            
            # Normalize row tags
            row_tags_lower = [str(t).lower() for t in row_tags]
            
            # Check for matches
            for ut in user_tags:
                if any(ut in rt for rt in row_tags_lower): # Partial match allowed (e.g. 'sport' in 'top sport')
                    boost += 0.05 # 5% boost per matching tag category
            
            return min(boost, 0.20) # Max 20% boost from tags

        # Apply boost
        tag_boosts = filtered_df.apply(calculate_tag_boost, axis=1).to_numpy()
        scores += tag_boosts

    # 6. Retrieve Top Results
    top_k = min(5, len(filtered_df))
    top_indices_local = scores.argsort()[-top_k:][::-1] # Dit zijn indices in de filtered lijst (0 tot len(filtered))

    results = []
    
    # Bepaal welke kolommen we teruggeven op basis van taal
    lang_suffix = "_en" if language.upper() == "EN" else "_nl"
    
    for local_idx in top_indices_local:
        local_idx = int(local_idx)
        # Haal de echte row op
        row = filtered_df.iloc[local_idx]
        score = float(scores[local_idx])
        
        # --- Explainable AI Logic (The 'Why') ---
        # Haal de beschrijving op in de juiste taal voor de uitleg
        desc_col = f'description{lang_suffix}'
        short_desc_col = f'shortdescription{lang_suffix}'
        
        # Fallback naar Engels als NL leeg is (of andersom)
        description_text = row.get(desc_col, "")
        if not description_text:
             description_text = row.get('description_en', "")
        
        short_description_text = row.get(short_desc_col, "")
        if not short_description_text:
             short_description_text = row.get('shortdescription_en', "")

        full_text_for_reason = f"{short_description_text} {description_text}"
        sentences = nltk.sent_tokenize(full_text_for_reason)
        
        ai_reason = "Match based on general profile overlap."
        if language.upper() == "NL":
             ai_reason = "Match op basis van je profiel."

        if sentences:
            sent_embeddings = model.encode(sentences, convert_to_tensor=True)
            sent_scores = util.cos_sim(student_embedding, sent_embeddings)[0]
            best_sent_idx = sent_scores.argmax().item()
            best_sentence = sentences[best_sent_idx].strip()
            
            if len(best_sentence) > 200:
                best_sentence = best_sentence[:197] + "..."
            
            if language.upper() == "NL":
                ai_reason = f'💡 AI-Inzicht: "...{best_sentence}..."'
            else:
                ai_reason = f'💡 AI-Insight: "...{best_sentence}..."'

        # Location Check
        loc_str = str(row['location'])
        if student.preferred_location and student.preferred_location.lower() in loc_str.lower():
            loc_check = "✅ Locatie match" if language.upper() == "NL" else "✅ Location match"
        else:
            loc_check = f"📍 in {loc_str}"

        # Naam ophalen
        name_col = f'name{lang_suffix}'
        module_name = row.get(name_col, row['name_en']) # fallback naar EN

        results.append({
            "ID": str(row['_id']),
            "Module_Name": module_name,
            "Description": short_description_text if short_description_text else description_text[:100] + "...",
            "Score": round(score, 2),
            "AI_Reason": ai_reason,
            "Details": {
                "ects": int(row['studycredit']) if pd.notna(row['studycredit']) else 0,
                "location": loc_str
            }
        })

    return {
        "recommendations": results,
        "number_of_results": len(results),
        "status": "success"
    }

@app.post("/api/refresh-data")
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

@app.post("/api/train")
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