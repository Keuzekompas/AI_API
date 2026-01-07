import pandas as pd
import joblib
import torch
import io
import pickle
import nltk
from pymongo import MongoClient
from ..config import settings
from .state import state

# CPU Unpickler Helper
class CpuUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        return super().find_class(module, name)

def clean_tags_helper(tags):
    try: 
        from ast import literal_eval
        if isinstance(tags, list): return ", ".join(tags)
        return ", ".join(literal_eval(tags))
    except (ValueError, SyntaxError): 
        return str(tags)
    except Exception:
        return str(tags)

def _load_model():
    print(f"Loading model from: {settings.MODEL_PATH}")
    try:
        try:
            state.model = joblib.load(settings.MODEL_PATH)
        except Exception as e:
            if "CUDA" in str(e):
                print("CUDA mismatch detected. Retrying with custom CPU Unpickler...")
                with open(settings.MODEL_PATH, 'rb') as f:
                    state.model = CpuUnpickler(f).load()
            else:
                raise e
        print("✅ Model loaded successfully.")
    except (FileNotFoundError, IsADirectoryError):
        print(f"❌ Error: Model file not found at {settings.MODEL_PATH}")
        state.model = None
    except Exception as e:
        print(f"❌ Critical Error: Could not load model. {e}")
        state.model = None

def _process_dataframe(df):
    required_cols = [
        'name_en', 'description_en', 'shortdescription_en', 'learningoutcomes_en', 'module_tags_en', 
        'name_nl', 'description_nl', 'module_tags_nl',
        'studycredit', 'location'
    ]
    
    for col in required_cols:
        if col not in df.columns:
            df[col] = None

    # Vul lege waardes op
    text_cols = ['description_en', 'shortdescription_en', 'learningoutcomes_en', 'name_en', 'description_nl', 'name_nl']
    df[text_cols] = df[text_cols].fillna('')
    
    # Tags opschonen
    df['clean_tags_en'] = df['module_tags_en'].apply(clean_tags_helper)
    df['clean_tags_nl'] = df['module_tags_nl'].apply(clean_tags_helper)

    # AI Context genereren
    if 'ai_context' not in df.columns:
        df['ai_context'] = (
            "Title (EN): " + df['name_en'] + ". " +
            "Titel (NL): " + df['name_nl'] + ". " +
            "Tags: " + df['clean_tags_en'] + ", " + df['clean_tags_nl'] + ". " +
            "Description (EN): " + df['shortdescription_en'] + " " + df['description_en'] + ". " +
            "Beschrijving (NL): " + df['description_nl']
        )
    return df

def _load_database_data():
    print(f"Connecting to MongoDB: {settings.DB_NAME} -> {settings.COLLECTION_NAME}...")
    try:
        client = MongoClient(settings.MONGO_URI)
        db = client[settings.DB_NAME]
        collection = db[settings.COLLECTION_NAME]
        
        data_from_mongo = list(collection.find())
        
        if len(data_from_mongo) > 0:
            df = pd.DataFrame(data_from_mongo)
            print(f"✅ Dataset loaded from MongoDB ({len(df)} records).")
            state.df = _process_dataframe(df)
            
            # Generate Embeddings
            if state.model:
                print("Generating embeddings for database...")
                state.module_embeddings = state.model.encode(state.df['ai_context'].tolist(), convert_to_tensor=True)
                print("✅ Embeddings ready.")
        else:
            print("⚠️ Warning: MongoDB collection is empty.")
            state.df = None

    except Exception as e:
        print(f"❌ Error connecting to MongoDB: {e}")
        state.df = None

def load_data_and_model():
    _load_model()
    _load_database_data()
