import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from dotenv import load_dotenv
import os
import joblib

# Load settings
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "KeuzeKompas")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "modules")

def _fetch_data_from_db():
    print("Connecting to MongoDB...")
    try:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        data = list(collection.find())
        if not data:
            print("No data found in MongoDB.")
            return None
        return data
    except PyMongoError as e:
        print(f"Database error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error during data retrieval: {e}")
        return None

def _preprocess_data(data):
    df = pd.DataFrame(data)
    print(f"✅ {len(df)} modules loaded from database.")

    required_cols = [
        'module_tags_en', 'module_tags_nl', 
        'shortdescription_en', 'description_en', 'name_en', 
        'shortdescription_nl', 'description_nl', 'name_nl'
    ]
    
    for col in required_cols:
        if col not in df.columns: df[col] = ""
    
    df[required_cols] = df[required_cols].fillna('')

    def clean_tags(tags):
        try:
            from ast import literal_eval
            if isinstance(tags, list): return ", ".join(tags)
            return ", ".join(literal_eval(tags))
        except (ValueError, SyntaxError, TypeError):
            return str(tags)
        except Exception:
            return str(tags)

    df['clean_tags_en'] = df['module_tags_en'].apply(clean_tags)
    df['clean_tags_nl'] = df['module_tags_nl'].apply(clean_tags)
    return df

def _process_row(row):
    examples = []
    # --- ENGELS ---
    tags_en = str(row['clean_tags_en'])
    desc_en = (str(row['shortdescription_en']) + " " + str(row['description_en'])).strip()
    title_en = str(row['name_en']).strip()

    if len(tags_en) > 2 and len(desc_en) > 10:
        examples.append(InputExample(texts=[tags_en, desc_en]))
    if len(title_en) > 2 and len(desc_en) > 10:
        examples.append(InputExample(texts=[title_en, desc_en]))

    # --- NEDERLANDS ---
    tags_nl = str(row['clean_tags_nl'])
    if len(tags_nl) < 2: 
        tags_nl = tags_en
        
    desc_nl = (str(row['shortdescription_nl']) + " " + str(row['description_nl'])).strip()
    title_nl = str(row['name_nl']).strip()
    
    if len(desc_nl) > 10:
        if len(tags_nl) > 2:
                examples.append(InputExample(texts=[tags_nl, desc_nl]))
        if len(title_nl) > 2:
                examples.append(InputExample(texts=[title_nl, desc_nl]))
    return examples

def _create_training_examples(df):
    train_examples = []
    print(" Preparing training data (only EN and NL)...")
    
    for _, row in df.iterrows():
        train_examples.extend(_process_row(row))
    
    return train_examples

def _train_model(model, train_examples):
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16, num_workers=0)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    print(f"Start Fine-Tuning on {len(train_examples)} pairs (CPU mode)...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=3,
        warmup_steps=10,
        show_progress_bar=True
    )
    return model

def _save_model(model):
    output_path = os.path.join(os.path.dirname(__file__), 'model', 'model.joblib')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Saving model to {output_path}...")
    joblib.dump(model, output_path)
    print("Training complete! Restart the API or wait for auto-reload.")

def train_and_save_model():
    data = _fetch_data_from_db()
    if not data:
        return

    df = _preprocess_data(data)
    
    model_name = 'paraphrase-multilingual-MiniLM-L12-v2' 
    print(f"Loading base model: {model_name}...")
    model = SentenceTransformer(model_name)

    train_examples = _create_training_examples(df)
    if not train_examples:
        print("Not enough data to train.")
        return

    print(f"Generated {len(train_examples)} training pairs.")
    
    trained_model = _train_model(model, train_examples)
    _save_model(trained_model)

if __name__ == "__main__":
    train_and_save_model()