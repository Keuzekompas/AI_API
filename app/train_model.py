import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from pymongo import MongoClient
from dotenv import load_dotenv
import os
import joblib

# Laad settings
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "KeuzeKompas")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "modules")

def train_and_save_model():
    print("🔌 Verbinden met MongoDB...")
    try:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        
        # 1. Data Ophalen
        data = list(collection.find())
    except Exception as e:
        print(f"❌ Database error: {e}")
        return

    if not data:
        print("❌ Geen data gevonden in MongoDB. Training gestopt.")
        return

    df = pd.DataFrame(data)
    print(f"✅ {len(df)} modules geladen uit database.")

    # 2. Data Schoonmaken
    required_cols = [
        'module_tags_en', 'module_tags_nl', 
        'shortdescription_en', 'description_en', 'name_en', 
        'shortdescription_nl', 'description_nl', 'name_nl'
    ]
    
    for col in required_cols:
        if col not in df.columns: df[col] = ""
    
    # Fill NA
    df[required_cols] = df[required_cols].fillna('')

    def clean_tags(tags):
        try:
            from ast import literal_eval
            if isinstance(tags, list): return ", ".join(tags)
            return ", ".join(literal_eval(tags))
        except: return str(tags)

    df['clean_tags_en'] = df['module_tags_en'].apply(clean_tags)
    df['clean_tags_nl'] = df['module_tags_nl'].apply(clean_tags)

    # 3. Model Laden
    # We gebruiken een meertalig model dat goed is in EN en NL.
    # 'paraphrase-multilingual-MiniLM-L12-v2' is compact en snel.
    model_name = 'paraphrase-multilingual-MiniLM-L12-v2' 
    print(f"🧠 Basismodel laden: {model_name}...")
    model = SentenceTransformer(model_name)

    # 4. Trainingsdata Maken
    train_examples = []
    print("🏋️  Trainingsdata voorbereiden (alleen EN en NL)...")
    
    for index, row in df.iterrows():
        # --- ENGELS ---
        tags_en = str(row['clean_tags_en'])
        desc_en = (str(row['shortdescription_en']) + " " + str(row['description_en'])).strip()
        title_en = str(row['name_en']).strip()

        if len(tags_en) > 2 and len(desc_en) > 10:
            train_examples.append(InputExample(texts=[tags_en, desc_en]))
        if len(title_en) > 2 and len(desc_en) > 10:
            train_examples.append(InputExample(texts=[title_en, desc_en]))

        # --- NEDERLANDS ---
        tags_nl = str(row['clean_tags_nl'])
        # Fallback naar EN tags als NL tags leeg zijn, want tags zijn vaak universeel
        if len(tags_nl) < 2: 
            tags_nl = tags_en
            
        desc_nl = (str(row['shortdescription_nl']) + " " + str(row['description_nl'])).strip()
        title_nl = str(row['name_nl']).strip()
        
        # Alleen toevoegen als er daadwerkelijk NL content is
        if len(desc_nl) > 10:
            if len(tags_nl) > 2:
                 train_examples.append(InputExample(texts=[tags_nl, desc_nl]))
            if len(title_nl) > 2:
                 train_examples.append(InputExample(texts=[title_nl, desc_nl]))

    if not train_examples:
        print("❌ Te weinig data om te trainen.")
        return

    print(f"Generated {len(train_examples)} training pairs.")

    # 5. Training Starten
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    print(f"🚀 Start Fine-Tuning op {len(train_examples)} paren (CPU mode)...")
    # Epochs iets verlaagd naar 3 voor snelheid, is vaak genoeg voor fine-tuning
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=3,
        warmup_steps=10,
        show_progress_bar=True
    )

    # 6. Model Opslaan
    output_path = os.path.join(os.path.dirname(__file__), 'model', 'model.joblib')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"💾 Model opslaan naar {output_path}...")
    joblib.dump(model, output_path)
    print("✅ Training klaar! Herstart de API of wacht op auto-reload.")

if __name__ == "__main__":
    train_and_save_model()