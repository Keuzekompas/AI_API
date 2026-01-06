import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from pymongo import MongoClient
from dotenv import load_dotenv
import os
import joblib # Zorg dat joblib geïmporteerd is

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
    for col in ['module_tags_en', 'shortdescription_en', 'description_en', 'name_en', 'name_nl', 'description_nl']:
        if col not in df.columns: df[col] = ""
    
    # Fill NA
    text_cols = ['shortdescription_en', 'description_en', 'name_en', 'name_nl', 'description_nl']
    df[text_cols] = df[text_cols].fillna('')

    def clean_tags(tags):
        try:
            from ast import literal_eval
            if isinstance(tags, list): return ", ".join(tags)
            return ", ".join(literal_eval(tags))
        except: return str(tags)

    df['clean_tags'] = df['module_tags_en'].apply(clean_tags)

    # 3. Model Laden
    # We laten 'device' weg, SentenceTransformer pakt automatisch CPU (of CUDA als je ooit Nvidia koopt)
    model_name = 'paraphrase-multilingual-MiniLM-L12-v2' 
    print(f"🧠 Basismodel laden: {model_name}...")
    model = SentenceTransformer(model_name)

    # 4. Trainingsdata Maken
    train_examples = []
    print("🏋️  Trainingsdata voorbereiden...")
    
    for index, row in df.iterrows():
        # Engels
        tags = str(row['clean_tags'])
        desc_en = str(row['shortdescription_en']) + " " + str(row['description_en'])
        title_en = str(row['name_en'])

        if len(tags) > 2 and len(desc_en) > 10:
            train_examples.append(InputExample(texts=[tags, desc_en]))
        if len(title_en) > 2 and len(desc_en) > 10:
            train_examples.append(InputExample(texts=[title_en, desc_en]))

        # Nederlands (probeer data te pakken als het er is)
        desc_nl = str(row.get('description_nl', ''))
        title_nl = str(row.get('name_nl', ''))
        
        if len(tags) > 2 and len(desc_nl) > 10:
             train_examples.append(InputExample(texts=[tags, desc_nl]))
        if len(title_nl) > 2 and len(desc_nl) > 10:
             train_examples.append(InputExample(texts=[title_nl, desc_nl]))

    if not train_examples:
        print("❌ Te weinig data om te trainen.")
        return

    # 5. Training Starten
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    print(f"🚀 Start Fine-Tuning op {len(train_examples)} paren (CPU mode)...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=4,
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