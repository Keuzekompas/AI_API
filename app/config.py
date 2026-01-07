import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Database
    MONGO_URI = os.getenv("MONGO_URI")
    DB_NAME = os.getenv("DB_NAME")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME")
    
    # Model
    MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'model.joblib')

    # Algorithm Weights & Boosts
    LOCATION_BOOST = 0.10      # 10% bonus for location match
    TAG_BOOST_PER_MATCH = 0.05 # 5% bonus per matching tag
    MAX_TAG_BOOST = 0.15       # Max 15% total bonus from tags

settings = Settings()
