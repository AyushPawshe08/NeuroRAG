import os
from dotenv import load_dotenv

load_dotenv()

DATASET_DIR = os.getenv("DATASET_DIR", "./data/train")
BIOMEDCLIP_MODEL = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
EMBED_DIM = 512
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "brain_mri_cases")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
TOP_K = int(os.getenv("TOP_K", "5"))