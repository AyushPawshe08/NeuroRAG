"""
Central configuration for the Brain Tumor RAG system.
Reads from environment variables (see .env.example).
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- Dataset ---
# Root folder containing class subfolders: glioma/, meningioma/, notumor/, pituitary/
DATASET_DIR = os.getenv("DATASET_DIR", "./data/train")

# --- BioMedCLIP ---
BIOMEDCLIP_MODEL = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
EMBED_DIM = 512  # BioMedCLIP image embedding dimension

# --- Qdrant ---
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)  # only needed for Qdrant Cloud
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "brain_mri_cases")

# --- Gemini ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")  # update later if you want a different one

# --- Retrieval ---
TOP_K = int(os.getenv("TOP_K", "5"))