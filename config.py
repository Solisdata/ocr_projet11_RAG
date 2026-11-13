"""Paramètres partagés par toutes les étapes du projet.

Les valeurs marquées os.getenv(...) peuvent être remplacées dans le fichier .env.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Données à récupérer ---
GEO_CITY = os.getenv("GEO_CITY", "Nantes").strip()
HISTORY_DAYS = 365  # garde les événements terminés depuis moins d'un an et ceux à venir

# --- API OpenDataSoft ---
OPENDATASOFT_API_URL = (
    "https://public.opendatasoft.com/api/explore/v2.1"
    "/catalog/datasets/evenements-publics-openagenda/records"
)
API_TIMEOUT = 15  # secondes maximum par requête
API_PAGE_SIZE = 100  # événements par page
API_MAX_RECORDS = 10000  # plafond total

# --- Modèles Ollama (locaux) ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "embeddinggemma").strip()
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "qwen3:1.7b").strip()

# --- Fichiers (construits depuis la racine du projet) ---
PROJECT_PATH = Path(__file__).resolve().parent
DATA_PATH = PROJECT_PATH / "data" / "evenements_nantes.csv"
CLEANED_DATA_PATH = PROJECT_PATH / "data" / "evenements_nantes_clean.csv"
FAISS_INDEX_PATH = PROJECT_PATH / "data" / "faiss_index"
