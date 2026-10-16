"""Paramètres partagés par les étapes de récupération et d'indexation."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# La période conserve les événements terminés récemment et ceux à venir.
HISTORY_DAYS = 365
GEO_CITY = os.getenv("GEO_CITY", "Nantes").strip()
GEO_REGION = os.getenv("GEO_REGION", "Pays de la Loire").strip()
# Réglages des appels paginés à l'API OpenDataSoft.
API_TIMEOUT = 15
API_PAGE_SIZE = 100
API_MAX_RECORDS = 10000

OPENDATASOFT_API_URL = (
    "https://public.opendatasoft.com/api/explore/v2.1"
    "/catalog/datasets/evenements-publics-openagenda/records"
)

# Les chemins sont construits depuis la racine du projet, quel que soit le dossier courant.
PROJECT_PATH = Path(__file__).resolve().parent
DATA_PATH = PROJECT_PATH / "data" / "evenements_nantes.csv"
CLEANED_DATA_PATH = PROJECT_PATH / "data" / "evenements_nantes_clean.csv"
FAISS_INDEX_PATH = PROJECT_PATH / "data" / "faiss_index"
