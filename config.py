#données opendatasoft

import os 
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

#filtre événements récents Nantes
HISTORY_DAYS = 365
GEO_CITY = os.getenv("GEO_CITY", "Nantes").strip() 
GEO_REGION = os.getenv("GEO_REGION", "Pays de la Loire").strip()

OPENDATASOFT_API_URL = (
    "https://public.opendatasoft.com/api/explore/v2.1"
    "/catalog/datasets/evenements-publics-openagenda/records"
)

#API_KEY = os.getenv("API_KEY", "").strip()  # API publique, pas besoin d'authentification pour accéder aux données publiques.


#chemin sauvegarde CSV
DATA_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "evenements_nantes.csv"
)