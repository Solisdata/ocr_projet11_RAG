# je réécupère des événements culturels, les lieux, des dates, des descriptions.

import requests
import pandas as pd
import os
from datetime import datetime, timedelta
from config import GEO_CITY, HISTORY_DAYS, OPENDATASOFT_API_URL


#api_key = os.environ.get("API_KEY") -- API publique, pas besoin d'authentification pour accéder aux données publiques.
# headers = {
#     "Authorization": f"Apikey {api_key}",
#     "Content-Type": "application/json"
# 



# 1.Définir la période : un an d'historique
start_date_filter = (
    datetime.now() - timedelta(days=HISTORY_DAYS)
).strftime("%Y-%m-%d")

# 2.Construire les filtres OpenDataSoft
where_clause = (
    f"location_city='{GEO_CITY}' AND "
    f"lastdate_end >= '{start_date_filter}'"
)

 
# 3. Interroger l'API : ne permet que de récupérer 100 événements par requête, donc il faud donc faire une boucle. Récupérer toutes les pages
all_events = []
offset = 0
limit = 100

while offset < 10000:
    params = {
        "lang": "fr",
        "limit": limit,
        "offset": offset,
        "where": where_clause,
        "order_by": "lastdate_end ASC",
    }

    response = requests.get(
        OPENDATASOFT_API_URL,
        params=params,
        timeout=15,
    )

    response.raise_for_status()

    results = response.json().get("results", [])

    if not results:
        break

    all_events.extend(results)

    print(f"{len(all_events)} événements récupérés")

    if len(results) < limit:
        break

    offset += limit

# 4. Transformer les résultats en DataFrame
df = pd.DataFrame(all_events)

# 5. Afficher les résultats
print(f"\nVille : {GEO_CITY}")
print(f"Date de début de l'historique : {start_date_filter}")
print(f"Total récupéré : {len(df)} événements")
print(df.head())

df.to_csv("../data/evenements_nantes.csv", index=False, encoding="utf-8-sig")

print("Fichier CSV sauvegardé !")