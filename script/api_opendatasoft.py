"""Récupère les événements OpenAgenda publiés sur OpenDataSoft : python -m script.api_opendatasoft"""

from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from config import (
    API_MAX_RECORDS,
    API_PAGE_SIZE,
    API_TIMEOUT,
    DATA_PATH,
    GEO_CITY,
    HISTORY_DAYS,
    OPENDATASOFT_API_URL,
)



EVENT_FIELDS = (
    "title_fr,description_fr,conditions_fr,location_name,location_address,"
    "location_city,firstdate_begin,lastdate_end,age_min,age_max,canonicalurl,uid,"
    "timings"  # liste JSON des séances : [{"begin": ..., "end": ...}, ...]
)


def fetch_events(city=GEO_CITY, history_days=HISTORY_DAYS):
    """Télécharge toutes les pages correspondant à la ville et à la période."""
    # La borne porte sur la date de fin : événements à venir et événements
    # terminés depuis moins d'un an sont ainsi inclus.
    date_limit = (datetime.now(timezone.utc) - timedelta(days=history_days)).date()
    filters = [
        f"location_city='{city}'",
        f"lastdate_end >= '{date_limit.isoformat()}'",
    ]

    events = []
    for offset in range(0, API_MAX_RECORDS, API_PAGE_SIZE):
        params = {
            "lang": "fr",
            "limit": API_PAGE_SIZE,
            "offset": offset,
            "where": " AND ".join(filters),
            "order_by": "lastdate_end ASC",
            "select": EVENT_FIELDS,
        }
        # Chaque requête a un délai maximal pour éviter de bloquer indéfiniment.
        response = requests.get(
            OPENDATASOFT_API_URL, params=params, timeout=API_TIMEOUT
        )
        response.raise_for_status()
        page = response.json().get("results", [])
        if not page:
            break

        events.extend(page)

        # Une page incomplète indique qu'il n'y a plus de résultats à lire.
        if len(page) < API_PAGE_SIZE:
            print(f"{len(events)} événements récupérés")
            break

    return pd.DataFrame(events)


def save_events(data, output_path=DATA_PATH):
    """Enregistre les données brutes en CSV avec les accents préservés."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"{len(data)} événements enregistrés dans {output_path}")


def main():
    """Lance le téléchargement puis l'enregistrement du CSV brut."""
    events = fetch_events()
    print(f"Ville : {GEO_CITY} | Total récupéré : {len(events)} événement(s)")
    save_events(events)


if __name__ == "__main__":
    main()
