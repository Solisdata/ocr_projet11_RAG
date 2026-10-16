
# lancer les tests : python -m pytest tests/test_data.py -v

from datetime import datetime, timedelta

import pandas as pd
from config import DATA_PATH, GEO_CITY, HISTORY_DAYS

# Critères attendus


def load_data():
    """Charge les événements depuis le CSV."""
    assert DATA_PATH.exists(), f"CSV introuvable : {DATA_PATH}"
    df = pd.read_csv(DATA_PATH)
    assert not df.empty, "Le CSV ne contient aucun événement."
    return df


def test_required_columns():
    """Vérifie la présence des colonnes indispensables."""
    df = load_data()

    required_columns = [
        "uid",
        "title_fr",
        "firstdate_begin",
        "lastdate_end",
        "location_city",
    ]

    for column in required_columns:
        assert column in df.columns, f"Colonne manquante : {column}"


def test_titles_are_present():
    """Vérifie que les événements ont un titre."""
    df = load_data()

    titles = df["title_fr"].fillna("").astype(str).str.strip()

    assert (titles != "").all(), "Certains événements n'ont pas de titre."


def test_dates_are_valid():
    """Vérifie que les dates sont présentes et interprétables."""
    df = load_data()

    start_dates = pd.to_datetime(
        df["firstdate_begin"], errors="coerce", utc=True
    )
    end_dates = pd.to_datetime(
        df["lastdate_end"], errors="coerce", utc=True
    )

    assert start_dates.notna().all(), "Des dates de début sont invalides."
    assert end_dates.notna().all(), "Des dates de fin sont invalides."

    assert (end_dates >= start_dates).all(), (
        "Certains événements se terminent avant leur début."
    )


def test_events_are_in_expected_period():
    """Vérifie que les événements ne sont pas trop anciens."""
    df = load_data()

    end_dates = pd.to_datetime(
        df["lastdate_end"], errors="coerce", utc=True
    )

    date_limit = (datetime.now() - timedelta(days=HISTORY_DAYS)).strftime("%Y-%m-%d")

    assert (end_dates.dt.strftime("%Y-%m-%d") >= date_limit).all(), (
        "Certains événements sont terminés depuis plus d'un an."
    )


def test_events_are_in_expected_city():
    """Vérifie que tous les événements sont associés à la ville configurée."""
    df = load_data()

    cities = df["location_city"].fillna("").astype(str).str.strip().str.casefold()

    assert (cities == GEO_CITY.casefold()).all(), (
        f"Certains événements ne sont pas associés à {GEO_CITY} "
        "ou n'ont pas de ville renseignée."
    )


def test_uids_are_present_and_unique():
    """Vérifie que chaque événement possède un identifiant unique."""
    df = load_data()

    uids = df["uid"].fillna("").astype(str).str.strip()

    assert (uids != "").all(), "Certains événements n'ont pas d'identifiant."
    assert uids.is_unique, "Des identifiants d'événements sont dupliqués."
