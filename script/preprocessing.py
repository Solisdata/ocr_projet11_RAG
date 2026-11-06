#Nettoie le CSV brut des événements pour les étapes suivantes du RAG
# Lancer le script : python -m script.preprocessing

import logging
from pathlib import Path

import pandas as pd

from config import CLEANED_DATA_PATH, DATA_PATH, GEO_CITY, HISTORY_DAYS

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = (
    "uid",
    "title_fr",
    "firstdate_begin",
    "lastdate_end",
    "location_city",
)

def format_age(value) -> str:
    """Normalise une borne d'âge et convertit les valeurs absentes en chaîne vide."""
    if pd.isna(value) or not str(value).strip():
        return ""
    try:
        numeric_age = float(value)
        if numeric_age.is_integer():
            return str(int(numeric_age))
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def format_age_range(age_min, age_max) -> str:
    """Construit le libellé d'âge conseillé à partir des bornes disponibles."""
    age_min = format_age(age_min)
    age_max = format_age(age_max)
    if age_min and age_max:
        return f"{age_min} à {age_max} ans"
    if age_min:
        return f"À partir de {age_min} ans"
    if age_max:
        return f"Jusqu'à {age_max} ans"
    return ""


def preprocess_data(
    data: pd.DataFrame,
    city: str = GEO_CITY,
    history_days: int = HISTORY_DAYS,
) -> pd.DataFrame:
    """Nettoie les champs et écarte les événements incomplets ou invalides."""
    logger.info("Début du prétraitement : %s événements reçus.", len(data))

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data]
    if missing_columns:
        raise ValueError(
            "Colonnes indispensables absentes : " + ", ".join(missing_columns)
        )

    cleaned = data.copy()

    # Normalise les âges ici pour que les CSV prétraités soient prêts à indexer.
    for column in ("age_min", "age_max"):
        if column not in cleaned:
            cleaned[column] = ""
        cleaned[column] = cleaned[column].map(format_age)
    cleaned["age_range"] = [
        format_age_range(age_min, age_max)
        for age_min, age_max in zip(cleaned["age_min"], cleaned["age_max"])
    ]

    for column in cleaned.columns:
        if pd.api.types.is_string_dtype(cleaned[column].dtype):
            cleaned[column] = (
                cleaned[column]
                .astype("string")
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
                .fillna("")
            )

    if pd.api.types.is_float_dtype(cleaned["uid"].dtype):
        cleaned["uid"] = cleaned["uid"].map(
            lambda value: (
                str(int(value))
                if pd.notna(value) and value % 1 == 0
                else value
            )
        )
    cleaned["uid"] = cleaned["uid"].astype("string").str.strip()
    valid_rows = cleaned["uid"].notna() & cleaned["uid"].ne("")
    for column in ("title_fr", "location_city"):
        valid_rows &= cleaned[column].ne("")
    missing_required_count = int((~valid_rows).sum())
    cleaned = cleaned.loc[valid_rows].copy()
    logger.info(
        "%s événements écartés pour titre ou ville manquante.",
        missing_required_count,
    )

    start_dates = pd.to_datetime(
        cleaned["firstdate_begin"], errors="coerce", utc=True
    )
    end_dates = pd.to_datetime(cleaned["lastdate_end"], errors="coerce", utc=True)
    parsed_dates = start_dates.notna() & end_dates.notna()
    invalid_date_count = int((~parsed_dates).sum())
    reversed_dates = parsed_dates & end_dates.lt(start_dates)
    reversed_date_count = int(reversed_dates.sum())
    valid_dates = parsed_dates & ~reversed_dates
    cleaned = cleaned.loc[valid_dates].copy()
    logger.info(
        "%s événements écartés pour dates invalides, %s dates inversées.",
        invalid_date_count,
        reversed_date_count,
    )

    # La ville est comparée sans tenir compte des majuscules ni espaces en trop.
    city_matches = (
        cleaned["location_city"].astype("string").str.strip().str.casefold()
        == city.strip().casefold()
    )
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=history_days)
    end_dates = pd.to_datetime(cleaned["lastdate_end"], errors="coerce", utc=True)
    in_period = end_dates >= cutoff
    outside_filter_count = int((~(city_matches & in_period)).sum())
    cleaned = cleaned.loc[city_matches & in_period].copy()
    logger.info(
        "%s événements écartés car hors de la ville '%s' ou trop anciens.",
        outside_filter_count,
        city,
    )

    duplicate_count = int(cleaned.duplicated(subset="uid").sum())
    cleaned = cleaned.drop_duplicates(subset="uid", keep="first")
    logger.info("%s doublon(s) trouvé(s) et supprimé(s).", duplicate_count)
    logger.info(
        "Prétraitement terminé : %s événement(s) conservé(s) sur %s.",
        len(cleaned),
        len(data),
    )
    return cleaned.reset_index(drop=True)


def preprocess_csv(
    input_path: Path = DATA_PATH,
    output_path: Path = CLEANED_DATA_PATH,
) -> pd.DataFrame:
    """Lit, nettoie et enregistre les événements sans modifier le fichier source."""
    data = pd.read_csv(input_path, encoding="utf-8-sig")
    logger.info("CSV source chargé : %s (%s événements).", input_path, len(data))
    cleaned = preprocess_data(data)

    if cleaned.empty:
        raise ValueError(f"Aucun événement valide à enregistrer depuis {input_path}.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(
        "CSV nettoyé enregistré : %s (%s événements).",
        output_path,
        len(cleaned),
    )
    return cleaned


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    preprocess_csv()
