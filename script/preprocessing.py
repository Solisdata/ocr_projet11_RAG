"""Nettoie le CSV brut des événements : python -m script.preprocessing"""

import logging

import pandas as pd

from config import CLEANED_DATA_PATH, DATA_PATH, GEO_CITY, HISTORY_DAYS

logger = logging.getLogger(__name__)

COLONNES_OBLIGATOIRES = (
    "uid",
    "title_fr",
    "firstdate_begin",
    "lastdate_end",
    "location_city",
)


# ---------------------------------------------------------------------------
# Âges
# ---------------------------------------------------------------------------

def format_age(value):
    """Transforme 6.0 en "6" et une valeur vide en ""."""
    if pd.isna(value) or not str(value).strip():
        return ""
    try:
        age = float(value)
        if age.is_integer():
            return str(int(age))
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def format_age_range(age_min, age_max):
    """Construit un libellé lisible : "6 à 12 ans", "À partir de 6 ans"..."""
    age_min = format_age(age_min)
    age_max = format_age(age_max)
    if age_min and age_max:
        return f"{age_min} à {age_max} ans"
    if age_min:
        return f"À partir de {age_min} ans"
    if age_max:
        return f"Jusqu'à {age_max} ans"
    return ""


# ---------------------------------------------------------------------------
# Étapes du nettoyage
# ---------------------------------------------------------------------------

def verifier_colonnes(data):
    """Arrête le traitement si une colonne indispensable manque."""
    manquantes = [colonne for colonne in COLONNES_OBLIGATOIRES if colonne not in data]
    if manquantes:
        raise ValueError("Colonnes indispensables absentes : " + ", ".join(manquantes))


def normaliser_ages(data):
    """Uniformise age_min / age_max et ajoute la colonne age_range."""
    for colonne in ("age_min", "age_max"):
        if colonne not in data:
            data[colonne] = ""
        data[colonne] = data[colonne].map(format_age)
    data["age_range"] = [
        format_age_range(age_min, age_max)
        for age_min, age_max in zip(data["age_min"], data["age_max"])
    ]
    return data


def nettoyer_texte(data):
    """Supprime les espaces en trop dans toutes les colonnes de texte."""
    for colonne in data.columns:
        if pd.api.types.is_string_dtype(data[colonne].dtype):
            data[colonne] = (
                data[colonne]
                .astype("string")
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
                .fillna("")
            )

    # Un uid lu comme nombre (101.0) devient le texte "101".
    data["uid"] = data["uid"].map(
        lambda uid: str(int(uid)) if isinstance(uid, float) and uid.is_integer() else uid
    )
    data["uid"] = data["uid"].astype("string").str.strip().fillna("")
    return data


def supprimer_incomplets(data):
    """Écarte les événements sans identifiant, sans titre ou sans ville."""
    complets = (data["uid"] != "") & (data["title_fr"] != "") & (data["location_city"] != "")
    logger.info("%s événements écartés pour titre ou ville manquante.", (~complets).sum())
    return data[complets]


def supprimer_dates_invalides(data):
    """Écarte les dates illisibles et les événements qui finissent avant de commencer."""
    debut = pd.to_datetime(data["firstdate_begin"], errors="coerce", utc=True)
    fin = pd.to_datetime(data["lastdate_end"], errors="coerce", utc=True)

    lisibles = debut.notna() & fin.notna()
    inversees = lisibles & (fin < debut)
    logger.info(
        "%s événements écartés pour dates invalides, %s dates inversées.",
        (~lisibles).sum(),
        inversees.sum(),
    )
    return data[lisibles & ~inversees]


def garder_ville_et_periode(data, city=GEO_CITY, history_days=HISTORY_DAYS):
    """Garde la ville choisie et les événements terminés depuis moins d'un an.

    L'API applique déjà ces filtres : on les revérifie ici par sécurité.
    """
    bonne_ville = data["location_city"].str.casefold() == city.strip().casefold()
    date_limite = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=history_days)
    fin = pd.to_datetime(data["lastdate_end"], utc=True)
    dans_periode = fin >= date_limite

    a_garder = bonne_ville & dans_periode
    logger.info(
        "%s événements écartés car hors de la ville '%s' ou trop anciens.",
        (~a_garder).sum(),
        city,
    )
    return data[a_garder]


def supprimer_doublons(data):
    """Garde un seul événement par identifiant."""
    nb_doublons = data.duplicated(subset="uid").sum()
    logger.info("%s doublon(s) trouvé(s) et supprimé(s).", nb_doublons)
    return data.drop_duplicates(subset="uid", keep="first")


# ---------------------------------------------------------------------------
# Enchaînement
# ---------------------------------------------------------------------------

def preprocess_data(data, city=GEO_CITY, history_days=HISTORY_DAYS):
    """Applique toutes les étapes du nettoyage, dans l'ordre."""
    logger.info("Début du prétraitement : %s événements reçus.", len(data))
    verifier_colonnes(data)

    cleaned = data.copy()
    cleaned = normaliser_ages(cleaned)
    cleaned = nettoyer_texte(cleaned)
    cleaned = supprimer_incomplets(cleaned)
    cleaned = supprimer_dates_invalides(cleaned)
    cleaned = garder_ville_et_periode(cleaned, city, history_days)
    cleaned = supprimer_doublons(cleaned)

    logger.info(
        "Prétraitement terminé : %s événement(s) conservé(s) sur %s.",
        len(cleaned),
        len(data),
    )
    return cleaned.reset_index(drop=True)


def preprocess_csv(input_path=DATA_PATH, output_path=CLEANED_DATA_PATH):
    """Lit le CSV brut, le nettoie et enregistre le CSV nettoyé."""
    data = pd.read_csv(input_path, encoding="utf-8-sig")
    cleaned = preprocess_data(data)
    if cleaned.empty:
        raise ValueError(f"Aucun événement valide à enregistrer depuis {input_path}.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info("CSV nettoyé enregistré : %s (%s événements).", output_path, len(cleaned))
    return cleaned


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    preprocess_csv()
