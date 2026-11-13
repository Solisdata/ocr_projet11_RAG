"""Transforme les événements en vecteurs Ollama et les enregistre dans FAISS."""

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import (
    CLEANED_DATA_PATH,
    OLLAMA_BASE_URL,
    OLLAMA_EMBEDDING_MODEL,
    OLLAMA_FAISS_INDEX_PATH,
)


# PARAMÈTRES À MODIFIER
# Les valeurs par défaut se trouvent aussi dans config.py et peuvent être
# remplacées dans le fichier .env.
FICHIER_EVENEMENTS = CLEANED_DATA_PATH
MODELE_EMBEDDING = OLLAMA_EMBEDDING_MODEL
ADRESSE_OLLAMA = OLLAMA_BASE_URL
DOSSIER_INDEX = OLLAMA_FAISS_INDEX_PATH
TAILLE_LOT = 32  # Réduire cette valeur si Ollama manque de mémoire.


def load_events(input_path=FICHIER_EVENEMENTS):
    """Étape 1 : lire le CSV nettoyé des événements."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"Fichier absent : {input_path}. Lance d'abord le prétraitement."
        )

    events = pd.read_csv(input_path, encoding="utf-8-sig").fillna("")
    if events.empty:
        raise ValueError(f"Aucun événement à vectoriser dans {input_path}.")
    return events


def _as_text(value):
    """Convertit une valeur CSV en texte sans écrire 'nan' ou 'None'."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def create_event_documents(events):
    """Étape 2 : créer un document LangChain à partir de chaque ligne CSV."""
    documents = []

    for event in events.to_dict(orient="records"):
        # Ces champs forment le texte que le modèle d'embedding va vectoriser.
        age_min = _as_text(event.get("age_min", ""))
        age_max = _as_text(event.get("age_max", ""))
        age_range = _as_text(event.get("age_range", ""))

        champs_du_document = [
            ("Événement", event.get("title_fr", "")),
            ("Description", event.get("description_fr", "")),
            ("Conditions", event.get("conditions_fr", "")),
            ("Lieu", event.get("location_name", "")),
            ("Adresse", event.get("location_address", "")),
            ("Ville", event.get("location_city", "")),
            ("Âge conseillé", age_range),
            (
                "Dates",
                f"{event.get('firstdate_begin', '')} - {event.get('lastdate_end', '')}",
            ),
        ]

        contenu = "\n".join(
            f"{label} : {_as_text(value)}"
            for label, value in champs_du_document
            if _as_text(value).strip(" -")
        )

        # Les métadonnées servent à retrouver le titre, la date ou le lien.
        metadata = {
            "uid": _as_text(event.get("uid", "")),
            "title": _as_text(event.get("title_fr", "")),
            "city": _as_text(event.get("location_city", "")),
            "start_date": _as_text(event.get("firstdate_begin", "")),
            "end_date": _as_text(event.get("lastdate_end", "")),
            "age_min": age_min,
            "age_max": age_max,
            "url": _as_text(event.get("canonicalurl", "")),
        }

        if contenu.strip():
            documents.append(Document(page_content=contenu, metadata=metadata))

    return documents


def create_embeddings():
    """Étape 3 : préparer le modèle Ollama qui transforme le texte en vecteurs."""
    try:
        from langchain_ollama import OllamaEmbeddings
    except ImportError as error:
        raise RuntimeError(
            "Installe les dépendances avec : "
            "python -m pip install -r requirements.txt"
        ) from error

    return OllamaEmbeddings(model=MODELE_EMBEDDING, base_url=ADRESSE_OLLAMA)


def build_vector_store(documents, index_path=DOSSIER_INDEX):
    """Étape 4 : vectoriser les documents et enregistrer l'index FAISS."""
    if not documents:
        raise ValueError("Aucun document valide à vectoriser.")

    embeddings = create_embeddings()
    index = None
    total = len(documents)

    # Envoyer tous les événements d'un coup peut faire planter le service Ollama.
    for debut in range(0, total, TAILLE_LOT):
        fin = min(debut + TAILLE_LOT, total)
        lot = documents[debut:fin]

        try:
            index_du_lot = FAISS.from_documents(lot, embeddings)
        except Exception as error:
            raise RuntimeError(
                f"Ollama a échoué sur les événements {debut + 1} à {fin}. "
                f"Vérifie le modèle {MODELE_EMBEDDING}, puis relance l'indexation."
            ) from error

        if index is None:
            index = index_du_lot
        else:
            index.merge_from(index_du_lot)

        print(f"Embeddings créés : {fin} / {total}", flush=True)

    index_path.mkdir(parents=True, exist_ok=True)
    index.save_local(str(index_path))
    return index


def main():
    """Lancer toutes les étapes d'indexation dans l'ordre."""
    events = load_events()
    documents = create_event_documents(events)
    build_vector_store(documents)
    print(f"{len(documents)} événement(s) indexé(s) dans {DOSSIER_INDEX}")


if __name__ == "__main__":
    main()
