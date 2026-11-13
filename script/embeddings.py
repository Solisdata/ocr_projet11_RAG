"""Transforme les événements en vecteurs et les enregistre dans FAISS : python -m script.embeddings"""

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from config import (
    CLEANED_DATA_PATH,
    FAISS_INDEX_PATH,
    OLLAMA_BASE_URL,
    OLLAMA_EMBEDDING_MODEL,
)

TAILLE_LOT = 32  # événements envoyés à Ollama à la fois (réduire si Ollama manque de mémoire)


def load_events(input_path=CLEANED_DATA_PATH):
    """Étape 1 : lire le CSV nettoyé."""
    if not input_path.exists():
        raise FileNotFoundError(f"Fichier absent : {input_path}. Lance d'abord le prétraitement.")

    events = pd.read_csv(input_path, encoding="utf-8-sig", dtype=str).fillna("")
    if events.empty:
        raise ValueError(f"Aucun événement à vectoriser dans {input_path}.")
    return events


def create_event_documents(events):
    """Étape 2 : créer un document par événement (un événement = un vecteur)."""
    documents = []

    for event in events.fillna("").astype(str).to_dict(orient="records"):
        # Le texte qui sera vectorisé : tout ce qui décrit l'événement.
        champs = [
            ("Événement", event.get("title_fr", "")),
            ("Description", event.get("description_fr", "")),
            ("Conditions", event.get("conditions_fr", "")),
            ("Lieu", event.get("location_name", "")),
            ("Adresse", event.get("location_address", "")),
            ("Ville", event.get("location_city", "")),
            ("Âge conseillé", event.get("age_range", "")),
            ("Dates", f"{event.get('firstdate_begin', '')} - {event.get('lastdate_end', '')}"),
        ]
        contenu = "\n".join(
            f"{label} : {valeur.strip()}" for label, valeur in champs if valeur.strip(" -")
        )
        if not contenu:
            continue

        # Les métadonnées ne sont pas vectorisées : elles servent à afficher les sources.
        metadata = {
            "uid": event.get("uid", ""),
            "title": event.get("title_fr", ""),
            "city": event.get("location_city", ""),
            "start_date": event.get("firstdate_begin", ""),
            "end_date": event.get("lastdate_end", ""),
            "age_min": event.get("age_min", ""),
            "age_max": event.get("age_max", ""),
            "url": event.get("canonicalurl", ""),
            # Séances réelles (JSON) : un atelier "de septembre à décembre" n'a lieu que certains jours.
            "timings": event.get("timings", ""),
        }
        documents.append(Document(page_content=contenu, metadata=metadata))

    return documents


def create_embeddings():
    """Étape 3 : le modèle Ollama qui transforme un texte en vecteur."""
    return OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)


def build_vector_store(documents, index_path=FAISS_INDEX_PATH):
    """Étape 4 : vectoriser les documents par lots et enregistrer l'index FAISS."""
    if not documents:
        raise ValueError("Aucun document valide à vectoriser.")

    embeddings = create_embeddings()
    index = None

    # Envoyer les 10 000 événements d'un coup peut faire planter Ollama : on procède par lots.
    for debut in range(0, len(documents), TAILLE_LOT):
        lot = documents[debut : debut + TAILLE_LOT]
        index_du_lot = FAISS.from_documents(lot, embeddings)
        if index is None:
            index = index_du_lot
        else:
            index.merge_from(index_du_lot)
        print(f"Embeddings créés : {debut + len(lot)} / {len(documents)}", flush=True)

    index_path.mkdir(parents=True, exist_ok=True)
    index.save_local(str(index_path))
    return index


def main():
    events = load_events()
    documents = create_event_documents(events)
    build_vector_store(documents)
    print(f"{len(documents)} événement(s) indexé(s) dans {FAISS_INDEX_PATH}")


if __name__ == "__main__":
    main()
