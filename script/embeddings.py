"""Prépare les événements comme documents et les indexe dans FAISS."""
#Lancer le script : python -m script.embeddings

import os

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings

from config import CLEANED_DATA_PATH, FAISS_INDEX_PATH

#vectorisation


def load_events(input_path=CLEANED_DATA_PATH):
    """Charge le CSV cleané et le convertit en DataFrame Pandas."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"Fichier absent : {input_path}. Lance d'abord le prétraitement."
        )
    events = pd.read_csv(input_path, encoding="utf-8-sig").fillna("")
    if events.empty:
        raise ValueError(f"Aucun événement à vectoriser dans {input_path}.")
    return events


def create_event_documents(events):
    """Transforme chaque ligne en texte à chercher et métadonnées associées."""
    documents = []
    for _, event in events.iterrows():
        # On met les informations utiles dans le texte transmis au modèle.
        age_min = str(event.get("age_min", "")).strip()
        age_max = str(event.get("age_max", "")).strip()
        age_range = str(event.get("age_range", "")).strip()

        text_parts = [
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
        page_content = "\n".join(
            f"{label} : {value}"
            for label, value in text_parts
            if str(value).strip(" -")
        )


        metadata = {
            "uid": str(event.get("uid", "")),
            "title": str(event.get("title_fr", "")),
            "city": str(event.get("location_city", "")),
            "start_date": str(event.get("firstdate_begin", "")),
            "end_date": str(event.get("lastdate_end", "")),
            "age_min": age_min,
            "age_max": age_max,
            "url": str(event.get("canonicalurl", "")),
        }
        if page_content.strip():
            documents.append(Document(page_content=page_content, metadata=metadata))
    return documents

#embeddings et vectore store 
def build_vector_store(documents, index_path=FAISS_INDEX_PATH):
    """Crée les embeddings Mistral et enregistre l'index de recherche FAISS."""
    if not os.getenv("MISTRAL_API_KEY", "").strip():
        raise EnvironmentError(
            "La variable MISTRAL_API_KEY manque dans le fichier .env."
        )
    if not documents:
        raise ValueError("Aucun document valide à vectoriser.")

    # Le modèle transforme le texte de chaque événement en vecteur numérique.
    embeddings = MistralAIEmbeddings(model="mistral-embed")
    vector_store = FAISS.from_documents(documents, embeddings)

    # FAISS écrit ses fichiers dans le dossier de l'index.
    index_path.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(index_path))
    return vector_store


def main():
    """Lit le CSV et construit l'index vectoriel complet."""
    events = load_events()
    documents = create_event_documents(events)
    build_vector_store(documents)
    print(f"{len(documents)} événement(s) indexé(s) dans {FAISS_INDEX_PATH}")


if __name__ == "__main__":
    main()
