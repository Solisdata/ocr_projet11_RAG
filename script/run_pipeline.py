"""Lance les trois étapes de préparation et d'indexation des événements."""

from script.api_opendatasoft import fetch_events, save_events
from script.embeddings import build_vector_store, create_event_documents, load_events
from script.preprocessing import preprocess_csv


def main():
    """Récupère les événements, les nettoie, puis crée l'index FAISS."""
    # Première étape : télécharger les événements et sauvegarder le CSV brut.
    print("Étape 1/3 : récupération des événements OpenAgenda")
    events = fetch_events()
    if events.empty:
        raise ValueError("L'API n'a renvoyé aucun événement pour ces filtres.")
    save_events(events)

    # Deuxième étape : nettoyer les données et appliquer les filtres de qualité.
    print("Étape 2/3 : nettoyage des événements")
    preprocess_csv()

    # Troisième étape : créer les vecteurs et les enregistrer dans FAISS.
    print("Étape 3/3 : création de l'index vectoriel")
    cleaned_events = load_events()
    documents = create_event_documents(cleaned_events)
    build_vector_store(documents)

    print(f"Pipeline terminé : {len(documents)} événement(s) indexé(s).")


if __name__ == "__main__":
    main()
