"""Chatbot d'événements : recherche dans FAISS, puis réponse avec Mistral."""

import argparse
import os
from pathlib import Path
import sys

# Sur Windows, utilise les certificats racines installés dans le système.
# Cette injection conserve la vérification TLS et doit précéder les clients HTTP.
import truststore

truststore.inject_into_ssl()

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

# Permet de lancer ce fichier directement avec : python script/chatbot.py
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import FAISS_INDEX_PATH


# -----------------------------------------------------------------------------
# PARAMÈTRES MODIFIABLES
# Tu peux changer ces valeurs pour ajuster le comportement du chatbot.
# -----------------------------------------------------------------------------
NOMBRE_EVENEMENTS = 5  # Nombre de résultats FAISS fournis au modèle.
MODELE_MISTRAL = os.getenv("MISTRAL_CHAT_MODEL", "mistral-small-latest")
MODELE_EMBEDDING = "mistral-embed"  # Doit rester le même que pour créer l'index.
TEMPERATURE = 0.2  # Bas = réponses plus stables ; haut = réponses plus variées.

# Ces instructions disent au modèle comment utiliser les événements retrouvés.
INSTRUCTIONS = """
Tu es un assistant qui recommande des événements.
Réponds en français et adapte tes suggestions à la demande.
Utilise uniquement les informations des événements fournis.
Si les informations ne suffisent pas, dis-le. N'invente pas de détails.
""".strip()


def creer_prompt():
    """Prépare le format du message envoyé à Mistral."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", INSTRUCTIONS + "\n\nÉvénements trouvés :\n{contexte}"),
            ("human", "{question}"),
        ]
    )


class EventChatbot:
    """Relie la recherche FAISS au modèle Mistral."""

    def __init__(
        self,
        index_path=FAISS_INDEX_PATH,
        k=NOMBRE_EVENEMENTS,
        model_name=MODELE_MISTRAL,
    ):
        # 1. Vérifie les éléments nécessaires avant de démarrer.
        if not os.getenv("MISTRAL_API_KEY", "").strip():
            raise EnvironmentError("Ajoute MISTRAL_API_KEY dans le fichier .env.")

        index_path = Path(index_path).resolve()
        if not (index_path / "index.faiss").is_file() or not (
            index_path / "index.pkl"
        ).is_file():
            raise FileNotFoundError(
                f"Index FAISS introuvable dans {index_path}. "
                "Lance d'abord python -m script.embeddings."
            )

        # 2. Charge le modèle d'embeddings et l'index déjà créé.
        embeddings = MistralAIEmbeddings(model=MODELE_EMBEDDING)
        self.index = FAISS.load_local(
            str(index_path),
            embeddings,
            # L'index local du projet contient un fichier pickle.
            allow_dangerous_deserialization=True,
        )

        # 3. Prépare la recherche : k est le nombre d'événements à récupérer.
        self.retriever = self.index.as_retriever(search_kwargs={"k": k})

        # 4. Prépare Mistral et la chaîne LangChain prompt -> modèle -> texte.
        modele = ChatMistralAI(model_name=model_name, temperature=TEMPERATURE)
        parser = StrOutputParser()
        self.chaine = creer_prompt() | modele | parser

    def ask(self, question):
        """Cherche des événements et retourne (réponse, documents trouvés)."""
        question = question.strip()
        if not question:
            raise ValueError("Écris une question avant de l'envoyer.")

        # 5. Recherche dans FAISS avec le texte de la question.
        documents = self.retriever.invoke(question)

        # 6. Assemble les événements et les donne au modèle avec la question.
        contexte = "\n\n---\n\n".join(doc.page_content for doc in documents)
        reponse = self.chaine.invoke({"question": question, "contexte": contexte})
        return reponse, documents


def main():
    """Mode terminal facultatif ; l'interface principale est app.py (Streamlit)."""
    parser = argparse.ArgumentParser(description="Chatbot de recommandations d'événements")
    parser.add_argument("--index-path", default=str(FAISS_INDEX_PATH))
    parser.add_argument("--k", type=int, default=NOMBRE_EVENEMENTS)
    parser.add_argument("--model", default=MODELE_MISTRAL)
    args = parser.parse_args()

    if args.k < 1:
        parser.error("--k doit être supérieur ou égal à 1.")

    try:
        chatbot = EventChatbot(args.index_path, k=args.k, model_name=args.model)
    except (EnvironmentError, FileNotFoundError) as erreur:
        parser.exit(1, f"Erreur : {erreur}\n")

    print("Pose une question sur les événements. Tape 'q' pour quitter.\n")
    while True:
        try:
            question = input("Vous : ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nÀ bientôt !")
            break

        if question.lower() in {"q", "quit", "exit"}:
            print("À bientôt !")
            break
        if not question:
            continue

        try:
            reponse, documents = chatbot.ask(question)
            print(f"\nAssistant : {reponse}")
            for document in documents:
                titre = document.metadata.get("title", "").strip()
                url = document.metadata.get("url", "").strip()
                if titre or url:
                    print(f"- {titre} {url}".strip())
            print()
        except Exception as erreur:
            print(f"Erreur : {erreur}\n")


if __name__ == "__main__":
    main()
