"""Chatbot RAG : recherche dans FAISS puis réponse rédigée par Ollama : python -m script.chatbot"""

import time
from datetime import datetime, timedelta

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings

from config import (
    FAISS_INDEX_PATH,
    OLLAMA_BASE_URL,
    OLLAMA_CHAT_MODEL,
    OLLAMA_EMBEDDING_MODEL,
)
from script.periode import FUSEAU, chevauche, detecter_periode, formater_jour, formater_seance, seances_dans

NOMBRE_EVENEMENTS = 5  # nombre d'événements transmis au LLM (le "k" du top-k)
TEMPERATURE = 0.2  # faible = réponses plus factuelles

INSTRUCTIONS = """
Tu es un assistant qui recommande des événements.
La date du jour est {current_date} (fuseau Europe/Paris).
{period}
Les dates ont DÉJÀ été vérifiées : chaque événement fourni a au moins une séance
pendant la période. N'écarte jamais un événement à cause de ses dates.
Pour chaque suggestion, donne le jour et l'horaire indiqués dans "Séances".
Réponds en français et choisis les événements qui correspondent le mieux au reste
de la demande (lieu, public, type de sortie...).
Utilise uniquement les informations des événements fournis. N'invente pas de détails.
""".strip()

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", INSTRUCTIONS + "\n\nÉvénements trouvés :\n{context}"),
        ("human", "{question}"),
    ]
)


def distance_to_similarity(distance):
    """Convertit la distance FAISS en similarité cosinus (1 = identique, 0 = sans rapport).

    L'index (IndexFlatL2) renvoie la distance euclidienne AU CARRÉ : plus elle est
    petite, plus l'événement est proche. Les vecteurs d'embeddinggemma étant
    normalisés (longueur 1), on a : distance² = 2 - 2 × cosinus.
    """
    return max(0.0, 1 - float(distance) / 2)


def contexte_evenement(document, debut, fin, periode_demandee, max_seances=5):
    """Texte de l'événement + ses séances dans la période, en clair (le LLM n'a pas à lire du JSON)."""
    seances = seances_dans(document.metadata, debut, fin)
    if not seances:
        return document.page_content
    titre = "Séances pendant la période demandée" if periode_demandee else "Prochaines séances"
    lignes = [f"- {formater_seance(d, f)}" for d, f in seances[:max_seances]]
    if len(seances) > max_seances:
        lignes.append(f"- ... et {len(seances) - max_seances} autre(s)")
    # La ligne "Dates : <ISO UTC> - <ISO UTC>" (début et fin de toute la série) fait croire
    # aux petits modèles que l'événement est hors période : on la remplace par les séances.
    texte = "\n".join(
        ligne for ligne in document.page_content.splitlines() if not ligne.startswith("Dates :")
    )
    return f"{texte}\n{titre} :\n" + "\n".join(lignes)


class EventChatbot:
    """Assemble les quatre briques du RAG : index, recherche, LLM, chaîne."""

    def __init__(self, model_name=OLLAMA_CHAT_MODEL, k=NOMBRE_EVENEMENTS, index_path=FAISS_INDEX_PATH):
        if not (index_path / "index.faiss").exists():
            raise FileNotFoundError(
                f"Index FAISS introuvable dans {index_path}. Lance d'abord : python -m script.embeddings"
            )

        # 1. Charger l'index, avec le même modèle d'embedding que celui qui l'a créé.
        embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
        index = FAISS.load_local(
            str(index_path),
            embeddings,
            allow_dangerous_deserialization=True,  # index.pkl est créé par notre propre pipeline
        )

        # 2. La recherche : on garde l'index pour obtenir les k événements les plus
        #    proches de la question ET leur score de similarité.
        self.index = index
        self.k = k

        # 3. Le LLM local qui rédige la réponse.
        llm = ChatOllama(model=model_name, base_url=OLLAMA_BASE_URL, temperature=TEMPERATURE)

        # 4. La chaîne : prompt rempli -> LLM -> texte.
        self.chain = PROMPT | llm | StrOutputParser()

    def ask(self, question):
        """Renvoie la réponse, les événements utilisés et le temps de chaque étape."""
        question = question.strip()
        if not question:
            raise ValueError("Écris une question avant de l'envoyer.")

        # Filtre par date : l'embedding ne comprend pas "ce week-end", on filtre donc
        # sur les dates des métadonnées. Sans période citée, on écarte les événements passés.
        maintenant = datetime.now(FUSEAU)
        periode = detecter_periode(question, maintenant)
        if periode:
            nom, debut_periode, fin_periode, expression = periode
            texte_recherche = question.replace(expression, " ").strip(" ?!.,") or question
            consigne_periode = (
                f'Période demandée ("{nom}") : du {formater_jour(debut_periode)} '
                f"au {formater_jour(fin_periode - timedelta(seconds=1))} inclus."
            )
        else:
            debut_periode, fin_periode = maintenant, datetime.max.replace(tzinfo=FUSEAU)
            texte_recherche = question
            consigne_periode = "Aucune période précise demandée : propose les prochaines séances."

        def filtre_dates(metadata):
            return chevauche(metadata, debut_periode, fin_periode)

        # Retrieval : la question est vectorisée puis comparée à TOUT l'index FAISS
        # (fetch_k = taille de l'index), on ne garde que les k meilleurs dans la période.
        debut = time.perf_counter()
        resultats = self.index.similarity_search_with_score(
            texte_recherche, k=self.k, filter=filtre_dates, fetch_k=self.index.index.ntotal
        )
        documents = [
            Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "score": distance_to_similarity(distance)},
            )
            for doc, distance in resultats
        ]
        temps_recherche = time.perf_counter() - debut

        # Generation : le LLM répond à partir des événements trouvés.
        debut = time.perf_counter()
        answer = self.chain.invoke(
            {
                "question": question,
                "context": "\n\n---\n\n".join(
                    contexte_evenement(doc, debut_periode, fin_periode, bool(periode)) for doc in documents
                ),
                "current_date": formater_jour(maintenant),
                "period": consigne_periode,
            }
        )
        temps_generation = time.perf_counter() - debut

        timings = {
            "search": temps_recherche,
            "generation": temps_generation,
            "total": temps_recherche + temps_generation,
        }
        return answer, documents, timings


def main():
    """Petit chat dans le terminal. Tape 'q' pour quitter."""
    chatbot = EventChatbot()
    print("Pose une question sur les événements. Tape 'q' pour quitter.\n")

    while True:
        question = input("Vous : ").strip()
        if question.lower() in {"q", "quit", "exit"}:
            print("À bientôt !")
            break
        if not question:
            continue

        answer, documents, _ = chatbot.ask(question)
        print(f"\nAssistant : {answer}\n")
        for document in documents:
            print(f"- {document.metadata.get('title', '')} {document.metadata.get('url', '')}")
        print()


if __name__ == "__main__":
    main()
