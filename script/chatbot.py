"""Chatbot RAG : recherche dans FAISS puis réponse rédigée par Ollama : python -m script.chatbot"""

import re
import time
from datetime import datetime, timedelta

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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
MAX_TOKENS_REPONSE = 500  # garde-fou : le temps de réponse est proportionnel au nombre de mots générés

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
Commence toujours par UNE courte phrase d'introduction, écrite avec tes propres mots
et adaptée à la question, avant la liste.
Puis sois concis : 3 suggestions au maximum, 2 ou 3 lignes chacune (nom, jour et horaire,
lieu, prix si connu), sans conclusion.
""".strip()

# Rappelé juste avant la question quand il y a un historique : un petit modèle suit
# mieux une consigne placée à côté de la question qu'en haut du prompt.
CONSIGNE_RELANCE = (
    "(Question de suivi sur les événements que tu viens de proposer. Réponds précisément "
    "à cette question pour ces événements, sans recopier ta réponse précédente.)\n"
)

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", INSTRUCTIONS + "\n\nÉvénements trouvés :\n{context}"),
        MessagesPlaceholder("history"),
        ("human", "{relance}{question}"),
    ]
)

NOMBRE_MESSAGES_MEMOIRE = 4  # mémoire courte : les 2 derniers échanges (question + réponse)


def preparer_historique(historique):
    """Garde les derniers messages de la conversation au format attendu par LangChain."""
    return [
        ("human" if message["role"] == "user" else "ai", message["content"])
        for message in (historique or [])[-NOMBRE_MESSAGES_MEMOIRE:]
    ]


# Une question de suivi est courte et renvoie à la réponse précédente ("c'est payant ?",
# "et le premier ?", "à quelle heure ?"). Sinon, c'est une nouvelle demande.
MOTS_RELANCE = re.compile(
    r"^(et|où|quand|combien|comment|à quelle|a quelle|quel(le)? est)\b"
    r"|\b(c'est|ça|cela|ceci|il|elle|ils|elles|y|celui|celle|ceux|lequel|laquelle"
    r"|premier|première|deuxième|second|seconde|troisième|dernier|dernière)\b"
)
MAX_MOTS_RELANCE = 10


def est_relance(question, historique):
    """True si la question prolonge l'échange précédent, False si c'est une nouvelle demande."""
    if not historique:
        return False
    texte = question.lower().replace("’", "'")
    return len(texte.split()) <= MAX_MOTS_RELANCE and bool(MOTS_RELANCE.search(texte))


def derniere_question(historique):
    """Dernière question posée par l'utilisateur avant celle-ci (ou None)."""
    for message in reversed(historique or []):
        if message["role"] == "user":
            return message["content"]
    return None


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
        #    reasoning=False coupe le mode "thinking" de qwen3 : sinon le modèle rédige
        #    d'abord un long raisonnement caché (x4 à x5 sur le temps de réponse).
        llm = ChatOllama(
            model=model_name,
            base_url=OLLAMA_BASE_URL,
            temperature=TEMPERATURE,
            reasoning=False,
            num_predict=MAX_TOKENS_REPONSE,
        )

        # 4. La chaîne : prompt rempli -> LLM -> texte.
        self.chain = PROMPT | llm | StrOutputParser()

    def ask(self, question, maintenant=None, historique=None):
        """Renvoie la réponse, les événements utilisés et le temps de chaque étape.

        maintenant : date simulée de la question (utile pour l'évaluation) ; par défaut, l'heure actuelle.
        historique : messages précédents [{"role": "user"|"assistant", "content": ...}].
        """
        entrees, documents, temps_recherche = self.retrieve(question, maintenant, historique)

        # Generation : le LLM répond à partir des événements trouvés.
        debut = time.perf_counter()
        answer = self.chain.invoke(entrees)
        temps_generation = time.perf_counter() - debut

        timings = {
            "search": temps_recherche,
            "generation": temps_generation,
            "total": temps_recherche + temps_generation,
        }
        return answer, documents, timings

    def ask_stream(self, question, historique=None):
        """Comme ask, mais la réponse arrive morceau par morceau (affichage au fil de l'eau).

        Renvoie (flux de texte, événements utilisés, temps de recherche).
        """
        entrees, documents, temps_recherche = self.retrieve(question, historique=historique)
        return self.chain.stream(entrees), documents, temps_recherche

    def retrieve(self, question, maintenant=None, historique=None):
        """Recherche les événements et prépare les entrées du prompt."""
        question = question.strip()
        if not question:
            raise ValueError("Écris une question avant de l'envoyer.")

        # Mémoire pour la recherche : une relance comme "c'est payant ?" ne dit rien du
        # sujet. On cherche donc avec la question précédente + la nouvelle, ce qui
        # retrouve les mêmes événements (et la même période) sans appel LLM en plus.
        # Nouvelle demande (autre type d'événement) : on repart de zéro, sans historique.
        if not est_relance(question, historique):
            historique = None
        precedente = derniere_question(historique)
        texte_complet = f"{precedente} {question}" if precedente else question

        # Filtre par date : l'embedding ne comprend pas "ce week-end", on filtre donc
        # sur les dates des métadonnées. Sans période citée, on écarte les événements passés.
        maintenant = maintenant or datetime.now(FUSEAU)
        periode = detecter_periode(question, maintenant) or detecter_periode(texte_complet, maintenant)
        if periode:
            nom, debut_periode, fin_periode, expression = periode
            texte_recherche = texte_complet.replace(expression, " ").strip(" ?!.,") or texte_complet
            consigne_periode = (
                f'Période demandée ("{nom}") : du {formater_jour(debut_periode)} '
                f"au {formater_jour(fin_periode - timedelta(seconds=1))} inclus."
            )
        else:
            debut_periode, fin_periode = maintenant, datetime.max.replace(tzinfo=FUSEAU)
            texte_recherche = texte_complet
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

        entrees = {
            "question": question,
            "history": preparer_historique(historique),
            "relance": CONSIGNE_RELANCE if precedente else "",
            "context": "\n\n---\n\n".join(
                contexte_evenement(doc, debut_periode, fin_periode, bool(periode)) for doc in documents
            ),
            "current_date": formater_jour(maintenant),
            "period": consigne_periode,
        }
        return entrees, documents, temps_recherche


def main():
    """Petit chat dans le terminal. Tape 'q' pour quitter."""
    chatbot = EventChatbot()
    print("Pose une question sur les événements. Tape 'q' pour quitter.\n")
    historique = []

    while True:
        question = input("Vous : ").strip()
        if question.lower() in {"q", "quit", "exit"}:
            print("À bientôt !")
            break
        if not question:
            continue

        answer, documents, _ = chatbot.ask(question, historique=historique)
        historique += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
        print(f"\nAssistant : {answer}\n")
        for document in documents:
            print(f"- {document.metadata.get('title', '')} {document.metadata.get('url', '')}")
        print()


if __name__ == "__main__":
    main()
