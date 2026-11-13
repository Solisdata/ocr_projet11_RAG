"""Local event chatbot: FAISS retrieval plus Ollama chat generation."""

import argparse
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from config import (
    OLLAMA_BASE_URL,
    OLLAMA_CHAT_MODEL,
    OLLAMA_EMBEDDING_MODEL,
    OLLAMA_FAISS_INDEX_PATH,
)


# Main settings to modify.
NUMBER_OF_EVENTS = 5
TEMPERATURE = 0.2

INSTRUCTIONS = """
Tu es un assistant qui recommande des \u00e9v\u00e9nements.
R\u00e9ponds en fran\u00e7ais et adapte tes suggestions \u00e0 la demande.
Utilise uniquement les informations des \u00e9v\u00e9nements fournis.
Si les informations ne suffisent pas, dis-le. N'invente pas de d\u00e9tails.
""".strip()


def create_prompt():
    """Prepare the system instructions, event context, and user question."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", INSTRUCTIONS + "\n\n\u00c9v\u00e9nements trouv\u00e9s :\n{context}"),
            ("human", "{question}"),
        ]
    )


class EventChatbot:
    """Search the local FAISS index and ask a local Ollama model to answer."""

    def __init__(
        self,
        index_path=OLLAMA_FAISS_INDEX_PATH,
        k=NUMBER_OF_EVENTS,
        model_name=OLLAMA_CHAT_MODEL,
    ):
        index_path = Path(index_path).resolve()
        if not (index_path / "index.faiss").is_file() or not (
            index_path / "index.pkl"
        ).is_file():
            raise FileNotFoundError(
                f"Index FAISS Ollama introuvable dans {index_path}. "
                "Lance d'abord : python -m script.embeddings"
            )

        try:
            from langchain_ollama import ChatOllama, OllamaEmbeddings
        except ImportError as error:
            raise RuntimeError(
                "Installe les d\u00e9pendances : python -m pip install -r requirements.txt"
            ) from error

        # The query and indexed documents must use the same embedding model.
        embeddings = OllamaEmbeddings(
            model=OLLAMA_EMBEDDING_MODEL,
            base_url=OLLAMA_BASE_URL,
        )
        try:
            query_dimension = len(embeddings.embed_query("v\u00e9rification"))
        except Exception as error:
            raise RuntimeError(
                "Ollama est inaccessible ou le mod\u00e8le d'embedding est absent. "
                "D\u00e9marre Ollama, puis lance : "
                f"ollama pull {OLLAMA_EMBEDDING_MODEL}"
            ) from error

        # Keep the existing Mistral index untouched; this branch uses its own.
        self.index = FAISS.load_local(
            str(index_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        if self.index.index.d != query_dimension:
            raise RuntimeError(
                "L'index n'utilise pas le mod\u00e8le d'embedding Ollama configur\u00e9. "
                "Reconstruis-le avec : python -m script.embeddings"
            )

        self.retriever = self.index.as_retriever(search_kwargs={"k": k})
        try:
            self.model = ChatOllama(
                model=model_name,
                base_url=OLLAMA_BASE_URL,
                temperature=TEMPERATURE,
                validate_model_on_init=True,
            )
        except Exception as error:
            raise RuntimeError(
                "Ollama est inaccessible ou le mod\u00e8le de chat est absent. "
                "D\u00e9marre Ollama, puis lance : "
                f"ollama pull {model_name}"
            ) from error

        self.chain = create_prompt() | self.model | StrOutputParser()

    def ask(self, question):
        """Return the answer and the FAISS documents used as sources."""
        question = question.strip()
        if not question:
            raise ValueError("\u00c9cris une question avant de l'envoyer.")

        answer, documents, _ = self.ask_with_timings(question)
        return answer, documents

    def ask_with_timings(self, question):
        """Return the answer, sources, and time spent in each main step."""
        question = question.strip()
        if not question:
            raise ValueError("Écris une question avant de l'envoyer.")

        # Includes converting the question to an embedding and searching FAISS.
        search_started_at = time.perf_counter()
        documents = self.retriever.invoke(question)
        search_time = time.perf_counter() - search_started_at

        context = "\n\n---\n\n".join(doc.page_content for doc in documents)
        generation_started_at = time.perf_counter()
        answer = self.chain.invoke({"question": question, "context": context})
        generation_time = time.perf_counter() - generation_started_at

        timings = {
            "search": search_time,
            "generation": generation_time,
            "total": search_time + generation_time,
        }
        return answer, documents, timings


def main():
    parser = argparse.ArgumentParser(
        description="Chatbot local de recommandations d'\u00e9v\u00e9nements"
    )
    parser.add_argument("--index-path", default=str(OLLAMA_FAISS_INDEX_PATH))
    parser.add_argument("--k", type=int, default=NUMBER_OF_EVENTS)
    parser.add_argument("--model", default=OLLAMA_CHAT_MODEL)
    args = parser.parse_args()

    if args.k < 1:
        parser.error("--k doit \u00eatre sup\u00e9rieur ou \u00e9gal \u00e0 1.")

    try:
        chatbot = EventChatbot(args.index_path, k=args.k, model_name=args.model)
    except (FileNotFoundError, RuntimeError) as error:
        parser.exit(1, f"Erreur : {error}\n")

    print("Pose une question sur les \u00e9v\u00e9nements. Tape 'q' pour quitter.\n")
    while True:
        try:
            question = input("Vous : ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\u00c0 bient\u00f4t !")
            break

        if question.lower() in {"q", "quit", "exit"}:
            print("\u00c0 bient\u00f4t !")
            break
        if not question:
            continue

        try:
            answer, documents = chatbot.ask(question)
            print(f"\nAssistant : {answer}")
            for document in documents:
                title = document.metadata.get("title", "").strip()
                url = document.metadata.get("url", "").strip()
                if title or url:
                    print(f"- {title} {url}".strip())
            print()
        except Exception as error:
            print(f"Erreur : {error}\n")


if __name__ == "__main__":
    main()
