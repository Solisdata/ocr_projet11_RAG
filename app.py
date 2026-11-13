"""Interface web du chatbot : python -m streamlit run app.py"""

import time

import streamlit as st

from config import OLLAMA_CHAT_MODEL, OLLAMA_EMBEDDING_MODEL
from script.chatbot import EventChatbot

st.set_page_config(page_title="Événements à Nantes", page_icon="🎭")
st.title("Événements à Nantes")
st.caption("Pose une question, je cherche les événements qui correspondent.")


@st.cache_resource(show_spinner="Chargement de l'index FAISS et du modèle...")
def get_chatbot(model_name):
    """Crée le chatbot une seule fois, puis le réutilise à chaque question."""
    return EventChatbot(model_name=model_name)


def format_sources(documents):
    """Liste les événements utilisés, avec leur score de similarité et un lien, sans doublon."""
    sources = []
    for document in documents:
        title = document.metadata.get("title", "") or "Voir l'événement"
        url = document.metadata.get("url", "")
        score = document.metadata.get("score")
        source = f"- [{title}]({url})" if url else f"- {title}"
        if score is not None:
            source += f" — similarité : **{score:.2f}**"
        if source not in sources:
            sources.append(source)
    return sources


def format_timings(timings):
    return (
        f"Recherche (embedding + FAISS) : {timings['search']:.2f} s · "
        f"Réponse Ollama : {timings['generation']:.2f} s · "
        f"Total : {timings['total']:.2f} s"
    )


DEFINITION_SIMILARITE = (
    "**Similarité** : proximité de sens entre ta question et l'événement, de 0 (sans rapport) "
    "à 1 (identique). C'est la similarité cosinus entre leurs deux vecteurs d'embedding. "
    "Elle sert surtout à comparer les événements entre eux."
)


def show_sources(sources):
    """Liste dépliable des événements consultés, suivie de la définition du score."""
    with st.expander("Événements consultés"):
        for source in sources:
            st.markdown(source)
        st.caption(DEFINITION_SIMILARITE)


def show_message(message):
    """Affiche un message du chat, avec ses sources et ses temps s'il en a."""
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            show_sources(message["sources"])
        if message.get("timings"):
            st.caption(format_timings(message["timings"]))


# --- Barre latérale : choix du modèle ---
with st.sidebar:
    st.header("Paramètres")
    model_name = st.text_input("Modèle de chat Ollama", value=OLLAMA_CHAT_MODEL)
    st.caption("Les embeddings et le modèle de chat tournent en local avec Ollama.")
    st.code(f"ollama pull {OLLAMA_EMBEDDING_MODEL}\nollama pull {model_name}", language="powershell")

try:
    chatbot = get_chatbot(model_name)
except Exception as error:
    st.error(str(error))
    st.info(
        "Vérifie qu'Ollama est démarré, que les modèles sont téléchargés "
        "et que l'index a été créé avec : python -m script.embeddings"
    )
    st.stop()

# --- Historique de la conversation ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    show_message(message)

# --- Nouvelle question ---
question = st.chat_input("Ex. Une sortie gratuite ce week-end pour des enfants")
if question:
    # Mémoire : les échanges précédents (texte seul), avant d'ajouter la nouvelle question.
    historique = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    user_message = {"role": "user", "content": question}
    st.session_state.messages.append(user_message)
    show_message(user_message)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Recherche d'événements..."):
                flux, documents, temps_recherche = chatbot.ask_stream(question, historique=historique)
            # La réponse s'affiche mot à mot dès que le modèle commence à écrire.
            debut = time.perf_counter()
            answer = st.write_stream(flux)
            temps_generation = time.perf_counter() - debut
        except Exception as error:
            st.error(f"La réponse n'a pas pu être générée : {error}")
            st.stop()

        assistant_message = {
            "role": "assistant",
            "content": answer,
            "sources": format_sources(documents),
            "timings": {
                "search": temps_recherche,
                "generation": temps_generation,
                "total": temps_recherche + temps_generation,
            },
        }
        if assistant_message["sources"]:
            show_sources(assistant_message["sources"])
        st.caption(format_timings(assistant_message["timings"]))
    st.session_state.messages.append(assistant_message)
