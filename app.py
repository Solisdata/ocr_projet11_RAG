"""Interface web du chatbot : python -m streamlit run app.py"""

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


def show_message(message):
    """Affiche un message du chat, avec ses sources et ses temps s'il en a."""
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Événements consultés"):
                for source in message["sources"]:
                    st.markdown(source)
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
    user_message = {"role": "user", "content": question}
    st.session_state.messages.append(user_message)
    show_message(user_message)

    with st.spinner("Recherche d'événements et rédaction de la réponse..."):
        try:
            answer, documents, timings = chatbot.ask(question)
        except Exception as error:
            st.error(f"La réponse n'a pas pu être générée : {error}")
            st.stop()

    assistant_message = {
        "role": "assistant",
        "content": answer,
        "sources": format_sources(documents),
        "timings": timings,
    }
    st.session_state.messages.append(assistant_message)
    show_message(assistant_message)
