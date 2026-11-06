"""Interface web Streamlit du chatbot de recommandations d'événements."""

# Configure les certificats système avant de charger les clients HTTP de LangChain.
import truststore

truststore.inject_into_ssl()

import streamlit as st

from script.chatbot import EventChatbot
from config import FAISS_INDEX_PATH


st.set_page_config(page_title="Événements à Nantes", page_icon="🎭")
st.title("🎭 Recommandations d’événements")
st.caption("Décris tes envies pour découvrir des événements à Nantes.")


@st.cache_resource(show_spinner="Chargement de l’index FAISS…")
def get_chatbot(k):
    return EventChatbot(index_path=FAISS_INDEX_PATH, k=k)


with st.sidebar:
    st.header("Paramètres")
    number_of_results = st.slider(
        "Événements à consulter", min_value=1, max_value=10, value=5
    )
    st.caption("Le modèle est défini par MISTRAL_CHAT_MODEL (ou mistral-small-latest).")
    st.caption("Chaque question utilise l’API Mistral.")

try:
    chatbot = get_chatbot(number_of_results)
except (EnvironmentError, FileNotFoundError) as error:
    st.error(str(error))
    st.info("Vérifie MISTRAL_API_KEY dans .env et la présence de data/faiss_index.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        for source in message.get("sources", []):
            st.markdown(source)

question = st.chat_input("Ex. Une sortie gratuite ce week-end pour des enfants")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Recherche d’événements et préparation de la réponse…"):
            try:
                answer, documents = chatbot.ask(question)
            except Exception as error:
                st.error(f"La réponse n’a pas pu être générée : {error}")
            else:
                st.markdown(answer)
                sources = []
                for document in documents:
                    title = document.metadata.get("title", "").strip()
                    url = document.metadata.get("url", "").strip()
                    if url:
                        source = f"- [{title or 'Voir l’événement'}]({url})"
                    elif title:
                        source = f"- {title}"
                    else:
                        continue
                    if source not in sources:
                        sources.append(source)
                if sources:
                    with st.expander("Événements consultés"):
                        for source in sources:
                            st.markdown(source)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
