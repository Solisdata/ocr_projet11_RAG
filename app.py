"""Streamlit interface for the local Ollama event chatbot."""

import time

import streamlit as st

from config import OLLAMA_CHAT_MODEL, OLLAMA_FAISS_INDEX_PATH
from script.chatbot import EventChatbot


st.set_page_config(page_title="\u00c9v\u00e9nements \u00e0 Nantes", page_icon="\U0001f3ad")
st.title("Evénements à Nantes")
st.caption("Je recherche un évenement.")


@st.cache_resource(show_spinner="Chargement du mod\u00e8le local et de FAISS...")
def get_chatbot(k, model_name):
    return EventChatbot(
        index_path=OLLAMA_FAISS_INDEX_PATH,
        k=k,
        model_name=model_name,
    )


with st.sidebar:
    st.header("Param\u00e8tres")
    number_of_results = st.slider(
        "\u00c9v\u00e9nements \u00e0 consulter", min_value=1, max_value=10, value=5
    )
    model_name = st.text_input("Mod\u00e8le de chat Ollama", value=OLLAMA_CHAT_MODEL)
    st.caption("Le mod\u00e8le de chat et les embeddings tournent localement avec Ollama.")
    st.code("ollama pull embeddinggemma\nollama pull " + model_name, language="powershell")
    st.caption("Autres mod\u00e8les à tester dans un terminal :")
    st.code(
        "ollama run qwen3:8b  # polyvalent, excellent en multilingue\n"
        "ollama run gpt-oss:20b  # raisonnement, si 16 Go de RAM/VRAM\n"
        "ollama run mistral-small3.2  # option Mistral, open-weight",
        language="powershell",
    )

try:
    chatbot = get_chatbot(number_of_results, model_name)
except (FileNotFoundError, RuntimeError) as error:
    st.error(str(error))
    st.info(
        "V\u00e9rifie qu'Ollama est d\u00e9marr\u00e9, que les mod\u00e8les sont t\u00e9l\u00e9charg\u00e9s "
        "et que l'index a \u00e9t\u00e9 cr\u00e9\u00e9 avec python -m script.embeddings."
    )
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        for source in message.get("sources", []):
            st.markdown(source)
        if "search_time" in message and "generation_time" in message:
            st.caption(
                f"Recherche (embedding + FAISS) : {message['search_time']:.2f} s · "
                f"Réponse Ollama : {message['generation_time']:.2f} s · "
                f"Total : {message['processing_time']:.2f} s"
            )
        elif "processing_time" in message:
            st.caption(f"Temps de traitement : {message['processing_time']:.2f} s")

question = st.chat_input("Ex. Une sortie gratuite ce week-end pour des enfants")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Recherche d'\u00e9v\u00e9nements et pr\u00e9paration de la r\u00e9ponse..."):
            started_at = time.perf_counter()
            try:
                answer, documents, timings = chatbot.ask_with_timings(question)
            except Exception as error:
                processing_time = time.perf_counter() - started_at
                st.error(f"La r?ponse n'a pas pu ?tre g?n?r?e : {error}")
                st.caption(f"Temps avant l'erreur : {processing_time:.2f} s")
            else:
                processing_time = time.perf_counter() - started_at
                st.markdown(answer)
                sources = []
                for document in documents:
                    title = document.metadata.get("title", "").strip()
                    url = document.metadata.get("url", "").strip()
                    if url:
                        source = f"- [{title or 'Voir l\u2019\u00e9v\u00e9nement'}]({url})"
                    elif title:
                        source = f"- {title}"
                    else:
                        continue
                    if source not in sources:
                        sources.append(source)
                if sources:
                    with st.expander("\u00c9v\u00e9nements consult\u00e9s"):
                        for source in sources:
                            st.markdown(source)
                st.caption(
                    f"Recherche (embedding + FAISS) : {timings['search']:.2f} s ? "
                    f"R?ponse Ollama : {timings['generation']:.2f} s ? "
                    f"Total : {processing_time:.2f} s"
                )
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "processing_time": processing_time,
                        "search_time": timings["search"],
                        "generation_time": timings["generation"],
                    }
                )
