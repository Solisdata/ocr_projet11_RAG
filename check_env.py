"""Vérifie que toutes les bibliothèques du projet s'importent : python check_env.py"""

import sys

print("Python :", sys.version.split()[0])
print("Interpréteur :", sys.executable)

import faiss
import langchain_community
import langchain_ollama
import pandas
import requests
import streamlit
from dotenv import load_dotenv

print("faiss     :", faiss.__version__)
print("pandas    :", pandas.__version__)
print("streamlit :", streamlit.__version__)
print("Tous les imports sont OK")
