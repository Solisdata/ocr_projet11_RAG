import sys

print("Python :", sys.version.split()[0])
print("Interpréteur :", sys.executable)

import langchain
import langchain_core
import langchain_community
import langchain_text_splitters
import langchain_mistralai
import mistralai
import faiss
import pandas
import requests
import dotenv
import pytest

from langchain_community.vectorstores import FAISS
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

print("langchain :", langchain.__version__)
print("faiss     :", faiss.__version__)
print("pandas    :", pandas.__version__)
print("Tous les imports sont OK")