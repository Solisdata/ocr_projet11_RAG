# Projet 11 — Recherche d’événements à Nantes

Ce projet récupère des événements depuis OpenAgenda, nettoie les données et crée un index vectoriel pour pouvoir les rechercher par sens.

## Fonctionnement

Le pipeline suit trois étapes :

1. `script/api_opendatasoft.py` récupère les événements et les enregistre dans `data/evenements_nantes.csv`.
2. `script/preprocessing.py` nettoie les champs, normalise les âges, crée le libellé `age_range` et filtre les événements invalides, trop anciens ou hors de Nantes. Le résultat est enregistré dans `data/evenements_nantes_clean.csv`.
3. `script/embeddings.py` transforme chaque ligne nettoyée en vecteur avec Ollama, puis enregistre un index FAISS local.

Chaque ligne du CSV représente un événement et devient un document. Les métadonnées du document gardent notamment l’identifiant, le titre, les dates, l’âge et l’URL de l’événement.

## Lancer le projet

Créer un environnement virtuel et installer les dépendances :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Installer Ollama pour Windows depuis https://ollama.com/download, puis télécharger les modèles :

```powershell
ollama pull embeddinggemma
ollama pull qwen3:4b
```

EmbeddingGemma vectorise les événements et les questions. Qwen3 rédige les réponses. Ces valeurs peuvent être changées dans `.env` :

```text
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=embeddinggemma
OLLAMA_CHAT_MODEL=qwen3:4b
```

Lancer toutes les étapes :

```powershell
python -m script.run_pipeline
```

Pour exécuter une étape seule :

```powershell
python -m script.api_opendatasoft
python -m script.preprocessing
python -m script.embeddings
```

Pour inspecter l'index FAISS local (nombre de vecteurs, dimension, métadonnées et
contenu de quelques documents), sans contacter Ollama :

```powershell
python -m script.read_index
python -m script.read_index --limit 10
```

Le script lit `data/faiss_index_ollama/` par défaut. `index.pkl` est chargé avec
`pickle` par LangChain : n'inspecte que les index créés par ce projet ou dont
la provenance est fiable.

## Chatbot de recommandations

Après avoir installé Ollama et téléchargé les modèles, reconstruire l'index avec les embeddings Ollama :

```powershell
python -m script.embeddings
```

L'index Ollama est enregistré dans `data/faiss_index_ollama/`. L'ancien index `data/faiss_index/` reste intact. Un index créé avec Mistral ne peut pas servir avec les embeddings Ollama : il faut le reconstruire.

Lancer le chatbot en terminal ou dans Streamlit :

```powershell
python -m script.chatbot
python -m streamlit run app.py
```

Le chatbot recherche les événements dans FAISS puis envoie le contexte au modèle Ollama local. La barre latérale Streamlit permet de régler le nombre de résultats et le modèle de chat.

La pipeline complète récupère toujours les événements depuis OpenDataSoft. La création des embeddings et les réponses tournent localement avec Ollama :

```powershell
python -m script.run_pipeline
```

## Tests

Lancer les tests locaux :

```powershell
python -m pytest tests/ -v
```

Les tests d'embeddings utilisent un faux encodeur et ne nécessitent pas qu'Ollama soit démarré. Pour tester les embeddings réels, démarrer Ollama et télécharger `embeddinggemma`, puis lancer dans PowerShell :

```powershell
$env:RUN_OLLAMA_INTEGRATION = "1"
python -m pytest tests/test_embeddings.py -v
```
